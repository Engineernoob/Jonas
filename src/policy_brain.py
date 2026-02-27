# src/policy_brain.py
import os
from dataclasses import dataclass
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

CHECKPOINT = "data/policy.pt"
DEVICE = "mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu")

@dataclass
class Policy:
    temperature: float  # ~[0.45..0.70] after bounds
    verbosity: int      # 0 short, 1 normal, 2 long
    humor: float        # 0..1
    recall_k: int       # 0..5

class SmallMLP(nn.Module):
    """
    Accepts ANY embedding dim via a projection layer -> 512, then MLP.
    Prevents shape mismatch when swapping embedding models.
    """
    def __init__(self, inp_dim: int, proj_dim: int = 512, hidden: int = 512):
        super().__init__()
        self.proj = nn.Linear(inp_dim, proj_dim)
        self.fc1 = nn.Linear(proj_dim, hidden)
        self.fc2 = nn.Linear(hidden, hidden)
        self.out = nn.Linear(hidden, 64)

        nn.init.xavier_uniform_(self.proj.weight)
        nn.init.kaiming_normal_(self.fc1.weight)
        nn.init.kaiming_normal_(self.fc2.weight)
        nn.init.xavier_uniform_(self.out.weight)

    def forward(self, x):
        x = F.relu(self.proj(x))
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.out(x)

class PolicyBrain:
    def __init__(self, emb_dim: int, ckpt_path: str = CHECKPOINT):
        self.device = torch.device(DEVICE)
        self.model = SmallMLP(inp_dim=emb_dim).to(self.device)
        self.opt = torch.optim.Adam(self.model.parameters(), lr=1e-3)
        self.ckpt_path = ckpt_path

        self.last_features = None
        self.last_policy: Policy | None = None  # for smoothing

        if os.path.exists(ckpt_path):
            try:
                self.model.load_state_dict(torch.load(ckpt_path, map_location=self.device))
            except Exception:
                # checkpoint may not match if emb_dim changed; ignore
                pass

    def _to_tensor(self, arr: np.ndarray) -> torch.Tensor:
        t = torch.from_numpy(arr).float().to(self.device)
        if t.ndim == 1:
            t = t.unsqueeze(0)
        return t

    @staticmethod
    def _sigmoid(x: float) -> float:
        return float(1 / (1 + np.exp(-x)))

    @staticmethod
    def _clamp(v: float, lo: float, hi: float) -> float:
        return max(lo, min(hi, v))

    def predict(self, emb: np.ndarray) -> Policy:
        x = self._to_tensor(emb)
        with torch.no_grad():
            out = self.model(x).squeeze(0).detach().cpu().numpy()

        self.last_features = emb

        # ---- Decode heads (simple + stable) ----
        # Temperature: keep narrow to avoid drama (especially on 1B models)
        temp_raw = self._sigmoid(out[3])
        temperature = 0.45 + 0.25 * temp_raw  # 0.45..0.70

        # Verbosity: categorical
        verbosity = int(np.argmax(out[4:7]) % 3)

        # Humor: 0..1, but we cap it a bit to prevent corny “performative” tone
        humor = self._sigmoid(out[7])
        humor = self._clamp(humor, 0.0, 0.65)

        # Recall: 0..5
        recall_k = int(round(self._sigmoid(out[8]) * 5))

        pol = Policy(
            temperature=float(temperature),
            verbosity=int(verbosity),
            humor=float(humor),
            recall_k=int(recall_k),
        )

        # ---- Smoothing (keeps Jonas consistent) ----
        if self.last_policy is not None:
            # Smooth temperature heavily
            pol.temperature = 0.8 * self.last_policy.temperature + 0.2 * pol.temperature

            # Keep verbosity stable unless it’s a big user message (handled in main.py)
            pol.verbosity = self.last_policy.verbosity

            # Smooth humor
            pol.humor = 0.85 * self.last_policy.humor + 0.15 * pol.humor

            # Recall can vary but keep it from swinging wildly
            pol.recall_k = int(round(0.7 * self.last_policy.recall_k + 0.3 * pol.recall_k))

        self.last_policy = pol
        return pol

    def update_from_rating(self, rating: float) -> bool:
        """
        Online learning: train a simple quality signal to match rating.
        This nudges the network without turning the system unstable.
        """
        if self.last_features is None:
            return False

        x = self._to_tensor(self.last_features)
        self.model.train()
        out = self.model(x)  # (1,64)

        quality = torch.sigmoid(out[:, 16:32].mean(dim=1))
        target = torch.tensor([float(rating)], dtype=quality.dtype, device=self.device)
        loss = F.mse_loss(quality, target)

        self.opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
        self.opt.step()

        try:
            os.makedirs(os.path.dirname(self.ckpt_path), exist_ok=True)
            torch.save(self.model.state_dict(), self.ckpt_path)
        except Exception:
            pass

        return True