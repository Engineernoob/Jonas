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
    temperature: float  # [0.2..1.0]
    verbosity: int      # 0 short,1 normal,2 long
    humor: float        # 0..1
    recall_k: int       # 0..5

class SmallMLP(nn.Module):
    """
    Accepts ANY embedding dim via a projection layer -> 512, then MLP.
    This prevents shape mismatch when you swap embedding models.
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
    PRESETS = ["operator", "brother", "executive"]

    def __init__(self, emb_dim: int, ckpt_path: str = CHECKPOINT):
        self.device = torch.device(DEVICE)
        self.model = SmallMLP(inp_dim=emb_dim).to(self.device)
        self.opt = torch.optim.Adam(self.model.parameters(), lr=1e-3)
        self.ckpt_path = ckpt_path

        self.last_features = None

        if os.path.exists(ckpt_path):
            try:
                self.model.load_state_dict(torch.load(ckpt_path, map_location=self.device))
            except Exception:
                # if checkpoint was created with a different emb_dim, ignore it
                pass

    def _to_tensor(self, arr: np.ndarray) -> torch.Tensor:
        t = torch.from_numpy(arr).float().to(self.device)
        if t.ndim == 1:
            t = t.unsqueeze(0)
        return t

    def predict(self, emb: np.ndarray) -> Policy:
        x = self._to_tensor(emb)
        with torch.no_grad():
            out = self.model(x).squeeze(0).detach().cpu().numpy()

        self.last_features = emb

        preset_idx = int(np.argmax(out[0:3]) % 3)
        preset = self.PRESETS[preset_idx]

        temp = 0.35 + 0.4 * (1 / (1 + np.exp(-out[3])))
        verbosity = int(np.argmax(out[4:7]) % 3)
        humor = float(1 / (1 + np.exp(-out[7])))
        recall_k = int(round((1 / (1 + np.exp(-out[8]))) * 5))

        return Policy(
            preset=preset,
            temperature=float(temp),
            verbosity=int(verbosity),
            humor=float(humor),
            recall_k=int(recall_k),
        )

    def update_from_rating(self, rating: float) -> bool:
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