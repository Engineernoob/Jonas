# src/main.py
"""
Jonas CLI: Local-only, minimal loop.
Commands:
  /quit or /exit
  /reset    -> clears DB
  /rate X   -> rate last response 0.0..1.0 (trains policy brain)
  /history  -> prints recent turns
"""
import os
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from .ollama_client import ollama_chat
from .encoder import text_to_embedding
from .policy_brain import PolicyBrain
from .persona import build_system_prompt
from .memory import add_turn, recent_turns, recall_similar, _conn

console = Console()
MODEL = os.getenv("JONAS_MODEL", "llama3.2:1b")
EMBED_MODEL = os.getenv("JONAS_EMBED", "nomic-embed-text")

def pretty_recent(limit=10):
    rows = recent_turns(limit)
    t = Table(show_header=True, header_style="bold magenta")
    t.add_column("role")
    t.add_column("text")
    for r in rows:
        t.add_row(r["role"], r["text"][:200])
    console.print(t)

def main():
    console.print(Panel("[bold]Jonas v0 — local[/bold]\nType /quit to exit, /rate 0.8 to teach him.", title="Jonas"))

    # Probe embedding dimension once so the policy brain matches your embed model
    probe = text_to_embedding("probe", model=EMBED_MODEL)
    brain = PolicyBrain(emb_dim=int(probe.shape[0]))

    while True:
        user = console.input("[cyan]You[/cyan]: ").strip()
        if not user:
            continue
        if user in ("/quit", "/exit"):
            console.print("Bye.")
            break
        if user == "/reset":
            # clear DB
            cur = _conn.cursor()
            cur.execute("DELETE FROM turns")
            _conn.commit()
            console.print("[yellow]Cleared conversation history.[/yellow]")
            continue
        if user.startswith("/history"):
            pretty_recent(20)
            continue
        if user.startswith("/rate"):
            parts = user.split()
            if len(parts) < 2:
                console.print("[red]Usage: /rate <0.0-1.0>[/red]")
                continue
            try:
                val = float(parts[1])
                val = max(0.0, min(1.0, val))
                trained = brain.update_from_rating(val)
                console.print(Panel(f"Trained: {trained} (rating={val})", title="Rate"))
            except Exception as e:
                console.print("[red]Invalid rating.[/red]", e)
            continue

        # Normal turn
        # 1) embed -> features
        try:
            emb = text_to_embedding(user, model=EMBED_MODEL)
        except Exception as e:
            console.print("[red]Embedding failed, proceeding without embedding.[/red]", e)
            emb = None

        if emb is not None:
            add_turn("user", user, embedding=emb)
        else:
            add_turn("user", user)

        # 2) policy predict (defaults to operator)
        if emb is not None:
            pol = brain.predict(emb)
        else:
            # fallback policy
            from types import SimpleNamespace
            pol = SimpleNamespace(preset="operator", temperature=0.6, verbosity=1, humor=0.2, recall_k=2)

        # 3) recall
        recalls = []
        if emb is not None and pol.recall_k > 0:
            recs = recall_similar(emb, top_k=pol.recall_k)
            recalls = recs

        # 4) build system prompt + messages
        system_prompt = build_system_prompt(humor=pol.humor)
        messages = []
        # inject relevant recalls
        if recalls:
            rec_block = "Relevant memories (brief):\n" + "\n".join(f"- {r}" for r in recalls)
            # keep it as a user message so LLM sees it as context
            messages.append({"role": "user", "content": rec_block})

        # add recent conversation turns for short-term context
        recent = recent_turns(limit=6)
        for r in recent:
            messages.append({"role": r["role"], "content": r["text"]})

        messages.append({"role": "user", "content": user})

        # decide tokens by verbosity
        num_predict = 160 if pol.verbosity == 0 else 320 if pol.verbosity == 1 else 512

        # 5) call ollama
        try:
            reply = ollama_chat(model=MODEL, system=system_prompt, messages=messages,
                                temperature=pol.temperature, num_predict=num_predict)
        except Exception as e:
            console.print("[red]LLM call failed:[/red]", e)
            reply = "Sorry, I couldn't generate a reply (local LLM error)."

        # 6) store assistant turn
        # also embed assistant reply for memory
        try:
            rep_emb = text_to_embedding(reply, model=EMBED_MODEL)
            add_turn("assistant", reply, embedding=rep_emb)
        except Exception:
            add_turn("assistant", reply)

        console.print(Panel(reply, title=f"Jonas ({pol.preset})", border_style="green"))

if __name__ == "__main__":
    main()