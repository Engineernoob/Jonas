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
from .memory import (
    add_turn,
    recent_turns,
    recall_similar,
    add_summary,
    turn_count,
    recent_block,
    _conn,
)

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
    console.print(
        Panel(
            "[bold]Jonas v0 — local[/bold]\nType /quit to exit, /rate 0.8 to teach him.",
            title="Jonas",
        )
    )

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

        # 1) embed user input (features)
        try:
            emb = text_to_embedding(user, model=EMBED_MODEL)
        except Exception as e:
            console.print("[red]Embedding failed, proceeding without embedding.[/red]", e)
            emb = None

        # Store user turn
        if emb is not None:
            add_turn("user", user, embedding=emb)
        else:
            add_turn("user", user)

        # 2) policy predict
        if emb is not None:
            pol = brain.predict(emb)
        else:
            # fallback policy (no preset in unified persona)
            from types import SimpleNamespace
            pol = SimpleNamespace(temperature=0.6, verbosity=1, humor=0.2, recall_k=2)

        # 3) semantic recall
        recalls = []
        if emb is not None and getattr(pol, "recall_k", 0) > 0:
            recalls = recall_similar(emb, top_k=pol.recall_k)

        # 4) build system prompt + messages
        system_prompt = build_system_prompt(humor=pol.humor)
        messages = []

        # --- inject latest working-memory summary (if available) ---
        cur = _conn.cursor()
        cur.execute("SELECT text FROM turns WHERE role='summary' ORDER BY id DESC LIMIT 1")
        row = cur.fetchone()
        if row and row[0]:
            messages.append(
                {
                    "role": "user",
                    "content": "Conversation memory summary (use as context, do not repeat verbatim):\n"
                    + row[0],
                }
            )

        # inject relevant recalls
        if recalls:
            rec_block = "Relevant memories (brief):\n" + "\n".join(f"- {r}" for r in recalls)
            messages.append({"role": "user", "content": rec_block})

        # add recent conversation turns for short-term context
        recent = recent_turns(limit=6)

        # avoid duplicating the current user input in context
        if recent and recent[-1]["role"] == "user" and recent[-1]["text"] == user:
            recent = recent[:-1]

        for r in recent:
            messages.append({"role": r["role"], "content": r["text"]})

        messages.append({"role": "user", "content": user})

        # decide tokens by verbosity
        num_predict = 160 if pol.verbosity == 0 else 320 if pol.verbosity == 1 else 512

        # 5) call ollama
        try:
            reply = ollama_chat(
                model=MODEL,
                system=system_prompt,
                messages=messages,
                temperature=pol.temperature,
                num_predict=num_predict,
            )
        except Exception as e:
            console.print("[red]LLM call failed:[/red]", e)
            reply = "Sorry, I couldn't generate a reply (local LLM error)."

        # 6) store assistant turn (with embedding if possible)
        try:
            rep_emb = text_to_embedding(reply, model=EMBED_MODEL)
            add_turn("assistant", reply, embedding=rep_emb)
        except Exception:
            add_turn("assistant", reply)

        console.print(Panel(reply, title="Jonas", border_style="green"))

        # --- working memory compression every 10 turns (user+assistant) ---
        try:
            if turn_count() % 10 == 0:
                block = recent_block(limit=10)  # list of (role, text)
                convo_text = "\n".join(f"{role}: {text}" for role, text in block)

                summary = ollama_chat(
                    model=MODEL,
                    system="You compress conversations into durable working memory. Output 5 concise bullet points max.",
                    messages=[
                        {
                            "role": "user",
                            "content": (
                                "Summarize the conversation below into 3–5 bullet points.\n"
                                "Include: user goals, key topics, preferences, open loops.\n"
                                "No fluff. No quotes. No stage directions.\n\n"
                                f"{convo_text}"
                            ),
                        }
                    ],
                    temperature=0.2,
                    num_predict=180,
                )
                add_summary(summary)
                console.print("[dim]Working memory updated.[/dim]")
        except Exception:
            pass


if __name__ == "__main__":
    main()