# src/memory.py
"""
Tiny SQLite conversation memory. Stores turns and some metadata.
Also provides a simple semantic recall using stored embeddings (cosine similarity).
"""
import sqlite3
import os
import numpy as np

DB_PATH = "data/chat.db"

def ensure_db():
    os.makedirs("data", exist_ok=True)
    conn = sqlite3.connect(DB_PATH, check_same_thread=False)
    cur = conn.cursor()
    cur.execute("""
    CREATE TABLE IF NOT EXISTS turns (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        role TEXT,
        text TEXT,
        embedding BLOB,
        metadata TEXT,
        ts DATETIME DEFAULT CURRENT_TIMESTAMP
    )
    """)
    conn.commit()
    return conn

_conn = ensure_db()

def add_turn(role: str, text: str, embedding: np.ndarray = None, metadata: str = None):
    emb_blob = embedding.tobytes() if embedding is not None else None
    cur = _conn.cursor()
    cur.execute("INSERT INTO turns (role, text, embedding, metadata) VALUES (?, ?, ?, ?)",
                (role, text, emb_blob, metadata))
    _conn.commit()

def add_summary(text: str):
    cur = _conn.cursor()
    cur.execute(
       "INSERT INTO turns (role, text) VALUES (?, ?)",
        ("summary", text),
    )
    _conn.commit()

def turn_count():
    cur = _conn.cursor
    cur.execute("SELECT COUNT(*) FROM turns WHERE role IN ('user', 'assistant')")
    return cur.fetchone()[0]

def recent_block(limit=10):
    cur = _conn.cursor()
    cur.execute(
        "SELECT role, text FROM turns WHERE role IN ('user','assistant') ORDER BY id DESC LIMIT ?",
        (limit,),
    )
    rows = cur.fetchall()
    rows.reverse()
    return rows

def recent_turns(limit=8):
    cur = _conn.cursor()
    cur.execute("SELECT role, text FROM turns ORDER BY id DESC LIMIT ?", (limit,))
    rows = cur.fetchall()
    # return in chronological order
    return list(reversed([{"role": r[0], "text": r[1]} for r in rows]))

def recall_similar(embedding: np.ndarray, top_k=3):
    """
    Brute-force cosine over stored embeddings. Works fine at small scale.
    """
    if embedding is None:
        return []
    cur = _conn.cursor()
    cur.execute("SELECT id, text, embedding FROM turns WHERE embedding IS NOT NULL")
    rows = cur.fetchall()
    if not rows:
        return []
    sims = []
    for rid, text, emb_blob in rows:
        vec = np.frombuffer(emb_blob, dtype=np.float32)
        # cosine:
        denom = (np.linalg.norm(vec) * np.linalg.norm(embedding))
        if denom == 0:
            score = 0.0
        else:
            score = float(np.dot(vec, embedding) / denom)
        sims.append((score, text))
    sims.sort(key=lambda x: x[0], reverse=True)
    return [t for s, t in sims[:top_k]]