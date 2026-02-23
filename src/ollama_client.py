# src/ollama_client.py
import requests
from typing import List, Dict, Optional

OLLAMA_BASE = "http://localhost:11434"
CHAT_URL = OLLAMA_BASE + "/api/chat"
EMBED_URL = OLLAMA_BASE + "/api/embeddings"

def ollama_chat(model: str, system: str, messages: List[Dict[str, str]],
                temperature: float = 0.7, num_predict: int = 320, timeout: int = 120) -> str:
    payload = {
        "model": model,
        "messages": [{"role": "system", "content": system}] + messages,
        "stream": False,
        "options": {"temperature": float(temperature), "num_predict": int(num_predict)}
    }
    r = requests.post(CHAT_URL, json=payload, timeout=timeout)
    r.raise_for_status()
    return r.json()["message"]["content"]

def ollama_embed(model: str, text: str, timeout: int = 30):
    payload = {"model": model, "prompt": text}
    r = requests.post(EMBED_URL, json=payload, timeout=timeout)
    r.raise_for_status()
    return r.json()["embedding"]