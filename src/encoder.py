# src/encoder.py
"""
Wrapper to get a fixed-size embedding for text using Ollama's embedding model.
Assumes you pulled a local embeddings model (e.g., nomic-embed-text).
"""
from typing import List
import numpy as np
from .ollama_client import ollama_embed

EMBED_MODEL_DEFAULT = "nomic-embed-text"

def text_to_embedding(text: str, model: str = EMBED_MODEL_DEFAULT) -> np.ndarray:
    vec = ollama_embed(model, text)
    arr = np.array(vec, dtype=np.float32)
    # Normalize to unit vector to make cosine simple
    norm = np.linalg.norm(arr)
    if norm == 0:
        return arr
    return arr / norm