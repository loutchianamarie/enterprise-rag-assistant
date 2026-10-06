"""Swappable local embedding backends. Hash mode is for offline smoke tests."""

import hashlib
import re
from typing import Protocol

import numpy as np


class Embedder(Protocol):
    name: str

    def embed(self, texts: list[str]) -> np.ndarray: ...


class HashEmbedder:
    """A deterministic lexical baseline, not a semantic model."""

    name = "hash-v2"

    STOPWORDS = frozenset({
        "a", "an", "and", "are", "at", "be", "can", "do", "for", "from",
        "how", "in", "is", "of", "on", "or", "should", "the", "to", "was",
        "were", "what", "when", "where", "which", "who", "why", "with",
    })

    def embed(self, texts: list[str]) -> np.ndarray:
        vectors = np.zeros((len(texts), 1024), dtype=np.float32)
        for row, value in enumerate(texts):
            for token in re.findall(r"[\w]+", value.casefold()):
                if token in self.STOPWORDS:
                    continue
                digest = hashlib.sha256(token.encode()).digest()
                column = int.from_bytes(digest[:4], "big") % vectors.shape[1]
                vectors[row, column] += 1.0
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        return vectors / np.maximum(norms, 1e-12)


class FastEmbedder:
    def __init__(self, model: str):
        try:
            from fastembed import TextEmbedding
        except ImportError as exc:
            raise RuntimeError("Install the semantic extra: pip install -e '.[semantic]'") from exc
        self.model = TextEmbedding(model_name=model)
        self.name = f"fastembed:{model}"

    def embed(self, texts: list[str]) -> np.ndarray:
        matrix = np.asarray(list(self.model.embed(texts)), dtype=np.float32)
        norms = np.linalg.norm(matrix, axis=1, keepdims=True)
        return matrix / np.maximum(norms, 1e-12)


def make_embedder(backend: str, model: str) -> Embedder:
    if backend == "hash":
        return HashEmbedder()
    if backend == "fastembed":
        return FastEmbedder(model)
    raise ValueError(f"Unsupported embedding backend: {backend}")
