"""Persistent exact cosine vector search for small document collections."""

import json
import threading
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .embeddings import Embedder


@dataclass(frozen=True)
class Hit:
    source_id: str
    title: str
    text: str
    access_group: str
    score: float


class VectorStore:
    def __init__(self, path: Path, embedder: Embedder):
        self.path = path
        self.embedder = embedder
        self.lock = threading.RLock()
        self.rows: list[dict] = []
        if path.exists():
            data = json.loads(path.read_text(encoding="utf-8"))
            if data["embedder"] != embedder.name:
                raise ValueError("Index embedding backend differs; rebuild the index")
            self.rows = data["rows"]

    def add(self, title: str, access_group: str, chunks: list[str]) -> int:
        if not chunks:
            return 0
        vectors = self.embedder.embed(chunks)
        with self.lock:
            for index, (text, vector) in enumerate(zip(chunks, vectors, strict=True)):
                self.rows.append(
                    {
                        "source_id": f"{len(self.rows) + 1}",
                        "title": title,
                        "chunk": index + 1,
                        "text": text,
                        "access_group": access_group,
                        "vector": vector.tolist(),
                    }
                )
            self.path.parent.mkdir(parents=True, exist_ok=True)
            temp = self.path.with_suffix(self.path.suffix + ".tmp")
            temp.write_text(
                json.dumps({"embedder": self.embedder.name, "rows": self.rows}),
                encoding="utf-8",
            )
            temp.replace(self.path)
        return len(chunks)

    def search(self, query: str, groups: set[str], limit: int = 3) -> list[Hit]:
        with self.lock:
            visible = [row for row in self.rows if row["access_group"] in groups]
        if not visible:
            return []
        vector = self.embedder.embed([query])[0]
        matrix = np.asarray([row["vector"] for row in visible], dtype=np.float32)
        if matrix.shape[1] != len(vector):
            raise ValueError("Index vector dimension differs; rebuild the index")
        scores = matrix @ vector
        order = np.argsort(-scores)[:limit]
        return [
            Hit(
                source_id=visible[i]["source_id"],
                title=visible[i]["title"],
                text=visible[i]["text"],
                access_group=visible[i]["access_group"],
                score=float(scores[i]),
            )
            for i in order
            if scores[i] >= 0.12
        ]
