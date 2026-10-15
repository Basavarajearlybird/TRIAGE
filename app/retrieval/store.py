"""In-process numpy cosine index over the seed corpus."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .embed import embed
from .seed import SEED_CORPUS


@dataclass
class Hit:
    text: str
    score: float


class Store:
    def __init__(self, docs: list[str]) -> None:
        self.texts = list(docs)
        self.vecs = embed(self.texts)

    def search(self, query: str, top_k: int = 4, min_score: float = 0.2) -> list[Hit]:
        q = embed([query])[0]
        sims = self.vecs @ q / (
            np.linalg.norm(self.vecs, axis=1) * (np.linalg.norm(q) + 1e-9) + 1e-9
        )
        return [Hit(self.texts[i], float(sims[i]))
                for i in np.argsort(-sims)[:top_k] if sims[i] >= min_score]


_store: Store | None = None


def get_store() -> Store:
    global _store
    if _store is None:
        _store = Store(SEED_CORPUS)
    return _store
