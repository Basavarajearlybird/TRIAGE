"""Semantic routing memory.

Records the outcome of every routed request keyed by its prompt embedding, so a
new request can ask how similar past prompts were handled. Near-neighbours that
the small model handled at low risk without escalating let the pre-filter take
the fast path and skip the escalation probes.

In-process only: it exists for the duration of one experiment run, so repeated
runs of a script start from the same empty state.
"""
from __future__ import annotations

import numpy as np

from ..retrieval.embed import embed

_MAT: np.ndarray | None = None          # (N, dim) prompt embeddings
_META: list[dict] = []                  # parallel row metadata


def reset() -> None:
    global _MAT, _META
    _MAT, _META = None, []


def record(question: str, tier: int, escalated: bool, risk: float, model: str) -> None:
    global _MAT
    v = embed([question])[0][None, :]
    _MAT = v if _MAT is None else np.vstack([_MAT, v])
    _META.append({"tier": tier, "escalated": escalated, "risk": risk, "model": model})


def lookup(question: str, k: int = 5, min_similarity: float = 0.6) -> dict:
    """Aggregate stats over the k nearest past prompts above `min_similarity`."""
    if _MAT is None:
        return {"n": 0, "hit": False}
    q = embed([question])[0]
    sims = _MAT @ q / (np.linalg.norm(_MAT, axis=1) * (np.linalg.norm(q) + 1e-9) + 1e-9)
    order = np.argsort(-sims)[:k]
    neigh = [(float(sims[i]), _META[i]) for i in order if sims[i] >= min_similarity]
    if not neigh:
        return {"n": 0, "hit": False, "max_similarity": round(float(sims[order[0]]), 4)}
    return {
        "n": len(neigh),
        "hit": True,
        "max_similarity": round(neigh[0][0], 4),
        "mean_risk": round(sum(m["risk"] for _, m in neigh) / len(neigh), 4),
        "escalation_rate": round(sum(m["escalated"] for _, m in neigh) / len(neigh), 4),
        "mean_tier": round(sum(m["tier"] for _, m in neigh) / len(neigh), 2),
    }
