"""Dependency-free hashing bag-of-words embeddings (384-d, L2-normalised).

zlib.crc32 rather than the builtin hash(): str hashing is salted per process
(PYTHONHASHSEED), which made retrieval scores and every similarity derived from
them differ between runs of the same experiment.
"""
from __future__ import annotations

import re
import zlib

import numpy as np

_DIM = 384
_WORD = re.compile(r"[a-z0-9]+")


def embed(texts: list[str]) -> np.ndarray:
    out = np.zeros((len(texts), _DIM), dtype=np.float32)
    for i, t in enumerate(texts):
        for tok in _WORD.findall(t.lower()):
            h = zlib.crc32(tok.encode())
            out[i, h % _DIM] += 1.0
            out[i, (h // _DIM) % _DIM] += 0.5   # second bucket reduces collisions
    norms = np.linalg.norm(out, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return out / norms


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na == 0 or nb == 0:
        return 0.0
    return float(np.dot(a, b) / (na * nb))
