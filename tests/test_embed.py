"""The embedder must be stable across processes: it feeds retrieval scores and
every similarity derived from them, so a salted hash would make two runs of the
same experiment disagree."""
import subprocess
import sys

from app.retrieval.embed import cosine, embed


def _vector_in_subprocess(text: str) -> list[float]:
    """A fresh interpreter gets a different PYTHONHASHSEED."""
    out = subprocess.run(
        [sys.executable, "-c",
         f"from app.retrieval.embed import embed;"
         f"print([round(float(x), 6) for x in embed([{text!r}])[0]])"],
        capture_output=True, text=True, check=True)
    return eval(out.stdout)


def test_embedding_is_identical_in_a_fresh_process():
    here = [round(float(x), 6) for x in embed(["the capital of France"])[0]]
    assert here == _vector_in_subprocess("the capital of France")
    assert any(here)


def test_embeddings_are_normalised():
    v = embed(["photosynthesis converts sunlight"])[0]
    assert abs(cosine(v, v) - 1.0) < 1e-6


def test_related_text_scores_above_unrelated():
    q = embed(["what is the capital of France"])[0]
    near = embed(["Paris is the capital of France"])[0]
    far = embed(["mitochondria generate cellular ATP"])[0]
    assert cosine(q, near) > cosine(q, far)
