"""Black-box scoring functions. The hash-fallback embedder is not deterministic
across processes, so assert relative properties / ranges, not exact values."""
from types import SimpleNamespace

from app.providers.base import GenResult
from app.signals import proxy


def _hit(text, score):
    return SimpleNamespace(text=text, score=score, source="test")


def test_text_similarity_bounds_and_identity():
    assert proxy.text_similarity("hello world", "hello world") > 0.9
    lo = proxy.text_similarity("quantum chromodynamics", "banana bread recipe")
    assert 0.0 <= lo <= 1.0
    assert proxy.text_similarity("hello world", "hello world") >= lo


def test_uncertainty_prefers_logprobs_high_conf_is_low_u():
    g = GenResult(text="Paris", provider="mock", model="m", tokens_in=1, tokens_out=1,
                  logprobs=[-0.01, -0.02])  # prob ~ 1 -> low uncertainty
    u, detail = proxy.uncertainty(g, [])
    assert detail["mode"] == "logprob"
    assert 0.0 <= u < 0.1


def test_uncertainty_falls_back_to_resample_entropy_without_logprobs():
    g = GenResult(text="x", provider="mock", model="m", tokens_in=1, tokens_out=1, logprobs=None)
    u, detail = proxy.uncertainty(g, ["totally different one", "unrelated other text"])
    assert detail["mode"] == "resample_entropy"
    assert 0.0 <= u <= 1.0


def test_instability_zero_when_all_samples_agree():
    val, detail = proxy.instability("the answer is 42", ["the answer is 42", "the answer is 42"])
    assert val == 0.0
    assert detail["agreement"] == 1.0


def test_instability_high_when_samples_disagree():
    val, _ = proxy.instability("alpha beta gamma",
                               ["completely unrelated wording", "yet another distinct phrase"])
    assert val > 0.0


def test_retrieval_disagreement_no_evidence_is_zero():
    val, detail = proxy.retrieval_disagreement("anything", [])
    assert val == 0.0
    assert detail["reason"] == "no_evidence"


def test_evidence_sufficiency_scales_with_hits():
    none_val, _ = proxy.evidence_sufficiency([], 0.2)
    assert none_val == 0.0
    weak_val, wdet = proxy.evidence_sufficiency([_hit("a", 0.05)], 0.2)
    assert weak_val == 0.1 and wdet["reason"] == "all_below_min_score"
    strong_val, _ = proxy.evidence_sufficiency(
        [_hit("a", 0.9), _hit("b", 0.8), _hit("c", 0.7)], 0.2)
    assert strong_val > weak_val
