"""Verify JSON parser and its deterministic fallback."""
import asyncio
from types import SimpleNamespace

from app.verify.verify import _parse_json, verify


def test_parse_json_extracts_object_from_surrounding_prose():
    obj = _parse_json('sure, here: {"verdict": "pass", "grounded": true} done')
    assert obj == {"verdict": "pass", "grounded": True}


def test_parse_json_returns_none_on_garbage():
    assert _parse_json("no json here at all") is None
    assert _parse_json("{not valid json}") is None


def test_parse_json_rejects_non_dict_json():
    assert _parse_json("[1, 2, 3]") is None


class _FakeClient:
    """Stand-in LLMClient whose generate() returns a fixed text."""
    def __init__(self, text):
        self._text = text

    async def generate(self, *a, **k):
        return SimpleNamespace(text=self._text, tokens_in=10, tokens_out=5)


def test_verify_parses_model_json_verdict():
    c = _FakeClient('{"verdict": "fail", "grounded": false, "revised": "corrected", "reason": "off"}')
    out = asyncio.run(verify(c, "q", "candidate", []))
    assert out["pass"] is False
    assert out["revised"] == "corrected"


def test_verify_falls_back_to_heuristic_when_unparseable():
    """No JSON in the model output -> deterministic heuristic verdict. With no
    evidence the heuristic passes (nothing to contradict) and keeps the candidate."""
    c = _FakeClient("this is not json")
    out = asyncio.run(verify(c, "q", "the candidate answer", []))
    assert out["pass"] is True
    assert out["revised"] == "the candidate answer"
    assert "heuristic fallback" in out["reason"]
