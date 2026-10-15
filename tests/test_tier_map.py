"""The tier map, and the rule that a conflicted answer skips retrieve."""
import asyncio

from app.config import load_router_config
from app.providers.base import GenResult
from app.router.router import TIER_NAMES, _assess
from app.schemas import ChatRequest, CostMetrics


def test_official_tier_numbering():
    assert TIER_NAMES == {0: "fast", 1: "retrieve", 2: "verify", 3: "abstain"}


def test_escalation_is_not_a_tier():
    assert TIER_NAMES[3] == "abstain"
    assert "escalate" not in TIER_NAMES.values()


_UNRELATED = [
    "Paris sits beside the river Seine.",
    "Quantum chromodynamics governs gluon interactions.",
    "Banana bread requires very ripe fruit.",
    "The halting problem is undecidable.",
    "Basalt columns form as lava cools slowly.",
]


class _DisagreeingClient:
    """Answers with unrelated text every call and returns no logprobs, so the
    resamples land in separate clusters and instability crosses the threshold."""

    def __init__(self):
        self.model_id = "stub"
        self.cost = CostMetrics()
        self.provider_label = "stub"
        self.n = 0

    async def generate(self, messages, temperature=0.2, max_tokens=512, want_logprobs=False):
        self.n += 1
        return GenResult(text=_UNRELATED[(self.n - 1) % len(_UNRELATED)],
                         provider="stub", model="stub", tokens_in=10, tokens_out=20)


def test_conflict_skips_retrieve_and_goes_to_verify():
    cfg = load_router_config()
    client = _DisagreeingClient()
    out = asyncio.run(_assess(client, ChatRequest(message="an ambiguous question"),
                              cfg, compute_full=True))

    assert out["signals"].instability >= cfg["tiers"]["instability_high"]
    assert out["should_retrieve"] is False      # conflict, so Tier 1 is skipped
    assert out["verify_trigger"] is True        # it goes straight to Tier 2
