"""LLM client used by the router, the signals and the verify pass. One instance
per routed request; accumulates cost across every call made serving it."""
from __future__ import annotations

import re

from .config import get_model
from .providers.base import GenResult, Message
from .providers.registry import get_registry
from .schemas import CostMetrics


class RateLimitError(Exception):
    """A live provider returned HTTP 429. Raised rather than answered from the
    mock, so a rate-limited benchmark run is never mistaken for a measurement."""

    def __init__(self, provider: str, retry_after: int | None) -> None:
        self.provider = provider
        self.retry_after = retry_after
        super().__init__(f"{provider} rate-limited (retry_after={retry_after})")


def estimate_cost(model_id: str, tokens_in: int, tokens_out: int) -> float:
    """USD from the catalog's per-1M-token prices. Free lanes are priced at 0."""
    m = get_model(model_id)
    if not m:
        return 0.0
    return (tokens_in * m.get("cost_in", 0.0) + tokens_out * m.get("cost_out", 0.0)) / 1_000_000


class LLMClient:
    def __init__(self, model_id: str) -> None:
        self.model_id = model_id
        self.registry = get_registry()
        self.cost = CostMetrics()
        self.provider_label = "mock"

    async def generate(
        self,
        messages: list[Message],
        temperature: float = 0.2,
        max_tokens: int = 512,
        want_logprobs: bool = False,
    ) -> GenResult:
        adapter, provider_model, label = self.registry.resolve(self.model_id)
        self.provider_label = label
        res = await adapter.generate(provider_model, messages, temperature, max_tokens, want_logprobs)

        if res.error and label != "mock":
            if res.error.startswith("RATE_LIMIT"):
                m = re.search(r"retry_after=(\d+)", res.error)
                raise RateLimitError(label, int(m.group(1)) if m else None)
            # Any other live-provider failure degrades to the mock so a long run
            # does not die on one bad response; provider_label records that it did.
            res = await self.registry.mock.generate(
                provider_model, messages, temperature, max_tokens, want_logprobs
            )
            self.provider_label = "mock"

        self.cost.tokens_in += res.tokens_in
        self.cost.tokens_out += res.tokens_out
        self.cost.est_cost_usd += estimate_cost(self.model_id, res.tokens_in, res.tokens_out)
        self.cost.llm_calls += 1
        return res
