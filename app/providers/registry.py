"""Resolves a catalog model id to a live adapter + provider-specific model name.

Falls back to the mock provider when the mapped provider has no key, so the
suite and the mock-mode demo run offline. A run served by "mock" is not a
measurement: the router refuses to escalate into it, and every script reports
the provider that answered.
"""
from __future__ import annotations

from ..config import get_model, get_settings
from .base import ProviderAdapter
from .mock import MockProvider
from .openai_compat import OpenAICompatAdapter


class ProviderRegistry:
    def __init__(self) -> None:
        s = get_settings()
        self.force_mock = s.force_mock
        self.mock = MockProvider()
        self._adapters: dict[str, ProviderAdapter] = {
            "openai": OpenAICompatAdapter("openai", s.openai_key, s.openai_base),
            "nvidia": OpenAICompatAdapter("nvidia", s.nvidia_key, s.nvidia_base),
            "groq": OpenAICompatAdapter("groq", s.groq_key, s.groq_base),
            # OpenRouter does not return token logprobs.
            "openrouter": OpenAICompatAdapter(
                "openrouter", s.openrouter_key, s.openrouter_base, supports_logprobs=False
            ),
        }

    def status(self) -> dict[str, bool]:
        return {name: a.available() for name, a in self._adapters.items()}

    def resolve(self, model_id: str) -> tuple[ProviderAdapter, str, str]:
        """Return (adapter, provider_model_name, provider_label)."""
        m = get_model(model_id)
        if m is None:
            return self.mock, model_id, "mock"
        if self.force_mock:
            return self.mock, m["provider_model"], "mock"

        adapter = self._adapters.get(m["provider"])
        if adapter and adapter.available():
            return adapter, m["provider_model"], m["provider"]

        for alt_provider, alt_name in (m.get("alt") or {}).items():
            alt = self._adapters.get(alt_provider)
            if alt and alt.available():
                return alt, alt_name, alt_provider

        return self.mock, m["provider_model"], "mock"


_registry: ProviderRegistry | None = None


def get_registry() -> ProviderRegistry:
    global _registry
    if _registry is None:
        _registry = ProviderRegistry()
    return _registry
