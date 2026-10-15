"""Provider adapter interface. Black-box only: no attention hooks, no weights.

`logprobs` carries per-token chosen-token logprobs when the provider exposes them
(OpenAI-compatible `logprobs`), else None, in which case uncertainty falls back to
resample entropy.
"""
from __future__ import annotations

import abc
from dataclasses import dataclass, field
from typing import Any, Optional


@dataclass
class GenResult:
    text: str
    provider: str
    model: str
    tokens_in: int = 0
    tokens_out: int = 0
    latency_ms: float = 0.0
    logprobs: Optional[list[float]] = None
    raw: dict[str, Any] = field(default_factory=dict)
    error: Optional[str] = None


Message = dict[str, str]  # {"role": ..., "content": ...}


class ProviderAdapter(abc.ABC):
    name: str = "base"

    def __init__(self, api_key: str = "", base_url: str = "") -> None:
        self.api_key = api_key
        self.base_url = base_url.rstrip("/")

    def available(self) -> bool:
        return bool(self.api_key)

    @abc.abstractmethod
    async def generate(
        self,
        model: str,
        messages: list[Message],
        temperature: float = 0.2,
        max_tokens: int = 512,
        want_logprobs: bool = False,
    ) -> GenResult:
        ...
