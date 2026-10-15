"""OpenAI-compatible /chat/completions adapter (OpenAI, NVIDIA NIM, Groq, OpenRouter).

Errors are returned on GenResult.error rather than raised; the caller decides
whether to fall back or surface them.
"""
from __future__ import annotations

import time
from typing import Any

import httpx

from .base import GenResult, Message, ProviderAdapter


class OpenAICompatAdapter(ProviderAdapter):
    def __init__(self, name: str, api_key: str, base_url: str, supports_logprobs: bool = True) -> None:
        super().__init__(api_key, base_url)
        self.name = name
        self.supports_logprobs = supports_logprobs

    async def generate(
        self,
        model: str,
        messages: list[Message],
        temperature: float = 0.2,
        max_tokens: int = 512,
        want_logprobs: bool = False,
    ) -> GenResult:
        payload: dict[str, Any] = {
            "model": model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        # gpt-5 reasoning models spend the whole completion budget thinking and
        # return empty content at chat-sized max_tokens.
        if model.startswith("gpt-5"):
            payload["reasoning_effort"] = "low"
        if want_logprobs and self.supports_logprobs:
            payload["logprobs"] = True
            payload["top_logprobs"] = 1

        headers = {"Authorization": f"Bearer {self.api_key}"}
        url = f"{self.base_url}/chat/completions"
        t0 = time.perf_counter()
        try:
            async with httpx.AsyncClient(timeout=150.0) as client:
                resp = await client.post(url, json=payload, headers=headers)
                # Some hosted models reject `logprobs` (422); retry without it and
                # let uncertainty fall back to resample entropy.
                if resp.status_code == 422 and "logprobs" in payload:
                    payload.pop("logprobs", None)
                    payload.pop("top_logprobs", None)
                    resp = await client.post(url, json=payload, headers=headers)
                # Reasoning models reject `max_tokens` and non-default temperature.
                for _ in range(2):
                    if resp.status_code != 400:
                        break
                    err = resp.text or ""
                    if "max_tokens" in err and "max_tokens" in payload:
                        payload["max_completion_tokens"] = payload.pop("max_tokens")
                    elif "temperature" in err and "temperature" in payload:
                        payload.pop("temperature")
                    else:
                        break
                    resp = await client.post(url, json=payload, headers=headers)
            latency = (time.perf_counter() - t0) * 1000
            if resp.status_code == 429:
                return GenResult(
                    text="", provider=self.name, model=model, latency_ms=latency,
                    error=f"RATE_LIMIT retry_after={resp.headers.get('retry-after', '')}: {resp.text[:200]}",
                )
            if resp.status_code >= 400:
                return GenResult(
                    text="", provider=self.name, model=model, latency_ms=latency,
                    error=f"HTTP {resp.status_code}: {resp.text[:300]}",
                )
            data = resp.json()
        except Exception as exc:  # network, timeout, malformed JSON
            return GenResult(
                text="", provider=self.name, model=model,
                latency_ms=(time.perf_counter() - t0) * 1000, error=str(exc),
            )

        # OpenRouter can answer HTTP 200 with an error body and no `choices`.
        if not data.get("choices"):
            err = data.get("error") or {}
            msg = err.get("message") if isinstance(err, dict) else str(err)
            code = err.get("code") if isinstance(err, dict) else None
            prefix = "RATE_LIMIT retry_after=: " if code in (429, "429") else ""
            return GenResult(
                text="", provider=self.name, model=model, latency_ms=latency,
                error=f"{prefix}{msg or 'empty choices in response'}"[:300],
            )

        choice = data["choices"][0]
        usage = data.get("usage") or {}
        return GenResult(
            text=((choice.get("message") or {}).get("content") or "").strip(),
            provider=self.name,
            model=model,
            tokens_in=usage.get("prompt_tokens", 0),
            tokens_out=usage.get("completion_tokens", 0),
            latency_ms=latency,
            logprobs=_extract_logprobs(choice),
            raw=data,
        )


def _extract_logprobs(choice: dict[str, Any]) -> list[float] | None:
    content = (choice.get("logprobs") or {}).get("content")
    if not content:
        return None
    out = [float(t["logprob"]) for t in content
           if isinstance(t, dict) and t.get("logprob") is not None]
    return out or None
