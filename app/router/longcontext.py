"""Tier-L: a long-context detection flag, orthogonal to the 0-3 reliability tiers.

Detection only. The chunked-partition / relay-token attention primitive from the
original multimodal plan is NOT implemented here (see CHARTER.md); this flags the
requests such a primitive would handle so their share of the workload is visible.
Retrieved evidence counts toward the length, since it also consumes context.
"""
from __future__ import annotations

_CHARS_PER_TOKEN = 4    # heuristic; avoids a tokenizer dependency

TIER_L_BUILT = False


def estimate_tokens(text: str, evidence: list | None = None) -> int:
    total = len(text or "") + sum(len(h.text) for h in (evidence or []))
    return max(1, total // _CHARS_PER_TOKEN)


def long_context_route(prompt: str, threshold_tokens: int, evidence: list | None = None) -> dict:
    approx = estimate_tokens(prompt, evidence)
    is_long = approx >= threshold_tokens
    return {
        "long_context": is_long,
        "approx_prompt_tokens": approx,
        "counts_evidence": bool(evidence),
        "threshold_tokens": threshold_tokens,
        "tier_l_built": TIER_L_BUILT,
        "strategy": "long-context-flag (detect-only; real primitive not built)" if is_long else "none",
    }
