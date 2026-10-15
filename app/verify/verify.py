"""Verify pass: a second LLM call that checks the candidate answer against the
retrieved evidence and returns a possibly-revised answer.

The model is asked for JSON. When it does not comply, a deterministic overlap
heuristic decides instead, so the pass always yields a verdict.
"""
from __future__ import annotations

import json
import re
from typing import Any

from ..llm import LLMClient
from ..signals.proxy import text_similarity

_SUPPORT_FLOOR = 0.35     # overlap with evidence needed to pass the heuristic

_PROMPT = """You are a verification model. Check whether the CANDIDATE answer to the \
QUESTION is (1) grounded in the EVIDENCE and (2) internally consistent.

QUESTION:
{question}

EVIDENCE:
{evidence}

CANDIDATE:
{candidate}

Reply with ONLY a JSON object, no prose:
{{"verdict": "pass" | "fail", "grounded": true|false, "revised": "<corrected or confirmed answer>", "reason": "<short>"}}"""


async def verify(client: LLMClient, question: str, candidate: str, evidence: list) -> dict[str, Any]:
    ev_text = "\n".join(f"- {h.text}" for h in evidence) or "(no evidence retrieved)"
    res = await client.generate(
        [{"role": "user", "content": _PROMPT.format(question=question, evidence=ev_text,
                                                    candidate=candidate)}],
        temperature=0.0, max_tokens=400)

    parsed = _parse_json(res.text)
    if parsed is None:
        support = max((text_similarity(candidate, h.text) for h in evidence), default=0.0)
        # With no evidence there is nothing for the candidate to contradict.
        passed = support >= _SUPPORT_FLOOR or not evidence
        parsed = {"verdict": "pass" if passed else "fail",
                  "grounded": support >= _SUPPORT_FLOOR,
                  "revised": candidate,
                  "reason": f"heuristic fallback (support={support:.2f})"}

    return {
        "pass": parsed.get("verdict", "pass") == "pass",
        "grounded": bool(parsed.get("grounded", True)),
        "revised": str(parsed.get("revised") or candidate).strip() or candidate,
        "reason": parsed.get("reason", ""),
    }


def _parse_json(text: str) -> dict[str, Any] | None:
    m = re.search(r"\{.*\}", text, re.DOTALL)
    if not m:
        return None
    try:
        obj = json.loads(m.group(0))
    except json.JSONDecodeError:
        return None
    return obj if isinstance(obj, dict) else None
