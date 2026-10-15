"""Tier -1: the predict-then-route pre-filter.

A training-free difficulty estimate from prompt lexical features plus the
semantic routing memory, computed before any generation:

  easy_direct  obviously easy AND memory says similar prompts were safe ->
               run the small model on a fast path, skip the deep probes
  hard_direct  obviously hard (roots, big numbers, multi-step) -> skip the small
               pass and go straight to the big model
  normal       ambiguous -> the full observe-then-allocate loop decides

This is the hybrid front-end: routing is not purely post-observation while it is
enabled. The output-signal escalator remains the safety net behind it, so a wrong
"easy" guess is still caught by the live signals. Every decision returns the
features that fired.
"""
from __future__ import annotations

import re
from typing import Any

_MATH = re.compile(r"[\d]+\s*[\+\-\*/x=%]|\bpercent\b|\bhow (?:much|many)\b|\bcalculate\b|\bcompute\b")
# "Hard math": arithmetic small models reliably flub. Roots, big numbers, factorials,
# powers. These alone should be enough to send the query straight to the big model.
_HARD_MATH = re.compile(
    r"\b(?:square root|cube root|nth root|under\s?root|underroot|sqrt|root of)\b|√|"
    r"\bfactorial\b|\d{5,}|\d+\s*\^\s*\d+|\d+\s*(?:squared|cubed)\b|\braised to\b"
)
_MULTISTEP = re.compile(r"\bstep by step\b|\bprove\b|\bderive\b|\bderivative\b|\bexplain why\b|\breason\b|\bif\b.*\bthen\b|\bin total\b|\bhow long\b")
_CODE = re.compile(r"```|\bdef \b|\bfunction\b|\bdebug\b|\bpython\b|\bsql\b|\bregex\b|\bcompile\b|\bstack trace\b")
_TRAP = re.compile(r"\bcurrent\b|\blatest\b|\bdefinitive\b|\bexact(?:ly)?\b|\bprecisely\b|\bobjective(?:ly)?\b|\bmeaning of life\b|\bright now\b")


def _lexical_difficulty(q: str) -> tuple[float, dict[str, Any]]:
    ql = q.lower()
    n_words = len(ql.split())
    feats = {
        "long_prompt": min(1.0, n_words / 60.0),          # long asks tend to be harder
        "hard_math": 1.0 if _HARD_MATH.search(ql) else 0.0,   # roots / big numbers / powers
        "math": 1.0 if _MATH.search(ql) else 0.0,
        "multistep": 1.0 if _MULTISTEP.search(ql) else 0.0,
        "code": 1.0 if _CODE.search(ql) else 0.0,
        "trap_markers": 1.0 if _TRAP.search(ql) else 0.0,
        "many_constraints": min(1.0, (ql.count(",") + ql.count(" and ")) / 4.0),
    }
    # hard_math alone clears hard_direct (0.70): this is exactly the class of prompt
    # that should skip the weak model and go straight to the strong one.
    w = {"long_prompt": 0.15, "hard_math": 0.72, "math": 0.30, "multistep": 0.28,
         "code": 0.30, "trap_markers": 0.22, "many_constraints": 0.15}
    score = min(1.0, sum(w[k] * feats[k] for k in feats))
    return score, {"features": {k: round(v, 3) for k, v in feats.items() if v}, "lexical": round(score, 4)}


def predict(question: str, mem: dict[str, Any], cfg: dict) -> dict[str, Any]:
    """Return {difficulty, route, detail}. `route` in {easy_direct, normal, hard_direct}."""
    lex, lex_detail = _lexical_difficulty(question)

    # Blend in the routing memory: if similar past prompts escalated, this one is
    # probably hard; if they were low-risk on the small model, probably easy.
    mem_difficulty = None
    if mem.get("hit") and mem.get("n", 0) >= cfg.get("memory_min_neighbours", 2):
        mem_difficulty = 0.5 * mem["escalation_rate"] + 0.5 * min(1.0, mem["mean_risk"] / 0.62)
        difficulty = 0.55 * lex + 0.45 * mem_difficulty
    else:
        difficulty = lex

    hard = cfg.get("hard_direct", 0.70)
    easy = cfg.get("easy_direct", 0.25)
    memory_safe = bool(
        mem.get("hit")
        and mem.get("n", 0) >= cfg.get("memory_min_neighbours", 2)
        and mem.get("escalation_rate", 1.0) <= cfg.get("safe_escalation_rate", 0.2)
        and mem.get("mean_risk", 1.0) <= cfg.get("safe_risk", 0.35)
    )

    if difficulty >= hard:
        route = "hard_direct"
    elif difficulty <= easy and memory_safe:
        route = "easy_direct"
    else:
        route = "normal"

    return {
        "difficulty": round(float(difficulty), 4),
        "route": route,
        "detail": {
            **lex_detail,
            "memory": mem if mem.get("hit") else {"hit": False},
            "memory_difficulty": None if mem_difficulty is None else round(mem_difficulty, 4),
            "memory_safe": memory_safe,
            "thresholds": {"hard_direct": hard, "easy_direct": easy},
        },
    }
