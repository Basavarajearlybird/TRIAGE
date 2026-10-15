"""Deterministic math tool (Tool-Aware Routing).

Parses a natural-language arithmetic / algebra request and computes the EXACT
answer with no LLM call, so brutal-arithmetic queries (square roots of huge
numbers, big multiplications, chained ops) are answered precisely and for ~free,
instead of being escalated to a slow reasoning model or abstained.

Safety: it only fires when the prompt reduces cleanly to a pure math expression
(numbers, + - * / **, sqrt/cbrt/factorial). Anything with leftover words (a word
problem, a knowledge question) is declined -> falls through to the normal router.
No `eval` of user text: we transform to a whitelisted expression, verify the
charset, wrap every literal as Decimal, and evaluate in an empty-builtins sandbox.
"""
from __future__ import annotations

import math
import re
from decimal import Decimal, getcontext
from typing import Any

getcontext().prec = 50

_LEADIN = re.compile(
    r"^\s*(?:so\s+)?(?:now\s+)?(?:please\s+)?(?:can you\s+)?"
    r"(?:what(?:'s| is)|whats|calculate|compute|evaluate|find|give me|"
    r"the value of|solve|tell me|how much is)\b", re.I)
_FILLER = re.compile(r"\b(?:then|please|result|answer|equal to|equals|value|the|an|a|of)\b", re.I)


def _preprocess(q: str) -> str:
    s = q.lower().strip().rstrip("?.! ")
    s = _LEADIN.sub("", s).strip()
    s = s.replace("$", "").replace("%", "").replace(",", " ")   # drop currency/thousands separators
    # roots
    s = re.sub(r"cube\s*root\s*(?:of)?", " cbrt ", s)
    s = re.sub(r"(?:square\s*root|squareroot|sqrt|under\s*root|underroot|nth\s*root|root)\s*(?:of)?", " sqrt ", s)
    # powers / factorial
    s = re.sub(r"factorial\s*of|\bfactorial\b", " fact ", s)
    s = re.sub(r"\b(\d+(?:\.\d+)?)\s*squared\b", r"(\1)**2", s)
    s = re.sub(r"\b(\d+(?:\.\d+)?)\s*cubed\b", r"(\1)**3", s)
    s = re.sub(r"to the power of|raised to(?: the power of)?|\^", " ** ", s)
    # word operators (NOT bare 'and', which is too ambiguous)
    s = re.sub(r"\bplus\b|\badded to\b|\badd\b|\bsum of\b|\bincreased by\b", " + ", s)
    s = re.sub(r"\bminus\b|\bsubtract(?:ed)?(?:\s*by)?\b|\bless\b|\bdecreased by\b|\btake away\b", " - ", s)
    s = re.sub(r"\btimes\b|\bmultiplied by\b|\bmultiply\b|\bproduct of\b|×|(?<=\d)\s*x\s*(?=\d)", " * ", s)
    s = re.sub(r"\bdivided by\b|\bdivide\b|÷", " / ", s)
    s = _FILLER.sub(" ", s)
    # attach function args: "sqrt 123" -> "sqrt(123)"
    s = re.sub(r"\bsqrt\s+(\d+(?:\.\d+)?)", r"sqrt(\1)", s)
    s = re.sub(r"\bcbrt\s+(\d+(?:\.\d+)?)", r"cbrt(\1)", s)
    s = re.sub(r"\bfact\s*\(?\s*(\d+)\s*\)?", r"fact(\1)", s)
    # trailing function with a preceding number ("2121 square root", "12 factorial")
    s = re.sub(r"(\d+(?:\.\d+)?)\s+sqrt\b", r"sqrt(\1)", s)
    s = re.sub(r"(\d+(?:\.\d+)?)\s+cbrt\b", r"cbrt(\1)", s)
    s = re.sub(r"(\d+)\s+fact\b", r"fact(\1)", s)
    return re.sub(r"\s+", " ", s).strip()


def _sqrt(x):
    if x < 0: raise ValueError("negative sqrt")
    return Decimal(x).sqrt()
def _cbrt(x): return Decimal(x) ** (Decimal(1) / Decimal(3))
def _fact(x):
    n = int(x)
    if n < 0 or n > 2000 or n != x: raise ValueError("bad factorial")
    return Decimal(math.factorial(n))


def solve(q: str) -> dict[str, Any]:
    expr = _preprocess(q)
    if not re.search(r"\d", expr):
        return {"handled": False}
    # must actually involve a computation, not just a bare number
    if not re.search(r"[+\-*/]|sqrt|cbrt|fact|\*\*", expr):
        return {"handled": False}
    # after removing our whitelisted function names, only math chars may remain
    residual = re.sub(r"sqrt|cbrt|fact", "", expr)
    if not re.fullmatch(r"[0-9.\s+\-*/()]*", residual):
        return {"handled": False}
    # wrap numeric literals as Decimal
    py = re.sub(r"(?<![\d.\w])(\d+(?:\.\d+)?)", r"D('\1')", expr)
    try:
        val = eval(py, {"__builtins__": {}}, {"D": Decimal, "sqrt": _sqrt, "cbrt": _cbrt, "fact": _fact})
    except Exception:
        return {"handled": False}
    if not isinstance(val, Decimal):
        try:
            val = Decimal(val)
        except Exception:
            return {"handled": False}
    exact = val == val.to_integral_value()
    return {"handled": True, "expression": expr, "answer": _fmt(val), "exact": exact}


def _fmt(val: Decimal) -> str:
    """Exact integer when the value is whole, else ~12 decimal places marked ≈."""
    if val == val.to_integral_value():
        return f"{int(val):,}"
    q = val.quantize(Decimal("1.000000000000"))
    s = format(q.normalize(), "f")
    return f"≈ {s}"
