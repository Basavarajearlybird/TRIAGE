"""On/off ablation of the three optional add-ons (pre-filter, semantic memory,
online learner) over a small labelled set.

Each config runs with the add-on state reset, so the configs are independent.
Measures accuracy, mean USD and compute cost, escalation rate, routing overhead
and latency. Runs on the free NVIDIA lane by default, so the whole ablation costs
nothing.

  PYTHONPATH=. python scripts/ablation.py --small llama-3.1-8b --big nemotron-super-49b

Writes data/ablation.json. Ten prompts is too small to detect a small effect;
see docs/results.md for what this run does and does not support.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import re

from app.config import load_router_config
from app.llm import RateLimitError
from app.router import memory, online
from app.router.router import route_and_answer
from app.schemas import ChatRequest

# Factual lookups, multi-step word problems the calculator declines, one
# reasoning item, and two unanswerable traps where correct == abstain.
CASES = [
    {"q": "What is the capital of France?", "gold": "paris", "kind": "text"},
    {"q": "Who wrote the play Hamlet?", "gold": "shakespeare", "kind": "text"},
    {"q": "What is the largest planet in our Solar System?", "gold": "jupiter", "kind": "text"},
    {"q": "A shop sells pens at 3 for 2 dollars. How much do 12 pens cost in dollars?", "gold": "8", "kind": "num"},
    {"q": "A car drives 60 mph for 2.5 hours then 40 mph for 1 hour. Total miles?", "gold": "190", "kind": "num"},
    {"q": "Tom has 5 boxes with 8 apples each and gives away 12. How many apples remain?", "gold": "28", "kind": "num"},
    {"q": "Is 91 a prime number? Answer yes or no.", "gold": "no", "kind": "text"},
    {"q": "A tank holds 240 liters and drains 15 liters per minute. Minutes to empty?", "gold": "16", "kind": "num"},
    {"q": "What are next week's winning lottery numbers?", "gold": None, "kind": "trap"},
    {"q": "What is the exact current population of Mars right now?", "gold": None, "kind": "trap"},
]

# Which add-ons each config disables.
CONFIGS = {
    "full": [],
    "no_prefilter": ["prefilter"],
    "no_memory": ["memory"],
    "no_online": ["online_learning"],
    "none_added": ["prefilter", "memory", "online_learning"],
}

ADDONS = ("prefilter", "memory", "online_learning")


def score(case: dict, resp) -> int:
    if case["kind"] == "trap":
        return int(resp.route.abstained)
    if resp.route.abstained:
        return 0
    answer = (resp.answer or "").lower()
    if case["kind"] == "num":
        nums = re.findall(r"-?\d[\d,]*(?:\.\d+)?", answer)
        return int(bool(nums) and nums[-1].replace(",", "") == case["gold"])
    return int(case["gold"] in answer)


async def _route(req: ChatRequest, tries: int = 5):
    """The free lane allows about 40 requests/min; back off and retry rather than
    scoring a rate limit as a wrong answer."""
    for i in range(tries):
        try:
            return await route_and_answer(req)
        except RateLimitError:
            if i == tries - 1:
                raise
            await asyncio.sleep(22)


async def run_config(name: str, disabled: list[str], args) -> dict:
    memory.reset()
    online.reset()
    cfg = load_router_config()          # the same cached object the router reads
    for addon in ADDONS:
        cfg[addon]["enabled"] = addon not in disabled

    totals = dict(correct=0, cost=0.0, compute=0.0, latency=0.0, overhead=0, escalated=0)
    rows = []
    for case in CASES:
        resp = await _route(ChatRequest(message=case["q"], model=args.small,
                                        escalate_to=args.big))
        await asyncio.sleep(1.2)        # stay under the free-lane rate cap
        correct = score(case, resp)
        totals["correct"] += correct
        totals["cost"] += resp.cost.est_cost_usd
        totals["compute"] += resp.cost.compute_units
        totals["latency"] += resp.latency_ms
        totals["overhead"] += resp.cost.routing_overhead_passes
        totals["escalated"] += int(resp.route.escalated)
        rows.append({"q": case["q"][:40], "tier": resp.route.tier_name,
                     "escalated": resp.route.escalated, "correct": correct,
                     "prefilter": resp.route.prefilter_route})

    for addon in ADDONS:               # restore for the next config
        cfg[addon]["enabled"] = True

    n = len(CASES)
    return {"config": name, "n": n, "accuracy": round(totals["correct"] / n, 3),
            "mean_cost_usd": round(totals["cost"] / n, 8),
            "mean_compute_units": round(totals["compute"] / n, 1),
            "escalation_rate": round(totals["escalated"] / n, 3),
            "mean_overhead_passes": round(totals["overhead"] / n, 2),
            "mean_latency_ms": round(totals["latency"] / n, 1), "rows": rows}


async def main_async(args) -> None:
    print(f"[ablation] small={args.small} big={args.big}  "
          f"{len(CASES)} cases x {len(CONFIGS)} configs")
    results = []
    for name, disabled in CONFIGS.items():
        r = await run_config(name, disabled, args)
        results.append(r)
        print(f"  {name:12} acc={r['accuracy']:.2f} esc={r['escalation_rate']:.2f} "
              f"overhead={r['mean_overhead_passes']:.2f} "
              f"compute={r['mean_compute_units']:.0f} "
              f"lat={r['mean_latency_ms']:.0f}ms cost=${r['mean_cost_usd']:.6f}")
    with open(args.out, "w") as f:
        json.dump({"small": args.small, "big": args.big, "cases": len(CASES),
                   "results": results}, f, indent=2)
    print(f"  -> {args.out}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--small", default="llama-3.1-8b")
    ap.add_argument("--big", default="nemotron-super-49b")
    ap.add_argument("--out", default="data/ablation.json")
    asyncio.run(main_async(ap.parse_args()))
