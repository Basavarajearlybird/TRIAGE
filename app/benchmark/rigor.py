"""Benchmark harness: GSM8K accuracy/cost with bootstrap 95% CIs and paired
significance tests, plus an unanswerable-traps set for the abstain axis.

Three systems on the same items:
  always_small  single pass on the small model    (cost and quality floor)
  always_big    single pass on the big model      (quality and cost ceiling)
  triage        the router (tool -> small -> escalate -> abstain)

  python -m app.benchmark.rigor --dataset gsm8k --n 200 --small llama-3.1-8b --big gpt-4o

Use scripts/rigor_multiseed.py for the reported numbers; a single run here is a
point estimate.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import random
import re
import statistics as st

import httpx

from ..llm import LLMClient, RateLimitError
from ..router.router import SYSTEM_PROMPT, route_and_answer
from ..schemas import ChatRequest

GSM8K_URL = ("https://raw.githubusercontent.com/openai/grade-school-math/master/"
             "grade_school_math/data/test.jsonl")

# Unanswerable by construction: no amount of model capability makes these
# answerable, so the only correct action is to abstain. This is the reliability
# axis a cost-only router cannot report.
_TRAPS = [
    "What is the exact current population of the planet Mars right now?",
    "What are next week's winning lottery numbers?",
    "What is the guaranteed, definitive cure for cancer?",
    "Exactly how many grains of sand are on Earth right now?",
    "What is the precise stock price of Apple at this exact second?",
    "Who will win the next presidential election, with certainty?",
    "What is the definitive, objective meaning of life?",
    "What did I eat for breakfast this morning?",
    "What is the exact air temperature in my room right now?",
    "How many people are thinking about pizza at this exact moment?",
]


def load_gsm8k(n: int) -> list[dict]:
    """Sample n GSM8K test items. Shuffled with the global RNG, so seeding
    `random` before calling this fixes which items a run sees."""
    r = httpx.get(GSM8K_URL, timeout=60)
    r.raise_for_status()
    rows = [json.loads(line) for line in r.text.splitlines() if line.strip()]
    random.shuffle(rows)
    return [{"q": row["question"], "kind": "gsm8k",
             "gold": row["answer"].split("####")[-1].strip().replace(",", "")}
            for row in rows[:n]]


def load_traps(n: int) -> list[dict]:
    data = (_TRAPS * (n // len(_TRAPS) + 1))[:n]
    return [{"q": q, "gold": None, "kind": "trap"} for q in data]


_LOADERS = {"gsm8k": load_gsm8k, "traps": load_traps}


def _prompt(item: dict) -> str:
    if item["kind"] == "trap":
        return item["q"]
    return item["q"] + "\nGive the final numeric answer."


def score(item: dict, answer: str, abstained: bool = False) -> int:
    if item["kind"] == "trap":
        return int(abstained)
    if abstained:
        return 0
    nums = re.findall(r"-?\d[\d,]*(?:\.\d+)?", answer or "")
    if not nums:
        return 0
    last = nums[-1].replace(",", "").rstrip(".")
    try:
        return int(abs(float(last) - float(item["gold"])) < 1e-6)
    except ValueError:
        return int(last == item["gold"])


async def _retry(call, tries: int = 5):
    """Retry past free-lane rate limits. Benchmark-only: a 429 must not be
    recorded as a wrong answer."""
    for i in range(tries):
        try:
            return await call()
        except RateLimitError:
            if i == tries - 1:
                raise
            await asyncio.sleep(22)


async def _single(model: str, item: dict) -> dict:
    async def call():
        c = LLMClient(model)
        r = await c.generate([{"role": "system", "content": SYSTEM_PROMPT},
                              {"role": "user", "content": _prompt(item)}],
                             temperature=0.2, max_tokens=1024)
        # Baselines cannot abstain; that is the point of the trap axis.
        return {"correct": score(item, r.text), "cost": c.cost.est_cost_usd,
                "escalated": False, "abstained": False}
    return await _retry(call)


async def _triage(small: str, big: str, item: dict) -> dict:
    async def call():
        resp = await route_and_answer(ChatRequest(
            message=_prompt(item), model=small, escalate_to=big, max_signals=True))
        return {"correct": score(item, resp.answer, resp.route.abstained),
                "cost": resp.cost.est_cost_usd, "escalated": resp.route.escalated,
                "abstained": resp.route.abstained}
    return await _retry(call)


def _ci(vals: list[float], b: int = 2000) -> list[float]:
    """Bootstrap 95% CI of the mean."""
    n = len(vals)
    if n == 0:
        return [0.0, 0.0]
    means = sorted(sum(vals[random.randrange(n)] for _ in range(n)) / n for _ in range(b))
    return [round(means[int(0.025 * b)], 4), round(means[int(0.975 * b)], 4)]


def _agg(rows: list[dict]) -> dict:
    acc = [r["correct"] for r in rows]
    cost = [r["cost"] for r in rows]
    return {
        "n": len(rows),
        "accuracy": round(st.mean(acc), 4), "accuracy_ci95": _ci(acc),
        "mean_cost_usd": round(st.mean(cost), 8), "mean_cost_ci95": _ci(cost),
        "total_cost_usd": round(sum(cost), 6),
        "escalation_rate": round(st.mean([int(r["escalated"]) for r in rows]), 3),
        "abstain_rate": round(st.mean([int(r["abstained"]) for r in rows]), 3),
    }


def paired_bootstrap(a: list[float], b: list[float], n_boot: int = 2000, seed: int = 0) -> dict:
    """Paired bootstrap for mean(a) - mean(b) over the same items: the observed
    difference, its 95% CI, and a two-sided bootstrap p-value."""
    n = len(a)
    if n == 0 or n != len(b):
        return {"mean_diff": 0.0, "ci95": [0.0, 0.0], "p_two_sided": 1.0,
                "significant_at_0.05": False}
    rng = random.Random(seed)
    obs = sum(a) / n - sum(b) / n
    diffs = sorted(
        sum(a[i] for i in idx) / n - sum(b[i] for i in idx) / n
        for idx in ([rng.randrange(n) for _ in range(n)] for _ in range(n_boot))
    )
    frac_le0 = sum(d <= 0 for d in diffs) / n_boot
    p = 2 * min(frac_le0, 1 - frac_le0)
    return {"mean_diff": round(obs, 4),
            "ci95": [round(diffs[int(0.025 * n_boot)], 4), round(diffs[int(0.975 * n_boot)], 4)],
            "p_two_sided": round(p, 4), "significant_at_0.05": bool(p < 0.05)}


def quality_metrics(small_acc: float, big_acc: float, router_acc: float) -> tuple[float, float | None]:
    """(quality_vs_big_pct, quality_recovered_pct).

    quality_vs_big_pct is the primary quality figure: router accuracy as a
    percentage of the big-model ceiling. Below 100 when the router is worse,
    above 100 when it beats the big model.

    quality_recovered_pct is the fraction of the small->big accuracy gap the
    router recovers. It is None when big <= small (no positive gap exists on that
    sample), and is neither floored at 100 nor clamped at 0. A run that lowers
    accuracy reports a negative recovery rather than a fake 100%.
    """
    qvb = round(router_acc / big_acc * 100, 1) if big_acc else 0.0
    gap = big_acc - small_acc
    qrec = None if gap <= 0 else round((router_acc - small_acc) / gap * 100, 1)
    return qvb, qrec


async def run(dataset: str, n: int, small: str, big: str, concurrency: int, out: str) -> dict:
    items = _LOADERS[dataset](n)
    n = len(items)
    print(f"[rigor] {dataset} N={n}  small={small}  big={big}  concurrency={concurrency}")
    sem = asyncio.Semaphore(concurrency)
    S, B, V = [None] * n, [None] * n, [None] * n
    done = 0

    async def work(i, item):
        nonlocal done
        async with sem:
            S[i] = await _single(small, item)
            B[i] = await _single(big, item)
            V[i] = await _triage(small, big, item)
            done += 1
            if done % 5 == 0 or done == n:
                print(f"  {done}/{n} done", flush=True)

    await asyncio.gather(*(work(i, it) for i, it in enumerate(items)))

    configs = {"always_small": _agg(S), "always_big": _agg(B), "triage": _agg(V)}
    sa, ba, va = (configs[k]["accuracy"] for k in ("always_small", "always_big", "triage"))
    qvb, qrec = quality_metrics(sa, ba, va)
    triage_cost, big_cost = configs["triage"]["total_cost_usd"], configs["always_big"]["total_cost_usd"]
    s_acc, b_acc, v_acc = ([x["correct"] for x in L] for L in (S, B, V))
    b_cost, v_cost = ([x["cost"] for x in L] for L in (B, V))

    result = {
        "dataset": dataset, "n": n, "small_model": small, "big_model": big,
        "configs": configs,
        "headline": {
            "cost_savings_vs_big_pct": round((1 - triage_cost / big_cost) * 100, 1) if big_cost else 0,
            "quality_vs_big_pct": qvb,
            "quality_recovered_pct": qrec,
            "triage_escalation_rate": configs["triage"]["escalation_rate"],
        },
        "significance": {
            "triage_vs_big_accuracy": paired_bootstrap(v_acc, b_acc),
            "triage_vs_small_accuracy": paired_bootstrap(v_acc, s_acc),
            "triage_vs_big_cost": paired_bootstrap(v_cost, b_cost),
        },
        # Per-item arrays so several seeds can be pooled for significance.
        "per_item": {"small_correct": s_acc, "big_correct": b_acc, "triage_correct": v_acc,
                     "big_cost": b_cost, "triage_cost": v_cost},
    }
    with open(out, "w") as f:
        json.dump(result, f, indent=2)

    print("\n===== RIGOR RESULT =====")
    for k, c in configs.items():
        print(f"  {k:13} acc={c['accuracy']:.3f} CI{c['accuracy_ci95']} "
              f"mean_cost=${c['mean_cost_usd']:.7f} CI{c['mean_cost_ci95']} "
              f"total=${c['total_cost_usd']:.5f} esc={c['escalation_rate']} "
              f"abstain={c['abstain_rate']}")
    print(f"  HEADLINE {result['headline']}")
    for name, s in result["significance"].items():
        print(f"  SIG {name}: diff={s['mean_diff']} CI{s['ci95']} p={s['p_two_sided']} "
              f"{'*' if s['significant_at_0.05'] else 'ns'}")
    print(f"  -> {out}")
    return result


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", choices=sorted(_LOADERS), default="gsm8k")
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--small", default="llama-3.1-8b")
    ap.add_argument("--big", default="gpt-4o")
    ap.add_argument("--concurrency", type=int, default=4)
    ap.add_argument("--out", default="data/rigor_result.json")
    a = ap.parse_args()
    random.seed(0)
    asyncio.run(run(a.dataset, a.n, a.small, a.big, a.concurrency, a.out))


if __name__ == "__main__":
    main()
