"""MT-Bench cost-quality curve.

One paid pass per question, then the whole threshold curve is computed offline:

  1. free small-model answer through the router (escalation off) + its risk signal
  2. paid big-model answer, which doubles as the always-big baseline
  3. judge both answers 1-10 (MT-Bench single-answer grading, cheap judge)
  4. sweep the escalation threshold t: escalate iff risk >= t -> cost(t), quality(t)

The curve is in-sample by construction. The reported headline comes from
scripts/mtbench_holdout.py, which picks the threshold on one half of the
questions and reports on the other; this script prints that held-out point too,
so a fresh run and the committed receipts are read the same way.

  PYTHONPATH=. python scripts/mtbench.py --small nemotron-nano-9b --big gpt-4o \
      --out data/mtbench_nemotron.json
"""
from __future__ import annotations

import argparse
import asyncio
import json
import random
import re

import httpx

from app.llm import LLMClient
from app.router.router import SYSTEM_PROMPT, route_and_answer
from app.schemas import ChatRequest
from scripts.mtbench_holdout import THRESHOLDS, bootstrap_ci, pick_threshold, point

QUESTIONS_URL = ("https://raw.githubusercontent.com/lm-sys/FastChat/main/"
                 "fastchat/llm_judge/data/mt_bench/question.jsonl")

JUDGE_TMPL = (
    "Please act as an impartial judge and evaluate the quality of the response "
    "provided by an AI assistant to the user question displayed below. Consider "
    "helpfulness, relevance, accuracy, depth, and level of detail. Be objective. "
    "After a short explanation (max 2 sentences), rate the response on a scale of "
    "1 to 10 strictly in this format: \"Rating: [[N]]\".\n\n"
    "[Question]\n{q}\n\n[The Start of Assistant's Answer]\n{a}\n[The End of Assistant's Answer]"
)


def load_questions() -> list[dict]:
    """MT-Bench first turns only."""
    r = httpx.get(QUESTIONS_URL, timeout=60)
    r.raise_for_status()
    return [{"id": q["question_id"], "cat": q["category"], "q": q["turns"][0]}
            for q in (json.loads(x) for x in r.text.splitlines() if x.strip())]


async def judge_score(judge: str, q: str, a: str) -> tuple[float, float]:
    """(score 1-10, judge cost in USD). An empty answer scores 1; an unparseable
    judge reply scores 5 rather than dropping the question."""
    if not (a or "").strip():
        return 1.0, 0.0
    c = LLMClient(judge)
    r = await c.generate([{"role": "user", "content": JUDGE_TMPL.format(q=q, a=a[:6000])}],
                         temperature=0.0, max_tokens=160)
    m = re.search(r"\[\[(\d+(?:\.\d+)?)\]\]", r.text or "")
    return (float(m.group(1)) if m else 5.0), c.cost.est_cost_usd


async def one(item: dict, args, sem: asyncio.Semaphore) -> dict:
    async with sem:
        resp = await route_and_answer(ChatRequest(
            message=item["q"], model=args.small, allow_escalation=False, max_signals=True))
        risk = 1.0 if resp.route.abstained else float(resp.route.abstain_risk)

        c = LLMClient(args.big)
        r = await c.generate([{"role": "system", "content": SYSTEM_PROMPT},
                              {"role": "user", "content": item["q"]}],
                             temperature=0.2, max_tokens=2048)

        small_score, jc1 = await judge_score(args.judge, item["q"], resp.answer)
        big_score, jc2 = await judge_score(args.judge, item["q"], r.text or "")
        print(f"  q{item['id']:3} [{item['cat']:10}] risk={risk:.2f} "
              f"small={small_score:.0f} big={big_score:.0f} "
              f"big_cost=${c.cost.est_cost_usd:.5f}", flush=True)
        return {"id": item["id"], "cat": item["cat"], "risk": round(risk, 3),
                "small_score": small_score, "big_score": big_score,
                "big_cost": c.cost.est_cost_usd, "judge_cost": jc1 + jc2}


def sweep(rows: list[dict]) -> list[dict]:
    """Cost and quality at every escalation threshold, over all questions."""
    total_big = sum(r["big_cost"] for r in rows)
    return [{**point(rows, t),
             "cost_usd": round(sum(r["big_cost"] for r in rows if r["risk"] >= t), 5),
             "total_big_cost_usd": round(total_big, 5)}
            for t in THRESHOLDS]


async def main_async(args) -> None:
    items = load_questions()
    print(f"[mtbench] {len(items)} questions  small={args.small}  big={args.big}  "
          f"judge={args.judge}")
    sem = asyncio.Semaphore(args.concurrency)
    rows = list(await asyncio.gather(*(one(i, args, sem) for i in items)))

    curve = sweep(rows)
    big_q = sum(r["big_score"] for r in rows) / len(rows)
    small_q = sum(r["small_score"] for r in rows) / len(rows)

    # Held-out headline: never tune on the reported half. Same split and rule as
    # scripts/mtbench_holdout.py applies to the committed receipts.
    order = sorted(range(len(rows)), key=lambda i: rows[i]["id"])
    random.Random(args.seed).shuffle(order)
    half = len(order) // 2
    tune = [rows[i] for i in order[:half]]
    test = [rows[i] for i in order[half:]]
    chosen = pick_threshold(tune)
    heldout = None
    if chosen is not None:
        heldout = {**point(test, chosen["threshold"]),
                   **bootstrap_ci(test, chosen["threshold"], args.seed),
                   "tuned_threshold": chosen["threshold"], "seed": args.seed,
                   "n_tune": len(tune), "n_test": len(test)}

    print("\n========== MT-BENCH COST-QUALITY CURVE (in-sample) ==========")
    print(f"  always-big   quality={big_q:.2f}/10  cost=${sum(r['big_cost'] for r in rows):.4f}")
    print(f"  always-small quality={small_q:.2f}/10  cost=$0 (free lane)")
    print(f"  {'thr':>5} {'esc%':>6} {'cost':>9} {'saved%':>7} {'vs-big%':>8}")
    for p in curve:
        print(f"  {p['threshold']:>5} {p['escalation_rate'] * 100:>5.0f}% "
              f"${p['cost_usd']:>8.4f} {p['cost_reduction_pct']:>6.1f}% "
              f"{p['quality_vs_big_pct']:>7.1f}%")
    if heldout:
        print(f"\n  HELD-OUT HEADLINE: {heldout['cost_reduction_pct']}% saved "
              f"(CI {heldout['cost_reduction_ci95']}) at {heldout['quality_vs_big_pct']}% of "
              f"{args.big} quality (CI {heldout['quality_vs_big_ci95']}), "
              f"{heldout['n_escalated']}/{heldout['n']} escalated "
              f"[threshold {heldout['tuned_threshold']} picked on the tuning half]")
    else:
        print("\n  No tuning threshold met quality >= 95% with a real escalation.")
    print(f"  (judge spend this run: ${sum(r['judge_cost'] for r in rows):.4f})")

    with open(args.out, "w") as f:
        json.dump({"n": len(rows), "small": args.small, "big": args.big, "judge": args.judge,
                   "always_big": {"quality": round(big_q, 3),
                                  "cost_usd": round(sum(r["big_cost"] for r in rows), 5)},
                   "always_small_quality": round(small_q, 3),
                   "curve": curve, "held_out_headline": heldout, "rows": rows}, f, indent=2)
    print(f"  -> {args.out}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--small", default="nemotron-nano-9b")
    ap.add_argument("--big", default="gpt-4o")
    ap.add_argument("--judge", default="gpt-4o-mini")
    ap.add_argument("--out", default="data/mtbench_run.json")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--concurrency", type=int, default=4)
    asyncio.run(main_async(ap.parse_args()))
