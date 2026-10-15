"""Live cost comparison on a fixed 30-query mixed workload: every query billed
on the big model, versus the same queries through the router.

Both lanes are real API calls, so this is a measured bill, not a projection.

  PYTHONPATH=. python scripts/showtime.py --small llama-3.1-8b --big gpt-4o

Writes data/showtime.json.
"""
from __future__ import annotations

import argparse
import asyncio
import json

from app.llm import LLMClient
from app.router.router import SYSTEM_PROMPT, route_and_answer
from app.schemas import ChatRequest

# 12 easy / 10 medium / 8 hard. Fixed set: the mix is what makes the bill
# comparable across runs.
QUERIES = [
    ("easy", "What is the capital of France?"),
    ("easy", "What does HTTP stand for?"),
    ("easy", "Translate 'good morning' to Spanish."),
    ("easy", "What is 17 x 24?"),
    ("easy", "Name the largest planet in the solar system."),
    ("easy", "Who wrote Romeo and Juliet?"),
    ("easy", "What is the chemical symbol for gold?"),
    ("easy", "Convert 100 fahrenheit to celsius."),
    ("easy", "What year did World War II end?"),
    ("easy", "Give me a one-line summary of what photosynthesis is."),
    ("easy", "What is the square root of 144?"),
    ("easy", "Rewrite this politely: 'send me the file now'."),
    ("med", "A bat and a ball cost $1.10 total; the bat costs $1 more than the ball. How much is the ball?"),
    ("med", "If a train travels 240 km in 3 hours, then speeds up by 20 km/h, how long does the next 300 km take?"),
    ("med", "Explain briefly why the sky is blue."),
    ("med", "I have 3 apples, eat one, buy a dozen more, give half of everything away. How many do I have?"),
    ("med", "What's the difference between a list and a tuple in Python, in two sentences?"),
    ("med", "If today is Tuesday, what day is it 100 days from now? Show the math."),
    ("med", "A shirt costs $25 after a 20% discount. What was the original price?"),
    ("med", "Explain compound interest with a small numeric example."),
    ("med", "Which is bigger, 2^10 or 10^3? Show why."),
    ("med", "Summarize the plot of Hamlet in three sentences."),
    ("hard", "Prove that the square root of 2 is irrational, step by step."),
    ("hard", "A farmer has chickens and rabbits, 35 heads and 94 legs total. Exactly how many of each? Show reasoning."),
    ("hard", "Three people check into a hotel room that costs $30... explain the missing dollar riddle rigorously."),
    ("hard", "Derive the formula for the sum of the first n odd numbers and prove it by induction."),
    ("hard", "Write a Python function that returns the longest palindromic substring, and explain its complexity."),
    ("hard", "Two trains 300 km apart head toward each other at 70 and 80 km/h; a bird flies between them at 120 km/h until they meet. Total distance the bird flies?"),
    ("hard", "Explain why the halting problem is undecidable, with a sketch of the diagonalization argument."),
    ("hard", "If 5 machines take 5 minutes to make 5 widgets, how long do 100 machines take to make 100 widgets? Then generalize to m machines, w widgets."),
]


async def main_async(args) -> None:
    rows = []
    for kind, q in QUERIES:
        # What the workload costs today: every query on the paid big model.
        c = LLMClient(args.big)
        await c.generate([{"role": "system", "content": SYSTEM_PROMPT},
                          {"role": "user", "content": q}], temperature=0.2, max_tokens=1024)

        resp = await route_and_answer(ChatRequest(
            message=q, model=args.small, escalate_to=args.big, max_signals=True))

        rows.append({"kind": kind, "q": q, "big": c.cost.est_cost_usd,
                     "triage": resp.cost.est_cost_usd, "esc": resp.route.escalated,
                     "tier": resp.route.tier_name})
        print(f"[{kind:4}] big=${c.cost.est_cost_usd:.6f}  "
              f"triage=${resp.cost.est_cost_usd:.6f}  esc={resp.route.escalated}  "
              f"tier={resp.route.tier_name}  | {q[:52]}")

    total_big = sum(r["big"] for r in rows)
    total_triage = sum(r["triage"] for r in rows)
    print(f"\n================ SHOWTIME RESULT (N={len(rows)}) ================")
    for kind in ("easy", "med", "hard"):
        sub = [r for r in rows if r["kind"] == kind]
        kb, kt = sum(r["big"] for r in sub), sum(r["triage"] for r in sub)
        print(f"  {kind:5} n={len(sub):2}  big=${kb:.5f}  triage=${kt:.5f}  "
              f"esc={sum(r['esc'] for r in sub)}  saved={(1 - kt / kb) * 100:5.1f}%")
    print(f"  always-{args.big}: ${total_big:.6f}")
    print(f"  triage         : ${total_triage:.6f} "
          f"({sum(r['esc'] for r in rows)} escalations)")
    print(f"  SAVED          : ${total_big - total_triage:.6f} = "
          f"{(1 - total_triage / total_big) * 100:.1f}%")

    with open(args.out, "w") as f:
        json.dump({"n": len(rows), "small": args.small, "big": args.big,
                   "always_big_usd": round(total_big, 6),
                   "triage_usd": round(total_triage, 6),
                   "saved_pct": round((1 - total_triage / total_big) * 100, 1),
                   "escalations": sum(r["esc"] for r in rows), "rows": rows}, f, indent=2)
    print(f"  -> {args.out}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--small", default="llama-3.1-8b")
    ap.add_argument("--big", default="gpt-4o")
    ap.add_argument("--out", default="data/showtime.json")
    asyncio.run(main_async(ap.parse_args()))
