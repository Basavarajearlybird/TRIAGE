"""Held-out re-scoring of an MT-Bench run.

Sweeping the escalation threshold over all 80 questions and reporting the best
point on those same 80 questions is tuning on the test set. This recomputes the
headline from the committed per-question receipts, with no new API spend:

  1. deterministic 50/50 split (ordered by question id, then seeded shuffle)
  2. pick the threshold on the TUNING half only: among points with quality
     >= 95% of the big model and a non-zero escalation rate, the one with the
     largest cost reduction
  3. report cost reduction and quality on the untouched TEST half, each with a
     bootstrap 95% CI

  PYTHONPATH=. python scripts/mtbench_holdout.py data/mtbench_nemotron.json [seed]
  PYTHONPATH=. python scripts/mtbench_holdout.py --multiseed data/mtbench_nemotron.json

Writes data/<name>_holdout.json (or _holdout_multiseed.json).
"""
from __future__ import annotations

import json
import random
import statistics as st
import sys

THRESHOLDS = [x / 100 for x in range(20, 100, 5)] + [1.01]
QUALITY_FLOOR_PCT = 95      # the RouteLLM comparison point


def point(rows: list[dict], t: float) -> dict:
    """Cost and quality if every question with risk >= t had been escalated."""
    total_big = sum(r["big_cost"] for r in rows) or 1e-12
    big_q = sum(r["big_score"] for r in rows) / len(rows) or 1e-12
    esc = [r for r in rows if r["risk"] >= t]
    cost = sum(r["big_cost"] for r in esc)
    qual = sum(r["big_score"] if r["risk"] >= t else r["small_score"] for r in rows) / len(rows)
    return {
        "threshold": t,
        "cost_reduction_pct": round((1 - cost / total_big) * 100, 1),
        "quality_vs_big_pct": round(qual / big_q * 100, 1),
        "escalation_rate": round(len(esc) / len(rows), 3),
        "n_escalated": len(esc),
        "n": len(rows),
    }


def pick_threshold(tune: list[dict]) -> dict | None:
    ok = [p for p in (point(tune, t) for t in THRESHOLDS)
          if p["quality_vs_big_pct"] >= QUALITY_FLOOR_PCT and p["escalation_rate"] > 0]
    return max(ok, key=lambda p: p["cost_reduction_pct"]) if ok else None


def bootstrap_ci(rows: list[dict], t: float, seed: int, b: int = 2000) -> dict:
    rng = random.Random(seed)
    n = len(rows)
    points = [point([rows[rng.randrange(n)] for _ in range(n)], t) for _ in range(b)]
    crs = sorted(p["cost_reduction_pct"] for p in points)
    qbs = sorted(p["quality_vs_big_pct"] for p in points)
    lo, hi = int(0.025 * b), int(0.975 * b)
    return {"cost_reduction_ci95": [crs[lo], crs[hi]],
            "quality_vs_big_ci95": [qbs[lo], qbs[hi]]}


def holdout(path: str, seed: int = 0, write: bool = True) -> dict:
    d = json.load(open(path))
    rows = d["rows"]
    order = sorted(range(len(rows)), key=lambda i: rows[i]["id"])   # stable base order
    random.Random(seed).shuffle(order)
    half = len(order) // 2
    tune = [rows[i] for i in order[:half]]
    test = [rows[i] for i in order[half:]]

    chosen = pick_threshold(tune)
    if chosen is None:
        raise SystemExit(f"[holdout] no tuning threshold met quality >= "
                         f"{QUALITY_FLOOR_PCT}% with a real escalation ({path}, seed {seed})")

    result = {
        "source": path, "small": d["small"], "big": d["big"], "judge": d.get("judge"),
        "method": "50/50 held-out split; threshold picked on tune, reported on test",
        "seed": seed, "n_tune": len(tune), "n_test": len(test),
        "tuned_threshold": chosen["threshold"],
        "held_out_headline": {**point(test, chosen["threshold"]),
                              **bootstrap_ci(test, chosen["threshold"], seed)},
        "in_sample_headline_for_reference": pick_threshold(rows),
    }
    h = result["held_out_headline"]
    print(f"[holdout] {path}  seed={seed}  tuned threshold={chosen['threshold']}")
    print(f"  HELD-OUT: {h['cost_reduction_pct']}% saved (CI {h['cost_reduction_ci95']}) "
          f"at {h['quality_vs_big_pct']}% of {d['big']} quality "
          f"(CI {h['quality_vs_big_ci95']}), {h['n_escalated']}/{h['n']} escalated")
    if write:
        out = path.replace(".json", "_holdout.json")
        with open(out, "w") as f:
            json.dump(result, f, indent=2)
        print(f"  -> {out}")
    return result


def multiseed(path: str, seeds: int = 5) -> dict:
    """Repeat the held-out split over several seeds. Escalation is rare on
    MT-Bench, so the spread across splits is the honest story: seeds where no
    threshold qualifies are skipped and counted."""
    runs = []
    for s in range(seeds):
        try:
            runs.append(holdout(path, s, write=False)["held_out_headline"])
        except SystemExit as e:
            print(e)
    if not runs:
        raise SystemExit(f"[multiseed] no seed produced a valid held-out headline for {path}")

    cr = [r["cost_reduction_pct"] for r in runs]
    qb = [r["quality_vs_big_pct"] for r in runs]
    esc = [r["n_escalated"] for r in runs]
    agg = {
        "source": path, "seeds": len(runs), "seeds_requested": seeds,
        "cost_reduction_pct": {"mean": round(st.mean(cr), 1), "min": min(cr), "max": max(cr),
                               "per_seed": cr},
        "quality_vs_big_pct": {"mean": round(st.mean(qb), 1), "min": min(qb), "max": max(qb),
                               "per_seed": qb},
        "n_escalated_per_test_half": {"mean": round(st.mean(esc), 1), "per_seed": esc},
    }
    out = path.replace(".json", "_holdout_multiseed.json")
    with open(out, "w") as f:
        json.dump(agg, f, indent=2)
    print(f"\n[multiseed] {path}: saved mean={agg['cost_reduction_pct']['mean']}% "
          f"(range {min(cr)}-{max(cr)}), quality mean={agg['quality_vs_big_pct']['mean']}% "
          f"(range {min(qb)}-{max(qb)}), escalations per test half {esc}\n  -> {out}")
    return agg


if __name__ == "__main__":
    args = sys.argv[1:]
    if args and args[0] == "--multiseed":
        multiseed(args[1] if len(args) > 1 else "data/mtbench_nemotron.json")
    else:
        holdout(args[0] if args else "data/mtbench_nemotron.json",
                int(args[1]) if len(args) > 1 else 0)
