"""Re-run the rigor benchmark over several fixed seeds and report every number
with a mean and 95% CI across seeds, plus significance tests pooled over all
items from all seeds. This produces the reported GSM8K numbers; a single run is
a point estimate and is not reported on its own.

  PYTHONPATH=. python scripts/rigor_multiseed.py --dataset gsm8k --n 30 \
      --small gpt-4o-mini --big gpt-4o --seeds 5

Writes data/rigor_<dataset>_multiseed.json. The small lane is free on the NVIDIA
models; only the big model and the escalations cost money.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import random
import statistics as st
import tempfile

from app.benchmark.rigor import paired_bootstrap, quality_metrics, run

CONFIGS = ("always_small", "always_big", "triage")


def _mean_ci(vals: list[float]) -> dict:
    """Normal-approximation CI across seeds (n is small, so this is indicative)."""
    m = st.mean(vals)
    if len(vals) < 2:
        return {"mean": round(m, 4), "ci95": [round(m, 4), round(m, 4)], "n_seeds": len(vals)}
    se = st.stdev(vals) / len(vals) ** 0.5
    return {"mean": round(m, 4), "std": round(st.stdev(vals), 4),
            "ci95": [round(m - 1.96 * se, 4), round(m + 1.96 * se, 4)],
            "per_seed": [round(v, 4) for v in vals], "n_seeds": len(vals)}


async def main_async(a) -> None:
    per_seed = []
    pool = {k: [] for k in ("small_correct", "big_correct", "triage_correct",
                            "big_cost", "triage_cost")}
    for seed in range(a.seeds):
        random.seed(seed)     # fixes which dataset items this seed samples
        print(f"\n===== SEED {seed} =====")
        res = await run(a.dataset, a.n, a.small, a.big, a.concurrency,
                        tempfile.mktemp(suffix=f"_seed{seed}.json"))
        per_seed.append(res)
        for k in pool:
            pool[k].extend(res["per_item"][k])

    def col(cfg, field):
        return [r["configs"][cfg][field] for r in per_seed]

    agg = {
        "dataset": a.dataset, "n_per_seed": a.n, "seeds": a.seeds,
        "small_model": a.small, "big_model": a.big,
        "accuracy": {c: _mean_ci(col(c, "accuracy")) for c in CONFIGS},
        "mean_cost_usd": {c: _mean_ci(col(c, "mean_cost_usd")) for c in CONFIGS},
        "escalation_rate": _mean_ci(col("triage", "escalation_rate")),
        "cost_savings_vs_big_pct": _mean_ci(
            [r["headline"]["cost_savings_vs_big_pct"] for r in per_seed]),
    }
    qvb, qrec = quality_metrics(agg["accuracy"]["always_small"]["mean"],
                                agg["accuracy"]["always_big"]["mean"],
                                agg["accuracy"]["triage"]["mean"])
    agg["quality_vs_big_pct"] = qvb
    agg["quality_recovered_pct"] = qrec
    agg["significance_pooled"] = {
        "triage_vs_big_accuracy": paired_bootstrap(pool["triage_correct"], pool["big_correct"]),
        "triage_vs_small_accuracy": paired_bootstrap(pool["triage_correct"], pool["small_correct"]),
        "triage_vs_big_cost": paired_bootstrap(pool["triage_cost"], pool["big_cost"]),
    }

    out = f"data/rigor_{a.dataset}_multiseed.json"
    with open(out, "w") as f:
        json.dump(agg, f, indent=2)

    print("\n########## MULTI-SEED SUMMARY ##########")
    for c in CONFIGS:
        print(f"  {c:13} acc mean={agg['accuracy'][c]['mean']} CI{agg['accuracy'][c]['ci95']}")
    print(f"  cost savings vs big: {agg['cost_savings_vs_big_pct']['mean']}% "
          f"CI{agg['cost_savings_vs_big_pct']['ci95']}")
    print(f"  quality_vs_big={qvb}%  quality_recovered={qrec}")
    for k, s in agg["significance_pooled"].items():
        print(f"  SIG {k}: diff={s['mean_diff']} CI{s['ci95']} p={s['p_two_sided']} "
              f"{'*' if s['significant_at_0.05'] else 'ns'}")
    print(f"  -> {out}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", choices=["gsm8k", "traps"], default="gsm8k")
    ap.add_argument("--n", type=int, default=30)
    ap.add_argument("--small", default="gpt-4o-mini")
    ap.add_argument("--big", default="gpt-4o")
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--concurrency", type=int, default=2)
    asyncio.run(main_async(ap.parse_args()))
