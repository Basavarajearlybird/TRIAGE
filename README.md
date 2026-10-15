# Triage

A reliability-aware router for text LLMs. Every request is answered first by a
cheap model; black-box signals computed on that model's *actual answer* decide
what happens next: ship it, retrieve, verify, escalate to a stronger model, or
abstain. There is no router training and no per-pair calibration, so any small
model can be paired with any large one.

The claim under test is that observing the cheap model's own output is a better
allocation signal than predicting difficulty from the prompt, and that it also
supports an axis a cost-only router cannot report: refusing to answer.

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="docs/assets/mtbench_curve_dark.svg">
  <img src="docs/assets/mtbench_curve_light.svg" alt="MT-Bench cost-quality curve: the in-sample threshold sweep over all 80 questions. The reported headline is the held-out point.">
</picture>

## Results

Every cost below is a measured API bill, with the per-question receipts in
`data/`. The MT-Bench headlines are held out: the escalation threshold is picked
on one half of the questions and reported on the other, over 5 splits.

| Result | Number | Receipts |
|---|---|---|
| MT-Bench, free Nemotron-Nano-9B → GPT-4o, held out, 5 seeds | 81.6% saved (75.7 to 87.9%) at 103.1% of GPT-4o quality (101.2 to 105.9%); 4 to 9 of 40 escalated | `data/mtbench_nemotron_holdout_multiseed.json` |
| MT-Bench, free Ministral-8B → GPT-4o, held out, 5 seeds | 96.0% saved (92 to 100%) at 98.4% of GPT-4o quality (96.0 to 100.3%); 0 to 3 of 40 escalated | `data/mtbench_gpt4o_holdout_multiseed.json` |
| GSM8K, gpt-4o-mini → GPT-4o, 5 seeds | 85.1% saved (CI 81.4 to 88.9%) at 0.960 accuracy against 0.900 for the big model (+0.06, *p*=0.017) | `data/rigor_gsm8k_multiseed.json` |
| 30-query mixed live bill | 69.5% saved ($0.0637 to $0.0195); 24 of 30 answers cost $0 | `data/showtime.json` |

Negative results, kept on purpose:

| What happened | Receipts |
|---|---|
| GSM8K with the escalation threshold too low: the router escalated 57.8% of queries and fell to 51.1% accuracy while both baselines held 91 to 93% | `data/rigor_gsm8k.json` |
| GSM8K after raising the threshold: accuracy recovered to the small model's 86.7%, but on that 30-item sample the router cost 29.1% *more* than always-big | `data/rigor_gsm8k_fixed.json` |
| The three optional add-ons left accuracy identical (0.80 in all five configurations) on a cold 10-prompt set, and spent 43% more compute than the router without them (2176 against 1521 mean compute units) | `data/ablation.json` |

An earlier version of this README claimed "95.3% saved at 100.3% quality" for the
Ministral pair. That point was chosen by sweeping the threshold on the same 80
questions it was reported on. It does not survive a held-out split: in 2 of the 5
splits the tuned threshold escalates nothing at all, and those splits score 99.4%
and 97.0% of GPT-4o quality. For this pair the honest reading is that the free
small model alone nearly matches GPT-4o on MT-Bench, and the routing contributes
little. The Nemotron pair, which escalates on 4 to 9 of 40 questions per split,
does survive.

`docs/results.md` states what each result does and does not support, and which
experiments were never run.

## How it works

```
request ─► calculator tool ──── exact arithmetic, $0, no model call
        └► small model ─── one pass + logprob uncertainty (free)
             │ confident  → ship it
             │ suspicious → deeper signals: resample agreement, self-check, evidence
             │ risky      → retrieve → verify (may revise) → escalate to the big model
             └ unanswerable → PENDING_REVIEW
```

The router makes its primary decision after reading the small model's answer, so
the decision runs on that answer's own signals. The prompt alone never sends a
request anywhere unless the pre-filter below is enabled. Three optional add-ons
sit around that core, all toggleable in `config/router.yaml`:

- a pre-filter that can short-circuit obviously easy or hard prompts before
  generation, which makes routing partly predictive while it is enabled;
- a semantic memory of past routing outcomes, intended to skip probes on prompt
  clusters already known to be safe;
- an online controller that nudges the escalation threshold from live outcomes
  without labels.

The ablation in `data/ablation.json` found all three neutral on accuracy and, on
that set, net negative on compute. Treat them as hypotheses. Nothing here shows
that they pay off, and the memory's intended saving needs a workload with
repeated prompts, which has not been run. The dollar cost was $0 in every
configuration because the ablation runs on a free lane, which is why the
compute-unit column is the one worth reading there.

Two guarantees hold whatever the configuration. No single signal can force an
abstain: the weighted risk must cross the threshold *and* at least two signals
must be individually elevated, in `app/router/abstain.py` and enforced by
`tests/test_abstain.py`. The router also never escalates into the mock provider.
A model with no provider key resolves to an offline mock, so escalating there
would fabricate a big-model answer. The router refuses, and every result records
which provider actually answered.

## Reproducing

```bash
python -m venv .venv && . .venv/bin/activate     # .venv\Scripts\activate on Windows
pip install -r requirements.txt
cp .env.example .env                             # add the keys you have
```

These need no keys and no network. They recompute the reported MT-Bench headlines
from the committed per-question receipts:

```bash
PYTHONPATH=. python scripts/mtbench_holdout.py --multiseed data/mtbench_nemotron.json
PYTHONPATH=. python scripts/mtbench_holdout.py --multiseed data/mtbench_gpt4o.json
PYTHONPATH=. python scripts/make_curve_svg.py data/mtbench_gpt4o.json --pair "free Ministral-8B → GPT-4o"
```

The two holdout commands reproduce the table above exactly, and the third
rewrites the committed figure byte for byte.

These call live APIs and cost money. They regenerate the receipts from scratch:

```bash
PYTHONPATH=. python scripts/mtbench.py --small nemotron-nano-9b --big gpt-4o \
    --out data/mtbench_nemotron.json                  # one paid pass per question
PYTHONPATH=. python scripts/rigor_multiseed.py --dataset gsm8k --n 30 \
    --small gpt-4o-mini --big gpt-4o --seeds 5        # CIs + significance
PYTHONPATH=. python scripts/showtime.py               # 30-query live bill
PYTHONPATH=. python scripts/ablation.py               # add-on on/off, free lane
```

Tests run on the mock provider, with no keys and no network:

```bash
PYTHONPATH=. pytest -q
```

They cover the abstain invariant, the tier map, the calculator and long-context
routes, the verify parser and its fallback, the quality metric's honesty guard,
and embedder determinism.

## Layout

```
app/
  router/       router.py (tiers, escalation) · abstain · prefilter · memory
                · online · longcontext
  signals/      black-box reliability signals
  verify/       grounding pass that can revise an answer
  providers/    OpenAI-compatible adapter · offline mock · registry
  retrieval/    hashing embedder + numpy index over a small seed corpus
  tools/        deterministic calculator
  benchmark/    rigor.py: GSM8K/traps harness, bootstrap CIs, significance
  llm.py        cost accounting across every call in one request
config/         models.yaml (catalog + prices) · router.yaml (all thresholds)
scripts/        experiment runners; each writes one file into data/
data/           committed receipts for every number reported above
docs/           results.md (provenance and limits) · assets/ (figures)
tests/
```

## Limits

- The signals are black-box only: logprobs and resampling. The target models are
  closed APIs, so weights and attention are unavailable.
- Thresholds are calibrated by hand, not learned. Instruction-tuned models
  compress their logprob uncertainty into a narrow band, so `signal_gate` and
  `tau_star` are set to that band. A per-model percentile calibration would be
  the proper fix and is not implemented.
- MT-Bench is partly saturated for modern small models. At "never escalate" the
  free small model alone already held 96.7% of GPT-4o quality for the Ministral
  pair and 93.3% for the Nemotron pair, so escalations are rare and the held-out
  intervals are wide, with only a handful of escalations per test half.
- The judge is gpt-4o-mini scoring 1 to 10, first turns only.
- Tier-L only detects. Long-context requests are counted and flagged, and nothing
  then handles them differently; the attention primitive from the original plan
  is not built.
- Four result files are frozen receipts. The Ministral MT-Bench run, the live
  bill, and both negative GSM8K runs used `ministral-8b` and `gpt-oss-120b`,
  since retired from the catalog, so they cannot be regenerated as committed. The
  Nemotron MT-Bench run, the GSM8K multi-seed run and the ablation can be. See
  `docs/results.md`.

`CHARTER.md` records what this project deliberately does not attempt and why.
