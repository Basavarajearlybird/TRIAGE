# Charter

This records what Triage is scoped to do, and what it deliberately does not
attempt. It exists because the project is a narrowing of an earlier plan, and the
difference should not be mistaken for missing work.

## The pivot

The original governing plan specified an output-aware router for multimodal
(image + text) answers, spending the team's own GPU compute on a local vLLM
substrate, scored by Trustworthy Goodput.

Triage is narrower on each of those three axes:

- It handles text only.
- It spends API dollars on third-party providers rather than GPU-seconds on a
  local serving substrate.
- Its signals are black-box: logprobs and resampling. The target models are
  closed APIs, so attention-based signals cannot be obtained.

This document charters that pivot. The current code is the deliverable, and
nothing in it is scaffolding toward the multimodal plan.

## In scope

- Observe-then-allocate routing over any small → big text-model pair.
- Black-box reliability signals and the multi-signal abstain invariant.
- A deterministic calculator route; the retrieve-and-verify loop.
- Measured cost and quality reporting, with confidence intervals, held-out
  evaluation, and negative results published alongside positive ones.

## Out of scope

These were not built. Under this charter they are not compliance gaps:

- Multimodal (image + text) reliability and the domain stress variants derived
  from it.
- The GPU-second cost model, the Trustworthy Goodput metric, and its three SLOs
  as the success measure. Triage reports API dollars as its headline currency;
  the compute-unit proxy in `CostMetrics` is a supplement.
- A local vLLM serving substrate: quantisation, prefix caching, chunked prefill,
  a version-frozen stack.
- White-box signal mode (attention-based ADS/TI/GIS) and the proxy-fidelity study
  that would compare it against the black-box signals. Closed weights make it
  impossible, so proxy mode is the product.
- The utility router with λ / predicted-cost optimisation, load-adaptive control,
  oracle and budget-matched baselines, and the long-context attention primitive.
  `app/router/longcontext.py` detects long-context requests and says so; it does
  not handle them differently.

## Obligations that survive the narrowing

- Every published number is a measured bill with its per-question receipts
  committed under `data/`.
- Negative results are published. Three of the seven result files are negative.
- Headline claims are evaluated on a held-out split, never tuned on the reported
  half.
- The no-single-signal abstain invariant is preserved and unit-tested.
- Experiments that were never run are named in `docs/results.md`. The presence of
  a runner is never left to imply that it was run.
