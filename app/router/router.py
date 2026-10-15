"""Reliability-conditioned allocation: the Triage router.

Tier map:
  0 fast     single pass on the small model
  1 retrieve uncertainty U > tau_star and no conflict
  2 verify   conflicted / unstable / under-evidenced -> grounding pass. A conflict
             goes straight here rather than through retrieve.
  3 abstain  PENDING_REVIEW when the multi-signal abstain policy fires and no
             intervention rescued the answer.

Tier-L (long context) is orthogonal and detect-only; see longcontext.py.

Escalation small -> big is a costed intervention inside the policy, not a tier:
run the cheap model, read its own proxy signals, trade up only when they say the
answer is unreliable. It is recorded on RouteDecision.escalated and paid for in
the aggregated cost; the served tier still reports the reliability level reached.
"""
from __future__ import annotations

import time

from ..config import default_small_model, get_model, load_router_config
from ..llm import LLMClient
from ..providers.registry import get_registry
from ..retrieval.store import get_store
from ..schemas import ChatRequest, ChatResponse, CostMetrics, RouteDecision, SignalSet
from ..signals import proxy
from ..tools import calculator
from ..verify.verify import verify
from . import memory, online, prefilter
from .abstain import should_abstain
from .longcontext import long_context_route

TIER_NAMES = {0: "fast", 1: "retrieve", 2: "verify", 3: "abstain"}

SYSTEM_PROMPT = ("You are a helpful, accurate assistant. Think step by step when the "
                 "question requires reasoning, then state the final answer.")

_ABSTAIN_MSG = (
    "I'm not confident enough to answer reliably, so this request has been routed "
    "to PENDING_REVIEW. See the signal panel for why."
)


def _degenerate(ans: str) -> bool:
    """A non-answer: empty, the '(no answer)' sentinel, or too short to be a reply."""
    a = (ans or "").strip()
    return a in ("", "(no answer)") or len(a) < 2


def pick_model(req: ChatRequest) -> str:
    if req.model and get_model(req.model):
        return req.model
    return default_small_model()["id"]


async def _assess(client: LLMClient, req: ChatRequest, cfg: dict, compute_full: bool) -> dict:
    """Run tiers 0-2 (main pass, signals, retrieve, verify) for ONE model.

    Returns the possibly verify-revised answer, the signals behind it, and the
    tier this model reached: everything the caller needs to decide abstain and
    escalation.
    """
    tcfg, pcfg, enabled = cfg["tiers"], cfg["proxy"], cfg["signals"]
    vcfg, rc = cfg["verify"], cfg["retrieval"]

    messages = [{"role": "system", "content": req.system or SYSTEM_PROMPT},
                {"role": "user", "content": req.message}]

    # -- Tier 0: main pass -------------------------------------------------- #
    main = await client.generate(messages, temperature=pcfg["base_temperature"],
                                 max_tokens=pcfg["max_tokens"], want_logprobs=True)
    answer = main.text or "(no answer)"
    main_tokens = max(main.tokens_out, 1)
    signals = SignalSet()

    # Free signal: uncertainty straight from the main pass's logprobs.
    if enabled["uncertainty"]:
        signals.uncertainty, signals.detail["uncertainty"] = proxy.uncertainty(main, [])

    # -- Cost-aware cascade gate -------------------------------------------- #
    # Extra LLM calls for the deep signals are only spent when the free signal is
    # already elevated, or when the provider returned no logprobs to read.
    gate = tcfg["signal_gate"]
    go_deep = compute_full and (not main.logprobs or signals.uncertainty >= gate)
    signals.detail["cascade"] = {"gate": gate, "free_U": signals.uncertainty,
                                 "deep_signals": go_deep, "had_logprobs": bool(main.logprobs)}

    if go_deep:
        samples: list[str] = []
        if enabled["instability"] or (enabled["uncertainty"] and not main.logprobs):
            for _ in range(pcfg["resamples"]):
                r = await client.generate(messages, temperature=pcfg["resample_temperature"],
                                          max_tokens=pcfg["max_tokens"])
                if r.text:
                    samples.append(r.text)
        if enabled["uncertainty"] and not main.logprobs:
            signals.uncertainty, signals.detail["uncertainty"] = proxy.uncertainty(main, samples)
        if enabled["instability"]:
            signals.instability, signals.detail["instability"] = proxy.instability(answer, samples)
        if enabled["contradiction"]:
            signals.contradiction, signals.detail["contradiction"] = await proxy.contradiction_probe(
                client, req.message, answer)

    conflict = (signals.contradiction >= tcfg["contradiction_high"]
                or signals.instability >= tcfg["instability_high"])

    # -- Tier 1: retrieve (uncertain AND NOT conflicted) -------------------- #
    tier = 0
    retrieved = False
    evidence: list = []
    should_retrieve = signals.uncertainty > tcfg["tau_star"] and not conflict
    if should_retrieve:
        evidence = get_store().search(req.message, top_k=rc["top_k"], min_score=rc["min_score"])
        retrieved = True
        tier = 1
        if compute_full and enabled["retrieval_disagreement"]:
            signals.retrieval_disagreement, signals.detail["retrieval_disagreement"] = (
                proxy.retrieval_disagreement(answer, evidence))
        if enabled["evidence_sufficiency"]:
            signals.evidence_sufficiency, signals.detail["evidence_sufficiency"] = (
                proxy.evidence_sufficiency(evidence, rc["min_score"]))

    # -- Tier 2: verify ----------------------------------------------------- #
    verify_trigger = (
        conflict
        or (retrieved and signals.evidence_sufficiency < tcfg["evidence_low"])
        or (retrieved and signals.retrieval_disagreement >= tcfg["retrieval_disagreement_high"])
    )
    # Cost guard: skip verify when the answer is long enough that a grounding pass
    # would cost more than cost_multiplier_cap x the main pass. The per-pass
    # overhead is a configured constant (verify.overhead_tokens), not a measurement.
    projected_mult = (main_tokens + vcfg["overhead_tokens"]) / main_tokens
    verified = False
    verify_pass = None
    if verify_trigger and projected_mult <= vcfg["cost_multiplier_cap"] and vcfg["max_rounds"] >= 1:
        if not retrieved:
            evidence = get_store().search(req.message, top_k=rc["top_k"], min_score=rc["min_score"])
            retrieved = True
            if enabled["evidence_sufficiency"]:
                signals.evidence_sufficiency, signals.detail["evidence_sufficiency"] = (
                    proxy.evidence_sufficiency(evidence, rc["min_score"]))
        v = await verify(client, req.message, answer, evidence)
        verified = True
        verify_pass = v["pass"]
        tier = 2
        signals.detail["verify"] = {"pass": v["pass"], "reason": v["reason"]}
        if v["pass"]:
            answer = v["revised"]

    return {"answer": answer, "signals": signals, "evidence": evidence, "tier": tier,
            "retrieved": retrieved, "verified": verified, "verify_pass": verify_pass,
            "should_retrieve": should_retrieve, "verify_trigger": verify_trigger}


async def route_and_answer(req: ChatRequest) -> ChatResponse:
    cfg = load_router_config()
    tcfg, ecfg = cfg["tiers"], cfg["escalation"]
    pfcfg, memcfg, olcfg = cfg["prefilter"], cfg["memory"], cfg["online_learning"]
    t_start = time.perf_counter()

    small_id = pick_model(req)
    big_id = (req.escalate_to or (get_model(small_id) or {}).get("escalate_to")
              or ecfg.get("default_target"))
    # Never escalate into the mock fallback (e.g. a paid model with no key set):
    # a fake escalation is worse than staying on the free lane.
    big_valid = bool(big_id and get_model(big_id) and big_id != small_id
                     and get_registry().resolve(big_id)[2] != "mock")

    prompt = (req.system or SYSTEM_PROMPT) + " " + req.message
    lc = long_context_route(prompt, tcfg["long_context_tokens"])

    # -- Tier T: deterministic tool, before any LLM spend ------------------- #
    if cfg["tools"]["calculator"]:
        sol = calculator.solve(req.message)
        if sol["handled"]:
            return _tool_response(sol, lc, t_start)

    # -- Tier -1: hybrid predict-then-route pre-filter ---------------------- #
    mem = (memory.lookup(req.message, memcfg["k"], memcfg["min_similarity"])
           if memcfg["enabled"] else {"hit": False, "n": 0})
    pf = (prefilter.predict(req.message, mem, pfcfg)
          if pfcfg["enabled"] else {"route": "normal", "difficulty": None, "detail": {}})

    escalation_on = bool(ecfg["enabled"] and req.allow_escalation and ecfg["max_escalations"] >= 1)
    prefiltered_to_big = pf["route"] == "hard_direct" and escalation_on and big_valid
    fast_path = pf["route"] == "easy_direct"

    # -- Assessment on the started model ------------------------------------ #
    # The deep probes are how we decide whether to escalate, so skip them when no
    # escalation decision is left to make: the easy fast path, or a pre-filtered
    # run that already started on the big model.
    start_id = big_id if prefiltered_to_big else small_id
    start_client = LLMClient(start_id)
    deep = req.max_signals and not fast_path and not prefiltered_to_big
    a = await _assess(start_client, req, cfg, deep)

    # Pre-filter went straight to the big model and it failed (timeout/empty):
    # fall back to the small model. A provider failure must never become an abstain.
    pf_big_failed = False
    if prefiltered_to_big and (_degenerate(a["answer"]) or start_client.provider_label == "mock"):
        fb_client = LLMClient(small_id)
        a_small = await _assess(fb_client, req, cfg, False)
        if not _degenerate(a_small["answer"]) and fb_client.provider_label != "mock":
            pf_big_failed = True
            a = a_small
            start_client = fb_client   # the failed big call cost ~0 tokens

    abstain_flag, risk, adetail = should_abstain(a["signals"], cfg["abstain"])
    rescued = a["verified"] and a["verify_pass"]
    would_abstain = abstain_flag and not rescued
    small_risk = risk                  # captured before escalation overwrites it
    # Observe-then-allocate: the cheap model visibly failed (no usable answer, or
    # verify rejected it), so spend up regardless of the risk score.
    force_escalate = _degenerate(a["answer"]) or a["verify_pass"] is False

    chosen = a
    clients = [start_client]
    models_used = [start_id]
    escalated = False
    escalated_to = None
    risk_high = online.current_risk_high(small_id, ecfg["risk_high"], olcfg)
    esc = {"enabled": ecfg["enabled"], "risk_high": risk_high, "small_risk": small_risk,
           "triggered": False, "prefilter_route": pf["route"]}

    if prefiltered_to_big:
        # Big-model compute was spent up front on a predicted-hard prompt.
        escalated = True
        escalated_to = big_id
        if pf_big_failed:
            models_used = [big_id, small_id]
        esc.update({"triggered": True, "via": "prefilter", "to": big_id,
                    "reason": "big model failed (timeout/empty) -> answered by the small model"
                              if pf_big_failed else
                              "pre-filter predicted hard -> direct to big model"})

    elif escalation_on and big_valid and (would_abstain or force_escalate or risk >= risk_high):
        big_client = LLMClient(big_id)
        # Light assessment on the big model: we want its answer, not another
        # escalation decision. The degenerate-answer guard still catches failure.
        b = await _assess(big_client, req, cfg, False)
        clients.append(big_client)
        escalated = True
        escalated_to = big_id
        models_used.append(big_id)
        if (_degenerate(b["answer"]) or big_client.provider_label == "mock") and not _degenerate(a["answer"]):
            # Big model failed: keep the small model's answer rather than serving a
            # mock answer or abstaining because of a provider problem.
            esc.update({"triggered": True, "via": "failure", "from": small_id, "to": big_id,
                        "reason": "big model failed (timeout/empty) -> served the small model's answer"})
        else:
            b_abstain, b_risk, adetail = should_abstain(b["signals"], cfg["abstain"])
            rescued = b["verified"] and b["verify_pass"]
            chosen, risk = b, b_risk
            would_abstain = b_abstain and not rescued
            esc.update({"triggered": True, "via": "failure" if force_escalate else "signals",
                        "from": small_id, "to": big_id, "big_risk": b_risk,
                        "reason": ("small model returned no usable answer" if force_escalate
                                   else "small-model risk high / would-abstain") + " -> traded up"})

    # -- Aggregate cost across every model touched -------------------------- #
    total = CostMetrics()
    for c in clients:
        total.tokens_in += c.cost.tokens_in
        total.tokens_out += c.cost.tokens_out
        total.est_cost_usd += c.cost.est_cost_usd
        total.llm_calls += c.cost.llm_calls
    # Decode costs roughly an order of magnitude more per token than prefill on
    # real hardware; the proxy weights it accordingly.
    total.compute_units = round(total.tokens_in + 10.0 * total.tokens_out, 1)
    total.routing_overhead_passes = max(0, total.llm_calls - 1)

    signals = chosen["signals"]
    signals.detail["escalation"] = esc
    signals.detail["abstain"] = adetail
    signals.detail["prefilter"] = pf["detail"]
    answer = chosen["answer"]

    # Retrieved documents also consume context window.
    if chosen["evidence"] and not lc["long_context"]:
        lc = long_context_route(prompt, tcfg["long_context_tokens"], chosen["evidence"])

    # Never serve a non-answer: if even the chosen model came back degenerate, abstain.
    abstained = would_abstain or _degenerate(answer)
    if abstained:
        tier = 3
        signals.detail["abstain"]["candidate_withheld"] = answer[:300]
        answer = _ABSTAIN_MSG
    else:
        tier = chosen["tier"]

    if memcfg["enabled"]:
        memory.record(req.message, tier, escalated, risk, models_used[-1])
    if not prefiltered_to_big:
        online.update(small_id, ecfg["risk_high"], escalated, small_risk, esc.get("big_risk"), olcfg)

    return ChatResponse(
        answer=answer,
        status="PENDING_REVIEW" if abstained else "OK",
        model=models_used[-1],
        provider=clients[-1].provider_label,
        route=RouteDecision(
            tier=tier, tier_name=TIER_NAMES[tier], long_context=lc["long_context"],
            retrieved=chosen["retrieved"], verified=chosen["verified"],
            escalated=escalated, escalated_to=escalated_to, models_used=models_used,
            prefilter_route=pf["route"], predicted_difficulty=pf["difficulty"],
            abstained=abstained, abstain_risk=risk,
            reason=_reason(escalated, prefiltered_to_big, chosen, abstained, lc,
                           small_id, escalated_to),
        ),
        signals=signals, cost=total,
        latency_ms=round((time.perf_counter() - t_start) * 1000, 1),
    )


def _tool_response(sol: dict, lc: dict, t_start: float) -> ChatResponse:
    """The calculator solved the request exactly: no model call, no cost."""
    signals = SignalSet()
    signals.detail["tool"] = {"name": "calculator", "expression": sol["expression"],
                              "exact": sol["exact"]}
    return ChatResponse(
        answer=sol["answer"], status="OK", model="calculator-tool", provider="tool",
        route=RouteDecision(
            tier=0, tier_name="tool", long_context=lc["long_context"], verified=True,
            models_used=["calculator"], prefilter_route="tool",
            reason=f"exact arithmetic via calculator tool (no LLM): "
                   f"{sol['expression']} = {sol['answer']}",
        ),
        signals=signals, cost=CostMetrics(),
        latency_ms=round((time.perf_counter() - t_start) * 1000, 1),
    )


def _reason(escalated, prefiltered_to_big, chosen, abstained, lc, small_id, big_id) -> str:
    bits = []
    if lc["long_context"]:
        bits.append(f"Tier-L long-context ({lc['approx_prompt_tokens']} tok, detect-only flag)")
    if abstained:
        bits.append("multi-signal risk over threshold; verify/escalation did not rescue -> abstain")
    elif prefiltered_to_big:
        bits.append(f"pre-filter predicted hard -> direct to big model {big_id} (skipped small pass)")
    elif escalated:
        bits.append(f"small model unreliable -> escalated {small_id} -> {big_id}")
    elif chosen["verified"]:
        bits.append("conflict/low-evidence -> verify pass")
    elif chosen["verify_trigger"]:
        bits.append("conflict/low-evidence, but verify skipped by the cost guard")
    elif chosen["should_retrieve"]:
        bits.append("U over tau_star -> retrieval")
    else:
        bits.append("confident single pass")
    return "; ".join(bits)
