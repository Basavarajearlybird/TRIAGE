"""Online escalation-threshold controller.

Tunes `escalation.risk_high` per small model from live outcomes, with no labels.
The feedback signal is self-supervised: an escalation was USEFUL if the big
model's abstain-risk came out materially below the small model's, WASTED if it
came out about the same. The controller steers the escalation rate toward
`target_escalation_rate`, with usefulness as an override, so it escalates more while it
keeps paying off, less when it is mostly wasted.

In-process only, like the routing memory: state lasts one experiment run.
"""
from __future__ import annotations

_STATE: dict[str, dict] = {}
_USEFUL_MARGIN = 0.08     # big risk must beat small risk by this to count as useful
_EMA = 0.2                # smoothing for the running rates


def reset() -> None:
    _STATE.clear()


def _model_state(model: str, default_risk_high: float) -> dict:
    return _STATE.setdefault(
        model, {"risk_high": default_risk_high, "escalation_rate": 0.0, "usefulness": 0.5, "n": 0}
    )


def current_risk_high(model: str, default_risk_high: float, cfg: dict) -> float:
    if not cfg.get("enabled", False):
        return default_risk_high
    return float(_model_state(model, default_risk_high)["risk_high"])


def update(model: str, default_risk_high: float, escalated: bool,
           small_risk: float, big_risk: float | None, cfg: dict) -> None:
    """Record one routed request's outcome and adapt the threshold."""
    if not cfg.get("enabled", False):
        return
    ms = _model_state(model, default_risk_high)
    ms["n"] += 1
    ms["escalation_rate"] = (1 - _EMA) * ms["escalation_rate"] + _EMA * float(escalated)
    if escalated and big_risk is not None:
        useful = float((small_risk - big_risk) >= _USEFUL_MARGIN)
        ms["usefulness"] = (1 - _EMA) * ms["usefulness"] + _EMA * useful

    rate, useful_ema = ms["escalation_rate"], ms["usefulness"]
    target = cfg.get("target_escalation_rate", 0.30)
    step = cfg.get("step", 0.02)
    if rate > target and useful_ema < 0.5:
        ms["risk_high"] = min(cfg.get("risk_high_ceil", 0.65), ms["risk_high"] + step)
    elif rate < target and useful_ema >= 0.5:
        ms["risk_high"] = max(cfg.get("risk_high_floor", 0.25), ms["risk_high"] - step)
    ms["risk_high"] = round(ms["risk_high"], 4)
