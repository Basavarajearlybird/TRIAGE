"""Abstain policy: a pure function of the signal set.

No single signal may force an abstain. Both guards must hold: the weighted risk
crosses `abstain_threshold`, and at least `min_elevated_signals` distinct signals
are individually above `min_elevated_level`. tests/test_abstain.py enforces this.
"""
from __future__ import annotations

from ..schemas import SignalSet


def abstain_risk(signals: SignalSet, cfg: dict) -> tuple[float, dict]:
    w = cfg["weights"]
    components = {
        "uncertainty": signals.uncertainty,
        "instability": signals.instability,
        "contradiction": signals.contradiction,
        "retrieval_disagreement": signals.retrieval_disagreement,
        "evidence_insufficiency": 1.0 - signals.evidence_sufficiency,
    }
    total_w = sum(w.values()) or 1.0
    risk = sum(w[k] * components[k] for k in components) / total_w
    return round(risk, 4), components


def should_abstain(signals: SignalSet, cfg: dict) -> tuple[bool, float, dict]:
    risk, components = abstain_risk(signals, cfg)
    elevated = {k: v for k, v in components.items() if v >= cfg["min_elevated_level"]}
    enough_signals = len(elevated) >= cfg["min_elevated_signals"]
    decision = (risk >= cfg["abstain_threshold"]) and enough_signals
    detail = {
        "risk": risk,
        "threshold": cfg["abstain_threshold"],
        "elevated_signals": list(elevated.keys()),
        "elevated_count": len(elevated),
        "min_elevated_signals": cfg["min_elevated_signals"],
        "components": {k: round(v, 4) for k, v in components.items()},
        "single_signal_guard": "passed" if not (decision and len(elevated) < 2) else "violated",
    }
    return decision, risk, detail
