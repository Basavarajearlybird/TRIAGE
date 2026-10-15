"""The hard invariant named explicitly in the directive: NO single signal may
force an abstain. Plus the abstain risk formula."""
from app.config import load_router_config
from app.router.abstain import abstain_risk, should_abstain
from app.schemas import SignalSet


def _cfg():
    return load_router_config()["abstain"]


def test_risk_is_zero_when_all_signals_clean():
    s = SignalSet()  # defaults: everything 0 except evidence_sufficiency=1 -> insuff 0
    risk, comps = abstain_risk(s, _cfg())
    assert risk == 0.0
    assert comps["evidence_insufficiency"] == 0.0


def test_risk_is_weighted_average_of_components():
    cfg = _cfg()
    s = SignalSet(uncertainty=1.0, instability=1.0, contradiction=1.0,
                  retrieval_disagreement=1.0, evidence_sufficiency=0.0)
    risk, _ = abstain_risk(s, cfg)
    assert risk == 1.0  # all components 1.0 -> weighted mean 1.0


def test_single_maxed_signal_never_abstains():
    """Uncertainty pinned to 1.0, everything else clean: must NOT abstain."""
    cfg = _cfg()
    s = SignalSet(uncertainty=1.0)
    decision, risk, detail = should_abstain(s, cfg)
    assert decision is False
    assert detail["elevated_count"] == 1


def test_guard_blocks_abstain_even_if_one_signal_dominates_risk():
    """Directly exercise the min_elevated_signals guard: craft a config where a
    single signal is enough to push risk over the threshold, and confirm the
    guard STILL refuses to abstain because only one signal is elevated."""
    cfg = dict(_cfg())
    cfg["weights"] = {"uncertainty": 1.0, "instability": 0.0, "contradiction": 0.0,
                      "retrieval_disagreement": 0.0, "evidence_insufficiency": 0.0}
    s = SignalSet(uncertainty=1.0)  # risk -> 1.0, but only ONE elevated signal
    decision, risk, detail = should_abstain(s, cfg)
    assert risk >= cfg["abstain_threshold"]
    assert decision is False               # guard wins over the raw risk score
    assert detail["elevated_count"] == 1


def test_two_elevated_signals_over_threshold_do_abstain():
    cfg = _cfg()
    s = SignalSet(uncertainty=1.0, instability=1.0, contradiction=1.0,
                  retrieval_disagreement=1.0, evidence_sufficiency=0.0)
    decision, risk, detail = should_abstain(s, cfg)
    assert decision is True
    assert detail["elevated_count"] >= cfg["min_elevated_signals"]
    assert detail["single_signal_guard"] == "passed"
