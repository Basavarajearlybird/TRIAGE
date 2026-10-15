"""The quality-recovered metric must never report a fake 100% for a run that
lowered accuracy. data/rigor_gsm8k.json is the run that exposed this."""
from app.benchmark.rigor import quality_metrics


def test_router_worse_than_small_is_not_100_percent():
    # The exact broken case from data/rigor_gsm8k.json: router 51% vs baselines ~91-93%.
    qvb, qrec = quality_metrics(small_acc=0.9333, big_acc=0.9111, router_acc=0.5111)
    assert qvb == 56.1                # 51.1 / 91.11
    assert qrec is None               # no positive small->big gap to recover


def test_recovered_none_when_big_not_better_than_small():
    _, qrec = quality_metrics(0.90, 0.85, 0.88)
    assert qrec is None


def test_recovered_is_gap_fraction_when_big_beats_small():
    qvb, qrec = quality_metrics(small_acc=0.60, big_acc=0.80, router_acc=0.70)
    assert qrec == 50.0               # recovered half of the 20-point gap
    assert qvb == 87.5               # 0.70 / 0.80


def test_router_matching_small_recovers_zero_not_hundred():
    _, qrec = quality_metrics(small_acc=0.8667, big_acc=0.9333, router_acc=0.8667)
    assert qrec == 0.0


def test_router_beating_big_exceeds_100():
    qvb, qrec = quality_metrics(small_acc=0.60, big_acc=0.80, router_acc=0.90)
    assert qvb > 100
    assert qrec > 100
