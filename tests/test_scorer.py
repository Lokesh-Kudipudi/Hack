import pandas as pd
import pytest

from eval.scorer import FileScore, aggregate, metrics, score_file

N = 100
TS = pd.date_range("2020-01-01", periods=N, freq="5min")


def W(i, j):
    return (TS[i], TS[j])


def starts(*idx):
    s = [False] * N
    for i in idx:
        s[i] = True
    return s


def sf(pred, windows, probation=10):
    return score_file(TS, pred, windows, probation=probation)


def test_single_tp_with_delay():
    s = sf(starts(53), [W(50, 59)])
    assert (s.tp, s.fp, s.fn) == (1, 0, 0)
    assert s.delay_points == [3]
    assert s.delay_minutes == [15.0]
    assert s.delay_fraction == [pytest.approx(0.3)]


def test_inclusive_bounds():
    assert sf(starts(50), [W(50, 59)]).tp == 1
    assert sf(starts(59), [W(50, 59)]).tp == 1
    s = sf(starts(49, 60), [W(50, 59)])
    assert (s.tp, s.fp, s.fn) == (0, 2, 1)


def test_fp_and_fn():
    s = sf(starts(20), [W(50, 59)])
    assert (s.tp, s.fp, s.fn) == (0, 1, 1)


def test_duplicates_in_window_ignored():
    s = sf(starts(52, 55, 58), [W(50, 59)])
    assert (s.tp, s.fp, s.fn) == (1, 0, 0)
    assert s.delay_points == [2]  # first start counts


def test_multiple_windows():
    s = sf(starts(25, 52, 80), [W(20, 30), W(50, 59), W(70, 75)])
    assert (s.tp, s.fp, s.fn) == (2, 1, 1)


def test_probation_exclusion():
    # prediction in probation dropped; window fully in probation dropped
    s = sf(starts(5), [W(2, 8)], probation=10)
    assert (s.tp, s.fp, s.fn) == (0, 0, 0)
    # window straddling probation is clipped; delay measured from first scored point
    s = sf(starts(5, 12), [W(5, 15)], probation=10)
    assert (s.tp, s.fp, s.fn) == (1, 0, 0)
    assert s.delay_points == [2]
    assert s.delay_fraction == [pytest.approx(2 / 6)]


def test_default_probation_is_15_percent():
    s = score_file(TS, starts(14), [])  # probation = 15 for n=100
    assert s.fp == 0
    s = score_file(TS, starts(15), [])
    assert s.fp == 1


def test_no_anomaly_file():
    s = sf(starts(), [])
    assert (s.tp, s.fp, s.fn) == (0, 0, 0)
    m = metrics([s])
    assert m["precision"] is None and m["recall"] is None and m["f1"] is None
    m = metrics([sf(starts(40), [])])
    assert m["fp"] == 1 and m["precision"] == 0.0 and m["recall"] is None and m["f1"] == 0.0


def test_length_mismatch_raises():
    with pytest.raises(ValueError):
        score_file(TS, [False] * (N - 1), [])


def test_micro_aggregation():
    a = FileScore("c1/a.csv", "c1", tp=2, fp=1, fn=0, delay_points=[1, 3], delay_minutes=[5, 15], delay_fraction=[.1, .3])
    b = FileScore("c2/b.csv", "c2", tp=0, fp=1, fn=2)
    r = aggregate([a, b])
    o = r["overall"]
    assert (o["tp"], o["fp"], o["fn"]) == (2, 2, 2)
    assert o["precision"] == 0.5 and o["recall"] == 0.5 and o["f1"] == 0.5
    assert o["delay_points_mean"] == 2 and o["delay_minutes_median"] == 10
    assert r["per_category"]["c2"]["f1"] == 0.0
    assert r["per_category"]["c1"]["f1"] == pytest.approx(0.8)
    assert set(r["per_file"]) == {"c1/a.csv", "c2/b.csv"}
