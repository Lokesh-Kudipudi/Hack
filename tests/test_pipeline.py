"""Unit tests for Sentinel pipeline (Task K1)."""
import pandas as pd
import numpy as np
import pytest

from sentinel.pipeline import Sentinel, run_series
from sentinel.types import Result


def test_sentinel_initialization():
    s = Sentinel(n_total=1000)
    # min(floor(0.15 * 1000), 750) = 150
    assert s.probation_limit == 150
    assert s.threshold == 6.0


def test_probation_suppression():
    """Ensure no alerts or event starts occur during probation."""
    s = Sentinel(n_total=200)
    # Probation = 30 points
    dates = pd.date_range("2026-01-01", periods=200, freq="5min")

    for i in range(30):
        # Even if there is an extreme spike during probation, it must be suppressed
        val = 999.0 if i == 15 else 10.0
        r = s.update(dates[i], val)
        assert isinstance(r, Result)
        assert r.label == "NORMAL"
        assert not r.is_event_start
        assert r.score == 0.0
        assert r.evidence is None


def test_post_probation_spike_detection():
    """Inject a spike post-probation and verify detection, event start, and evidence."""
    np.random.seed(42)
    s = Sentinel(n_total=200)
    dates = pd.date_range("2026-01-01", periods=200, freq="5min")

    # Feed normal noise for 50 points (probation is 30)
    for i in range(50):
        val = 20.0 + np.random.normal(0, 0.5)
        r = s.update(dates[i], val)
        assert r.label == "NORMAL"

    # Inject massive spike at point 51
    spike_r = s.update(dates[51], 150.0)
    assert spike_r.label == "ANOMALOUS"
    assert spike_r.is_event_start is True
    assert spike_r.score > 0.5
    assert spike_r.evidence is not None
    assert spike_r.evidence["type"] == "spike"
    assert spike_r.evidence["top_detector"] == "RobustZ"
    assert "deviation_sigma" in spike_r.evidence


def test_event_merging():
    """Consecutive anomalous points must have only ONE event start."""
    s = Sentinel(n_total=200)
    dates = pd.date_range("2026-01-01", periods=200, freq="5min")

    for i in range(40):
        s.update(dates[i], 10.0)

    # 3 consecutive spikes
    r1 = s.update(dates[40], 100.0)
    r2 = s.update(dates[41], 100.0)
    r3 = s.update(dates[42], 100.0)

    assert r1.is_event_start is True
    assert r2.is_event_start is False
    assert r3.is_event_start is False


def test_run_series_dataframe():
    """Verify run_series output structure and length."""
    dates = pd.date_range("2026-01-01", periods=100, freq="5min")
    df = pd.DataFrame({"timestamp": dates, "value": np.ones(100) * 5.0})

    out = run_series(df)
    expected_cols = ["timestamp", "value", "score", "label", "is_event_start", "evidence"]
    assert list(out.columns) == expected_cols
    assert len(out) == 100
    assert out["score"].between(0.0, 1.0).all()
