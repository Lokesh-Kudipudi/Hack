"""Comprehensive robustness, fuzz, and edge-case unit tests for Sentinel (Task K15).

Tests:
- NaN, Inf, -Inf, None, and string values
- Duplicate and backward timestamps
- Large irregular time gaps
- Completely constant series (scale = 0 edge cases)
- Extreme magnitudes (numerical stability against overflow)
- Fuzz testing combining random corruptions
"""
import math
import random
import numpy as np
import pandas as pd
import pytest

from sentinel.pipeline import Sentinel
from sentinel.validate import (
    sanitize_timestamp,
    sanitize_value,
    sanitize_observation,
    MAX_SAFE_MAGNITUDE,
)


def test_sanitize_value_clean_numbers():
    val, flags = sanitize_value(42.5)
    assert val == 42.5
    assert not flags.has_data_quality_issue

    val_int, flags_int = sanitize_value(10)
    assert val_int == 10.0
    assert not flags_int.has_data_quality_issue


def test_sanitize_value_nan_none():
    val, flags = sanitize_value(float("nan"), fallback=5.0)
    assert val == 5.0
    assert flags.was_nan
    assert flags.has_data_quality_issue

    val_none, flags_none = sanitize_value(None, fallback=12.0)
    assert val_none == 12.0
    assert flags_none.was_missing

    val_str, flags_str = sanitize_value("not_a_number", fallback=7.0)
    assert val_str == 7.0
    assert flags_str.raw_type_invalid


def test_sanitize_value_inf_and_extreme():
    val_inf, flags_inf = sanitize_value(float("inf"))
    assert val_inf == MAX_SAFE_MAGNITUDE
    assert flags_inf.was_inf

    val_ninf, flags_ninf = sanitize_value(float("-inf"))
    assert val_ninf == -MAX_SAFE_MAGNITUDE
    assert flags_ninf.was_inf

    val_huge, flags_huge = sanitize_value(1e25)
    assert val_huge == MAX_SAFE_MAGNITUDE
    assert flags_huge.was_extreme_magnitude


def test_sanitize_timestamp_edge_cases():
    base_t = pd.Timestamp("2026-01-01 12:00:00")

    # Clean normal timestamp
    t_next = base_t + pd.Timedelta(minutes=5)
    clean_t, flags = sanitize_timestamp(t_next, last_t=base_t)
    assert clean_t == t_next
    assert not flags.is_duplicate_timestamp
    assert not flags.is_backward_timestamp
    assert flags.time_gap_seconds == 300.0

    # Duplicate timestamp
    clean_t2, flags2 = sanitize_timestamp(base_t, last_t=base_t)
    assert clean_t2 == base_t
    assert flags2.is_duplicate_timestamp

    # Backward timestamp
    t_prev = base_t - pd.Timedelta(minutes=10)
    clean_t3, flags3 = sanitize_timestamp(t_prev, last_t=base_t)
    assert clean_t3 == t_prev
    assert flags3.is_backward_timestamp

    # None timestamp
    clean_t4, flags4 = sanitize_timestamp(None, last_t=base_t)
    assert clean_t4 > base_t
    assert flags4.was_missing


def test_constant_series_never_divides_by_zero():
    """A flatline series (e.g. constant sensor at 42.0) must not crash or output NaN."""
    s = Sentinel(n_total=500)
    t = pd.Timestamp("2026-01-01 00:00:00")

    for i in range(500):
        t += pd.Timedelta(minutes=5)
        res = s.update(t, 42.0)
        assert math.isfinite(res.score), f"Score was not finite at step {i}: {res.score}"
        assert res.score >= 0.0 and res.score <= 1.0
        assert res.label in ("NORMAL", "ANOMALOUS")
        for dname, dscore in res.detector_scores.items():
            assert math.isfinite(dscore), f"{dname} emitted non-finite score {dscore}"


def test_nan_and_inf_stream_resilience():
    """Streaming NaN and Inf values must be handled smoothly without throwing exceptions."""
    s = Sentinel(n_total=200)
    t = pd.Timestamp("2026-01-01 00:00:00")

    # Feed normal points with occasional corruptions
    corrupt_inputs = [None, float("nan"), float("inf"), float("-inf"), "corrupt_str", 1e20]

    for i in range(300):
        t += pd.Timedelta(minutes=5)
        if i in (10, 25, 50, 100, 150, 220, 250):
            val = corrupt_inputs[i % len(corrupt_inputs)]
        else:
            val = 50.0 + math.sin(i * 0.1) * 5.0

        res = s.update(t, val)
        assert math.isfinite(res.score)
        assert 0.0 <= res.score <= 1.0
        assert res.label in ("NORMAL", "ANOMALOUS")


def test_timestamp_anomalies_resilience():
    """Duplicate, backward, and missing timestamps must not cause crashes."""
    s = Sentinel(n_total=100)
    t = pd.Timestamp("2026-01-01 00:00:00")

    for i in range(200):
        if i == 30:
            # Duplicate
            feed_t = t
        elif i == 45:
            # Backward jump
            feed_t = t - pd.Timedelta(hours=2)
        elif i == 60:
            # Massive gap forward
            feed_t = t + pd.Timedelta(days=14)
            t = feed_t
        elif i == 75:
            feed_t = None
        else:
            t += pd.Timedelta(minutes=5)
            feed_t = t

        val = 10.0 + (50.0 if i == 120 else 0.0)
        res = s.update(feed_t, val)
        assert math.isfinite(res.score)


def test_fuzz_stream():
    """Fuzz test combining random noise, corruptions, timestamp skips, and spikes."""
    random.seed(42)
    np.random.seed(42)

    s = Sentinel(n_total=200)
    t = pd.Timestamp("2026-01-01 00:00:00")

    weird_values = [
        None,
        float("nan"),
        float("inf"),
        float("-inf"),
        "bad",
        1e18,
        -1e18,
        0.0,
        1e-12,
    ]

    for i in range(400):
        rand_case = random.random()
        if rand_case < 0.05:
            val = random.choice(weird_values)
        elif rand_case < 0.10:
            val = 5000.0  # Massive spike
        else:
            val = float(np.random.normal(100.0, 10.0))

        if rand_case < 0.02:
            feed_t = None
        elif rand_case < 0.04:
            feed_t = t - pd.Timedelta(minutes=15)
        else:
            t += pd.Timedelta(minutes=random.choice([1, 5, 10, 60]))
            feed_t = t

        res = s.update(feed_t, val)
        assert math.isfinite(res.score), f"Step {i}: non-finite score {res.score}"
        assert 0.0 <= res.score <= 1.0, f"Step {i}: score out of bounds {res.score}"
        assert res.label in ("NORMAL", "ANOMALOUS")
