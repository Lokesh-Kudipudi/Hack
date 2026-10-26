"""Unit tests for SPOT and DSPOT Extreme Value Theory thresholding (Task K3)."""
import numpy as np
import pytest

from sentinel.spot import SPOT, DSPOT


def test_spot_initialization():
    """Verify SPOT fits initial threshold t and extreme threshold z_q > t."""
    np.random.seed(42)
    vals = list(np.random.normal(0, 1.0, size=500))
    spot = SPOT(q=1e-3, init_level=0.98)
    spot.fit_from_probation(vals)

    assert spot.is_initialized
    assert spot.t > 1.5  # 98th percentile of N(0, 1) is ~2.05
    assert spot.z_q > spot.t  # Extreme threshold must be strictly higher
    assert spot.sigma > 0


def test_spot_small_probation_fallback():
    """Ensure small probation windows don't crash and trigger fallback."""
    vals = [1.0, 2.0, 1.5, 3.0, 2.2]
    spot = SPOT(min_excesses=8)
    spot.fit_from_probation(vals)

    assert spot.is_initialized
    assert spot.z_q >= 3.0


def test_spot_false_alarm_rate():
    """Verify false alarm rate on Gaussian noise matches target risk level q."""
    np.random.seed(42)
    prob_vals = list(np.random.normal(0, 1.0, size=1000))
    # Target risk level q = 0.01 (1% false alarm rate)
    spot = SPOT(q=0.01, init_level=0.95)
    spot.fit_from_probation(prob_vals)

    test_vals = np.random.normal(0, 1.0, size=5000)
    alarms = 0
    for x in test_vals:
        _, is_alarm = spot.update(x)
        if is_alarm:
            alarms += 1

    observed_rate = alarms / len(test_vals)
    # Observed false alarm rate should be close to 0.01 (within [0.002, 0.03])
    assert 0.002 <= observed_rate <= 0.03


def test_spot_extreme_outlier():
    """Verify an extreme outlier returns is_alarm=True and score=1.0."""
    np.random.seed(42)
    prob_vals = list(np.random.normal(0, 1.0, size=500))
    spot = SPOT(q=1e-3, init_level=0.98)
    spot.fit_from_probation(prob_vals)

    score, is_alarm = spot.update(25.0)  # Extreme 25-sigma outlier
    assert is_alarm is True
    assert score >= 0.99


def test_dspot_drift_handling():
    """Verify DSPOT handles a linear drifting series and flags an added spike."""
    np.random.seed(42)
    # Series with a strong upward trend
    t = np.arange(200)
    trend = 0.5 * t
    noise = np.random.normal(0, 0.5, size=200)
    prob_vals = list(trend + noise)

    dspot = DSPOT(depth=20, q=1e-3)
    dspot.fit_from_probation(prob_vals)

    # Stream continuing the trend
    for i in range(200, 250):
        val = 0.5 * i + np.random.normal(0, 0.5)
        score, is_alarm = dspot.update(val)
        assert is_alarm is False  # Trend alone should not trigger alarms

    # Massive spike superimposed on top of current trend
    spike_val = (0.5 * 251) + 20.0
    score, is_alarm = dspot.update(spike_val)
    assert is_alarm is True
    assert score >= 0.95
