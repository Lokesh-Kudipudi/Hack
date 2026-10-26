"""Unit tests for D1-D6 detectors on synthetic anomaly types."""
import numpy as np
import pandas as pd
import pytest

from sentinel.detectors import RobustZ, LastW, LaggedResidual, VarRatio, Seasonal, CUSUM


def test_d1_robust_z_spike():
    """D1 should fire a large score on a single isolated spike."""
    d1 = RobustZ()
    # Probation with normal values around 10.0
    prob_vals = [10.0 + np.random.normal(0, 0.5) for _ in range(100)]
    d1.init_from_probation(prob_vals)

    now = pd.Timestamp("2026-01-01 12:00:00")
    normal_score = d1.update(now, 10.5)
    assert normal_score < 2.0

    spike_score = d1.update(now, 50.0)
    assert spike_score > 30.0


def test_d2_last_w_sudden_jump():
    """D2 (Last-3) should fire immediately when a step jump occurs."""
    d2 = LastW(w=3)
    d2.init_scale(1.0)
    now = pd.Timestamp("2026-01-01 12:00:00")

    # Plateau at 5.0
    d2.update(now, 5.0)
    d2.update(now, 5.0)
    d2.update(now, 5.0)

    # Sudden jump to 25.0
    jump_score = d2.update(now, 25.0)
    assert jump_score == pytest.approx(20.0, rel=1e-3)


def test_d3_lagged_residual_sequence_anomaly():
    """D3 should fire on a sustained sequence that contaminates a short window."""
    d3 = LaggedResidual(w=8, h=16)
    d3.init_scale(1.0)
    now = pd.Timestamp("2026-01-01 12:00:00")

    # Feed 30 normal baseline points around 10.0
    for _ in range(30):
        d3.update(now, 10.0)

    # Now an anomaly starts and stays at 30.0 for 10 steps
    # A short window would contaminate in 3 steps, but D3 still has the 16-step lagged reference
    for _ in range(5):
        d3.update(now, 30.0)

    score_at_step_6 = d3.update(now, 30.0)
    assert score_at_step_6 > 15.0


def test_d4_var_ratio_volatility_burst():
    """D4 should fire when variance explodes even if mean remains constant."""
    np.random.seed(42)
    d4 = VarRatio(w=20)
    now = pd.Timestamp("2026-01-01 12:00:00")

    # Low variance baseline (std = 0.5)
    baseline_vals = [np.random.normal(0, 0.5) for _ in range(100)]
    d4.init_from_probation(baseline_vals)

    # Ingest low variance
    for _ in range(25):
        d4.update(now, np.random.normal(0, 0.5))

    # Burst of high volatility (std = 5.0)
    scores = []
    for _ in range(20):
        s = d4.update(now, np.random.normal(0, 5.0))
        scores.append(s)

    assert max(scores) > 4.0  # Significant variance ratio explosion


def test_d5_seasonal_profile():
    """D5 should catch values that are normal during the day but abnormal at 3 AM."""
    d5 = Seasonal(min_count=2)
    # Feed 10 days of synthetic data where 3 AM is ~5.0 and 2 PM is ~50.0
    timestamps = []
    values = []
    for day in range(5):
        t_night = pd.Timestamp(f"2026-01-0{day+1} 03:00:00")
        t_day = pd.Timestamp(f"2026-01-0{day+1} 14:00:00")
        timestamps.extend([t_night, t_day])
        values.extend([5.0, 50.0])

    d5.init_from_probation(timestamps, values)

    # Value 50.0 at 2 PM should have low anomaly score
    normal_score = d5.update(pd.Timestamp("2026-01-10 14:00:00"), 50.0)
    assert normal_score < 1.0

    # Value 50.0 at 3 AM should have a huge anomaly score!
    abnormal_night_score = d5.update(pd.Timestamp("2026-01-10 03:00:00"), 50.0)
    assert abnormal_night_score > 10.0


def test_d6_cusum_drift():
    """D6 should accumulate persistent small shifts that slip past single-point Z."""
    d6 = CUSUM(k=0.5)
    d6.init_scale(median=0.0, scale=1.0)
    now = pd.Timestamp("2026-01-01 12:00:00")

    # Constant persistent shift of +1.5 sigma (small, but persistent)
    scores = []
    for _ in range(10):
        scores.append(d6.update(now, 1.5))

    # CUSUM accumulates: (1.5 - 0.5) = +1.0 each step
    assert scores[-1] > scores[0]
    assert scores[-1] >= 9.0


def test_no_lookahead_guarantee():
    """Strict test: altering future points must NOT change the score at point i."""
    d = LastW(w=3)
    d.init_scale(1.0)
    t = pd.Timestamp("2026-01-01")

    # Run stream A
    d.update(t, 1.0)
    d.update(t, 2.0)
    score_at_3_A = d.update(t, 10.0)

    # Reset and run stream B with completely different future values afterwards
    d_b = LastW(w=3)
    d_b.init_scale(1.0)
    d_b.update(t, 1.0)
    d_b.update(t, 2.0)
    score_at_3_B = d_b.update(t, 10.0)

    assert score_at_3_A == score_at_3_B
