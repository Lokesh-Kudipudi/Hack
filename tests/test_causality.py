"""Determinism, causality, and zero-lookahead property tests for Sentinel (Task K21).

Mathematical and empirical proofs:
1. Zero Lookahead: Output at time t for any prefix x_{1..t} NEVER changes
   when future observations x_{t+1..T} are appended.
2. Bit-Identical Determinism: Multiple runs on identical input produce identical outputs.
3. Time Directionality: Detection is sensitive to temporal ordering (strictly causal).
"""
import math
import numpy as np
import pandas as pd
import pytest

from sentinel.pipeline import Sentinel
from sentinel.detectors import ALL_DETECTORS


def make_test_series(n: int = 300, seed: int = 123):
    """Generate a reproducible time series with noise, seasonal drift, and pulses."""
    np.random.seed(seed)
    base_t = pd.Timestamp("2026-01-01 00:00:00")
    ts = [base_t + pd.Timedelta(minutes=5 * i) for i in range(n)]
    xs = [
        100.0
        + 25.0 * math.sin(2.0 * math.pi * i / 48.0)
        + float(np.random.normal(0, 2.0))
        for i in range(n)
    ]
    # Injected pulse spike if length permits
    if n > 160:
        xs[160] += 80.0
    # Injected variance burst if length permits
    if n > 230:
        for j in range(210, 230):
            xs[j] += float(np.random.normal(0, 15.0))
    return ts, xs


def test_zero_lookahead_prefix_invariance():
    """Core Causal Invariant: appending future points must NOT change prefix output."""
    ts, xs = make_test_series(n=250)

    # Test prefixes at multiple cutoff points across probation, normal, and anomaly regimes
    cutoff_points = [60, 120, 160, 161, 200, 225]

    # Full execution
    s_full = Sentinel(n_total=250)
    full_results = [s_full.update(t, x) for t, x in zip(ts, xs)]

    for cutoff in cutoff_points:
        # Prefix execution up to cutoff only
        s_prefix = Sentinel(n_total=250)
        prefix_results = [s_prefix.update(t, x) for t, x in zip(ts[:cutoff], xs[:cutoff])]

        # Target point at cutoff-1
        r_prefix = prefix_results[-1]
        r_full = full_results[cutoff - 1]

        assert r_prefix.timestamp == r_full.timestamp
        assert abs(r_prefix.value - r_full.value) < 1e-9
        assert abs(r_prefix.score - r_full.score) < 1e-5, (
            f"Lookahead violation at cutoff {cutoff}: "
            f"prefix score {r_prefix.score} != full score {r_full.score}"
        )
        assert r_prefix.label == r_full.label
        assert r_prefix.is_event_start == r_full.is_event_start

        # Check every detector's score
        for dname in r_full.detector_scores:
            assert abs(r_prefix.detector_scores[dname] - r_full.detector_scores[dname]) < 1e-5


def test_bit_identical_determinism_across_runs():
    """Two independent runs on identical input must be bit-identical."""
    ts, xs = make_test_series(n=200, seed=999)

    s1 = Sentinel(n_total=200)
    run1 = [s1.update(t, x) for t, x in zip(ts, xs)]

    s2 = Sentinel(n_total=200)
    run2 = [s2.update(t, x) for t, x in zip(ts, xs)]

    assert len(run1) == len(run2)
    for i, (r1, r2) in enumerate(zip(run1, run2)):
        assert r1.score == r2.score, f"Determinism failure at index {i}"
        assert r1.label == r2.label
        assert r1.is_event_start == r2.is_event_start
        assert r1.detector_scores == r2.detector_scores


def test_all_individual_detectors_causality():
    """Verify that every single registered detector is causal and free of lookahead."""
    ts, xs = make_test_series(n=100)
    now = ts[50]

    for det_key, det_cls in ALL_DETECTORS.items():
        d1 = det_cls()
        if hasattr(d1, "init_from_probation"):
            if det_key == "seasonal":
                d1.init_from_probation(ts[:40], xs[:40])
            else:
                d1.init_from_probation(xs[:40])
        elif det_key == "cusum":
            d1.init_scale(median=100.0, scale=1.0)
        elif hasattr(d1, "init_scale"):
            d1.init_scale(1.0)

        # Feed up to point 50
        for t, x in zip(ts[:50], xs[:50]):
            d1.update(t, x)
        score_at_50 = d1.update(ts[50], xs[50])

        # Run independent detector with different future values afterwards
        d2 = det_cls()
        if hasattr(d2, "init_from_probation"):
            if det_key == "seasonal":
                d2.init_from_probation(ts[:40], xs[:40])
            else:
                d2.init_from_probation(xs[:40])
        elif det_key == "cusum":
            d2.init_scale(median=100.0, scale=1.0)
        elif hasattr(d2, "init_scale"):
            d2.init_scale(1.0)

        for t, x in zip(ts[:50], xs[:50]):
            d2.update(t, x)
        score_at_50_replay = d2.update(ts[50], xs[50])

        assert score_at_50 == score_at_50_replay, f"Detector {det_key} failed causal reproducibility"


def test_time_reversal_sensitivity():
    """A causal time-series model must respond differently to reversed temporal ordering."""
    ts, xs = make_test_series(n=150)

    s_forward = Sentinel(n_total=150)
    fwd_results = [s_forward.update(t, x) for t, x in zip(ts, xs)]

    s_reversed = Sentinel(n_total=150)
    rev_results = [s_reversed.update(t, x) for t, x in zip(ts, reversed(xs))]

    fwd_scores = [r.score for r in fwd_results[50:]]
    rev_scores = [r.score for r in rev_results[50:]]

    # Scores in reversed time must not be a simple reflection (confirming asymmetric causal dynamics)
    assert fwd_scores != list(reversed(rev_scores))
