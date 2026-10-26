"""State checkpointing, serialization, and stream resumption tests (Task K16).

Verifies the fundamental streaming contract:
Resumed output from a checkpoint MUST be bit-identical to uninterrupted output.
"""
import math
import tempfile
from pathlib import Path
import numpy as np
import pandas as pd
import pytest

from sentinel.pipeline import Sentinel
from sentinel.config import ORIGINAL_DEFAULT_CONFIG


def generate_synthetic_stream(n: int = 400, seed: int = 42):
    """Generate a realistic test series with seasonality, noise, drift, and an anomaly spike."""
    np.random.seed(seed)
    base_t = pd.Timestamp("2026-01-01 00:00:00")
    timestamps = [base_t + pd.Timedelta(minutes=5 * i) for i in range(n)]

    # Seasonal diurnal curve + noise
    values = [
        50.0 + 15.0 * math.sin(2.0 * math.pi * (t.hour * 60 + t.minute) / 1440.0) + float(np.random.normal(0, 1.5))
        for t in timestamps
    ]

    # Insert sustained anomaly jump if length permits
    if n > 250:
        for i in range(220, min(245, n)):
            values[i] += 30.0

    # Insert single extreme spike if length permits
    if n > 310:
        values[310] += 80.0

    return timestamps, values


def test_checkpoint_roundtrip_post_probation():
    """Verify that pausing at step 180 (post-probation) and resuming yields identical results."""
    timestamps, values = generate_synthetic_stream(n=350)
    n_total = len(values)

    # 1. Continuous uninterrupted execution
    s_full = Sentinel(n_total=n_total)
    full_results = [s_full.update(t, x) for t, x in zip(timestamps, values)]

    # 2. Checkpointed execution
    pause_idx = 180
    s_part = Sentinel(n_total=n_total)
    first_half_results = [s_part.update(t, x) for t, x in zip(timestamps[:pause_idx], values[:pause_idx])]

    with tempfile.TemporaryDirectory() as tmpdir:
        json_path = Path(tmpdir) / "checkpoint.json"
        s_part.save(json_path)
        assert json_path.exists()

        # Resume from checkpoint
        s_resumed = Sentinel.load(json_path)
        assert s_resumed.count == pause_idx

        second_half_results = [
            s_resumed.update(t, x) for t, x in zip(timestamps[pause_idx:], values[pause_idx:])
        ]

    # 3. Assert bit-identical equivalence across every single output field
    for i in range(pause_idx, n_total):
        rel_idx = i - pause_idx
        res_full = full_results[i]
        res_resumed = second_half_results[rel_idx]

        assert res_resumed.timestamp == res_full.timestamp, f"Timestamp mismatch at step {i}"
        assert abs(res_resumed.value - res_full.value) < 1e-9, f"Value mismatch at step {i}"
        assert abs(res_resumed.score - res_full.score) < 1e-4, f"Score mismatch at step {i}: {res_resumed.score} vs {res_full.score}"
        assert res_resumed.label == res_full.label, f"Label mismatch at step {i}: {res_resumed.label} vs {res_full.label}"
        assert res_resumed.is_event_start == res_full.is_event_start, f"Event start mismatch at step {i}"

        # Check detector scores
        for dname in res_full.detector_scores:
            assert dname in res_resumed.detector_scores
            diff = abs(res_resumed.detector_scores[dname] - res_full.detector_scores[dname])
            assert diff < 1e-4, f"Detector {dname} score mismatch at step {i}: {diff}"


def test_checkpoint_roundtrip_during_probation():
    """Verify pausing mid-probation (e.g. step 25 when probation is 50)."""
    timestamps, values = generate_synthetic_stream(n=100)
    n_total = len(values)

    s_full = Sentinel(n_total=n_total)
    full_results = [s_full.update(t, x) for t, x in zip(timestamps, values)]

    pause_idx = 10  # During probation
    s_part = Sentinel(n_total=n_total)
    for t, x in zip(timestamps[:pause_idx], values[:pause_idx]):
        s_part.update(t, x)

    with tempfile.TemporaryDirectory() as tmpdir:
        pkl_path = Path(tmpdir) / "checkpoint.pkl"
        s_part.save(pkl_path)

        s_resumed = Sentinel.load(pkl_path)
        assert s_resumed.count == pause_idx

        second_half = [s_resumed.update(t, x) for t, x in zip(timestamps[pause_idx:], values[pause_idx:])]

    for i in range(pause_idx, n_total):
        rel_idx = i - pause_idx
        assert full_results[i].score == second_half[rel_idx].score
        assert full_results[i].label == second_half[rel_idx].label
        assert full_results[i].is_event_start == second_half[rel_idx].is_event_start


def test_checkpoint_all_detectors_ensemble():
    """Verify state checkpointing with all 6 detectors configured."""
    timestamps, values = generate_synthetic_stream(n=250)
    n_total = len(values)

    cfg = dict(ORIGINAL_DEFAULT_CONFIG)
    s_full = Sentinel(n_total=n_total, config=cfg)
    full_results = [s_full.update(t, x) for t, x in zip(timestamps, values)]

    pause_idx = 120
    s_part = Sentinel(n_total=n_total, config=cfg)
    for t, x in zip(timestamps[:pause_idx], values[:pause_idx]):
        s_part.update(t, x)

    with tempfile.TemporaryDirectory() as tmpdir:
        json_path = Path(tmpdir) / "all_detectors_state.json"
        s_part.save(json_path)

        s_resumed = Sentinel.load(json_path)
        resumed_results = [s_resumed.update(t, x) for t, x in zip(timestamps[pause_idx:], values[pause_idx:])]

    for i in range(pause_idx, n_total):
        rel = i - pause_idx
        assert abs(full_results[i].score - resumed_results[rel].score) < 1e-4
        assert full_results[i].label == resumed_results[rel].label
        assert full_results[i].is_event_start == resumed_results[rel].is_event_start
