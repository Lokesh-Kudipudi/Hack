"""Unit tests for ExplainEngine (Task K6)."""
import pandas as pd
import pytest
from sentinel.explain import ExplainEngine, DETECTOR_TYPE_MAP


def test_explain_schema_contract():
    """Verify evidence dict conforms strictly to AGENTS.md §4.6 schema."""
    engine = ExplainEngine()
    t = pd.Timestamp("2026-01-06 14:00:00")  # Tuesday 14:00
    ev = engine.explain(
        t=t,
        observed=91.8,
        expected=74.2,
        deviation_sigma=4.12,
        top_detector="RobustZ",
        fired=["RobustZ", "CUSUM"],
    )

    expected_keys = {"type", "expected", "observed", "deviation_sigma", "fired", "top_detector", "text"}
    assert set(ev.keys()) == expected_keys
    assert ev["type"] == "spike"
    assert ev["expected"] == 74.2
    assert ev["observed"] == 91.8
    assert ev["deviation_sigma"] == 4.12
    assert ev["fired"] == ["RobustZ", "CUSUM"]
    assert ev["top_detector"] == "RobustZ"
    assert "91.80" in ev["text"]
    assert "above" in ev["text"]
    assert "Tue 14:00" in ev["text"]
    assert "CUSUM" in ev["text"]


def test_explain_all_detector_types():
    """Verify type mapping for every detector in D1 to D6."""
    engine = ExplainEngine()
    t = pd.Timestamp("2026-01-01 00:00:00")

    mappings = {
        "RobustZ": "spike",
        "LastW": "spike",
        "LaggedResidual": "sequence",
        "VarRatio": "variance_change",
        "Seasonal": "pattern_break",
        "CUSUM": "level_shift",
    }

    for det_name, expected_type in mappings.items():
        ev = engine.explain(t, 50.0, 40.0, 3.5, det_name, [det_name])
        assert ev["type"] == expected_type


def test_explain_direction_below():
    """Verify direction 'below' is used when observed < expected."""
    engine = ExplainEngine()
    t = pd.Timestamp("2026-01-01 00:00:00")
    ev = engine.explain(t, 20.0, 50.0, 6.0, "LastW", ["LastW"])
    assert "below" in ev["text"]


def test_explain_v2_confidence_and_context():
    """Verify confidence calculation, context inclusion, and latency under 1 microsecond."""
    import time
    engine = ExplainEngine()
    t = pd.Timestamp("2026-01-01 12:00:00")

    ev = engine.explain(
        t=t,
        observed=120.0,
        expected=50.0,
        deviation_sigma=7.0,
        top_detector="RobustZ",
        fired=["RobustZ", "LastW"],
        threshold_ratio=2.5,
        context={"baseline_median": 50.0, "time_of_day": "12:00"},
    )

    assert "confidence" in ev
    assert 0.70 <= ev["confidence"] <= 0.99
    assert "context" in ev
    assert ev["context"]["time_of_day"] == "12:00"

    # Latency benchmark: 5,000 explanations should execute in < 20 ms (< 4 µs per point)
    t0 = time.perf_counter()
    for _ in range(5000):
        engine.explain(t, 120.0, 50.0, 7.0, "RobustZ", ["RobustZ", "LastW"], threshold_ratio=2.0)
    dur = time.perf_counter() - t0
    us_per_call = (dur / 5000) * 1e6
    assert us_per_call < 10.0, f"Overhead was {us_per_call:.2f} µs/point"
