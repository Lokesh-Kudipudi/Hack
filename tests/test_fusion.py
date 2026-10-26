"""Unit tests for FusionEngine (Task K4)."""
import pytest
from sentinel.fusion import FusionEngine


def test_fusion_max_strategy():
    """Verify max fusion takes maximum calibrated score and flags on single alarm if vote_k=1."""
    engine = FusionEngine(strategy="max", vote_k=1)
    scores = {"RobustZ": 0.3, "LastW": 0.85, "Seasonal": 0.1}
    ratios = {"RobustZ": 0.4, "LastW": 1.2, "Seasonal": 0.2}
    alarms = {"RobustZ": False, "LastW": True, "Seasonal": False}

    fused_score, is_anom, top_name, top_ratio, fired = engine.fuse(scores, ratios, alarms)

    assert fused_score == 0.85
    assert is_anom is True
    assert top_name == "LastW"
    assert top_ratio == 1.2
    assert fired == ["LastW"]


def test_fusion_consensus_voting_requires_k():
    """Verify vote_k=2 suppresses a single non-extreme alarm."""
    engine = FusionEngine(strategy="max", vote_k=2, extreme_ratio=1.5)
    scores = {"RobustZ": 0.5, "LastW": 0.8, "VarRatio": 0.2}
    # Only LastW alarms, with ratio 1.1 (< 1.5)
    ratios = {"RobustZ": 0.6, "LastW": 1.1, "VarRatio": 0.3}
    alarms = {"RobustZ": False, "LastW": True, "VarRatio": False}

    fused_score, is_anom, top_name, top_ratio, fired = engine.fuse(scores, ratios, alarms)

    assert fused_score == 0.8
    assert is_anom is False  # Suppressed by consensus vote
    assert top_name == "LastW"
    assert fired == ["LastW"]

    # When a second detector agrees:
    alarms["RobustZ"] = True
    ratios["RobustZ"] = 1.05
    _, is_anom_2, _, _, fired_2 = engine.fuse(scores, ratios, alarms)
    assert is_anom_2 is True  # 2 detectors fired
    assert len(fired_2) == 2


def test_fusion_extreme_outlier_bypass():
    """Verify single extreme outlier (ratio >= 1.5) bypasses consensus requirement."""
    engine = FusionEngine(strategy="max", vote_k=2, extreme_ratio=1.5)
    scores = {"RobustZ": 0.99, "LastW": 0.1}
    ratios = {"RobustZ": 3.5, "LastW": 0.2}
    alarms = {"RobustZ": True, "LastW": False}

    fused_score, is_anom, top_name, top_ratio, fired = engine.fuse(scores, ratios, alarms)

    assert is_anom is True  # Single detector >= 1.5x threshold bypasses vote_k
    assert top_name == "RobustZ"
    assert top_ratio == 3.5


def test_fusion_mean_strategy():
    """Verify mean strategy averages calibrated scores."""
    engine = FusionEngine(strategy="mean", vote_k=1)
    scores = {"D1": 0.2, "D2": 0.4, "D3": 0.6}
    ratios = {"D1": 0.2, "D2": 0.4, "D3": 0.6}
    alarms = {"D1": False, "D2": False, "D3": False}

    fused_score, is_anom, _, _, _ = engine.fuse(scores, ratios, alarms)
    assert round(fused_score, 4) == 0.4
    assert is_anom is False


def test_fusion_empty_input():
    """Verify safe handling of empty inputs."""
    engine = FusionEngine()
    fused_score, is_anom, top_name, top_ratio, fired = engine.fuse({}, {}, {})
    assert fused_score == 0.0
    assert is_anom is False
    assert fired == []
