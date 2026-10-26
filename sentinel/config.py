"""Default configuration for Sentinel."""
from typing import Any, Dict

PROBATION_FRACTION = 0.15
PROBATION_MAX = 750


def probation_length(n: int) -> int:
    """Matches eval/data.py: min(floor(0.15*n), 750)."""
    return min(int(PROBATION_FRACTION * n), PROBATION_MAX)


# Default = the event-F1 operating point chosen by leave-one-category-out tuning over 720 global configs
# (results/best_config.json, results/tuning.md): RobustZ + LaggedResidual, MAX fusion, per-detector SPOT risk.
# Full NAB corpus: event-F1 0.563 (P 0.675, R 0.483), held-out pooled 0.510, ~2.4 µs/pt.
# Other operating points: results/best_nab_config.json (NAB 59.06), results/best_joint_config.json.
DEFAULT_CONFIG: Dict[str, Any] = {
    "probation_fraction": PROBATION_FRACTION,
    "probation_max": PROBATION_MAX,
    "probation": 300,  # Fallback when n_total is unknown
    "fixed_threshold": 6.0,  # Base fallback threshold
    "debounce": 2,  # Require 2 consecutive outliers (or 1 extreme spike) (ADR-003)
    "detectors": ["robust_z", "lagged_residual"],
    "detector_thresholds": {
        "RobustZ": 6.0,
        "LastW": 2.0,
        "LaggedResidual": 2.0,
        "VarRatio": 1.5,
        "Seasonal": 5.0,
        "CUSUM": 8.0,
        "StreamingIsolationForest": 6.0,
    },
    "last_w": {"w": 3},
    "lagged_residual": {"w": 16, "h": 64},
    "var_ratio": {"w": 24},
    "seasonal": {"by": "hour", "alpha": 0.05, "min_count": 3},
    "cusum": {"k": 0.5},
    "baseline_alpha": 0.01,
    "use_spot": True,  # Task K3: EVT Extreme Value Theory tail calibration
    # Per-detector risk (ADR-009): clean short/long-horizon residuals need a looser q than noisy detectors
    "spot": {"init_level": 0.98, "q": 1e-6, "min_excesses": 10, "dspot_depth": 0,
             "q_per_detector": {"LastW": 1e-4, "LaggedResidual": 1e-4}},
    "fusion": "max",
    "vote_k": 1,
    "use_events": True,  # False = plain rising-edge events (no debounce / merge / cooldown)
    "events": {"merge_gap": 64, "cooldown": 600, "rebaseline_after": 80},
}

# Likhit's original K1–K8 defaults (event-F1 0.311, NAB 53.0), kept for reference / ablation.
ORIGINAL_DEFAULT_CONFIG: Dict[str, Any] = {
    "probation_fraction": PROBATION_FRACTION,
    "probation_max": PROBATION_MAX,
    "probation": 300,  # Fallback when n_total is unknown
    "fixed_threshold": 6.0,  # Base fallback threshold
    "debounce": 2,  # Require 2 consecutive outliers (or 1 extreme spike) (ADR-003)
    "detectors": ["robust_z", "last_w", "lagged_residual", "var_ratio", "seasonal", "cusum"],
    "detector_thresholds": {
        "RobustZ": 6.0,
        "LastW": 6.0,
        "LaggedResidual": 6.0,
        "VarRatio": 1.5,
        "Seasonal": 5.0,
        "CUSUM": 8.0,
        "StreamingIsolationForest": 6.0,
    },
    "last_w": {"w": 3},
    "lagged_residual": {"w": 16, "h": 64},
    "var_ratio": {"w": 24},
    "seasonal": {"by": "hour", "alpha": 0.05, "min_count": 3},
    "cusum": {"k": 0.5},
    "baseline_alpha": 0.01,
    "use_spot": True,  # Task K3: EVT Extreme Value Theory tail calibration
    "spot": {"init_level": 0.98, "q": 1e-4, "min_excesses": 10, "dspot_depth": 0,
             "q_per_detector": {}},  # e.g. {"LastW": 1e-4, "LaggedResidual": 1e-4}
    "fusion": "max",
    "vote_k": 2,
    "use_events": True,
    "events": {"merge_gap": 16, "cooldown": 200, "rebaseline_after": 80},
}
