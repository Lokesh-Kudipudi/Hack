"""Default configuration for Sentinel."""
from typing import Any, Dict

PROBATION_FRACTION = 0.15
PROBATION_MAX = 750


def probation_length(n: int) -> int:
    """Matches eval/data.py: min(floor(0.15*n), 750)."""
    return min(int(PROBATION_FRACTION * n), PROBATION_MAX)


# Default = the event-F1 operating point chosen by leave-one-category-out tuning (results/best_config.json,
# results/tuning.md): RobustZ + LaggedResidual, MAX fusion, per-detector SPOT risk (ADR-009), plus the phase-3
# recall mechanisms (ADR-010): spatial-extreme flag, debounce 3, cooldown = 0.5 x probation, extreme ratio 2.0.
# Full NAB corpus: event-F1 0.673, NAB 54.9, held-out pooled F1 0.654. NAB operating point: results/best_nab_config.json.
DEFAULT_CONFIG: Dict[str, Any] = {
    "probation_fraction": PROBATION_FRACTION,
    "probation_max": PROBATION_MAX,
    "probation": 300,  # Fallback when n_total is unknown
    "fixed_threshold": 6.0,  # Base fallback threshold
    "debounce": 3,  # Require 3 consecutive outliers (or 1 extreme spike / spatial extreme) (ADR-003, ADR-010)
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
    "var_ratio": {"w": 24, "two_sided": False},  # two_sided: also flag variance collapse / flatlines (K19)
    "seasonal": {"by": "hour", "alpha": 0.05, "min_count": 3, "weekly": False, "auto_period": False},
    "cusum": {"k": 0.5},
    "baseline_alpha": 0.01,
    "use_spot": True,  # Task K3: EVT Extreme Value Theory tail calibration
    # Per-detector risk (ADR-009): clean short/long-horizon residuals need a looser q than noisy detectors
    "spot": {"init_level": 0.98, "q": 1e-6, "min_excesses": 10, "dspot_depth": 0,
             "q_per_detector": {"LastW": 1e-4, "LaggedResidual": 1e-5}},
    "fusion": "max",
    "vote_k": 1,
    "use_events": True,  # False = plain rising-edge events (no debounce / merge / cooldown)
    # cooldown_frac > 0 replaces cooldown with frac x probation length; agree_k > 0 confirms on the first
    # point when that many detectors fire together (phase-3 recall/delay knobs, off by default)
    "events": {"merge_gap": 64, "cooldown": 600, "rebaseline_after": 80, "cooldown_frac": 0.5, "agree_k": 0},
    "extreme_ratio": 2.0,  # raw/threshold ratio that confirms a single detector immediately
    # new running min/max beyond tol x range -> alarm, once >= min_seed values have been seen
    "spatial": {"enabled": True, "tol": 0.05, "min_seed": 100},
    "smoothing": {"window": 1, "detectors": None},  # moving average of raw scores before SPOT (1 = off)
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
    "var_ratio": {"w": 24, "two_sided": False},  # two_sided: also flag variance collapse / flatlines (K19)
    "seasonal": {"by": "hour", "alpha": 0.05, "min_count": 3, "weekly": False, "auto_period": False},
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
