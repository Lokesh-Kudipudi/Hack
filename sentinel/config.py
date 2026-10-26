"""Default configuration for Sentinel."""
from typing import Any, Dict

PROBATION_FRACTION = 0.15
PROBATION_MAX = 750


def probation_length(n: int) -> int:
    """Matches eval/data.py: min(floor(0.15*n), 750)."""
    return min(int(PROBATION_FRACTION * n), PROBATION_MAX)


DEFAULT_CONFIG: Dict[str, Any] = {
    "probation_fraction": PROBATION_FRACTION,
    "probation_max": PROBATION_MAX,
    "probation": 300,  # Fallback when n_total is unknown
    "fixed_threshold": 4.5,  # Robust Z-score threshold for K1 baseline
    "detectors": ["robust_z"],
    "last_w": {"w": 3},
    "lagged_residual": {"w": 16, "h": 64},
    "var_ratio": {"w": 24},
    "seasonal": {"by": "hour", "alpha": 0.05, "min_count": 3},
    "cusum": {"k": 0.5},
    "baseline_alpha": 0.01,
    "use_spot": False,  # Turned to True in K3
    "spot": {"init_level": 0.98, "q": 1e-4, "min_excesses": 10, "dspot_depth": 0},
    "fusion": "max",
    "vote_k": 2,
    "use_events": True,
    "events": {"merge_gap": 12, "cooldown": 150, "rebaseline_after": 80},
}
