"""Sentinel anomaly detection pipeline.

Executes streaming ingestion through probation, detectors, thresholding,
event management, and explainability.
"""
from __future__ import annotations

import math
from typing import Any, Dict, List, Optional
import numpy as np
import pandas as pd

from sentinel.config import DEFAULT_CONFIG, probation_length
from sentinel.detectors import (
    ALL_DETECTORS,
    RobustZ,
    LastW,
    LaggedResidual,
    VarRatio,
    Seasonal,
    CUSUM,
)
from sentinel.types import Detector, Result

TYPE_MAP = {
    "RobustZ": "spike",
    "LastW": "spike",
    "LaggedResidual": "sequence",
    "VarRatio": "variance_change",
    "Seasonal": "pattern_break",
    "CUSUM": "level_shift",
}


class Sentinel:
    """Lightweight streaming anomaly detector registering D1-D6 detectors."""

    def __init__(self, n_total: Optional[int] = None, config: Optional[Dict[str, Any]] = None):
        cfg = dict(DEFAULT_CONFIG)
        if config:
            cfg.update(config)
        self.config = cfg

        if n_total is not None:
            self.probation_limit = probation_length(n_total)
        else:
            self.probation_limit = int(self.config.get("probation", 300))

        self.threshold = float(self.config.get("fixed_threshold", 6.0))
        self.debounce = int(self.config.get("debounce", 2))
        self.alpha = float(self.config.get("baseline_alpha", 0.01))

        # Instantiate detectors
        active_names = self.config.get("detectors", ["robust_z"])
        self.detectors: Dict[str, Detector] = {}

        if "robust_z" in active_names:
            self.detectors["RobustZ"] = RobustZ(alpha=self.alpha)
        if "last_w" in active_names:
            w = int(self.config.get("last_w", {}).get("w", 3))
            self.detectors["LastW"] = LastW(w=w)
        if "lagged_residual" in active_names:
            cfg_lag = self.config.get("lagged_residual", {})
            self.detectors["LaggedResidual"] = LaggedResidual(
                w=int(cfg_lag.get("w", 16)), h=int(cfg_lag.get("h", 64))
            )
        if "var_ratio" in active_names:
            w_var = int(self.config.get("var_ratio", {}).get("w", 24))
            self.detectors["VarRatio"] = VarRatio(w=w_var)
        if "seasonal" in active_names:
            cfg_s = self.config.get("seasonal", {})
            self.detectors["Seasonal"] = Seasonal(
                alpha=float(cfg_s.get("alpha", 0.05)), min_count=int(cfg_s.get("min_count", 3))
            )
        if "cusum" in active_names:
            k = float(self.config.get("cusum", {}).get("k", 0.5))
            self.detectors["CUSUM"] = CUSUM(k=k)

        # Ensure at least RobustZ is present for baseline scaling
        if "RobustZ" not in self.detectors:
            self.detectors["RobustZ"] = RobustZ(alpha=self.alpha)

        # Probation buffers
        self.count: int = 0
        self.probation_timestamps: List[pd.Timestamp] = []
        self.probation_values: List[float] = []

        # Event management state
        self.in_event: bool = False
        self.consecutive_anom: int = 0
        self.cooldown_remaining: int = 0
        self.cooldown_duration: int = int(self.config.get("events", {}).get("cooldown", 200))
        self.merge_gap: int = int(self.config.get("events", {}).get("merge_gap", 16))
        self.gap_count: int = 0

    def _finish_probation(self):
        """Seed all detectors from the collected probation observations."""
        d1: RobustZ = self.detectors["RobustZ"]  # type: ignore
        d1.init_from_probation(self.probation_values)
        scale = d1.scale
        median = d1.median

        if "LastW" in self.detectors:
            d2: LastW = self.detectors["LastW"]  # type: ignore
            d2.init_scale(scale)
        if "LaggedResidual" in self.detectors:
            d3: LaggedResidual = self.detectors["LaggedResidual"]  # type: ignore
            d3.init_scale(scale)
        if "VarRatio" in self.detectors:
            d4: VarRatio = self.detectors["VarRatio"]  # type: ignore
            d4.init_from_probation(self.probation_values)
        if "Seasonal" in self.detectors:
            d5: Seasonal = self.detectors["Seasonal"]  # type: ignore
            d5.init_from_probation(self.probation_timestamps, self.probation_values)
        if "CUSUM" in self.detectors:
            d6: CUSUM = self.detectors["CUSUM"]  # type: ignore
            d6.init_scale(median=median, scale=scale)

        # Release probation memory
        self.probation_timestamps = []
        self.probation_values = []

    def update(self, t: pd.Timestamp, x: float) -> Result:
        """Stream one point (timestamp, value) through Sentinel."""
        x = float(x)
        self.count += 1

        # Phase 1: Probation phase (never flag anomalies, only learn)
        if self.count <= self.probation_limit:
            self.probation_timestamps.append(t)
            self.probation_values.append(x)
            if self.count == self.probation_limit:
                self._finish_probation()
            return Result(
                timestamp=t,
                value=x,
                score=0.0,
                label="NORMAL",
                is_event_start=False,
                evidence=None,
                detector_scores={name: 0.0 for name in self.detectors},
            )

        # Phase 2: Post-probation detection across all active detectors
        raw_scores: Dict[str, float] = {}
        calibrated_scores: Dict[str, float] = {}

        for name, det in self.detectors.items():
            raw = det.update(t, x)
            raw_scores[name] = raw
            # Smooth erf mapping to [0, 1]
            cal_score = float(math.erf(raw / math.sqrt(2.0)))
            calibrated_scores[name] = max(0.0, min(1.0, cal_score))

        # MAX-Score Fusion
        fused_score = max(calibrated_scores.values()) if calibrated_scores else 0.0

        # Check threshold exceedances
        fired = [name for name, s in raw_scores.items() if s > self.threshold]
        is_anom = len(fired) > 0
        label = "ANOMALOUS" if is_anom else "NORMAL"

        # Determine top detector
        top_name = max(raw_scores, key=raw_scores.get)
        top_score = raw_scores[top_name]

        # Event manager logic with debounce (ADR-003)
        is_event_start = False
        if self.cooldown_remaining > 0:
            self.cooldown_remaining -= 1

        if is_anom:
            self.consecutive_anom += 1
            if not self.in_event and self.cooldown_remaining == 0:
                if self.consecutive_anom >= self.debounce or top_score >= (self.threshold * 1.5):
                    is_event_start = True
                    self.in_event = True
                    self.cooldown_remaining = self.cooldown_duration

            self.gap_count = 0
        else:
            self.consecutive_anom = 0
            if self.in_event:
                self.gap_count += 1
                if self.gap_count >= self.merge_gap:
                    self.in_event = False
                    self.gap_count = 0

            # Update baseline stats only on non-anomalous observations (anti-pollution)
            if "RobustZ" in self.detectors:
                self.detectors["RobustZ"].update_baseline(x)  # type: ignore
            if "Seasonal" in self.detectors:
                self.detectors["Seasonal"].update_baseline(t, x)  # type: ignore

        # Evidence generation
        evidence = None
        if label == "ANOMALOUS":
            top_det = self.detectors[top_name]
            expected_val = top_det.expected()
            if expected_val is None:
                expected_val = self.detectors["RobustZ"].expected() or 0.0

            anom_type = TYPE_MAP.get(top_name, "anomaly")
            direction = "above" if x >= expected_val else "below"
            evidence = {
                "type": anom_type,
                "expected": round(float(expected_val), 3),
                "observed": round(x, 3),
                "deviation_sigma": round(top_score, 2),
                "fired": fired if fired else [top_name],
                "top_detector": top_name,
                "text": f"Value {x:.2f} is {top_score:.1f}σ {direction} expected {expected_val:.2f} ({top_name} detected {anom_type}).",
            }

        return Result(
            timestamp=t,
            value=x,
            score=fused_score,
            label=label,
            is_event_start=is_event_start,
            evidence=evidence,
            detector_scores=calibrated_scores,
        )


def run_series(df: pd.DataFrame, config: Optional[Dict[str, Any]] = None) -> pd.DataFrame:
    """Convenience: run Sentinel over a whole DataFrame with columns timestamp, value."""
    df = df.copy()
    sentinel = Sentinel(n_total=len(df), config=config)
    ts = pd.to_datetime(df["timestamp"]).tolist()
    xs = df["value"].astype(float).tolist()

    res = [sentinel.update(t, x) for t, x in zip(ts, xs)]

    return pd.DataFrame({
        "timestamp": df["timestamp"].values,
        "value": xs,
        "score": [r.score for r in res],
        "label": [r.label for r in res],
        "is_event_start": [bool(r.is_event_start) for r in res],
        "evidence": [r.evidence for r in res],
    })
