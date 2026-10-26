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
from sentinel.types import Result


class Sentinel:
    """Lightweight streaming anomaly detector."""

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

        # Probation buffer
        self.count: int = 0
        self.probation_values: List[float] = []

        # Baseline stats learned during probation
        self.median: float = 0.0
        self.scale: float = 1.0  # 1.4826 * MAD

        # Event management state
        self.in_event: bool = False
        self.consecutive_anom: int = 0
        self.cooldown_remaining: int = 0
        self.cooldown_duration: int = int(self.config.get("events", {}).get("cooldown", 200))
        self.merge_gap: int = int(self.config.get("events", {}).get("merge_gap", 16))
        self.gap_count: int = 0

    def _finish_probation(self):
        """Estimate initial median and scale (1.4826 * MAD) from probation window."""
        arr = np.array(self.probation_values, dtype=float)
        if len(arr) == 0:
            self.median = 0.0
            self.scale = 1.0
            return

        self.median = float(np.median(arr))
        mad = float(np.median(np.abs(arr - self.median)))
        if mad < 1e-6:
            mad = float(np.std(arr))
        if mad < 1e-6:
            mad = 1.0

        self.scale = 1.4826 * mad
        # Clear probation values to release memory
        self.probation_values = []

    def update(self, t: pd.Timestamp, x: float) -> Result:
        """Stream one point (timestamp, value) through Sentinel."""
        x = float(x)
        self.count += 1

        # Phase 1: Probation phase (never flag anomalies, only learn)
        if self.count <= self.probation_limit:
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
                detector_scores={"RobustZ": 0.0},
            )

        # Phase 2: Post-probation detection
        # Robust Z-score (D1)
        residual = abs(x - self.median)
        raw_z = residual / (self.scale + 1e-9)

        # Continuous CDF score in [0, 1] using erf (smooth mapping for NAB sweeper)
        score = float(math.erf(raw_z / math.sqrt(2.0)))
        score = max(0.0, min(1.0, score))

        is_anom = raw_z > self.threshold
        label = "ANOMALOUS" if is_anom else "NORMAL"

        # Event manager logic with debounce (ADR-003)
        is_event_start = False
        if self.cooldown_remaining > 0:
            self.cooldown_remaining -= 1

        if is_anom:
            self.consecutive_anom += 1
            # Trigger event start if:
            # (1) Not already in an ongoing event
            # (2) Cooldown expired
            # (3) EITHER consecutive count >= debounce OR extreme single spike >= 1.5 * threshold
            if not self.in_event and self.cooldown_remaining == 0:
                if self.consecutive_anom >= self.debounce or raw_z >= (self.threshold * 1.5):
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
            # Only update running median on non-anomalous points (anti-pollution)
            self.median += self.alpha * (x - self.median)

        # Evidence generation
        evidence = None
        if label == "ANOMALOUS":
            direction = "above" if x >= self.median else "below"
            evidence = {
                "type": "spike",
                "expected": round(self.median, 3),
                "observed": round(x, 3),
                "deviation_sigma": round(raw_z, 2),
                "fired": ["RobustZ"],
                "top_detector": "RobustZ",
                "text": f"Value {x:.2f} is {raw_z:.1f}σ {direction} expected baseline {self.median:.2f}.",
            }

        return Result(
            timestamp=t,
            value=x,
            score=score,
            label=label,
            is_event_start=is_event_start,
            evidence=evidence,
            detector_scores={"RobustZ": score},
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
