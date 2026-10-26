"""Streaming Anomaly Detectors (D1 to D6).

All detectors are strictly causal, operate in O(1) time complexity, and implement
the shared Detector interface contract.
"""
from __future__ import annotations

import collections
import math
from typing import Dict, List, Optional
import numpy as np
import pandas as pd

from sentinel.types import Detector


class RobustZ(Detector):
    """D1: Robust Z-Score against trailing median / MAD.

    Catches isolated spikes and sudden extreme pulses.
    """
    name: str = "RobustZ"

    def __init__(self, alpha: float = 0.01):
        self.alpha = alpha
        self.median: float = 0.0
        self.scale: float = 1.0  # 1.4826 * MAD
        self.is_initialized: bool = False

    def init_from_probation(self, values: List[float]):
        """Initialize median and scale from probation window values."""
        arr = np.array(values, dtype=float)
        if len(arr) == 0:
            return
        self.median = float(np.median(arr))
        mad = float(np.median(np.abs(arr - self.median)))
        if mad < 1e-6:
            mad = float(np.std(arr))
        if mad < 1e-6:
            mad = 1.0
        self.scale = 1.4826 * mad
        self.is_initialized = True

    def update(self, t: pd.Timestamp, x: float) -> float:
        if not self.is_initialized:
            return 0.0
        raw_score = abs(x - self.median) / (self.scale + 1e-9)
        return float(raw_score)

    def update_baseline(self, x: float):
        """Called on non-anomalous points to adapt to slow drift."""
        if self.is_initialized:
            self.median += self.alpha * (x - self.median)

    def expected(self) -> Optional[float]:
        return self.median if self.is_initialized else None

    def reset_baseline(self):
        """Called when a level shift is confirmed."""
        pass


class LastW(Detector):
    """D2: Last-W squared difference against immediate preceding window mean.

    Catches sudden step jumps and instantaneous discontinuities.
    Paper: ICLR 2026 Definition 5 (Last-w baseline).
    """
    name: str = "LastW"

    def __init__(self, w: int = 3):
        self.w = max(1, w)
        self.buffer = collections.deque(maxlen=self.w)
        self.running_sum: float = 0.0
        self.scale: float = 1.0

    def init_scale(self, scale: float):
        self.scale = max(1e-6, scale)

    def update(self, t: pd.Timestamp, x: float) -> float:
        raw_score = 0.0
        if len(self.buffer) > 0:
            mean = self.running_sum / len(self.buffer)
            # Standardized deviation from immediate preceding mean
            raw_score = abs(x - mean) / (self.scale + 1e-9)

        # Update buffer
        if len(self.buffer) == self.w:
            self.running_sum -= self.buffer[0]
        self.buffer.append(x)
        self.running_sum += x
        return float(raw_score)

    def expected(self) -> Optional[float]:
        if len(self.buffer) > 0:
            return float(self.running_sum / len(self.buffer))
        return None

    def reset_baseline(self):
        self.buffer.clear()
        self.running_sum = 0.0


class LaggedResidual(Detector):
    """D3: Lagged Residual against a reference window that ended h steps ago.

    Catches sequence anomalies and sustained level shifts that contaminate
    short-horizon windows.
    Paper: ICLR 2026 Section 6.1 (Multi-horizon anomaly detection).
    """
    name: str = "LaggedResidual"

    def __init__(self, w: int = 16, h: int = 64):
        self.w = max(1, w)
        self.h = max(1, h)
        self.total_len = self.h + self.w
        self.buffer = collections.deque(maxlen=self.total_len)
        self.lagged_sum: float = 0.0
        self.scale: float = 1.0

    def init_scale(self, scale: float):
        self.scale = max(1e-6, scale)

    def update(self, t: pd.Timestamp, x: float) -> float:
        raw_score = 0.0
        # If we have at least h + 1 points, we have an un-contaminated lagged reference
        if len(self.buffer) > self.h:
            lagged_count = min(self.w, len(self.buffer) - self.h)
            # Slice oldest lagged_count elements from buffer
            # Since buffer is small (max ~80), summing is O(w) <= 16 operations
            lagged_mean = sum(self.buffer[i] for i in range(lagged_count)) / lagged_count
            raw_score = abs(x - lagged_mean) / (self.scale + 1e-9)

        self.buffer.append(x)
        return float(raw_score)

    def expected(self) -> Optional[float]:
        if len(self.buffer) > self.h:
            lagged_count = min(self.w, len(self.buffer) - self.h)
            return float(sum(self.buffer[i] for i in range(lagged_count)) / lagged_count)
        return None

    def reset_baseline(self):
        self.buffer.clear()


class VarRatio(Detector):
    """D4: Local window variance ratio against baseline variance.

    Catches volatility bursts, noise explosions, and sensor fluttering.
    Paper: ICLR 2026 Definition 4 (Var-w baseline).
    """
    name: str = "VarRatio"

    def __init__(self, w: int = 24):
        self.w = max(2, w)
        self.buffer = collections.deque(maxlen=self.w)
        self.running_sum: float = 0.0
        self.running_sumsq: float = 0.0
        self.baseline_var: float = 1.0
        self.is_initialized: bool = False

    def init_from_probation(self, values: List[float]):
        arr = np.array(values, dtype=float)
        if len(arr) < 2:
            return
        self.baseline_var = float(np.var(arr))
        if self.baseline_var < 1e-6:
            self.baseline_var = 1.0
        self.is_initialized = True

    def update(self, t: pd.Timestamp, x: float) -> float:
        raw_score = 0.0
        if len(self.buffer) == self.w:
            old = self.buffer[0]
            self.running_sum -= old
            self.running_sumsq -= old * old

        self.buffer.append(x)
        self.running_sum += x
        self.running_sumsq += x * x

        if self.is_initialized and len(self.buffer) >= 8:
            n = len(self.buffer)
            mean = self.running_sum / n
            local_var = max(0.0, (self.running_sumsq / n) - (mean * mean))
            ratio = (local_var + 1e-6) / (self.baseline_var + 1e-6)
            # Higher variance ratio -> higher anomaly score
            raw_score = max(0.0, math.sqrt(ratio) - 1.0)

        return float(raw_score)

    def expected(self) -> Optional[float]:
        if len(self.buffer) > 0:
            return float(self.running_sum / len(self.buffer))
        return None

    def reset_baseline(self):
        if len(self.buffer) >= 8:
            n = len(self.buffer)
            mean = self.running_sum / n
            self.baseline_var = max(1e-6, (self.running_sumsq / n) - (mean * mean))


class Seasonal(Detector):
    """D5: Hour-of-day seasonal profile deviation.

    Catches contextual anomalies (e.g. midday load at 3 AM).
    """
    name: str = "Seasonal"

    def __init__(self, alpha: float = 0.05, min_count: int = 3):
        self.alpha = alpha
        self.min_count = min_count
        # 24 hour buckets: each stores (running_mean, running_std, count)
        self.profiles: Dict[int, Dict[str, float]] = {
            h: {"mean": 0.0, "std": 1.0, "count": 0} for h in range(24)
        }

    def init_from_probation(self, timestamps: List[pd.Timestamp], values: List[float]):
        """Seed initial hourly profiles from probation data."""
        hour_data: Dict[int, List[float]] = collections.defaultdict(list)
        for ts, v in zip(timestamps, values):
            hour_data[ts.hour].append(float(v))

        for h, vals in hour_data.items():
            if len(vals) >= 2:
                arr = np.array(vals, dtype=float)
                m = float(np.mean(arr))
                s = float(np.std(arr))
                self.profiles[h] = {
                    "mean": m,
                    "std": max(0.1, s),
                    "count": len(vals)
                }

    def update(self, t: pd.Timestamp, x: float) -> float:
        h = t.hour
        prof = self.profiles[h]
        raw_score = 0.0
        if prof["count"] >= self.min_count:
            raw_score = abs(x - prof["mean"]) / (prof["std"] + 1e-6)

        return float(raw_score)

    def update_baseline(self, t: pd.Timestamp, x: float):
        """Update hourly baseline profile on non-anomalous points."""
        h = t.hour
        prof = self.profiles[h]
        diff = x - prof["mean"]
        prof["mean"] += self.alpha * diff
        prof["std"] += self.alpha * (abs(diff) - prof["std"])
        prof["count"] += 1

    def expected_for_time(self, t: pd.Timestamp) -> Optional[float]:
        prof = self.profiles[t.hour]
        if prof["count"] >= self.min_count:
            return prof["mean"]
        return None

    def expected(self) -> Optional[float]:
        return None

    def reset_baseline(self):
        pass


class CUSUM(Detector):
    """D6: Two-sided Cumulative Sum (CUSUM) Control Chart.

    Catches subtle persistent mean drift and permanent level shifts.
    """
    name: str = "CUSUM"

    def __init__(self, k: float = 0.5):
        self.k = k  # Slack / allowance factor in std units
        self.s_pos: float = 0.0
        self.s_neg: float = 0.0
        self.median: float = 0.0
        self.scale: float = 1.0
        self.is_initialized: bool = False

    def init_scale(self, median: float, scale: float):
        self.median = median
        self.scale = max(1e-6, scale)
        self.is_initialized = True

    def update(self, t: pd.Timestamp, x: float) -> float:
        if not self.is_initialized:
            return 0.0

        z = (x - self.median) / self.scale
        # Accumulate two-sided CUSUM drift with leaky decay when in-control
        if z > self.k:
            self.s_pos = min(25.0, self.s_pos + (z - self.k))
        else:
            self.s_pos = max(0.0, self.s_pos * 0.85 + (z - self.k))

        if -z > self.k:
            self.s_neg = min(25.0, self.s_neg + (-z - self.k))
        else:
            self.s_neg = max(0.0, self.s_neg * 0.85 + (-z - self.k))

        raw_score = max(self.s_pos, self.s_neg)
        return float(raw_score)

    def expected(self) -> Optional[float]:
        return self.median if self.is_initialized else None

    def reset_baseline(self):
        self.s_pos = 0.0
        self.s_neg = 0.0


ALL_DETECTORS = {
    "robust_z": RobustZ,
    "last_w": LastW,
    "lagged_residual": LaggedResidual,
    "var_ratio": VarRatio,
    "seasonal": Seasonal,
    "cusum": CUSUM,
}
