"""Streaming Peak-Over-Threshold (SPOT and DSPOT) using Extreme Value Theory.

References:
- Siffer et al., "Anomaly Detection in Streams with Extreme Value Theory" (KDD 2017)
- AGENTS.md §4.3
"""
from __future__ import annotations

import collections
import math
from typing import List, Optional, Tuple
import numpy as np


class SPOT:
    """Streaming Peak-Over-Threshold with Generalized Pareto Distribution (GPD).

    Adapts the extreme anomaly threshold dynamically without hardcoded cutoffs.
    """

    def __init__(
        self,
        q: float = 1e-4,
        init_level: float = 0.98,
        min_excesses: int = 8,
    ):
        self.q = q  # Desired risk level (false alarm probability)
        self.init_level = init_level  # Initial quantile for tail definition
        self.min_excesses = min_excesses

        self.t: float = 0.0  # Threshold separating bulk distribution from tail
        self.gamma: float = 0.0  # GPD shape parameter
        self.sigma: float = 1.0  # GPD scale parameter
        self.z_q: float = 1.0  # Calibrated alarm threshold for risk q

        self.n: int = 0  # Number of observations seen
        self.n_excess: int = 0  # Number of excesses above t

        # Running accumulators for Method of Moments GPD estimation
        self.sum_excess: float = 0.0
        self.sum_sq_excess: float = 0.0
        self.is_initialized: bool = False

    def fit_from_probation(self, values: List[float]) -> None:
        """Fit initial tail threshold and GPD parameters from probation window."""
        arr = np.array(values, dtype=float)
        self.n = len(arr)
        if self.n < 10 or float(np.max(arr)) < 1e-6:
            self.t = float(np.median(arr)) if len(arr) > 0 and float(np.max(arr)) >= 1e-6 else 1.0
            self.z_q = max(float(np.max(arr)) * 1.25, 3.0) if len(arr) > 0 else 3.0
            self.sigma = 1.0
            self.is_initialized = True
            return

        # Adaptive initial quantile (lower to 0.95 if sample is small)
        level = self.init_level if self.n >= 100 else 0.95
        self.t = float(np.quantile(arr, level))

        # Extract excesses Y = {x - t | x > t}
        excesses = arr[arr > self.t] - self.t
        self.n_excess = len(excesses)

        if self.n_excess > 0:
            self.sum_excess = float(np.sum(excesses))
            self.sum_sq_excess = float(np.sum(excesses * excesses))

        # Fallback if too few excesses in small probation
        if self.n_excess < self.min_excesses:
            # Empirical fallback: 99.9th percentile or 1.15x maximum
            self.z_q = max(float(np.quantile(arr, 0.999)), float(np.max(arr)) * 1.15)
            self.sigma = max(1e-6, float(np.std(arr)))
            self.gamma = 0.0
            self.is_initialized = True
            return

        # Fit GPD via Method of Moments
        self._fit_gpd_mom()
        self._compute_alarm_threshold()
        self.is_initialized = True

    def _fit_gpd_mom(self) -> None:
        """Method of Moments fit for Generalized Pareto Distribution."""
        if self.n_excess < 2:
            return
        m = self.sum_excess / self.n_excess
        v = max(1e-8, (self.sum_sq_excess / self.n_excess) - (m * m))

        ratio = (m * m) / v
        self.gamma = float(0.5 * (1.0 - ratio))
        self.sigma = float(0.5 * m * (ratio + 1.0))

        # Clamp parameters to physically valid GPD ranges
        self.sigma = max(1e-6, self.sigma)
        self.gamma = max(-0.5, min(0.5, self.gamma))

    def _compute_alarm_threshold(self) -> None:
        """Compute the extreme quantile threshold z_q for risk level q."""
        if self.n == 0 or self.n_excess == 0:
            return

        prob_ratio = (self.q * self.n) / max(1, self.n_excess)
        prob_ratio = max(1e-12, min(1.0, prob_ratio))

        if abs(self.gamma) > 1e-4:
            # GPD threshold formula
            factor = (prob_ratio ** (-self.gamma)) - 1.0
            zq = self.t + (self.sigma / self.gamma) * factor
        else:
            # Exponential tail limit as gamma -> 0
            zq = self.t - self.sigma * math.log(prob_ratio)

        # Ensure alarm threshold is at least above t
        self.z_q = max(self.t + 1e-4, zq)

    def update(self, x: float) -> Tuple[float, bool]:
        """Stream one observation.

        Returns:
            calibrated_score: normalized score in [0, 1] (>= 1.0 indicates anomaly)
            is_alarm: True if x exceeds extreme threshold z_q
        """
        x = float(x)
        self.n += 1

        if not self.is_initialized:
            return 0.0, False

        is_alarm = bool(x > self.z_q)

        # If it exceeds alarm threshold, do NOT corrupt the tail fit
        if not is_alarm and x > self.t:
            # Add excess to running tail statistics
            excess = x - self.t
            self.n_excess += 1
            self.sum_excess += excess
            self.sum_sq_excess += excess * excess

            # Periodically update GPD parameters if we have sufficient excesses
            if self.n_excess >= self.min_excesses and self.n_excess % 5 == 0:
                self._fit_gpd_mom()
                self._compute_alarm_threshold()

        # Compute calibrated tail score in [0, 1]
        calibrated_score = self.calculate_score(x)
        return calibrated_score, is_alarm

    def calculate_score(self, x: float) -> float:
        """Maps raw score into calibrated score in [0, 1]. Exactly 1.0 at alarm threshold z_q."""
        if x <= self.t:
            # In the bulk distribution (below tail threshold)
            ratio = max(0.0, x / max(1e-6, self.t))
            return 0.5 * ratio

        # In the tail: calculate tail exceedance probability
        excess = x - self.t
        if abs(self.gamma) > 1e-4:
            base = 1.0 + (self.gamma * excess) / max(1e-6, self.sigma)
            if base <= 0:
                p_tail = 1e-12
            else:
                p_tail = (self.n_excess / max(1, self.n)) * (base ** (-1.0 / self.gamma))
        else:
            p_tail = (self.n_excess / max(1, self.n)) * math.exp(-excess / max(1e-6, self.sigma))

        p_tail = max(1e-12, min(1.0, float(p_tail)))

        # Logarithmic score mapping: 0.5 at t, 1.0 at z_q (where p_tail == q)
        log_q = math.log(max(1e-12, self.q))
        log_p = math.log(p_tail)
        # Normalized score where score == 1.0 when log_p == log_q
        score = 0.5 + 0.5 * min(1.0, log_p / log_q)
        return float(max(0.0, min(1.0, score)))


class DSPOT:
    """Drifting Streaming Peak-Over-Threshold.

    Subtracts trailing local moving average/median to remove drift before applying SPOT.
    """

    def __init__(self, depth: int = 30, q: float = 1e-4, init_level: float = 0.98):
        self.depth = max(5, depth)
        self.buffer = collections.deque(maxlen=self.depth)
        self.running_sum: float = 0.0
        self.spot = SPOT(q=q, init_level=init_level)

    def fit_from_probation(self, values: List[float]):
        """Warm up moving average and fit SPOT on residuals."""
        residuals = []
        for x in values:
            mean = (self.running_sum / len(self.buffer)) if len(self.buffer) > 0 else x
            residuals.append(abs(x - mean))

            if len(self.buffer) == self.depth:
                self.running_sum -= self.buffer[0]
            self.buffer.append(x)
            self.running_sum += x

        self.spot.fit_from_probation(residuals)

    def update(self, x: float) -> Tuple[float, bool]:
        mean = (self.running_sum / len(self.buffer)) if len(self.buffer) > 0 else x
        residual = abs(x - mean)

        # Update running buffer
        if len(self.buffer) == self.depth:
            self.running_sum -= self.buffer[0]
        self.buffer.append(x)
        self.running_sum += x

        return self.spot.update(residual)
