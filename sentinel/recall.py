"""Recall / detection-delay mechanisms (phase 3, L19–L20). All causal, O(1) per point, off by default.

- SpatialExtreme: Numenta-style "spatial anomaly" — a value beyond the running min/max by more than
  tol × (max − min) is anomalous on its first point. Min/max are seeded from probation and updated online.
- ScoreSmoother: short moving average of a detector's raw score before SPOT (Numenta anomaly-likelihood
  idea): a sustained deviation accumulates, a single noisy point is diluted.
"""
from __future__ import annotations

import collections
from typing import Dict, Iterable, Optional


class SpatialExtreme:
    name = "SpatialExtreme"

    def __init__(self, tol: float = 0.05):
        self.tol = float(tol)
        self.lo: Optional[float] = None
        self.hi: Optional[float] = None

    def seed(self, values: Iterable[float]) -> None:
        for v in values:
            self._observe(v)

    def _observe(self, x: float) -> None:
        self.lo = x if self.lo is None or x < self.lo else self.lo
        self.hi = x if self.hi is None or x > self.hi else self.hi

    def update(self, x: float) -> bool:
        """True when x leaves the tolerated range; then the range is extended to include x."""
        flag = False
        if self.lo is not None and self.hi is not None and self.hi > self.lo:
            pad = self.tol * (self.hi - self.lo)
            flag = x > self.hi + pad or x < self.lo - pad
        self._observe(x)
        return flag


class ScoreSmoother:
    """Per-detector moving average of raw scores over the last `window` points (window <= 1: identity)."""

    def __init__(self, window: int = 1, detectors: Optional[Iterable[str]] = None):
        self.window = max(1, int(window))
        self.only = set(detectors) if detectors else None
        self._buf: Dict[str, collections.deque] = {}
        self._sum: Dict[str, float] = {}

    def applies(self, name: str) -> bool:
        return self.window > 1 and (self.only is None or name in self.only)

    def __call__(self, name: str, raw: float) -> float:
        if not self.applies(name):
            return raw
        buf = self._buf.get(name)
        if buf is None:
            buf = self._buf[name] = collections.deque(maxlen=self.window)
            self._sum[name] = 0.0
        if len(buf) == self.window:
            self._sum[name] -= buf[0]
        buf.append(raw)
        self._sum[name] += raw
        return self._sum[name] / len(buf)
