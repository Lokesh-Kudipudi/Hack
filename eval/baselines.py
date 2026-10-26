"""Reference detectors with the same streaming interface as sentinel.Sentinel.

Each exposes update(t, x) -> result with attributes score, label, is_event_start,
evidence (duck-typed like sentinel.types.Result). Labels: ANOMALOUS when
score >= threshold after probation; an event starts on each NORMAL->ANOMALOUS
edge (no merging/cooldown — that is Sentinel's job).
"""
from __future__ import annotations

import math
import zlib
from dataclasses import dataclass, field
from typing import Callable, Dict, Optional

import numpy as np

from eval.data import probation_length


@dataclass
class BaselineResult:
    timestamp: object
    value: float
    score: float
    label: str
    is_event_start: bool
    evidence: Optional[dict] = None
    detector_scores: dict = field(default_factory=dict)


class _Thresholded:
    """Turns a raw [0,1] score stream into labels + event starts."""

    def __init__(self, n_total: Optional[int], threshold: float):
        self.probation = probation_length(n_total) if n_total else 0
        self.threshold = threshold
        self.i = 0
        self.prev_anom = False

    def _score(self, t, x: float) -> float:
        raise NotImplementedError

    def update(self, t, x: float) -> BaselineResult:
        s = self._score(t, x)
        anom = self.i >= self.probation and s >= self.threshold
        start = anom and not self.prev_anom
        self.prev_anom = anom
        self.i += 1
        return BaselineResult(t, x, s, "ANOMALOUS" if anom else "NORMAL", start,
                              {"text": f"score {s:.4f} >= {self.threshold}"} if anom else None)


class NullDetector(_Thresholded):
    """Constant 0.5 (NAB's null baseline); never flags."""

    def __init__(self, n_total=None, threshold: float = 1.0, **_):
        super().__init__(n_total, threshold)

    def _score(self, t, x):
        return 0.5


class RandomDetector(_Thresholded):
    """Uniform random score, seeded per file for determinism."""

    def __init__(self, n_total=None, threshold: float = 0.99, seed: int = 42, key: str = "", **_):
        super().__init__(n_total, threshold)
        self.rng = np.random.default_rng(seed + zlib.crc32(key.encode()))

    def _score(self, t, x):
        return float(self.rng.random())


class WindowedGaussian(_Thresholded):
    """Port of NAB's windowedGaussian: score = 1 - Q(|z|) against the mean/std of the
    last `window` values, refreshed every `step` points once the window is full."""

    def __init__(self, n_total=None, threshold: float = 0.9999, window: int = 6400, step: int = 100, **_):
        super().__init__(n_total, threshold)
        self.window, self.step = window, step
        self.data: list = []
        self.buf: list = []
        self.mean, self.std = 0.0, 1.0

    def _refresh(self):
        self.mean = float(np.mean(self.data))
        self.std = float(np.std(self.data)) or 1e-6

    def _score(self, t, x):
        s = 0.0
        if self.data:
            z = abs(x - self.mean) / self.std
            s = 1.0 - 0.5 * math.erfc(z / math.sqrt(2))
        if len(self.data) < self.window:
            self.data.append(x)
            self._refresh()
        else:
            self.buf.append(x)
            if len(self.buf) == self.step:
                self.data = self.data[self.step:] + self.buf
                self.buf = []
                self._refresh()
        return s


BASELINES: Dict[str, Callable] = {
    "null": NullDetector,
    "random": RandomDetector,
    "windowed_gaussian": WindowedGaussian,
}
