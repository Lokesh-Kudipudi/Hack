"""Event-based scoring, exactly per AGENTS.md §5.

Per file, the probation prefix (first probation_length(n) points) is ignored:
GT windows ending inside it are dropped, windows straddling it are clipped to
start at the first scored point, and predictions inside it are discarded.

- predicted event = point with is_event_start=True
- TP window = GT window containing >=1 predicted start (inclusive); extra starts
  in the same window are ignored
- FP = predicted start outside every window
- FN = GT window with no predicted start
- delay (TP only) = first start in window - (clipped) window start, in points,
  minutes and as a fraction of the window width in points
"""
from __future__ import annotations

from dataclasses import dataclass, field
from statistics import mean, median
from typing import Dict, Iterable, List, Optional, Sequence

import numpy as np
import pandas as pd

from eval.data import Window, probation_length, window_indices


@dataclass
class FileScore:
    key: str
    category: str
    tp: int = 0
    fp: int = 0
    fn: int = 0
    delay_points: List[int] = field(default_factory=list)
    delay_minutes: List[float] = field(default_factory=list)
    delay_fraction: List[float] = field(default_factory=list)


def score_file(
    timestamps: Sequence,
    is_event_start: Sequence[bool],
    windows: Sequence[Window],
    key: str = "",
    category: str = "",
    probation: Optional[int] = None,
) -> FileScore:
    ts = pd.DatetimeIndex(pd.to_datetime(pd.Series(timestamps)))
    starts = np.asarray(is_event_start, dtype=bool)
    if len(starts) != len(ts):
        raise ValueError(f"{key}: {len(ts)} timestamps but {len(starts)} predictions")
    p = probation_length(len(ts)) if probation is None else probation

    gt = [(max(i, p), j) for i, j in window_indices(ts, windows) if j >= p]
    preds = np.flatnonzero(starts)
    preds = preds[preds >= p]

    fs = FileScore(key=key, category=category)
    in_window = np.zeros(len(preds), dtype=bool)
    for i, j in gt:
        hit = (preds >= i) & (preds <= j)
        in_window |= hit
        if not hit.any():
            fs.fn += 1
            continue
        fs.tp += 1
        first = int(preds[hit][0])
        fs.delay_points.append(first - i)
        fs.delay_minutes.append((ts[first] - ts[i]).total_seconds() / 60.0)
        fs.delay_fraction.append((first - i) / (j - i + 1))
    fs.fp = int((~in_window).sum())
    return fs


def score_run(df: pd.DataFrame, windows: Sequence[Window], key: str = "", category: str = "",
              probation: Optional[int] = None) -> FileScore:
    """Convenience: df from run_series() with columns timestamp, is_event_start."""
    return score_file(df["timestamp"], df["is_event_start"], windows, key, category, probation)


def _stat(fn, xs: List[float]) -> Optional[float]:
    return float(fn(xs)) if xs else None


def metrics(scores: Iterable[FileScore]) -> Dict:
    """Micro-averaged metrics over the given files. Undefined ratios are None."""
    scores = list(scores)
    tp, fp, fn = (sum(getattr(s, a) for s in scores) for a in ("tp", "fp", "fn"))
    dp = [d for s in scores for d in s.delay_points]
    dm = [d for s in scores for d in s.delay_minutes]
    df = [d for s in scores for d in s.delay_fraction]
    return {
        "files": len(scores),
        "tp": tp, "fp": fp, "fn": fn,
        "precision": tp / (tp + fp) if tp + fp else None,
        "recall": tp / (tp + fn) if tp + fn else None,
        # 2PR/(P+R) == 2TP/(2TP+FP+FN); this form is defined whenever anything happened
        "f1": 2 * tp / (2 * tp + fp + fn) if tp + fp + fn else None,
        "delay_points_mean": _stat(mean, dp), "delay_points_median": _stat(median, dp),
        "delay_minutes_mean": _stat(mean, dm), "delay_minutes_median": _stat(median, dm),
        "delay_fraction_mean": _stat(mean, df), "delay_fraction_median": _stat(median, df),
    }


def aggregate(scores: Iterable[FileScore]) -> Dict:
    """{'overall': metrics, 'per_category': {cat: metrics}, 'per_file': {key: metrics}}."""
    scores = list(scores)
    cats: Dict[str, List[FileScore]] = {}
    for s in scores:
        cats.setdefault(s.category, []).append(s)
    return {
        "overall": metrics(scores),
        "per_category": {c: metrics(v) for c, v in sorted(cats.items())},
        "per_file": {s.key: metrics([s]) for s in scores},
    }
