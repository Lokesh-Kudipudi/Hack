from dataclasses import dataclass, field
from typing import Optional
import pandas as pd


@dataclass
class Result:
    timestamp: pd.Timestamp
    value: float
    score: float                 # fused anomaly score in [0, 1]
    label: str                   # "NORMAL" | "ANOMALOUS"
    is_event_start: bool         # True only on the first point of a new anomaly event
    evidence: Optional[dict] = None   # filled when label == "ANOMALOUS"
    detector_scores: dict = field(default_factory=dict)  # name -> calibrated score in [0,1]


class Detector:                  # base class for D1..D6
    name: str = "base"

    def update(self, t: pd.Timestamp, x: float) -> float:
        """Raw score (higher = more anomalous), O(1)."""
        raise NotImplementedError

    def expected(self) -> Optional[float]:
        """Expected value (for evidence/plots), may be None."""
        return None

    def reset_baseline(self) -> None:
        """Called by event manager after confirmed level shift."""
        pass
