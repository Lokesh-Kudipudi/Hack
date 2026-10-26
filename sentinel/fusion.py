"""Multi-detector score and decision fusion engine.

References:
- AGENTS.md §4.4
- ADR-004: Multi-Horizon Fusion
"""
from __future__ import annotations

from typing import Dict, List, Optional, Tuple


class FusionEngine:
    """Fuses per-detector calibrated scores and alarms into a single decision.

    Supported strategies:
    - 'max': MAX-Score Fusion (default per §4.4). Fused score is the maximum of calibrated
             scores. Decision triggers if any detector alarms (or consensus if vote_k > 1).
    - 'vote': k-of-n consensus voting. Requires at least k detectors to alarm,
              or a single detector exceeding extreme_ratio (default 1.5x threshold).
    - 'mean': Arithmetic average of calibrated scores.
    """

    def __init__(
        self,
        strategy: str = "max",
        vote_k: int = 2,
        extreme_ratio: float = 1.5,
    ):
        self.strategy = strategy.lower()
        self.vote_k = max(1, int(vote_k))
        self.extreme_ratio = float(extreme_ratio)

    def fuse(
        self,
        calibrated_scores: Dict[str, float],
        norm_ratios: Dict[str, float],
        alarms: Dict[str, bool],
    ) -> Tuple[float, bool, str, float, List[str]]:
        """Combine per-detector metrics into an integrated result.

        Args:
            calibrated_scores: Map from detector name to normalized score in [0, 1].
            norm_ratios: Map from detector name to raw deviation ratio (raw / threshold).
            alarms: Map from detector name to boolean alarm flag.

        Returns:
            Tuple of:
            - fused_score: Combined scalar score in [0, 1].
            - is_anom: Whether an anomaly is flagged.
            - top_detector: Name of the detector with highest normalized deviation.
            - top_ratio: Deviation ratio of the top detector.
            - fired: List of detector names that flagged alarms.
        """
        if not calibrated_scores:
            return 0.0, False, "RobustZ", 0.0, []

        # Determine top detector by normalized deviation ratio
        top_detector = max(norm_ratios, key=norm_ratios.get) if norm_ratios else "RobustZ"
        top_ratio = norm_ratios.get(top_detector, 0.0)

        # Detectors that crossed alarm threshold
        fired = [name for name, is_al in alarms.items() if is_al]

        # Calculate combined scalar score
        if self.strategy == "mean":
            fused_score = sum(calibrated_scores.values()) / len(calibrated_scores)
        else:  # default 'max'
            fused_score = max(calibrated_scores.values())

        # Anomaly classification decision
        if self.strategy == "max" and self.vote_k == 1:
            # 1-of-n: any detector alarms
            is_anom = len(fired) >= 1
        else:
            # Consensus voting (vote_k >= 2): requires k votes OR single extreme outlier
            is_anom = (len(fired) >= self.vote_k) or (top_ratio >= self.extreme_ratio)

        return float(fused_score), is_anom, top_detector, float(top_ratio), fired
