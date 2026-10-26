"""Event lifecycle manager.

Handles event state, debouncing, gap merging, cooldown suppression,
and level-shift re-baselining.

References:
- AGENTS.md §4.5
- ADR-003: Event Debounce and Confirmation Protocol
"""
from __future__ import annotations

from typing import Optional, Tuple


class EventManager:
    """Manages continuous event states, debouncing, merging, and adaptation."""

    def __init__(
        self,
        debounce: int = 2,
        merge_gap: int = 16,
        cooldown: int = 200,
        rebaseline_after: int = 80,
        extreme_ratio: float = 1.5,
        agree_k: int = 0,
    ):
        self.debounce = max(1, int(debounce))
        self.merge_gap = max(1, int(merge_gap))
        self.cooldown_duration = max(0, int(cooldown))
        self.rebaseline_after = max(10, int(rebaseline_after))
        self.extreme_ratio = float(extreme_ratio)
        # L21: confirm on the first point when >= agree_k detectors fire together (0 = off)
        self.agree_k = int(agree_k)

        # State tracking
        self.in_event: bool = False
        self.consecutive_anom: int = 0
        self.cooldown_remaining: int = 0
        self.gap_count: int = 0
        self.total_event_points: int = 0

    def update(self, is_anom: bool, top_ratio: float = 1.0, n_fired: int = 0) -> Tuple[bool, bool]:
        """Update event state with current point's anomaly classification.

        Args:
            is_anom: Whether the current observation was flagged anomalous.
            top_ratio: Normalized deviation ratio of the top detector.
            n_fired: Number of detectors alarming on this point (agreement bypass).

        Returns:
            Tuple of:
            - is_event_start (bool): True if this point initiates a new event.
            - should_rebaseline (bool): True if persistent shift warrants detector re-baselining.
        """
        is_event_start = False
        should_rebaseline = False

        if self.cooldown_remaining > 0:
            self.cooldown_remaining -= 1

        if is_anom:
            self.consecutive_anom += 1
            self.gap_count = 0

            if not self.in_event:
                if self.cooldown_remaining == 0:
                    # Debounce confirmation or extreme outlier bypass (ADR-003)
                    agree = self.agree_k > 0 and n_fired >= self.agree_k
                    if self.consecutive_anom >= self.debounce or top_ratio >= self.extreme_ratio or agree:
                        is_event_start = True
                        self.in_event = True
                        self.total_event_points = self.consecutive_anom
                        self.cooldown_remaining = self.cooldown_duration
            else:
                self.total_event_points += 1
                # Check for sustained level shift needing re-baselining (AGENTS.md §4.5)
                if self.total_event_points >= self.rebaseline_after:
                    should_rebaseline = True
                    # Close event after adaptation to new baseline
                    self.in_event = False
                    self.total_event_points = 0
                    self.consecutive_anom = 0
        else:
            self.consecutive_anom = 0
            if self.in_event:
                self.gap_count += 1
                if self.gap_count >= self.merge_gap:
                    self.in_event = False
                    self.gap_count = 0
                    self.total_event_points = 0

        return is_event_start, should_rebaseline

    def reset(self):
        """Reset all internal state."""
        self.in_event = False
        self.consecutive_anom = 0
        self.cooldown_remaining = 0
        self.gap_count = 0
        self.total_event_points = 0
