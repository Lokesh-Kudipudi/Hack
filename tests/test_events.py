"""Unit tests for EventManager (Task K5)."""
import pytest
from sentinel.events import EventManager


def test_event_debounce():
    """Verify debounce requires confirmation unless extreme outlier."""
    em = EventManager(debounce=2, extreme_ratio=1.5, cooldown=50)

    # Point 1: mild anomaly (top_ratio = 1.1 < 1.5) -> debounced (not event start)
    start1, rebase1 = em.update(is_anom=True, top_ratio=1.1)
    assert start1 is False
    assert rebase1 is False
    assert em.in_event is False

    # Point 2: confirmed anomaly -> triggers event start
    start2, rebase2 = em.update(is_anom=True, top_ratio=1.1)
    assert start2 is True
    assert em.in_event is True

    # Point 3: continuing anomaly -> remains in event, no new event start
    start3, _ = em.update(is_anom=True, top_ratio=1.1)
    assert start3 is False


def test_extreme_outlier_bypass():
    """Verify extreme outlier (top_ratio >= 1.5) immediately triggers without waiting for debounce."""
    em = EventManager(debounce=3, extreme_ratio=1.5)

    start, _ = em.update(is_anom=True, top_ratio=2.5)
    assert start is True
    assert em.in_event is True


def test_gap_merging():
    """Verify anomalous points within merge_gap belong to the same event."""
    em = EventManager(debounce=1, merge_gap=5, cooldown=0)

    # Event starts
    start1, _ = em.update(is_anom=True)
    assert start1 is True

    # 3 normal points (gap <= 5)
    for _ in range(3):
        start, _ = em.update(is_anom=False)
        assert start is False
        assert em.in_event is True  # Still inside merge window

    # Anomaly resumes within merge_gap -> NO new event start
    start2, _ = em.update(is_anom=True)
    assert start2 is False
    assert em.in_event is True

    # Now 6 normal points (gap > 5) -> closes event
    for _ in range(6):
        em.update(is_anom=False)
    assert em.in_event is False


def test_rebaseline_after_sustained_shift():
    """Verify persistent anomaly triggers should_rebaseline after rebaseline_after points."""
    em = EventManager(debounce=1, rebaseline_after=15, cooldown=0)

    # Stream 14 continuous anomalous points
    for i in range(14):
        _, should_rebase = em.update(is_anom=True)
        assert should_rebase is False

    # 15th continuous anomalous point -> triggers rebaseline
    _, should_rebase_15 = em.update(is_anom=True)
    assert should_rebase_15 is True
    assert em.in_event is False  # Closes event after adapting
