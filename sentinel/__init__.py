"""Sentinel: Lightweight streaming time-series anomaly detection engine."""
from sentinel.types import Detector, Result
from sentinel.pipeline import Sentinel, run_series
from sentinel.config import DEFAULT_CONFIG

__all__ = ["Detector", "Result", "Sentinel", "run_series", "DEFAULT_CONFIG"]
