"""Sentinel: Lightweight, strictly causal, streaming time-series anomaly detection engine."""

__version__ = "0.3.0"

from sentinel.types import Detector, Result
from sentinel.pipeline import Sentinel, run_series
from sentinel.config import DEFAULT_CONFIG, ORIGINAL_DEFAULT_CONFIG, probation_length
from sentinel.pool import SentinelPool

__all__ = [
    "__version__",
    "Detector",
    "Result",
    "Sentinel",
    "SentinelPool",
    "run_series",
    "DEFAULT_CONFIG",
    "ORIGINAL_DEFAULT_CONFIG",
    "probation_length",
]
