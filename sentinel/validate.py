"""Input validation and sanitization for streaming time-series observations.

Guarantees:
1. Never crashes on invalid, missing, or malformed inputs.
2. Handles NaN, Inf, non-numeric types, missing values, extreme magnitudes.
3. Handles duplicate, backward, or unparseable timestamps.
4. Produces clean, sanitized (Timestamp, float) tuples and diagnostic flags.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Optional, Tuple
import numpy as np
import pandas as pd

# Safe numeric boundaries to prevent float overflow in quadratic variance buffers
MAX_SAFE_MAGNITUDE: float = 1e12
DEFAULT_FALLBACK_VALUE: float = 0.0


@dataclass(slots=True)
class ValidationFlags:
    """Diagnostic flags for sanitized observation data quality."""
    was_nan: bool = False
    was_inf: bool = False
    was_missing: bool = False
    was_extreme_magnitude: bool = False
    is_duplicate_timestamp: bool = False
    is_backward_timestamp: bool = False
    raw_type_invalid: bool = False
    time_gap_seconds: Optional[float] = None

    @property
    def has_data_quality_issue(self) -> bool:
        return (
            self.was_nan
            or self.was_inf
            or self.was_missing
            or self.was_extreme_magnitude
            or self.raw_type_invalid
            or self.is_backward_timestamp
        )


def sanitize_timestamp(
    t: Any,
    last_t: Optional[pd.Timestamp] = None,
    default_gap_seconds: float = 60.0,
) -> Tuple[pd.Timestamp, ValidationFlags]:
    """Sanitize timestamp input into a valid pd.Timestamp."""
    flags = ValidationFlags()

    if t is None:
        flags.was_missing = True
        if last_t is not None:
            clean_t = last_t + pd.Timedelta(seconds=default_gap_seconds)
        else:
            clean_t = pd.Timestamp.now()
        return clean_t, flags

    # Convert to pd.Timestamp
    if isinstance(t, pd.Timestamp):
        clean_t = t
    else:
        try:
            clean_t = pd.to_datetime(t)
            if pd.isna(clean_t):
                flags.was_missing = True
                clean_t = last_t + pd.Timedelta(seconds=default_gap_seconds) if last_t else pd.Timestamp.now()
        except Exception:
            flags.raw_type_invalid = True
            clean_t = last_t + pd.Timedelta(seconds=default_gap_seconds) if last_t else pd.Timestamp.now()

    # Mixed tz-aware / tz-naive streams: align to the previous timestamp so ordering comparisons work
    if last_t is not None and (clean_t.tzinfo is None) != (last_t.tzinfo is None):
        try:
            clean_t = clean_t.tz_convert(None) if last_t.tzinfo is None else clean_t.tz_localize(last_t.tzinfo)
        except Exception:
            pass

    # Check temporal ordering if last_t is available
    if last_t is not None:
        try:
            delta_sec = (clean_t - last_t).total_seconds()
            flags.time_gap_seconds = delta_sec
            if delta_sec == 0:
                flags.is_duplicate_timestamp = True
            elif delta_sec < 0:
                flags.is_backward_timestamp = True
        except Exception:
            pass

    return clean_t, flags


def sanitize_value(
    x: Any,
    fallback: Optional[float] = None,
) -> Tuple[float, ValidationFlags]:
    """Sanitize input value into a finite, safe float."""
    flags = ValidationFlags()

    # Handle None or missing
    if x is None:
        flags.was_missing = True
        val = fallback if fallback is not None else DEFAULT_FALLBACK_VALUE
        return float(val), flags

    # Try numeric conversion
    try:
        val = float(x)
    except (ValueError, TypeError):
        flags.raw_type_invalid = True
        val = fallback if fallback is not None else DEFAULT_FALLBACK_VALUE
        return float(val), flags

    # Check for NaN
    if math.isnan(val) or np.isnan(val):
        flags.was_nan = True
        val = fallback if fallback is not None else DEFAULT_FALLBACK_VALUE
        return float(val), flags

    # Check for Infinity
    if math.isinf(val) or np.isinf(val):
        flags.was_inf = True
        val = math.copysign(MAX_SAFE_MAGNITUDE, val)
        return float(val), flags

    # Check for extreme magnitudes (prevent overflow in x**2 variance accumulators)
    if abs(val) > MAX_SAFE_MAGNITUDE:
        flags.was_extreme_magnitude = True
        val = math.copysign(MAX_SAFE_MAGNITUDE, val)

    return float(val), flags


def sanitize_observation(
    t: Any,
    x: Any,
    last_t: Optional[pd.Timestamp] = None,
    last_val: Optional[float] = None,
    baseline_expected: Optional[float] = None,
) -> Tuple[pd.Timestamp, float, ValidationFlags]:
    """Sanitize both timestamp and value in a single causal call."""
    # Fast path for the common case (clean pd.Timestamp moving forward, finite in-range number):
    # same result as the general path, without per-point Timedelta / flag merging overhead.
    if type(t) is pd.Timestamp and type(x) in (float, int) and (last_t is None or (t.tzinfo is None) == (last_t.tzinfo is None)):
        v = float(x)
        if math.isfinite(v) and abs(v) <= MAX_SAFE_MAGNITUDE and (last_t is None or t > last_t):
            gap = None if last_t is None else (t.value - last_t.value) / 1e9
            return t, v, ValidationFlags(time_gap_seconds=gap)
    clean_t, t_flags = sanitize_timestamp(t, last_t=last_t)

    fallback = last_val if last_val is not None else baseline_expected
    clean_x, v_flags = sanitize_value(x, fallback=fallback)

    # Merge flags
    merged_flags = ValidationFlags(
        was_nan=v_flags.was_nan,
        was_inf=v_flags.was_inf,
        was_missing=v_flags.was_missing or t_flags.was_missing,
        was_extreme_magnitude=v_flags.was_extreme_magnitude,
        is_duplicate_timestamp=t_flags.is_duplicate_timestamp,
        is_backward_timestamp=t_flags.is_backward_timestamp,
        raw_type_invalid=v_flags.raw_type_invalid or t_flags.raw_type_invalid,
        time_gap_seconds=t_flags.time_gap_seconds,
    )

    return clean_t, clean_x, merged_flags
