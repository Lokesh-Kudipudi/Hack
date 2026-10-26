"""Explainability and evidence generation engine for Sentinel anomalies (Task K6, K20).

References:
- AGENTS.md §4.6
- ADR-007: Explainability and Evidence Schema
"""
from __future__ import annotations

import math
from typing import Any, Dict, List, Optional
import pandas as pd

DETECTOR_TYPE_MAP = {
    "RobustZ": "spike",
    "LastW": "spike",
    "LaggedResidual": "sequence",
    "VarRatio": "variance_change",
    "Seasonal": "pattern_break",
    "CUSUM": "level_shift",
    "StreamingIsolationForest": "multivariate_outlier",
}

TYPE_DESCRIPTIONS = {
    "spike": "isolated pulse or step jump",
    "sequence": "lagged sequence anomaly",
    "variance_change": "volatility/variance regime shift",
    "pattern_break": "diurnal/seasonal pattern break",
    "level_shift": "persistent level shift / mean drift",
    "multivariate_outlier": "boundary isolation anomaly",
}

_WEEKDAYS = ("Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun")


class ExplainEngine:
    """Generates structured evidence and human-readable text for anomalies."""

    def __init__(self, type_map: Optional[Dict[str, str]] = None):
        self.type_map = dict(type_map or DETECTOR_TYPE_MAP)

    def explain(
        self,
        t: pd.Timestamp,
        observed: float,
        expected: float,
        deviation_sigma: float,
        top_detector: str,
        fired: List[str],
        detector_score: Optional[float] = None,
        threshold_ratio: Optional[float] = None,
        confidence: Optional[float] = None,
        context: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Constructs evidence dictionary conforming to AGENTS.md §4.6 schema."""
        observed = float(observed)
        expected = float(expected)
        deviation_sigma = float(deviation_sigma)

        anom_type = self.type_map.get(top_detector, "anomaly")
        direction = "above" if observed >= expected else "below"

        # Time context (fast tuple lookup without strftime overhead)
        time_str = ""
        try:
            time_str = f" ({_WEEKDAYS[t.weekday()]} {t.hour:02d}:{t.minute:02d})"
        except Exception:
            pass

        # Primary explanation clause using specialized templates
        desc = TYPE_DESCRIPTIONS.get(anom_type, anom_type)
        if anom_type == "spike":
            text = f"Value {observed:.2f} is {deviation_sigma:.1f}σ {direction} expected {expected:.2f}{time_str} ({top_detector} detected {desc})."
        elif anom_type == "pattern_break":
            text = f"Value {observed:.2f} violates expected {expected:.2f}{time_str} seasonal profile ({deviation_sigma:.1f}σ {direction})."
        elif anom_type == "sequence":
            text = f"Value {observed:.2f} is {deviation_sigma:.1f}σ {direction} lagged baseline {expected:.2f}{time_str} (sustained sequence divergence)."
        else:
            text = f"Value {observed:.2f} is {deviation_sigma:.1f}σ {direction} expected {expected:.2f}{time_str} ({top_detector} detected {desc})."

        # Detector-specific context where the value deviation alone is misleading
        if anom_type == "variance_change" and detector_score is not None:
            text += f" Local volatility is {float(detector_score) + 1.0:.1f}x the baseline."
        elif anom_type == "level_shift" and detector_score is not None:
            text += f" Cumulative drift statistic {float(detector_score):.1f}."

        # Corroborating detector clause
        other_fired = [d for d in fired if d != top_detector]
        if other_fired:
            text += f" Corroborated by: {', '.join(other_fired)}."

        evidence: Dict[str, Any] = {
            "type": anom_type,
            "expected": round(expected, 3),
            "observed": round(observed, 3),
            "deviation_sigma": round(deviation_sigma, 2),
            "fired": list(fired) if fired else [top_detector],
            "top_detector": top_detector,
            "text": text,
        }

        # Optional extras beyond base schema
        if detector_score is not None:
            evidence["detector_score"] = round(float(detector_score), 3)

        if threshold_ratio is not None:
            evidence["threshold_ratio"] = round(float(threshold_ratio), 3)
            if confidence is None:
                # Calibrate confidence from threshold exceedance and consensus
                ratio_val = float(threshold_ratio)
                bonus = 0.08 * min(3, max(0, len(fired) - 1))
                conf = 0.50 + 0.35 * min(1.0, max(0.0, (ratio_val - 1.0) / 2.0)) + bonus
                confidence = round(min(0.99, max(0.50, conf)), 2)

        if confidence is not None:
            evidence["confidence"] = round(float(confidence), 2)

        if context is not None:
            evidence["context"] = dict(context)

        return evidence
