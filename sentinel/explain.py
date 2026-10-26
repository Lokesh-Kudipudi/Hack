"""Explainability and evidence generation engine for Sentinel anomalies.

References:
- AGENTS.md §4.6
- ADR-005: Explainability and Evidence Schema
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional
import pandas as pd

DETECTOR_TYPE_MAP = {
    "RobustZ": "spike",
    "LastW": "spike",
    "LaggedResidual": "sequence",
    "VarRatio": "variance_change",
    "Seasonal": "pattern_break",
    "CUSUM": "level_shift",
}

TYPE_DESCRIPTIONS = {
    "spike": "isolated pulse or step jump",
    "sequence": "lagged sequence anomaly",
    "variance_change": "volatility/variance regime shift",
    "pattern_break": "diurnal/seasonal pattern break",
    "level_shift": "persistent level shift / mean drift",
}


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
    ) -> Dict[str, Any]:
        """Constructs evidence dictionary conforming to AGENTS.md §4.6 schema."""
        observed = float(observed)
        expected = float(expected)
        deviation_sigma = float(deviation_sigma)

        anom_type = self.type_map.get(top_detector, "anomaly")
        direction = "above" if observed >= expected else "below"

        # Time context
        time_str = ""
        try:
            time_str = f" ({t.strftime('%a %H:%M')})"
        except Exception:
            pass

        # Primary explanation clause
        desc = TYPE_DESCRIPTIONS.get(anom_type, anom_type)
        text = f"Value {observed:.2f} is {deviation_sigma:.1f}σ {direction} expected {expected:.2f}{time_str} ({top_detector} detected {desc})."

        # Corroborating detector clause
        other_fired = [d for d in fired if d != top_detector]
        if other_fired:
            text += f" Corroborated by: {', '.join(other_fired)}."

        return {
            "type": anom_type,
            "expected": round(expected, 3),
            "observed": round(observed, 3),
            "deviation_sigma": round(deviation_sigma, 2),
            "fired": list(fired) if fired else [top_detector],
            "top_detector": top_detector,
            "text": text,
        }
