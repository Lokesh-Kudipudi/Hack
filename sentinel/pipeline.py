"""Sentinel anomaly detection pipeline.

Executes streaming ingestion through probation, detectors, thresholding,
event management, and explainability.
"""
from __future__ import annotations

import copy
import math
from typing import Any, Dict, List, Optional
import numpy as np
import pandas as pd

from sentinel.config import DEFAULT_CONFIG, probation_length
from sentinel.detectors import (
    ALL_DETECTORS,
    RobustZ,
    LastW,
    LaggedResidual,
    VarRatio,
    Seasonal,
    CUSUM,
    StreamingIsolationForest,
)
from sentinel.events import EventManager
from sentinel.explain import ExplainEngine
from sentinel.fusion import FusionEngine
from sentinel.spot import SPOT, DSPOT
from sentinel.types import Detector, Result

def _deep_merge(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    """Copy of base with override applied; nested dicts merge instead of being replaced."""
    out = copy.deepcopy(base)
    for k, v in override.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _deep_merge(out[k], v)
        else:
            out[k] = copy.deepcopy(v)
    return out


TYPE_MAP = {
    "RobustZ": "spike",
    "LastW": "spike",
    "LaggedResidual": "sequence",
    "VarRatio": "variance_change",
    "Seasonal": "pattern_break",
    "CUSUM": "level_shift",
}


class Sentinel:
    """Lightweight streaming anomaly detector registering D1-D6 detectors."""

    def __init__(self, n_total: Optional[int] = None, config: Optional[Dict[str, Any]] = None):
        self.config = _deep_merge(DEFAULT_CONFIG, config or {})

        if n_total is not None:
            self.probation_limit = probation_length(n_total)
        else:
            self.probation_limit = int(self.config.get("probation", 300))

        self.threshold = float(self.config.get("fixed_threshold", 6.0))
        self.debounce = int(self.config.get("debounce", 2))
        self.alpha = float(self.config.get("baseline_alpha", 0.01))
        self.detector_thresholds: Dict[str, float] = dict(self.config.get("detector_thresholds", {
            "RobustZ": self.threshold,
            "LastW": self.threshold,
            "LaggedResidual": self.threshold,
            "VarRatio": 1.5,
            "Seasonal": 5.0,
            "CUSUM": 8.0,
        }))

        # Instantiate detectors
        active_names = self.config.get("detectors", ["robust_z"])
        self.detectors: Dict[str, Detector] = {}

        if "robust_z" in active_names:
            self.detectors["RobustZ"] = RobustZ(alpha=self.alpha)
        if "last_w" in active_names:
            w = int(self.config.get("last_w", {}).get("w", 3))
            self.detectors["LastW"] = LastW(w=w)
        if "lagged_residual" in active_names:
            cfg_lag = self.config.get("lagged_residual", {})
            self.detectors["LaggedResidual"] = LaggedResidual(
                w=int(cfg_lag.get("w", 16)), h=int(cfg_lag.get("h", 64))
            )
        if "var_ratio" in active_names:
            w_var = int(self.config.get("var_ratio", {}).get("w", 24))
            self.detectors["VarRatio"] = VarRatio(w=w_var)
        if "seasonal" in active_names:
            cfg_s = self.config.get("seasonal", {})
            self.detectors["Seasonal"] = Seasonal(
                alpha=float(cfg_s.get("alpha", 0.05)), min_count=int(cfg_s.get("min_count", 3))
            )
        if "cusum" in active_names:
            k = float(self.config.get("cusum", {}).get("k", 0.5))
            self.detectors["CUSUM"] = CUSUM(k=k)
        if "isolation_forest" in active_names:
            self.detectors["StreamingIsolationForest"] = StreamingIsolationForest()

        # Ensure at least RobustZ is present for baseline scaling
        if "RobustZ" not in self.detectors:
            self.detectors["RobustZ"] = RobustZ(alpha=self.alpha)

        # SPOT EVT configuration
        self.use_spot: bool = bool(self.config.get("use_spot", False))
        spot_cfg = self.config.get("spot", {})
        self.spot_q: float = float(spot_cfg.get("q", 1e-4))
        # Optional per-detector risk level: clean detectors (LastW, LaggedResidual) need a looser q
        # than noisy ones (VarRatio, Seasonal) to fire at all; missing names fall back to spot.q.
        self.spot_q_per_detector: Dict[str, float] = {
            k: float(v) for k, v in (spot_cfg.get("q_per_detector") or {}).items()
        }
        self.spot_init_level: float = float(spot_cfg.get("init_level", 0.98))
        self.spot_min_excesses: int = int(spot_cfg.get("min_excesses", 8))
        self.dspot_depth: int = int(spot_cfg.get("dspot_depth", 0))

        self.spot_models: Dict[str, Any] = {}
        if self.use_spot:
            for name in self.detectors:
                if self.dspot_depth > 0:
                    self.spot_models[name] = DSPOT(
                        depth=self.dspot_depth, q=self.spot_q_per_detector.get(name, self.spot_q),
                        init_level=self.spot_init_level
                    )
                else:
                    self.spot_models[name] = SPOT(
                        q=self.spot_q_per_detector.get(name, self.spot_q),
                        init_level=self.spot_init_level,
                        min_excesses=self.spot_min_excesses,
                    )

        # Fusion Engine (Task K4)
        fusion_strategy = str(self.config.get("fusion", "max"))
        vote_k = int(self.config.get("vote_k", 2))
        self.fusion_engine = FusionEngine(strategy=fusion_strategy, vote_k=vote_k, extreme_ratio=1.5)

        # Event Manager (Task K5)
        evt_cfg = self.config.get("events", {})
        if self.config.get("use_events", True):
            self.event_manager = EventManager(
                debounce=int(self.config.get("debounce", 2)),
                merge_gap=int(evt_cfg.get("merge_gap", 16)),
                cooldown=int(evt_cfg.get("cooldown", 200)),
                rebaseline_after=int(evt_cfg.get("rebaseline_after", 80)),
                extreme_ratio=1.5,
            )
        else:
            # Ablation: no event lifecycle — every NORMAL->ANOMALOUS edge starts an event
            self.event_manager = EventManager(debounce=1, merge_gap=1, cooldown=0,
                                              rebaseline_after=int(evt_cfg.get("rebaseline_after", 80)),
                                              extreme_ratio=1.5)

        # Explain Engine (Task K6)
        self.explain_engine = ExplainEngine()

        # Hot-path pre-allocated buffers (Task K7 Performance)
        self._raw_scores: Dict[str, float] = {}
        self._norm_ratios: Dict[str, float] = {}
        self._calibrated_scores: Dict[str, float] = {}
        self._alarms: Dict[str, bool] = {}

        # Probation buffers
        self.count: int = 0
        self.probation_timestamps: List[pd.Timestamp] = []
        self.probation_values: List[float] = []

    @property
    def in_event(self) -> bool:
        return self.event_manager.in_event

    @in_event.setter
    def in_event(self, val: bool):
        self.event_manager.in_event = val

    def _finish_probation(self):
        """Seed all detectors and calibrate SPOT from probation observations."""
        d1: RobustZ = self.detectors["RobustZ"]  # type: ignore
        d1.init_from_probation(self.probation_values)
        scale = d1.scale
        median = d1.median

        if "LastW" in self.detectors:
            d2: LastW = self.detectors["LastW"]  # type: ignore
            d2.init_scale(scale)
        if "LaggedResidual" in self.detectors:
            d3: LaggedResidual = self.detectors["LaggedResidual"]  # type: ignore
            d3.init_scale(scale)
        if "VarRatio" in self.detectors:
            d4: VarRatio = self.detectors["VarRatio"]  # type: ignore
            d4.init_from_probation(self.probation_values)
        if "Seasonal" in self.detectors:
            d5: Seasonal = self.detectors["Seasonal"]  # type: ignore
            d5.init_from_probation(self.probation_timestamps, self.probation_values)
        if "CUSUM" in self.detectors:
            d6: CUSUM = self.detectors["CUSUM"]  # type: ignore
            d6.init_scale(median=median, scale=scale)
        if "StreamingIsolationForest" in self.detectors:
            d7: StreamingIsolationForest = self.detectors["StreamingIsolationForest"]  # type: ignore
            d7.init_from_probation(self.probation_values)

        # Warm up detectors on probation observations to populate sliding buffers
        # and collect normal score distributions for SPOT calibration
        probation_scores: Dict[str, List[float]] = {name: [] for name in self.detectors}
        for tp, xp in zip(self.probation_timestamps, self.probation_values):
            for name, det in self.detectors.items():
                s = det.update(tp, xp)
                probation_scores[name].append(s)

        # Reset CUSUM drift so it starts clean at 0 post-probation
        if "CUSUM" in self.detectors:
            self.detectors["CUSUM"].s_pos = 0.0  # type: ignore
            self.detectors["CUSUM"].s_neg = 0.0  # type: ignore

        if self.use_spot:
            for name in self.detectors:
                if name in self.spot_models:
                    self.spot_models[name].fit_from_probation(probation_scores[name])

        # Release probation memory
        self.probation_timestamps = []
        self.probation_values = []

    def update(self, t: pd.Timestamp, x: float) -> Result:
        """Stream one point (timestamp, value) through Sentinel."""
        x = float(x)
        self.count += 1

        # Phase 1: Probation phase (never flag anomalies, only learn)
        if self.count <= self.probation_limit:
            self.probation_timestamps.append(t)
            self.probation_values.append(x)
            if self.count == self.probation_limit:
                self._finish_probation()
            return Result(
                timestamp=t,
                value=x,
                score=0.0,
                label="NORMAL",
                is_event_start=False,
                evidence=None,
                detector_scores={name: 0.0 for name in self.detectors},
            )

        # Phase 2: Post-probation detection across all active detectors
        raw_scores = self._raw_scores
        raw_scores.clear()
        norm_ratios = self._norm_ratios
        norm_ratios.clear()
        calibrated_scores = self._calibrated_scores
        calibrated_scores.clear()
        alarms = self._alarms
        alarms.clear()

        for name, det in self.detectors.items():
            raw = det.update(t, x)
            raw_scores[name] = raw

            if self.use_spot and name in self.spot_models:
                cal_score, _ = self.spot_models[name].update(raw)
                base_th = self.detector_thresholds.get(name, self.threshold)
                zq = getattr(self.spot_models[name], "z_q", base_th)
                if hasattr(self.spot_models[name], "spot"):
                    zq = self.spot_models[name].spot.z_q
                # Reconcile dynamic EVT threshold with detector baseline threshold
                effective_zq = max(zq, base_th)
                ratio = raw / max(1e-6, effective_zq)
                norm_ratios[name] = ratio
                calibrated_scores[name] = cal_score
                alarms[name] = bool(raw > effective_zq)
            else:
                thresh = self.detector_thresholds.get(name, self.threshold)
                ratio = raw / max(1e-6, thresh)
                norm_ratios[name] = ratio
                # Calibrated score in [0, 1] using erf on normalized ratio
                cal_score = float(math.erf(ratio / math.sqrt(2.0)))
                calibrated_scores[name] = max(0.0, min(1.0, cal_score))
                alarms[name] = ratio > 1.0

        # Multi-detector fusion (Task K4)
        fused_score, is_anom, top_name, top_ratio, fired = self.fusion_engine.fuse(
            calibrated_scores, norm_ratios, alarms
        )
        top_score = raw_scores.get(top_name, 0.0)
        label = "ANOMALOUS" if is_anom else "NORMAL"

        # Event manager logic with debounce, gap merge, cooldown, and rebaseline (Task K5)
        is_event_start, should_rebaseline = self.event_manager.update(is_anom, top_ratio)

        if should_rebaseline:
            # Confirmed persistent level shift: adapt all detectors to new baseline (AGENTS.md §4.5)
            for det in self.detectors.values():
                det.reset_baseline()
            if "RobustZ" in self.detectors:
                self.detectors["RobustZ"].median = x

        # Update baseline stats only on non-anomalous observations (anti-pollution)
        if not is_anom:
            if "RobustZ" in self.detectors:
                self.detectors["RobustZ"].update_baseline(x)  # type: ignore
            if "Seasonal" in self.detectors:
                self.detectors["Seasonal"].update_baseline(t, x)  # type: ignore

        # Evidence generation (Task K6)
        evidence = None
        if label == "ANOMALOUS":
            top_det = self.detectors[top_name]
            expected_val = top_det.expected()
            if expected_val is None:
                expected_val = self.detectors["RobustZ"].expected() or 0.0

            # σ-deviation of the value itself, on the robust (MAD) scale learned in probation; the raw
            # detector score is reported separately (for VarRatio/CUSUM it is not a value deviation)
            robust_scale = getattr(self.detectors["RobustZ"], "scale", 1.0) or 1.0
            evidence = self.explain_engine.explain(
                t=t,
                observed=x,
                expected=expected_val,
                deviation_sigma=abs(x - expected_val) / max(robust_scale, 1e-9),
                top_detector=top_name,
                fired=fired,
                detector_score=top_score,
                threshold_ratio=top_ratio,
            )

        # Event-aligned continuous score:
        # Event start points (actual alarms) receive the peak calibrated score in [0.6, 1.0].
        # Non-alarm points remain strictly below the 0.5 decision boundary.
        if is_event_start:
            final_score = float(max(0.6, min(1.0, fused_score)))
        elif is_anom:
            final_score = float(min(0.48, fused_score * 0.45))
        else:
            final_score = float(min(0.30, fused_score * 0.30))

        return Result(
            timestamp=t,
            value=x,
            score=round(final_score, 4),
            label=label,
            is_event_start=is_event_start,
            evidence=evidence,
            detector_scores=dict(calibrated_scores),
        )


def run_series(df: pd.DataFrame, config: Optional[Dict[str, Any]] = None) -> pd.DataFrame:
    """Convenience: run Sentinel over a whole DataFrame with columns timestamp, value."""
    df = df.copy()
    sentinel = Sentinel(n_total=len(df), config=config)
    ts = pd.to_datetime(df["timestamp"]).tolist()
    xs = df["value"].astype(float).tolist()

    res = [sentinel.update(t, x) for t, x in zip(ts, xs)]

    return pd.DataFrame({
        "timestamp": df["timestamp"].values,
        "value": xs,
        "score": [r.score for r in res],
        "label": [r.label for r in res],
        "is_event_start": [bool(r.is_event_start) for r in res],
        "evidence": [r.evidence for r in res],
    })
