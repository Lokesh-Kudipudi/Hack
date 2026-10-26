"""Sentinel anomaly detection pipeline.

Executes streaming ingestion through probation, detectors, thresholding,
event management, and explainability.
"""
from __future__ import annotations

import copy
import math
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
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
from sentinel.explain import DETECTOR_TYPE_MAP, ExplainEngine
from sentinel.fusion import FusionEngine
from sentinel.recall import ScoreSmoother, SpatialExtreme
from sentinel.spot import SPOT, DSPOT
from sentinel.types import Detector, Result
from sentinel.validate import MAX_SAFE_MAGNITUDE, sanitize_observation

def _deep_merge(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    """Copy of base with override applied; nested dicts merge instead of being replaced.
    An empty dict replaces (e.g. {"spot": {"q_per_detector": {}}} clears the per-detector risks)."""
    out = copy.deepcopy(base)
    for k, v in override.items():
        if isinstance(v, dict) and v and isinstance(out.get(k), dict):
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
            cfg_v = self.config.get("var_ratio", {})
            self.detectors["VarRatio"] = VarRatio(w=int(cfg_v.get("w", 24)),
                                                  two_sided=bool(cfg_v.get("two_sided", False)))
        if "seasonal" in active_names:
            cfg_s = self.config.get("seasonal", {})
            self.detectors["Seasonal"] = Seasonal(
                alpha=float(cfg_s.get("alpha", 0.05)), min_count=int(cfg_s.get("min_count", 3)),
                weekly=bool(cfg_s.get("weekly", False)), auto_period=bool(cfg_s.get("auto_period", False)),
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
        # ratio (raw / threshold) above which a single detector confirms immediately (fusion + debounce bypass)
        self.extreme_ratio = float(self.config.get("extreme_ratio", 1.5))
        self.fusion_engine = FusionEngine(strategy=fusion_strategy, vote_k=vote_k, extreme_ratio=self.extreme_ratio)

        # Event Manager (Task K5)
        evt_cfg = self.config.get("events", {})
        cooldown = int(evt_cfg.get("cooldown", 200))
        if evt_cfg.get("cooldown_frac"):
            # L18: CAD-OSE-style rest period proportional to the stream (probation = 15% of it, max 750)
            cooldown = max(1, int(float(evt_cfg["cooldown_frac"]) * self.probation_limit))
        if self.config.get("use_events", True):
            self.event_manager = EventManager(
                debounce=int(self.config.get("debounce", 2)),
                merge_gap=int(evt_cfg.get("merge_gap", 16)),
                cooldown=cooldown,
                rebaseline_after=int(evt_cfg.get("rebaseline_after", 80)),
                extreme_ratio=self.extreme_ratio,
                agree_k=int(evt_cfg.get("agree_k", 0)),
            )
        else:
            # Ablation: no event lifecycle — every NORMAL->ANOMALOUS edge starts an event
            self.event_manager = EventManager(debounce=1, merge_gap=1, cooldown=0,
                                              rebaseline_after=int(evt_cfg.get("rebaseline_after", 80)),
                                              extreme_ratio=self.extreme_ratio)

        # Recall / delay mechanisms (L19, L20), off by default
        sp_cfg = self.config.get("spatial") or {}
        self.spatial: Optional[SpatialExtreme] = (SpatialExtreme(float(sp_cfg.get("tol", 0.05)),
                                                                  int(sp_cfg.get("min_seed", 100)))
                                                   if sp_cfg.get("enabled") else None)
        sm_cfg = self.config.get("smoothing") or {}
        self.smoother = ScoreSmoother(int(sm_cfg.get("window", 1)), sm_cfg.get("detectors"))

        # Explain Engine (Task K6)
        self.explain_engine = ExplainEngine({**DETECTOR_TYPE_MAP, SpatialExtreme.name: "spike"})

        # Hot-path pre-allocated buffers (Task K7 Performance)
        self._raw_scores: Dict[str, float] = {}
        self._norm_ratios: Dict[str, float] = {}
        self._calibrated_scores: Dict[str, float] = {}
        self._alarms: Dict[str, bool] = {}

        # Probation buffers
        self.count: int = 0
        self.probation_timestamps: List[pd.Timestamp] = []
        self.probation_values: List[float] = []

        # Stream history tracking for input validation
        self._last_t: Optional[pd.Timestamp] = None
        self._last_x: Optional[float] = None

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
                s = self.smoother(name, det.update(tp, xp))
                probation_scores[name].append(s)

        if self.spatial is not None:
            self.spatial.seed(self.probation_values)

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

    def update(self, t: Any, x: Any) -> Result:
        """Stream one point (timestamp, value) through Sentinel."""
        last_t = self._last_t
        if not (type(t) is pd.Timestamp and type(x) is float and math.isfinite(x) and abs(x) <= MAX_SAFE_MAGNITUDE
                and (last_t is None or t > last_t)):
            # slow path only for unusual input (NaN/inf, bad types, duplicate/backward timestamps)
            expected_base = (
                self.detectors["RobustZ"].expected()
                if "RobustZ" in self.detectors and hasattr(self.detectors["RobustZ"], "expected")
                else None
            )
            t, x, _ = sanitize_observation(
                t, x, last_t=last_t, last_val=self._last_x, baseline_expected=expected_base
            )
        self._last_t = t
        self._last_x = x
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

        smooth = self.smoother
        for name, det in self.detectors.items():
            raw = smooth(name, det.update(t, x))
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
        if self.spatial is not None and self.spatial.update(x):
            # L19: new extreme beyond the tolerated range -> anomalous now, confirmed without debounce
            is_anom = True
            fired = fired + [SpatialExtreme.name]
            top_ratio = max(top_ratio, self.extreme_ratio)
            fused_score = max(fused_score, 1.0 - 1e-6)
        top_score = raw_scores.get(top_name, 0.0)
        label = "ANOMALOUS" if is_anom else "NORMAL"

        # Event manager logic with debounce, gap merge, cooldown, and rebaseline (Task K5)
        is_event_start, should_rebaseline = self.event_manager.update(is_anom, top_ratio, len(fired))

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

    def to_dict(self) -> Dict[str, Any]:
        """Serialize internal pipeline state to dictionary."""
        from sentinel.state import pipeline_to_dict
        return pipeline_to_dict(self)

    @classmethod
    def from_dict(cls, state: Dict[str, Any]) -> Sentinel:
        """Restore pipeline from serialized dictionary state."""
        from sentinel.state import pipeline_from_dict
        return pipeline_from_dict(cls, state)

    def save(self, file_path: Union[str, Path]) -> None:
        """Checkpoint pipeline state to file (.json or .pkl)."""
        from sentinel.state import save_pipeline_state
        save_pipeline_state(self, file_path)

    @classmethod
    def load(cls, file_path: Union[str, Path]) -> Sentinel:
        """Load and restore Sentinel pipeline from checkpoint file."""
        from sentinel.state import load_pipeline_state
        return load_pipeline_state(cls, file_path)


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
