"""State checkpointing and restoration for Sentinel pipelines (Task K16).

Enables pausing, persisting, migrating, and resuming streaming detection pipelines
with bit-identical output continuity (resumed output == uninterrupted output).
"""
from __future__ import annotations

import collections
import json
import os
import pickle
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
import numpy as np
import pandas as pd


def _serialize_obj(val: Any) -> Any:
    """Helper to convert collections, numpy types, and Timestamps to JSON-serializable primitives."""
    if isinstance(val, (collections.deque, list, tuple)):
        return [_serialize_obj(x) for x in val]
    if isinstance(val, dict):
        return {k: _serialize_obj(v) for k, v in val.items()}
    if isinstance(val, (np.floating, float)):
        return float(val)
    if isinstance(val, (np.integer, int)):
        return int(val)
    if isinstance(val, (np.bool_, bool)):
        return bool(val)
    if isinstance(val, pd.Timestamp):
        return val.isoformat()
    if isinstance(val, np.ndarray):
        return val.tolist()
    return val


def detector_to_dict(det: Any) -> Dict[str, Any]:
    """Extract serializable state dict from any detector."""
    state: Dict[str, Any] = {"class_name": det.__class__.__name__}
    for attr in dir(det):
        if attr.startswith("_") or callable(getattr(det, attr)):
            continue
        val = getattr(det, attr)
        state[attr] = _serialize_obj(val)
    return state


def restore_detector_state(det: Any, state: Dict[str, Any]):
    """Restore detector state from state dict."""
    for attr, val in state.items():
        if attr == "class_name" or not hasattr(det, attr):
            continue
        orig = getattr(det, attr)
        if isinstance(orig, collections.deque):
            maxlen = orig.maxlen
            restored_deque = collections.deque(val, maxlen=maxlen)
            setattr(det, attr, restored_deque)
        elif isinstance(orig, dict) and len(orig) > 0 and all(isinstance(k, int) for k in orig.keys()):
            # Restore integer keys (e.g. Seasonal hourly profiles) converted to strings in JSON
            restored_dict = {int(k): v for k, v in val.items()}
            setattr(det, attr, restored_dict)
        else:
            setattr(det, attr, val)


def spot_to_dict(spot_obj: Any) -> Dict[str, Any]:
    """Serialize SPOT or DSPOT model state."""
    state: Dict[str, Any] = {"class_name": spot_obj.__class__.__name__}
    for attr in dir(spot_obj):
        if attr.startswith("_") or callable(getattr(spot_obj, attr)):
            continue
        val = getattr(spot_obj, attr)
        if attr == "spot" and val is not None:
            state["inner_spot"] = spot_to_dict(val)
        else:
            state[attr] = _serialize_obj(val)
    return state


def restore_spot_state(spot_obj: Any, state: Dict[str, Any]):
    """Restore SPOT / DSPOT state from dict."""
    for attr, val in state.items():
        if attr in ("class_name", "inner_spot") or not hasattr(spot_obj, attr):
            continue
        setattr(spot_obj, attr, val)
    if "inner_spot" in state and hasattr(spot_obj, "spot"):
        restore_spot_state(spot_obj.spot, state["inner_spot"])


def save_pipeline_state(pipeline: Any, file_path: Union[str, Path]) -> None:
    """Save full pipeline state to a file (.json or .pkl)."""
    state = pipeline_to_dict(pipeline)
    path = Path(file_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    if path.suffix.lower() == ".json":
        with open(path, "w", encoding="utf-8") as f:
            json.dump(state, f, indent=2)
    else:
        with open(path, "wb") as f:
            pickle.dump(state, f, protocol=pickle.HIGHEST_PROTOCOL)


def load_pipeline_state(pipeline_class: Any, file_path: Union[str, Path]) -> Any:
    """Load pipeline state from a file and return a fully restored pipeline instance."""
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"Checkpoint file not found: {path}")

    if path.suffix.lower() == ".json":
        with open(path, "r", encoding="utf-8") as f:
            state = json.load(f)
    else:
        with open(path, "rb") as f:
            state = pickle.load(f)

    return pipeline_from_dict(pipeline_class, state)


def pipeline_to_dict(pipeline: Any) -> Dict[str, Any]:
    """Serialize complete Sentinel pipeline state to a nested dictionary."""
    state: Dict[str, Any] = {
        "config": pipeline.config,
        "count": pipeline.count,
        "probation_limit": pipeline.probation_limit,
        "threshold": pipeline.threshold,
        "debounce": pipeline.debounce,
        "alpha": pipeline.alpha,
        "detector_thresholds": pipeline.detector_thresholds,
        "use_spot": pipeline.use_spot,
        "_last_t": pipeline._last_t.isoformat() if pipeline._last_t is not None else None,
        "_last_x": pipeline._last_x,
        "probation_timestamps": [t.isoformat() for t in pipeline.probation_timestamps],
        "probation_values": list(pipeline.probation_values),
        "detectors": {name: detector_to_dict(det) for name, det in pipeline.detectors.items()},
        "spot_models": {name: spot_to_dict(model) for name, model in pipeline.spot_models.items()},
        "event_manager": {
            attr: _serialize_obj(getattr(pipeline.event_manager, attr))
            for attr in dir(pipeline.event_manager)
            if not attr.startswith("_") and not callable(getattr(pipeline.event_manager, attr))
        },
    }
    return state


def pipeline_from_dict(pipeline_class: Any, state: Dict[str, Any]) -> Any:
    """Instantiate and restore a pipeline from a dictionary state."""
    # Instantiate clean pipeline with identical configuration
    pipeline = pipeline_class(config=state["config"])
    pipeline.count = state["count"]
    pipeline.probation_limit = state["probation_limit"]
    pipeline.threshold = state["threshold"]
    pipeline.debounce = state["debounce"]
    pipeline.alpha = state["alpha"]
    pipeline.detector_thresholds = state["detector_thresholds"]
    pipeline.use_spot = state["use_spot"]

    if state.get("_last_t"):
        pipeline._last_t = pd.Timestamp(state["_last_t"])
    pipeline._last_x = state.get("_last_x")

    pipeline.probation_timestamps = [pd.Timestamp(t) for t in state.get("probation_timestamps", [])]
    pipeline.probation_values = list(state.get("probation_values", []))

    # Restore detectors
    for name, det_state in state.get("detectors", {}).items():
        if name in pipeline.detectors:
            restore_detector_state(pipeline.detectors[name], det_state)

    # Restore SPOT models
    for name, spot_state in state.get("spot_models", {}).items():
        if name in pipeline.spot_models:
            restore_spot_state(pipeline.spot_models[name], spot_state)

    # Restore EventManager
    em_state = state.get("event_manager", {})
    for attr, val in em_state.items():
        if hasattr(pipeline.event_manager, attr):
            setattr(pipeline.event_manager, attr, val)

    return pipeline
