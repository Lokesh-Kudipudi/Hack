"""Regression tests for the final correctness audit (lokesh/final-docs)."""
import csv
import json

import numpy as np
import pandas as pd

from eval.run import sentinel_config
from sentinel import Sentinel
from sentinel.__main__ import main as cli_main
from sentinel.pipeline import run_series
from sentinel.validate import sanitize_observation


def _series(n=1500, seed=0):
    rng = np.random.default_rng(seed)
    ts = pd.date_range("2020-01-01", periods=n, freq="5min")
    return ts, rng.normal(10.0, 1.0, n)


def test_dotted_config_keys_apply_in_sentinel():
    # results/best_*.json use dotted keys; passed straight to Sentinel they used to be silently ignored
    cfg = {"events.agree_k": 2, "spot.q_per_detector": {"LaggedResidual": 1e-3}, "spot.q": 1e-5}
    s = Sentinel(n_total=1000, config=cfg)
    assert s.event_manager.agree_k == 2
    assert s.spot_models["LaggedResidual"].q == 1e-3
    assert s.spot_models["RobustZ"].q == 1e-5
    assert s.config["spot"]["init_level"] == 0.98  # siblings of a dotted key are kept
    assert not any("." in k for k in s.config)


def test_dotted_config_same_as_eval_path():
    ts, xs = _series()
    xs[1200:1260] += 8.0
    df = pd.DataFrame({"timestamp": ts, "value": xs})
    cfg = {"events.cooldown_frac": 0.1, "spot.q_per_detector": {"LaggedResidual": 1e-4}, "debounce": 2}
    a, b = run_series(df, cfg), run_series(df, sentinel_config(cfg))
    assert a["score"].tolist() == b["score"].tolist()
    assert a["is_event_start"].tolist() == b["is_event_start"].tolist()


def test_spatial_only_event_is_attributed_to_spatial_extreme():
    ts, xs = _series(seed=1)
    # gentle new maximum: beyond the running range + 5% but far below RobustZ's 6σ base threshold
    xs = np.clip(xs, 7.0, 13.0)
    xs[1300] = 13.5
    s = Sentinel(n_total=len(xs))
    res = [s.update(t, float(x)) for t, x in zip(ts, xs)]
    r = res[1300]
    assert r.is_event_start
    ev = r.evidence
    assert ev["fired"] == ["SpatialExtreme"]
    assert ev["top_detector"] == "SpatialExtreme"
    assert "new high" in ev["text"]
    assert ev["context"]["range_high"] == 13.0
    assert ev["threshold_ratio"] > 1.0
    assert "Corroborated by" not in ev["text"]


def test_mixed_timezone_timestamps_do_not_crash():
    s = Sentinel(n_total=100)
    s.update(pd.Timestamp("2020-01-01"), 1.0)
    r = s.update(pd.Timestamp("2020-01-01 00:05", tz="UTC"), 1.0)
    assert r.timestamp == pd.Timestamp("2020-01-01 00:05")
    t, x, flags = sanitize_observation(pd.Timestamp("2020-01-01 00:10", tz="UTC"), 2.0,
                                       last_t=pd.Timestamp("2020-01-01 00:05"))
    assert t == pd.Timestamp("2020-01-01 00:10") and x == 2.0 and not flags.is_backward_timestamp


def test_cli_format_csv(tmp_path):
    ts, xs = _series(n=400)
    xs[350] += 30
    inp = tmp_path / "s.csv"
    pd.DataFrame({"timestamp": ts, "value": xs}).to_csv(inp, index=False)
    out = tmp_path / "o.csv"
    assert cli_main(["-i", str(inp), "-o", str(out), "--format", "csv"]) == 0
    rows = list(csv.DictReader(open(out, newline="")))
    assert len(rows) == 400
    assert {"timestamp", "value", "score", "label", "is_event_start", "evidence"} <= set(rows[0])
    flagged = [r for r in rows if r["label"] == "ANOMALOUS"]
    assert flagged and json.loads(flagged[0]["evidence"])["type"]
