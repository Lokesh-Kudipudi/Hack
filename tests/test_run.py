import json

import numpy as np
import pandas as pd
import pytest

from eval import leaderboard, run
from eval.baselines import NullDetector, RandomDetector, WindowedGaussian
from eval.data import REPO_ROOT, load_series

N = 400
TS = pd.date_range("2020-01-01", periods=N, freq="5min")


def _series(spike_at=None):
    x = np.sin(np.arange(N) / 10) + np.random.default_rng(0).normal(0, 0.05, N)
    if spike_at is not None:
        x[spike_at] += 20
    return pd.DataFrame({"timestamp": TS, "value": x})


@pytest.fixture
def corpus(tmp_path):
    (tmp_path / "cat").mkdir()
    _series(300).to_csv(tmp_path / "cat" / "spike.csv", index=False)
    _series().to_csv(tmp_path / "cat" / "flat.csv", index=False)
    (tmp_path / "labels").mkdir()
    (tmp_path / "labels" / "combined_windows.json").write_text(json.dumps(
        {"cat/spike.csv": [[str(TS[295]), str(TS[310])]], "cat/flat.csv": []}))
    return tmp_path


def test_null_never_flags():
    d = NullDetector(n_total=N)
    assert all(d.update(t, 1.0).label == "NORMAL" for t in TS)


def test_random_is_deterministic():
    a = [RandomDetector(N, key="k").update(t, 0).score for t in TS[:5]]
    b = [RandomDetector(N, key="k").update(t, 0).score for t in TS[:5]]
    assert a == b


def test_event_start_on_rising_edge_only_after_probation():
    df = _series(300)
    df.loc[301:303, "value"] += 20  # 4-point anomaly -> one event
    out, _ = run.run_file(lambda n, k: WindowedGaussian(n_total=n), df)
    assert out["is_event_start"].sum() == 1
    assert out.loc[300, "is_event_start"] and out.loc[300:303, "label"].eq("ANOMALOUS").all()
    assert not out.loc[: int(0.15 * N) - 1, "label"].eq("ANOMALOUS").any()
    assert list(out.columns) == ["timestamp", "value", "score", "label", "is_event_start", "evidence"]


def test_evaluate_and_save(corpus, tmp_path, monkeypatch):
    res = run.evaluate(run.make_factory("windowed_gaussian"), corpus)
    o = res["overall"]
    assert o["tp"] == 1 and o["fn"] == 0
    assert res["runtime"]["points"] == 2 * N and res["runtime"]["us_per_point"] > 0
    assert set(res["nab"]) == {"standard", "reward_low_FP_rate", "reward_low_FN_rate"}

    monkeypatch.setattr(run, "RUNS_DIR", tmp_path / "runs")
    p = run.save_run("windowed_gaussian", "unit test", {}, None, res)
    rec = json.loads(p.read_text())
    assert rec["detector"] == "windowed_gaussian" and rec["overall"]["tp"] == 1
    md = leaderboard.render(leaderboard.load_runs(tmp_path / "runs"))
    assert "windowed_gaussian" in md and "unit test" in md


def test_unknown_detector():
    with pytest.raises(ValueError):
        run.make_factory("nope")


NAB = REPO_ROOT / "data" / "NAB"


@pytest.mark.skipif(not (NAB / "results").is_dir(), reason="NAB not cloned")
def test_windowed_gaussian_matches_nab_reference():
    key = "realKnownCause/machine_temperature_system_failure.csv"  # has a backward time jump
    df = load_series(NAB / "data" / key)
    d = WindowedGaussian(len(df))
    ours = np.array([d.update(t, x).score for t, x in zip(df["timestamp"], df["value"])])
    ref = pd.read_csv(NAB / "results" / "windowedGaussian" / key.replace("/", "/windowedGaussian_"))
    np.testing.assert_allclose(ours, ref["anomaly_score"].values, atol=1e-9)


def test_benchmark_on_tiny_series():
    from eval.benchmark import bench_detector
    series = [("c/a.csv", list(TS), list(_series(300)["value"]))]
    r = bench_detector("windowed_gaussian", series)
    assert r["points"] == N and r["us_per_point_mean"] > 0 and r["points_per_sec"] > 0


def test_tune_grid_and_loco():
    from eval import tune
    from eval.scorer import FileScore
    cfgs = tune.expand({"fusion": [("max", 1), ("vote", 2)], "spot_q": [1e-4]})
    assert cfgs == [{"fusion": "max", "vote_k": 1, "spot_q": 1e-4}, {"fusion": "vote", "vote_k": 2, "spot_q": 1e-4}]
    # config 0 wins on A, config 1 wins on B: each held-out category gets the other's winner
    per_cfg = [[FileScore("A/x", "A", tp=1), FileScore("B/y", "B", fp=3, fn=1)],
               [FileScore("A/x", "A", fn=1), FileScore("B/y", "B", tp=1)]]
    lo = tune.loco(cfgs, per_cfg)
    assert lo["per_category"]["A"]["config_index"] == 1 and lo["per_category"]["A"]["held_out_f1"] == 0.0
    assert lo["per_category"]["B"]["config_index"] == 0 and lo["per_category"]["B"]["held_out_f1"] == 0.0
    assert lo["mean_held_out_f1"] == 0.0


def test_ablation_configs_use_handoff_flags():
    from eval.ablation import configs
    allowed = {"detectors", "use_spot", "use_events", "fusion", "vote_k", "threshold"}
    for _, _, _, cfg in configs():
        assert set(cfg) <= allowed
