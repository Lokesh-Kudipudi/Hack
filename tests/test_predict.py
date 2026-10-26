import json

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("sentinel.pipeline")

from eval import predict  # noqa: E402

N = 3000
TS = pd.date_range("2020-01-01", periods=N, freq="5min")


@pytest.fixture
def folder(tmp_path):
    rng = np.random.default_rng(0)
    x = 10 + np.sin(2 * np.pi * np.arange(N) / 288) + rng.normal(0, 0.1, N)
    x[2000:2010] += 8
    (tmp_path / "in" / "catA").mkdir(parents=True)
    pd.DataFrame({"timestamp": TS, "value": x}).to_csv(tmp_path / "in" / "catA" / "s1.csv", index=False)
    pd.DataFrame({"timestamp": TS, "value": 10 + rng.normal(0, 0.1, N)}).to_csv(tmp_path / "in" / "catA" / "s2.csv",
                                                                                index=False)
    (tmp_path / "labels.json").write_text(json.dumps({"catA/s1.csv": [[str(TS[1990]), str(TS[2040])]],
                                                      "catA/s2.csv": []}))
    return tmp_path


def test_predict_folder_writes_outputs_and_scores(folder):
    out = folder / "out"
    s = predict.main(["--input", str(folder / "in"), "--out", str(out), "--config", "default",
                      "--windows", str(folder / "labels.json"), "--combined", "-q"])
    assert s["files"] == 2 and s["points"] == 2 * N
    per = pd.read_csv(out / "catA" / "s1.csv")
    assert list(per.columns) == ["timestamp", "value", "score", "label", "is_event_start", "evidence_type",
                                 "evidence_text", "evidence"]
    assert len(per) == N and per["score"].between(0, 1).all()
    anom = per[per["label"] == "ANOMALOUS"]
    assert len(anom) and anom["evidence_text"].fillna("").str.len().gt(0).all()
    ev = pd.read_csv(out / "events.csv")
    assert ((ev["file"] == "catA/s1.csv") & pd.to_datetime(ev["timestamp"]).between(TS[1990], TS[2040])).any()
    assert len(pd.read_csv(out / "predictions.csv")) == 2 * N
    summary = json.loads((out / "summary.json").read_text())
    assert summary["metrics"]["tp"] == 1
    assert summary["runtime"]["us_per_point"] > 0


def test_predict_single_file(folder):
    s = predict.main(["--input", str(folder / "in" / "catA" / "s2.csv"), "--out", str(folder / "o2"),
                      "--config", "default", "-q"])
    assert s["files"] == 1 and (folder / "o2" / "catA" / "s2.csv").is_file()
