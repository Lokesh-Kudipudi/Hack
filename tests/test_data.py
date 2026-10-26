import json
from pathlib import Path

import pandas as pd
import pytest

from eval import data


def _write_csv(path: Path, n: int = 10, start: str = "2020-01-01") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    ts = pd.date_range(start, periods=n, freq="5min")
    pd.DataFrame({"timestamp": ts, "value": range(n)}).to_csv(path, index=False)


@pytest.fixture
def starter(tmp_path):
    """Starter-pack style: data/<cat>/<cat>/*.csv plus __MACOSX junk and a labels file."""
    _write_csv(tmp_path / "catA" / "catA" / "a1.csv")
    _write_csv(tmp_path / "catB" / "catB" / "b1.csv", n=20)
    _write_csv(tmp_path / "catB" / "__MACOSX" / "catB" / "._b1.csv")
    labels = {"catA/a1.csv": [["2020-01-01 00:10:00.000000", "2020-01-01 00:20:00.000000"]], "catB/b1.csv": []}
    (tmp_path / "labels").mkdir()
    (tmp_path / "labels" / "combined_windows.json").write_text(json.dumps(labels))
    return tmp_path


def test_list_files_nested_and_skips_macosx(starter):
    files = data.list_files(starter)
    assert [f.key for f in files] == ["catA/a1.csv", "catB/b1.csv"]
    assert [f.key for f in data.list_files(starter, categories=["catB"])] == ["catB/b1.csv"]


def test_nab_clone_layout_ignores_results(tmp_path):
    _write_csv(tmp_path / "data" / "realX" / "x.csv")
    _write_csv(tmp_path / "results" / "det" / "realX" / "det_x.csv")
    (tmp_path / "labels").mkdir()
    assert [f.key for f in data.list_files(tmp_path)] == ["realX/x.csv"]


def test_load_series_types(starter):
    df = data.load_series(starter / "catA" / "catA" / "a1.csv")
    assert list(df.columns) == ["timestamp", "value"]
    assert pd.api.types.is_datetime64_any_dtype(df["timestamp"])
    assert df["value"].dtype == float and len(df) == 10


def test_windows_and_indices(starter):
    wp = data.windows_path(starter)
    assert wp == starter / "labels" / "combined_windows.json"
    w = data.load_windows(wp)
    df = data.load_series(starter / "catA" / "catA" / "a1.csv")
    assert data.window_indices(df["timestamp"], w["catA/a1.csv"]) == [(2, 4)]
    # window entirely outside the series is dropped
    out = [(pd.Timestamp("2030-01-01"), pd.Timestamp("2030-01-02"))]
    assert data.window_indices(df["timestamp"], out) == []


def test_iter_corpus(starter):
    items = list(data.iter_corpus(starter))
    assert len(items) == 2
    info, df, wins = items[0]
    assert info.key == "catA/a1.csv" and len(wins) == 1


def test_env_override(starter, monkeypatch):
    monkeypatch.setenv("SENTINEL_DATA", str(starter))
    assert data.data_root() == starter.resolve()


@pytest.mark.parametrize("n,expected", [(100, 15), (4032, 604), (5000, 750), (22695, 750), (0, 0)])
def test_probation_length(n, expected):
    assert data.probation_length(n) == expected


NAB = data.REPO_ROOT / "data" / "NAB"


@pytest.mark.skipif(not (NAB / "labels").is_dir(), reason="NAB not cloned")
def test_full_nab_corpus():
    files = data.list_files(NAB)
    assert len(files) == 58
    assert len({f.category for f in files}) == 7
    w = data.load_windows(data.windows_path(NAB))
    assert set(w) == {f.key for f in files}
    assert sum(len(v) for v in w.values()) == 116


def test_file_order_preserved(tmp_path):
    p = tmp_path / "c" / "x.csv"
    p.parent.mkdir()
    p.write_text("timestamp,value\n2020-01-01 00:10:00,1\n2020-01-01 00:05:00,2\n2020-01-01 00:15:00,3\n")
    df = data.load_series(p)
    assert df["value"].tolist() == [1.0, 2.0, 3.0]
    w = [(pd.Timestamp("2020-01-01 00:10:00"), pd.Timestamp("2020-01-01 00:15:00"))]
    assert data.window_indices(df["timestamp"], w) == [(0, 2)]
