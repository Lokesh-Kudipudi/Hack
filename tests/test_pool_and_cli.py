"""Unit tests for SentinelPool multi-stream management and CLI execution (Task K17)."""
import json
import subprocess
import sys
import tempfile
from pathlib import Path
import pandas as pd
import pytest

from sentinel.pool import SentinelPool
from sentinel.__main__ import main


def test_sentinel_pool_lifecycle():
    pool = SentinelPool()
    assert len(pool) == 0

    base_t = pd.Timestamp("2026-01-01 00:00:00")

    # Feed observations across 3 distinct streams
    streams = ["sensor_alpha", "sensor_beta", "sensor_gamma"]
    for i in range(150):
        t = base_t + pd.Timedelta(minutes=5 * i)
        for s_idx, sid in enumerate(streams):
            val = (s_idx + 1) * 20.0 + (50.0 if i == 120 and sid == "sensor_alpha" else 0.0)
            res = pool.update(sid, t, val)
            assert res.value == val

    assert len(pool) == 3
    assert set(pool.stream_ids) == set(streams)

    # Check memory report
    rep = pool.memory_report()
    assert rep["active_streams"] == 3
    assert rep["total_points_processed"] == 150 * 3
    assert "sensor_alpha" in rep["streams"]
    assert rep["streams"]["sensor_alpha"]["points"] == 150

    # Close one stream
    assert pool.close_stream("sensor_gamma") is True
    assert len(pool) == 2
    assert "sensor_gamma" not in pool.stream_ids


def test_sentinel_pool_save_and_load():
    pool = SentinelPool()
    base_t = pd.Timestamp("2026-01-01 00:00:00")

    for i in range(50):
        t = base_t + pd.Timedelta(minutes=5 * i)
        pool.update("host_a", t, float(i))
        pool.update("host_b", t, float(i * 2))

    with tempfile.TemporaryDirectory() as tmpdir:
        pool.save_pool(tmpdir)
        loaded_pool = SentinelPool.load_pool(tmpdir)

        assert len(loaded_pool) == 2
        assert "host_a" in loaded_pool.stream_ids
        assert "host_b" in loaded_pool.stream_ids


def test_cli_csv_file_execution():
    with tempfile.TemporaryDirectory() as tmpdir:
        csv_path = Path(tmpdir) / "test_stream.csv"
        out_path = Path(tmpdir) / "output.jsonl"

        df = pd.DataFrame({
            "timestamp": pd.date_range("2026-01-01", periods=60, freq="5min"),
            "value": [25.0] * 59 + [250.0],
        })
        df.to_csv(csv_path, index=False)

        exit_code = main(["--input", str(csv_path), "--output", str(out_path)])
        assert exit_code == 0
        assert out_path.exists()

        lines = out_path.read_text(encoding="utf-8").strip().splitlines()
        assert len(lines) == 60
        last_rec = json.loads(lines[-1])
        assert "score" in last_rec
        assert "label" in last_rec


def test_cli_multi_stream_execution():
    with tempfile.TemporaryDirectory() as tmpdir:
        csv_path = Path(tmpdir) / "multi_stream.csv"
        out_path = Path(tmpdir) / "multi_out.jsonl"

        rows = []
        for i in range(30):
            t = f"2026-01-01 00:{i:02d}:00"
            rows.append({"stream": "s1", "timestamp": t, "value": 10.0})
            rows.append({"stream": "s2", "timestamp": t, "value": 20.0})

        pd.DataFrame(rows).to_csv(csv_path, index=False)

        exit_code = main([
            "--input", str(csv_path),
            "--output", str(out_path),
            "--stream-col", "stream",
        ])
        assert exit_code == 0
        lines = out_path.read_text(encoding="utf-8").strip().splitlines()
        assert len(lines) == 60
        first_rec = json.loads(lines[0])
        assert "stream_id" in first_rec
