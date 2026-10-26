"""Command Line Interface (CLI) for Sentinel (Task K17).

Usage:
  python -m sentinel --input data.csv --output results.jsonl
  cat stream.jsonl | python -m sentinel > flagged.jsonl
  python -m sentinel --input multi_stream.csv --stream-col stream_id
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, Optional
import pandas as pd

from sentinel.pipeline import Sentinel
from sentinel.pool import SentinelPool
from sentinel import __version__


def parse_args(args: Optional[list] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="sentinel",
        description=f"Sentinel v{__version__} — Streaming Time-Series Anomaly Detection Engine",
    )
    parser.add_argument(
        "-i", "--input",
        type=str,
        default="-",
        help="Input CSV or JSONL file path (default: '-' for stdin)",
    )
    parser.add_argument(
        "-o", "--output",
        type=str,
        default="-",
        help="Output JSONL or CSV file path (default: '-' for stdout)",
    )
    parser.add_argument(
        "-c", "--config",
        type=str,
        default=None,
        help="Optional path to custom JSON configuration file",
    )
    parser.add_argument(
        "--stream-col",
        type=str,
        default=None,
        help="Optional column name identifying distinct streams in a multiplexed dataset",
    )
    parser.add_argument(
        "--time-col",
        type=str,
        default="timestamp",
        help="Timestamp column name (default: 'timestamp')",
    )
    parser.add_argument(
        "--val-col",
        type=str,
        default="value",
        help="Value column name (default: 'value')",
    )
    parser.add_argument(
        "--format",
        choices=["jsonl", "csv"],
        default="jsonl",
        help="Output stream format (default: jsonl)",
    )
    parser.add_argument(
        "--anomalies-only",
        action="store_true",
        help="Only output flagged ANOMALOUS points",
    )
    parser.add_argument(
        "-v", "--version",
        action="version",
        version=f"Sentinel {__version__}",
    )
    return parser.parse_args(args)


def main(args: Optional[list] = None) -> int:
    parsed = parse_args(args)

    # Load custom config if provided
    config: Optional[Dict[str, Any]] = None
    if parsed.config:
        with open(parsed.config, "r", encoding="utf-8") as f:
            config = json.load(f)

    # Determine input stream source
    is_stdin = parsed.input == "-" or not parsed.input
    is_stdout = parsed.output == "-" or not parsed.output

    out_file = None if is_stdout else open(parsed.output, "w", encoding="utf-8")

    def emit_result(r_dict: Dict[str, Any]):
        if parsed.anomalies_only and r_dict.get("label") != "ANOMALOUS":
            return
        line = json.dumps(r_dict)
        if is_stdout:
            sys.stdout.write(line + "\n")
            sys.stdout.flush()
        else:
            out_file.write(line + "\n")

    t_start = time.perf_counter()
    n_pts = 0
    n_anom = 0

    try:
        # Multi-stream routing mode
        if parsed.stream_col:
            pool = SentinelPool(default_config=config)
            df = pd.read_csv(sys.stdin if is_stdin else parsed.input)
            for _, row in df.iterrows():
                sid = str(row[parsed.stream_col])
                t = row[parsed.time_col]
                x = row[parsed.val_col]
                res = pool.update(sid, t, x)
                n_pts += 1
                if res.label == "ANOMALOUS":
                    n_anom += 1

                emit_result({
                    "stream_id": sid,
                    "timestamp": str(res.timestamp),
                    "value": res.value,
                    "score": res.score,
                    "label": res.label,
                    "is_event_start": res.is_event_start,
                    "evidence": res.evidence,
                })
        else:
            # Single stream streaming mode
            if not is_stdin and parsed.input.endswith(".csv"):
                df = pd.read_csv(parsed.input)
                sentinel = Sentinel(n_total=len(df), config=config)
                for _, row in df.iterrows():
                    res = sentinel.update(row[parsed.time_col], row[parsed.val_col])
                    n_pts += 1
                    if res.label == "ANOMALOUS":
                        n_anom += 1
                    emit_result({
                        "timestamp": str(res.timestamp),
                        "value": res.value,
                        "score": res.score,
                        "label": res.label,
                        "is_event_start": res.is_event_start,
                        "evidence": res.evidence,
                        "detectors": res.detector_scores,
                    })
            else:
                # Stdin or JSONL streaming line-by-line
                sentinel = Sentinel(config=config)
                stream_in = sys.stdin if is_stdin else open(parsed.input, "r", encoding="utf-8")
                for line in stream_in:
                    line = line.strip()
                    if not line:
                        continue
                    if line.startswith("{"):
                        record = json.loads(line)
                        t = record.get(parsed.time_col)
                        x = record.get(parsed.val_col)
                    else:
                        # Simple comma separated fallback
                        parts = line.split(",")
                        t, x = parts[0].strip(), parts[1].strip()

                    res = sentinel.update(t, x)
                    n_pts += 1
                    if res.label == "ANOMALOUS":
                        n_anom += 1
                    emit_result({
                        "timestamp": str(res.timestamp),
                        "value": res.value,
                        "score": res.score,
                        "label": res.label,
                        "is_event_start": res.is_event_start,
                        "evidence": res.evidence,
                    })

    finally:
        if out_file:
            out_file.close()

    elapsed = time.perf_counter() - t_start
    rate = n_pts / max(1e-6, elapsed)
    us_per_pt = (elapsed / max(1, n_pts)) * 1e6
    sys.stderr.write(
        f"\n[Sentinel v{__version__}] Finished: {n_pts} points processed, "
        f"{n_anom} anomalies flagged in {elapsed:.3f}s ({us_per_pt:.2f} µs/pt, {rate:,.0f} pts/sec)\n"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
