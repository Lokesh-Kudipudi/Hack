"""Inference time of NAB's own detector implementations vs Sentinel, on the same streams and hardware.

Runs the detectors that ship as plain Python in data/NAB/nab/detectors through NAB's AnomalyDetector API
(constructor + initialize + handleRecord) and times only the per-point handleRecord call; Sentinel is timed
on its per-point update() the same way. Single process, no I/O in the timed loop.

Not runnable here (reported as such): Numenta HTM / numentaTM (nupic, Python 2), HTM Java (Java build),
ARTime (Julia via JuliaCall), Random Cut Forest (AWS Kinesis Analytics service), Twitter ADVec (R package).

    python -m eval.nab_runtime [--max-points 4032] [--budget 120]

Writes results/runs/<ts>_nab_runtime.json and docs/results/inference_runtime_comparison.md.
"""
from __future__ import annotations

import argparse
import gc
import importlib
import json
import platform
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np
import pandas as pd

from eval.data import list_files, load_series
from eval.nab_score import NAB_DIR
from eval.run import REPO_ROOT, RUNS_DIR, _git_sha, make_factory

# NAB leaderboard key -> (module under nab.detectors, class name)
RUNNABLE = {
    "contextOSE": ("context_ose.context_ose_detector", "ContextOSEDetector"),
    "knncad": ("knncad.knncad_detector", "KnncadDetector"),
    "relativeEntropy": ("relative_entropy.relative_entropy_detector", "RelativeEntropyDetector"),
    "earthgeckoSkyline": ("earthgecko_skyline.earthgecko_skyline_detector", "EarthgeckoSkylineDetector"),
    "skyline": ("skyline.skyline_detector", "SkylineDetector"),
    "bayesChangePt": ("bayes_changept.bayes_changept_detector", "BayesChangePtDetector"),
    "expose": ("expose.expose_detector", "ExposeDetector"),
    "windowedGaussian": ("gaussian.windowedGaussian_detector", "WindowedGaussianDetector"),
    "random": ("random.random_detector", "RandomDetector"),
    "null": ("null.null_detector", "NullDetector"),
}
NOT_RUNNABLE = {
    "ARTime": "Julia package via JuliaCall",
    "numenta": "nupic (Python 2 only)",
    "numentaTM": "nupic (Python 2 only)",
    "htmjava": "Java build of htm.java",
    "randomCutForest": "AWS Kinesis Analytics cloud service",
    "twitterADVec": "R package",
}
# one representative stream per category with anomalies (first --max-points points of each)
STREAMS = ["artificialWithAnomaly/art_daily_jumpsup.csv", "realAWSCloudwatch/ec2_cpu_utilization_825cc2.csv",
           "realAdExchange/exchange-2_cpc_results.csv", "realKnownCause/nyc_taxi.csv",
           "realTraffic/occupancy_6005.csv", "realTweets/Twitter_volume_AAPL.csv"]


class _DataSet:
    """Minimal stand-in for nab.corpus.DataFile (only .data is used by the detectors)."""

    def __init__(self, df: pd.DataFrame):
        self.data = df


def _nab_class(module: str, cls: str):
    if not hasattr(np, "asscalar"):  # removed in NumPy 1.23; EXPoSE still calls it
        np.asscalar = lambda a: a.item()
    if str(NAB_DIR) not in sys.path:
        sys.path.insert(0, str(NAB_DIR))
    return getattr(importlib.import_module(f"nab.detectors.{module}"), cls)


def _time_loop(fn, items) -> np.ndarray:
    clock = time.perf_counter_ns
    out = np.empty(len(items), dtype=np.int64)
    gc.disable()
    try:
        for i, it in enumerate(items):
            t0 = clock()
            fn(it)
            out[i] = clock() - t0
    finally:
        gc.enable()
    return out


def bench_nab(name: str, streams: List[pd.DataFrame], budget_s: float) -> Dict:
    module, cls = RUNNABLE[name]
    klass = _nab_class(module, cls)
    calls, files_done, t_start = [], 0, time.perf_counter()
    for df in streams:
        det = klass(dataSet=_DataSet(df), probationaryPercent=0.15)
        det.initialize()
        records = df.to_dict("records")
        calls.append(_time_loop(det.handleRecord, records))
        files_done += 1
        if time.perf_counter() - t_start > budget_s:
            break
    c = np.concatenate(calls) / 1e3
    return {"detector": name, "points": int(len(c)), "files": files_done, "us_per_point_mean": float(c.mean()),
            "us_per_point_p50": float(np.percentile(c, 50)), "us_per_point_p99": float(np.percentile(c, 99))}


def bench_sentinel(streams: List[pd.DataFrame], config: Optional[dict] = None) -> Dict:
    factory = make_factory("sentinel", config or {})
    calls = []
    for df in streams:
        det = factory(len(df), "")
        pts = list(zip(df["timestamp"].tolist(), df["value"].tolist()))
        calls.append(_time_loop(lambda p: det.update(p[0], p[1]), pts))
    c = np.concatenate(calls) / 1e3
    return {"detector": "Sentinel (engine default)", "points": int(len(c)), "files": len(streams),
            "us_per_point_mean": float(c.mean()), "us_per_point_p50": float(np.percentile(c, 50)),
            "us_per_point_p99": float(np.percentile(c, 99))}


def render(rec: Dict) -> str:
    rows = sorted(rec["results"], key=lambda r: r["us_per_point_mean"])
    sent = next((r["us_per_point_mean"] for r in rows if r["detector"].startswith("Sentinel")), None)
    lines = ["# Inference time: NAB detectors vs Sentinel (same streams, same machine)", "",
             "> **About this report:** Reports single-process per-point inference times for Sentinel and runnable NAB detectors on the same six streams.", "",
             f"Generated by `python -m eval.nab_runtime` ({rec['created']}, sha {rec['git_sha']}, {rec['machine']}, "
             f"Python {rec['python']}). Streams: first {rec['max_points']} points of {len(rec['streams'])} files "
             f"({', '.join(rec['streams'])}). Timed: NAB `handleRecord()` / Sentinel `update()` per point, "
             f"single process; detectors over the {rec['budget_s']:.0f}s budget stop after the file in progress.", "",
             "| Detector | µs/point (mean) | p50 | p99 | vs Sentinel | points timed |", "|---|---|---|---|---|---|"]
    for r in rows:
        x = f"{r['us_per_point_mean'] / sent:.1f}x" if sent else "—"
        lines.append(f"| {r['detector']} | **{r['us_per_point_mean']:.1f}** | {r['us_per_point_p50']:.1f} | "
                     f"{r['us_per_point_p99']:.1f} | {x} | {r['points']:,} ({r['files']} files) |")
    lines += ["", "Not runnable on this machine:", ""]
    lines += [f"- {k}: {v}" for k, v in rec["not_runnable"].items()]
    lines.append("")
    return "\n".join(lines)


def main(argv: Optional[Sequence[str]] = None) -> Dict:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--max-points", type=int, default=4032)
    ap.add_argument("--budget", type=float, default=120.0, help="seconds per detector before stopping early")
    ap.add_argument("--detectors", nargs="+", default=list(RUNNABLE))
    args = ap.parse_args(argv)

    paths = {f.key: f.path for f in list_files()}
    streams = [load_series(paths[k]).head(args.max_points).reset_index(drop=True) for k in STREAMS if k in paths]
    results = [bench_sentinel(streams)]
    print(f"{'Sentinel (engine default)':28s} {results[0]['us_per_point_mean']:10.1f} µs/pt")
    for name in args.detectors:
        try:
            r = bench_nab(name, streams, args.budget)
        except Exception as e:  # missing optional dependency etc.
            print(f"{name:28s} failed: {type(e).__name__}: {e}")
            continue
        results.append(r)
        print(f"{name:28s} {r['us_per_point_mean']:10.1f} µs/pt  ({r['points']} pts)")
    rec = {"kind": "nab_runtime", "created": datetime.now().isoformat(timespec="seconds"), "git_sha": _git_sha(),
           "machine": f"{platform.machine()} {platform.system()}", "python": platform.python_version(),
           "max_points": args.max_points, "budget_s": args.budget, "streams": STREAMS, "results": results,
           "not_runnable": NOT_RUNNABLE}
    RUNS_DIR.mkdir(parents=True, exist_ok=True)
    (RUNS_DIR / f"{datetime.now():%Y%m%d-%H%M%S}_nab_runtime.json").write_text(json.dumps(rec, indent=1))
    (REPO_ROOT / "docs" / "results" / "inference_runtime_comparison.md").write_text(render(rec))
    print("wrote docs/results/inference_runtime_comparison.md")
    return rec


if __name__ == "__main__":
    main()
