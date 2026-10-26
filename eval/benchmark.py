"""Inference runtime: µs per update() call, single core, no I/O (data pre-loaded).

    python -m eval.benchmark                                  # all detectors, full corpus
    python -m eval.benchmark --detectors sentinel --repeat 3

Writes results/runs/<ts>_benchmark_<tag>.json; eval.leaderboard renders the latest one.
"""
from __future__ import annotations

import argparse
import gc
import json
import platform
import time
from datetime import datetime
from typing import Dict, List, Optional, Sequence

import numpy as np

from eval.data import iter_corpus
from eval.run import DETECTORS, RUNS_DIR, REPO_ROOT, _git_sha, make_factory, sentinel_source


def bench_detector(detector: str, series: List, config: Optional[dict] = None, repeat: int = 1) -> Dict:
    factory = make_factory(detector, config)
    per_call: List[np.ndarray] = []
    totals = []
    for _ in range(repeat):
        total_ns, n = 0, 0
        calls = []
        for key, ts, xs in series:
            det = factory(len(xs), key)
            update = det.update
            clock = time.perf_counter_ns
            buf = np.empty(len(xs), dtype=np.int64)
            gc.disable()
            try:
                t_file = clock()
                for i in range(len(xs)):
                    t0 = clock()
                    update(ts[i], xs[i])
                    buf[i] = clock() - t0
                total_ns += clock() - t_file
            finally:
                gc.enable()
            n += len(xs)
            calls.append(buf)
        totals.append(total_ns / n / 1e3)
        per_call.append(np.concatenate(calls))
    calls = np.concatenate(per_call) / 1e3
    best = min(totals)  # loop overhead incl. timer calls; best of repeats
    return {
        "detector": detector,
        "points": int(len(calls) / repeat),
        "us_per_point_mean": float(calls.mean()),
        "us_per_point_p50": float(np.percentile(calls, 50)),
        "us_per_point_p99": float(np.percentile(calls, 99)),
        "us_per_point_max": float(calls.max()),
        "points_per_sec": 1e6 / float(calls.mean()),
        "loop_us_per_point_best": best,
        **({"sentinel_source": sentinel_source()} if detector == "sentinel" else {}),
    }


def main(argv: Optional[Sequence[str]] = None) -> Dict:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--detectors", nargs="+", default=DETECTORS)
    ap.add_argument("--categories", nargs="+")
    ap.add_argument("--repeat", type=int, default=1)
    ap.add_argument("--tag", default="corpus")
    ap.add_argument("--no-save", action="store_true")
    args = ap.parse_args(argv)

    series = [(i.key, df["timestamp"].tolist(), df["value"].tolist())
              for i, df, _ in iter_corpus(categories=args.categories, windows={})]
    rows = []
    for d in args.detectors:
        try:
            r = bench_detector(d, series, repeat=args.repeat)
        except ImportError as e:
            print(f"skip {d}: {e}")
            continue
        rows.append(r)
        print(f"{d:18s} mean={r['us_per_point_mean']:7.2f} µs  p50={r['us_per_point_p50']:6.2f}  "
              f"p99={r['us_per_point_p99']:7.2f}  {r['points_per_sec']:,.0f} pts/s")
    record = {
        "kind": "benchmark", "created": datetime.now().isoformat(timespec="seconds"), "tag": args.tag,
        "git_sha": _git_sha(), "machine": f"{platform.machine()} {platform.processor()} {platform.system()}",
        "python": platform.python_version(), "categories": args.categories, "results": rows,
    }
    if not args.no_save:
        path = RUNS_DIR / f"{datetime.now():%Y%m%d-%H%M%S}_benchmark_{args.tag}.json"
        RUNS_DIR.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(record, indent=1))
        print("saved", path.relative_to(REPO_ROOT))
    return record


if __name__ == "__main__":
    main()
