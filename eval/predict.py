"""Run Sentinel on any folder of `timestamp,value` CSVs (e.g. the hidden test windows) and write predictions.

    python -m eval.predict --input data/starter --out outputs/predictions
    python -m eval.predict --input some_file.csv --config results/best_config.json
    python -m eval.predict --input data/starter --windows labels.json   # also score if labels exist

Outputs (in --out):
  <category>/<name>.csv   per point: timestamp,value,score,label,is_event_start,evidence_type,evidence_text,evidence
  events.csv              one row per detected event (file, timestamp, score, type, top detector, text)
  predictions.csv         (--combined) every point of every file in one table, with a `file` column
  summary.json            config, files, points, events, runtime, and metrics when labels were given

Streaming: every file goes through a fresh Sentinel one point at a time, in file order; nothing looks ahead.
"""
from __future__ import annotations

import argparse
import json
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import pandas as pd

from eval import scorer
from eval.data import FileInfo, REPO_ROOT, list_files, load_series, load_windows, windows_path
from eval.run import make_factory, run_file

DEFAULT_CONFIG = REPO_ROOT / "results" / "best_config.json"
DEFAULT_OUT = REPO_ROOT / "outputs" / "predictions"  # *.csv is gitignored outside results/


def find_inputs(path: Path) -> List[FileInfo]:
    path = Path(path)
    if path.is_file():
        return [FileInfo(key=f"{path.parent.name}/{path.name}", category=path.parent.name, name=path.name, path=path)]
    return list_files(path)


def load_config(arg: Optional[str]) -> dict:
    if arg is None:
        return json.loads(DEFAULT_CONFIG.read_text()) if DEFAULT_CONFIG.is_file() else {}
    if arg.lower() in ("default", "none", "{}"):
        return {}
    p = Path(arg)
    return json.loads(p.read_text() if p.is_file() else arg)


def to_rows(out: pd.DataFrame) -> pd.DataFrame:
    ev = out["evidence"]
    return pd.DataFrame({
        "timestamp": out["timestamp"],
        "value": out["value"],
        "score": out["score"],
        "label": out["label"],
        "is_event_start": out["is_event_start"],
        "evidence_type": [e.get("type") if e else "" for e in ev],
        "evidence_text": [e.get("text") if e else "" for e in ev],
        "evidence": [json.dumps(e, default=str) if e else "" for e in ev],
    })


def predict(inputs: Sequence[FileInfo], config: dict, out_dir: Path, combined: bool = False,
            windows: Optional[Dict] = None, verbose: bool = True) -> Dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    factory = make_factory("sentinel", config)
    events, combined_parts, file_scores = [], [], []
    n_points, secs = 0, 0.0
    t_start = time.perf_counter()
    for info in inputs:
        df = load_series(info.path)
        out, s = run_file(factory, df, info.key)
        n_points += len(df)
        secs += s
        rows = to_rows(out)
        dest = out_dir / info.category / info.name
        dest.parent.mkdir(parents=True, exist_ok=True)
        rows.to_csv(dest, index=False)
        if combined:
            combined_parts.append(rows.assign(file=info.key)[["file", *rows.columns]])
        for _, r in out[out["is_event_start"]].iterrows():
            e = r["evidence"] or {}
            events.append({"file": info.key, "timestamp": r["timestamp"], "value": r["value"], "score": r["score"],
                           "type": e.get("type", ""), "top_detector": e.get("top_detector", ""),
                           "text": e.get("text", "")})
        if windows is not None and info.key in windows:
            file_scores.append(scorer.score_run(out, windows[info.key], info.key, info.category))
        if verbose:
            n_ev = int(out["is_event_start"].sum())
            print(f"  {info.key:60s} {len(df):7d} pts  {n_ev:3d} events  {1e6 * s / max(len(df), 1):5.1f} µs/pt")

    pd.DataFrame(events, columns=["file", "timestamp", "value", "score", "type", "top_detector", "text"]) \
        .to_csv(out_dir / "events.csv", index=False)
    if combined and combined_parts:
        pd.concat(combined_parts, ignore_index=True).to_csv(out_dir / "predictions.csv", index=False)
    summary = {
        "created": datetime.now().isoformat(timespec="seconds"),
        "config": config, "files": len(inputs), "points": n_points, "events": len(events),
        "runtime": {"update_seconds": secs, "us_per_point": 1e6 * secs / n_points if n_points else None,
                    "wall_seconds": time.perf_counter() - t_start},
    }
    if file_scores:
        summary["metrics"] = scorer.aggregate(file_scores)["overall"]
        summary["metrics_files_scored"] = len(file_scores)
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=1, default=str))
    return summary


def main(argv: Optional[Sequence[str]] = None) -> Dict:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--input", required=True, help="CSV file or folder of CSVs (any nesting)")
    ap.add_argument("--out", default=str(DEFAULT_OUT))
    ap.add_argument("--config", help="Sentinel config JSON file / inline JSON; default results/best_config.json; "
                                     "'default' = engine defaults")
    ap.add_argument("--windows", help="NAB-format windows JSON to score against (optional); 'auto' = look in --input")
    ap.add_argument("--combined", action="store_true", help="also write one predictions.csv with every point")
    ap.add_argument("-q", "--quiet", action="store_true")
    args = ap.parse_args(argv)

    inputs = find_inputs(Path(args.input))
    if not inputs:
        raise SystemExit(f"no CSV files found under {args.input}")
    windows = None
    if args.windows:
        wp = windows_path(Path(args.input)) if args.windows == "auto" else Path(args.windows)
        windows = load_windows(wp) if wp else None
    config = load_config(args.config)
    out_dir = Path(args.out)
    print(f"{len(inputs)} files -> {out_dir}  (config: {json.dumps(config) if config else 'engine defaults'})")
    s = predict(inputs, config, out_dir, args.combined, windows, verbose=not args.quiet)
    print(f"{s['points']:,} points, {s['events']} events, {s['runtime']['us_per_point']:.1f} µs/pt")
    if "metrics" in s:
        m = s["metrics"]
        f = lambda v: "—" if v is None else f"{v:.3f}"  # noqa: E731
        print(f"event-F1={f(m['f1'])}  P={f(m['precision'])}  R={f(m['recall'])}  "
              f"(TP {m['tp']}, FP {m['fp']}, FN {m['fn']})")
    return s


if __name__ == "__main__":
    main()
