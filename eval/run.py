"""Run a detector over the corpus, score it, and save results/runs/<ts>_<detector>_<tag>.json.

    python -m eval.run --detector sentinel --tag "d1 only"
    python -m eval.run --detector windowed_gaussian --tag baseline
    python -m eval.run --detector sentinel --config cfg.json --categories realTweets --no-save
"""
from __future__ import annotations

import argparse
import json
import re
import subprocess
import time
from datetime import datetime
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence

import pandas as pd

from eval import nab_score, scorer
from eval.baselines import BASELINES
from eval.data import REPO_ROOT, data_root, iter_corpus, load_windows, windows_path

RUNS_DIR = REPO_ROOT / "results" / "runs"
DETECTORS = ["sentinel", *BASELINES]

# factory(n_total, key, config) -> object with update(t, x) -> Result-like
Factory = Callable[[int, str, dict], object]


def sentinel_config(overrides: Optional[dict] = None) -> dict:
    """Full Sentinel config: sentinel.config.DEFAULT_CONFIG deep-merged with overrides.

    Sentinel itself does a shallow update, so a partial nested dict (e.g. {"spot": {"q": 1e-5}})
    would drop the other defaults; here nested dicts merge. Dotted keys ("spot.q") set nested values.
    Eval-only key "thr_scale" multiplies every entry of detector_thresholds.
    """
    import copy
    from sentinel.config import DEFAULT_CONFIG

    cfg = copy.deepcopy(DEFAULT_CONFIG)

    def merge(dst: dict, src: dict) -> None:
        for k, v in src.items():
            if isinstance(v, dict) and v and isinstance(dst.get(k), dict):  # {} replaces
                merge(dst[k], v)
            else:
                dst[k] = copy.deepcopy(v)

    overrides = dict(overrides or {})
    scale = overrides.pop("thr_scale", None)
    for k, v in overrides.items():
        if "." in k:
            *path, leaf = k.split(".")
            d = cfg
            for p in path:
                d = d.setdefault(p, {})
            d[leaf] = v
        else:
            merge(cfg, {k: v})
    if scale is not None:
        cfg["detector_thresholds"] = {k: v * scale for k, v in cfg["detector_thresholds"].items()}
        cfg["fixed_threshold"] = cfg.get("fixed_threshold", 6.0) * scale
    return cfg


def make_factory(detector: str, config: Optional[dict] = None) -> Factory:
    config = dict(config or {})
    if detector == "sentinel":
        from sentinel.pipeline import Sentinel  # lazy: eval works before sentinel/ exists
        full = sentinel_config(config)
        return lambda n, key, cfg=full: Sentinel(n_total=n, config=cfg)
    if detector in BASELINES:
        cls = BASELINES[detector]
        return lambda n, key, cfg=config: cls(n_total=n, key=key, **cfg)
    raise ValueError(f"unknown detector {detector!r}; choose from {DETECTORS}")


def run_file(factory: Factory, df: pd.DataFrame, key: str = "") -> tuple[pd.DataFrame, float]:
    """Stream one series through a fresh detector. Returns (results df, seconds in update())."""
    det = factory(len(df), key)
    ts, xs = df["timestamp"].tolist(), df["value"].tolist()
    update = det.update
    t0 = time.perf_counter()
    res = [update(t, x) for t, x in zip(ts, xs)]
    elapsed = time.perf_counter() - t0
    out = pd.DataFrame({
        "timestamp": df["timestamp"].values,
        "value": xs,
        "score": [r.score for r in res],
        "label": [r.label for r in res],
        "is_event_start": [bool(r.is_event_start) for r in res],
        "evidence": [r.evidence for r in res],
    })
    return out, elapsed


def evaluate(factory: Factory, root: Optional[Path] = None, categories: Optional[Sequence[str]] = None,
             verbose: bool = False) -> Dict:
    root = Path(root) if root else data_root()
    windows = load_windows(windows_path(root))
    file_scores: List[scorer.FileScore] = []
    nab_runs: List[nab_score.Run] = []
    nab_start_runs: List[nab_score.Run] = []
    total_s, total_n, events = 0.0, 0, 0
    for info, df, wins in iter_corpus(root, categories, windows):
        out, secs = run_file(factory, df, info.key)
        total_s += secs
        total_n += len(df)
        events += int(out["is_event_start"].sum())
        fs = scorer.score_run(out, wins, info.key, info.category)
        file_scores.append(fs)
        nab_runs.append((info.key, out["timestamp"], out["score"].clip(0, 1)))
        nab_start_runs.append((info.key, out["timestamp"], out["is_event_start"].astype(float)))
        if verbose:
            print(f"  {info.key:60s} tp={fs.tp} fp={fs.fp} fn={fs.fn}  {1e6 * secs / len(df):6.1f} µs/pt")
    result = scorer.aggregate(file_scores)
    result["nab"] = {p: {k: r[k] for k in ("score", "threshold")}
                     for p, r in nab_score.nab_score_all_profiles(nab_runs, windows).items()}
    # NAB on the detector's actual decisions (score 1 at event starts, 0 elsewhere): penalises
    # each extra event, not every point of a sustained alarm
    result["nab_events"] = {k: v for k, v in nab_score.nab_score(nab_start_runs, windows, threshold=0.5).items()
                            if k in ("score", "raw")}
    result["runtime"] = {
        "points": total_n,
        "update_seconds": total_s,
        "us_per_point": 1e6 * total_s / total_n if total_n else None,
        "points_per_sec": total_n / total_s if total_s else None,
    }
    result["events_predicted"] = events
    return result


def _git_sha() -> str:
    try:
        sha = subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], cwd=REPO_ROOT, text=True).strip()
        dirty = subprocess.call(["git", "diff", "--quiet", "HEAD", "--", "sentinel", "eval"], cwd=REPO_ROOT)
        return sha + ("-dirty" if dirty else "")
    except Exception:
        return "unknown"


def sentinel_source() -> str:
    """'tracked' when sentinel/ is in git, else 'local-untracked' (e.g. Lokesh's local stand-in)."""
    try:
        out = subprocess.check_output(["git", "ls-files", "sentinel"], cwd=REPO_ROOT, text=True)
        return "tracked" if out.strip() else "local-untracked"
    except Exception:
        return "unknown"


def save_run(detector: str, tag: str, config: dict, categories, result: Dict) -> Path:
    now = datetime.now()
    slug = re.sub(r"[^A-Za-z0-9]+", "-", tag).strip("-")[:40] or "run"
    RUNS_DIR.mkdir(parents=True, exist_ok=True)
    path = RUNS_DIR / f"{now:%Y%m%d-%H%M%S}_{detector}_{slug}.json"
    record = {
        "run_id": path.stem, "created": now.isoformat(timespec="seconds"),
        "detector": detector, "tag": tag, "config": config, "categories": categories,
        "git_sha": _git_sha(), **result,
        **({"sentinel_source": sentinel_source()} if detector == "sentinel" else {}),
    }
    path.write_text(json.dumps(record, indent=1, default=str))
    return path


def _fmt(v, nd=3):
    return "—" if v is None else f"{v:.{nd}f}"


def main(argv: Optional[Sequence[str]] = None) -> Dict:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--detector", required=True, choices=DETECTORS)
    ap.add_argument("--tag", default="")
    ap.add_argument("--config", help="JSON file or inline JSON passed to the detector")
    ap.add_argument("--categories", nargs="+")
    ap.add_argument("--data", help="corpus root (default: $SENTINEL_DATA / data/NAB / data)")
    ap.add_argument("--no-save", action="store_true")
    ap.add_argument("-v", "--verbose", action="store_true")
    args = ap.parse_args(argv)

    config = {}
    if args.config:
        p = Path(args.config)
        config = json.loads(p.read_text() if p.is_file() else args.config)

    result = evaluate(make_factory(args.detector, config), args.data, args.categories, args.verbose)
    o, rt = result["overall"], result["runtime"]
    print(f"{args.detector} [{args.tag}]  F1={_fmt(o['f1'])}  P={_fmt(o['precision'])}  R={_fmt(o['recall'])}  "
          f"TP={o['tp']} FP={o['fp']} FN={o['fn']}  delay_med={_fmt(o['delay_points_median'], 1)} pts  "
          f"NAB={_fmt(result['nab']['standard']['score'], 2)}  NAB(starts)={_fmt(result['nab_events']['score'], 2)}  {_fmt(rt['us_per_point'], 2)} µs/pt")
    for cat, m in result["per_category"].items():
        print(f"  {cat:24s} F1={_fmt(m['f1'])}  TP={m['tp']} FP={m['fp']} FN={m['fn']}")
    if not args.no_save:
        print("saved", save_run(args.detector, args.tag, config, args.categories, result).relative_to(REPO_ROOT))
    return result


if __name__ == "__main__":
    main()
