"""Official NAB score from per-point anomaly scores, via data/NAB/nab/sweeper.py.

Mirrors NAB's runner: corpus-wide threshold optimisation (optimizer.py), then
normalisation 100 * (raw - null) / (perfect - null), where null is the constant
0.5 detector at its own optimal threshold and perfect = #windows * tpWeight.

Unlike our event scorer, NAB counts every point with score >= threshold as a
detection (FP points are penalised individually, a window counts once, earlier
detections in a window earn more).

CLI (validate against the published leaderboard):
    python -m eval.nab_score --nab-results windowedGaussian numenta null
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import pandas as pd

from eval.data import REPO_ROOT, Window, data_root, load_windows, windows_path

NAB_DIR = REPO_ROOT / "data" / "NAB"

# key, timestamps, per-point scores in [0, 1]
Run = Tuple[str, Sequence, Sequence[float]]


def _sweeper_module():
    if not (NAB_DIR / "nab" / "sweeper.py").is_file():
        raise FileNotFoundError(f"NAB not found at {NAB_DIR}; git clone https://github.com/numenta/NAB data/NAB")
    if str(NAB_DIR) not in sys.path:
        sys.path.insert(0, str(NAB_DIR))
    from nab import sweeper  # noqa: E402  (sweeper.py only needs the stdlib)
    return sweeper


def load_profiles() -> Dict[str, Dict[str, float]]:
    with open(NAB_DIR / "config" / "profiles.json") as f:
        return {k: v["CostMatrix"] for k, v in json.load(f).items()}


def _anomaly_rows(runs: Iterable[Run], windows: Dict[str, List[Window]], cost: Dict[str, float]) -> list:
    sw = _sweeper_module().Sweeper(probationPercent=0.15, costMatrix=cost)
    rows = []
    for key, ts, scores in runs:
        ts = list(pd.to_datetime(pd.Series(ts)))
        rows.extend(sw.calcSweepScore(ts, list(map(float, scores)), windows.get(key, []), key))
    return sw, rows


def _pick(by_threshold: list, threshold: Optional[float]):
    if threshold is None:
        return max(by_threshold, key=lambda r: r.score)
    # rows are in descending threshold order; row(t) = all points with score >= t
    chosen = by_threshold[0]
    for r in by_threshold:
        if r.threshold >= threshold:
            chosen = r
        else:
            break
    return chosen


def raw_score(runs: Iterable[Run], windows: Dict[str, List[Window]], profile: str = "standard",
              threshold: Optional[float] = None):
    """(ThresholdScore, n_windows). threshold=None optimises it over the corpus like NAB."""
    runs = list(runs)
    sw, rows = _anomaly_rows(runs, windows, load_profiles()[profile])
    n_windows = sum(len(windows.get(k, [])) for k, _, _ in runs)
    return _pick(sw.calcScoreByThreshold(rows), threshold), n_windows


def nab_score(runs: Iterable[Run], windows: Dict[str, List[Window]], profile: str = "standard",
              threshold: Optional[float] = None) -> Dict:
    runs = list(runs)
    best, n_windows = raw_score(runs, windows, profile, threshold)
    null, _ = raw_score([(k, ts, [0.5] * len(ts)) for k, ts, _ in runs], windows, profile)
    perfect = n_windows * load_profiles()[profile]["tpWeight"]
    return {
        "profile": profile,
        "score": 100 * (best.score - null.score) / (perfect - null.score),
        "threshold": best.threshold,
        "raw": best.score, "null_raw": null.score, "perfect": perfect,
        # point counts at the chosen threshold (NAB semantics, not event counts)
        "tp_points": best.tp, "fp_points": best.fp, "fn_points": best.fn, "tn_points": best.tn,
    }


def nab_score_all_profiles(runs: Iterable[Run], windows: Dict[str, List[Window]]) -> Dict[str, Dict]:
    runs = list(runs)
    return {p: nab_score(runs, windows, p) for p in load_profiles()}


def runs_from_nab_results(detector: str, keys: Iterable[str]) -> List[Run]:
    """Read NAB's published per-point scores: results/<det>/<cat>/<det>_<file>.csv."""
    out = []
    for key in keys:
        cat, name = key.split("/", 1)
        df = pd.read_csv(NAB_DIR / "results" / detector / cat / f"{detector}_{name}", parse_dates=["timestamp"])
        out.append((key, df["timestamp"], df["anomaly_score"]))
    return out


def main(argv: Optional[Sequence[str]] = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--nab-results", nargs="+", required=True, help="detector names under data/NAB/results/")
    ap.add_argument("--profile", default="standard")
    args = ap.parse_args(argv)

    windows = load_windows(windows_path(data_root()))
    with open(NAB_DIR / "results" / "final_results.json") as f:
        published = json.load(f)
    for det in args.nab_results:
        r = nab_score(runs_from_nab_results(det, sorted(windows)), windows, args.profile)
        pub = published.get(det, {}).get(args.profile)
        print(f"{det:20s} ours={r['score']:7.2f}  published={pub if pub is None else round(pub, 2)}  "
              f"threshold={r['threshold']:.4f}")


if __name__ == "__main__":
    main()
