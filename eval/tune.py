"""Global hyperparameter search with leave-one-category-out (LOCO) validation.

For every config in the grid the whole corpus is run once (parallel over configs) and per-file
event counts are kept. For each held-out category the config with the best micro event-F1 on the
other 6 categories is selected and evaluated on the held-out one. Reported:
  - held-out F1 per category and their mean (categories with windows only)
  - pooled held-out micro F1 (TP/FP/FN summed over all held-out evaluations)
The final proposal is the best config on all 7 categories.

    python -m eval.tune [--grid small|default] [--jobs 8] [--tag name]

Writes results/runs/<ts>_tune_<tag>.json, results/tuning.md and results/best_config.json (proposal for
sentinel/config.py).
"""
from __future__ import annotations

import argparse
import itertools
import json
import os
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime
from typing import Dict, List, Optional, Sequence

from eval import nab_score, scorer
from eval.data import iter_corpus
from eval.run import REPO_ROOT, RUNS_DIR, _git_sha, make_factory, run_file, sentinel_source

ALL6 = ["robust_z", "last_w", "lagged_residual", "var_ratio", "seasonal", "cusum"]

# Keys follow sentinel/config.py; dotted keys set nested values, "thr_scale" scales detector_thresholds
# (see eval.run.sentinel_config).
GRIDS = {
    "small": {"fusion": [("max", 1), ("vote", 2)], "spot.q": [1e-4, 1e-6], "events.cooldown": [200]},
    "default": {
        "detectors": [ALL6, [d for d in ALL6 if d != "var_ratio"], ALL6 + ["isolation_forest"]],
        "fusion": [("max", 1), ("vote", 2), ("vote", 3)],
        "spot.q": [1e-4, 1e-6],
        "thr_scale": [0.75, 1.0, 1.5],
        "debounce": [1, 2, 4],
        "events.cooldown": [200, 600],
        "events.merge_gap": [16, 64],
    },
    # per-detector SPOT risk + base thresholds for the short/long-horizon detectors (LastW, LaggedResidual)
    "perdet": {
        "spot.q_per_detector": [{}, {"LastW": 1e-4, "LaggedResidual": 1e-4}, {"LastW": 1e-3, "LaggedResidual": 1e-4},
                                {"LastW": 1e-4, "LaggedResidual": 1e-4, "VarRatio": 1e-7, "Seasonal": 1e-7}],
        "detector_thresholds": [{}, {"LastW": 2.0, "LaggedResidual": 2.0}, {"LastW": 4.0, "LaggedResidual": 4.0}],
        "fusion": [("max", 1), ("vote", 2), ("vote", 3)],
        "thr_scale": [1.0, 1.5],
        "debounce": [1, 2],
        "events.cooldown": [400, 600, 1000],
        "spot.q": [1e-6],
        "events.merge_gap": [64],
    },
}


def expand(grid: Dict[str, list]) -> List[dict]:
    keys = list(grid)
    out = []
    for combo in itertools.product(*(grid[k] for k in keys)):
        cfg = {}
        for k, v in zip(keys, combo):
            if k == "fusion":
                cfg["fusion"], cfg["vote_k"] = v
            else:
                cfg[k] = v
        out.append(cfg)
    return out


_CORPUS = None


def _corpus(categories):
    global _CORPUS
    if _CORPUS is None:
        _CORPUS = list(iter_corpus(categories=categories))
    return _CORPUS


def run_config(args) -> List[dict]:
    detector, cfg, categories = args
    factory = make_factory(detector, cfg)
    out = []
    runs, windows = [], {}
    for info, df, wins in _corpus(categories):
        res, _ = run_file(factory, df, info.key)
        fs = scorer.score_run(res, wins, info.key, info.category)
        out.append(fs.__dict__)
        runs.append((info.key, res["timestamp"], res["score"].clip(0, 1)))
        windows[info.key] = wins
    nab = nab_score.nab_score(runs, windows)["score"]
    return {"files": out, "nab": nab}


def _f1(fss: List[scorer.FileScore]) -> float:
    m = scorer.metrics(fss)
    return m["f1"] or 0.0


def loco(cfgs: List[dict], per_cfg: List[List[scorer.FileScore]]) -> Dict:
    cats = sorted({fs.category for fs in per_cfg[0]})
    held, pooled = {}, []
    for c in cats:
        train = [_f1([fs for fs in fss if fs.category != c]) for fss in per_cfg]
        best = max(range(len(cfgs)), key=lambda i: train[i])
        test = [fs for fs in per_cfg[best] if fs.category == c]
        pooled += test
        m = scorer.metrics(test)
        held[c] = {"config_index": best, "config": cfgs[best], "train_f1": train[best],
                   "held_out_f1": m["f1"], "tp": m["tp"], "fp": m["fp"], "fn": m["fn"]}
    with_windows = [v["held_out_f1"] for v in held.values() if v["tp"] + v["fn"] > 0]
    return {"per_category": held,
            "mean_held_out_f1": sum(with_windows) / len(with_windows) if with_windows else None,
            "pooled_held_out": scorer.metrics(pooled)}


def main(argv: Optional[Sequence[str]] = None) -> Dict:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--detector", default="sentinel")
    ap.add_argument("--grid", default="default", choices=list(GRIDS))
    ap.add_argument("--base", help="JSON config merged under every grid point")
    ap.add_argument("--jobs", type=int, default=max(1, (os.cpu_count() or 2) - 1))
    ap.add_argument("--categories", nargs="+")
    ap.add_argument("--tag", default="loco")
    args = ap.parse_args(argv)

    base = json.loads(args.base) if args.base else {}
    cfgs = [{**base, **c} for c in expand(GRIDS[args.grid])]
    print(f"{len(cfgs)} configs x corpus, {args.jobs} workers")
    with ProcessPoolExecutor(args.jobs) as ex:
        raw = list(ex.map(run_config, [(args.detector, c, args.categories) for c in cfgs]))
    per_cfg = [[scorer.FileScore(**d) for d in r["files"]] for r in raw]
    nabs = [r["nab"] for r in raw]

    full = [{**scorer.metrics(fss), "nab": nabs[i]} for i, fss in enumerate(per_cfg)]
    best_nab = max(range(len(cfgs)), key=lambda i: nabs[i])
    order = sorted(range(len(cfgs)), key=lambda i: -(full[i]["f1"] or 0))
    best = order[0]
    lo = loco(cfgs, per_cfg)
    now = datetime.now()
    record = {
        "kind": "tune", "created": now.isoformat(timespec="seconds"), "tag": args.tag, "detector": args.detector,
        "git_sha": _git_sha(), "sentinel_source": sentinel_source() if args.detector == "sentinel" else None,
        "grid": GRIDS[args.grid], "base": base, "n_configs": len(cfgs),
        "best_config": cfgs[best], "best_full_corpus": full[best], "loco": lo,
        "best_nab_config": cfgs[best_nab], "best_nab_full_corpus": full[best_nab],
        "top10": [{"config": cfgs[i], **{k: full[i][k] for k in ("f1", "precision", "recall", "tp", "fp", "fn", "nab")}}
                  for i in order[:10]],
    }
    RUNS_DIR.mkdir(parents=True, exist_ok=True)
    path = RUNS_DIR / f"{now:%Y%m%d-%H%M%S}_tune_{args.tag}.json"
    path.write_text(json.dumps(record, indent=1))
    (REPO_ROOT / "results" / "tuning.md").write_text(render_md(record))
    (REPO_ROOT / "results" / "best_config.json").write_text(json.dumps(cfgs[best], indent=1) + "\n")

    print(f"best (all categories): F1={full[best]['f1']:.3f} NAB={nabs[best]:.1f}  {cfgs[best]}")
    print(f"best NAB: NAB={nabs[best_nab]:.1f} F1={full[best_nab]['f1']:.3f}  {cfgs[best_nab]}")
    print(f"LOCO mean held-out F1={lo['mean_held_out_f1']:.3f}  pooled held-out F1={lo['pooled_held_out']['f1']:.3f}")
    for c, v in lo["per_category"].items():
        print(f"  {c:24s} held-out F1={v['held_out_f1'] if v['held_out_f1'] is not None else float('nan'):.3f}  "
              f"(train {v['train_f1']:.3f})  fp={v['fp']}")
    print("saved", path.relative_to(REPO_ROOT), "results/tuning.md, results/best_config.json")
    return record


def _f(v, nd=3):
    return "—" if v is None else f"{v:.{nd}f}"


def render_md(r: Dict) -> str:
    lo = r["loco"]
    lines = ["# Hyperparameter tuning (leave-one-category-out)", "",
             f"Generated by `python -m eval.tune` ({r['created']}, sha {r['git_sha']}, sentinel: {r['sentinel_source']}). "
             f"{r['n_configs']} global configs; grid: `{json.dumps(r['grid'])}`; base: `{json.dumps(r['base'])}`.", "",
             f"- **Mean held-out event-F1 (categories with anomalies): {_f(lo['mean_held_out_f1'])}**",
             f"- Pooled held-out event-F1: {_f(lo['pooled_held_out']['f1'])} "
             f"(TP {lo['pooled_held_out']['tp']}, FP {lo['pooled_held_out']['fp']}, FN {lo['pooled_held_out']['fn']})",
             f"- Best config on all categories (in-sample F1 {_f(r['best_full_corpus']['f1'])}, "
             f"NAB {_f(r['best_full_corpus'].get('nab'), 1)}): `{json.dumps(r['best_config'])}`",
             f"- Best NAB config (NAB {_f(r.get('best_nab_full_corpus', {}).get('nab'), 1)}, "
             f"F1 {_f(r.get('best_nab_full_corpus', {}).get('f1'))}): `{json.dumps(r.get('best_nab_config'))}`",
             "", "| Held-out category | Selected config (tuned on the other 6) | Train F1 | Held-out F1 | TP | FP | FN |",
             "|---|---|---|---|---|---|---|"]
    for c, v in lo["per_category"].items():
        lines.append(f"| {c} | `{json.dumps(v['config'])}` | {_f(v['train_f1'])} | **{_f(v['held_out_f1'])}** | "
                     f"{v['tp']} | {v['fp']} | {v['fn']} |")
    lines += ["", "## Top 10 configs (full corpus, in-sample)", "", "| F1 | P | R | TP | FP | FN | NAB | Config |",
              "|---|---|---|---|---|---|---|---|"]
    for t in r["top10"]:
        lines.append(f"| {_f(t['f1'])} | {_f(t['precision'])} | {_f(t['recall'])} | {t['tp']} | {t['fp']} | {t['fn']} | "
                     f"{_f(t.get('nab'), 1)} | "
                     f"`{json.dumps(t['config'])}` |")
    lines.append("")
    return "\n".join(lines)


if __name__ == "__main__":
    main()
