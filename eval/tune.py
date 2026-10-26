"""Global hyperparameter search with leave-one-category-out (LOCO) validation.

For every config in the grid the whole corpus is run once (parallel over configs) and per-file
event counts are kept. For each held-out category the config with the best micro event-F1 on the
other 6 categories is selected and evaluated on the held-out one. Reported:
  - held-out F1 per category and their mean (categories with windows only)
  - pooled held-out micro F1 (TP/FP/FN summed over all held-out evaluations)
The final proposal is the best config on all 7 categories.

    python -m eval.tune [--grid small|default] [--jobs 8] [--tag name]

Writes results/runs/<ts>_tune_<tag>.json and results/tuning.md.
"""
from __future__ import annotations

import argparse
import itertools
import json
import os
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime
from typing import Dict, List, Optional, Sequence

from eval import scorer
from eval.data import iter_corpus
from eval.run import REPO_ROOT, RUNS_DIR, _git_sha, make_factory, run_file, sentinel_source

GRIDS = {
    "small": {"fusion": [("max", 1), ("vote", 2)], "spot_q": [1e-4, 1e-5], "cooldown": [300]},
    "default": {
        "detectors": [
            ["RobustZ", "LastW", "LaggedResidual", "VarRatio", "Seasonal", "CUSUM"],
            ["RobustZ", "LastW", "LaggedResidual", "VarRatio", "Seasonal"],
            ["LastW", "LaggedResidual", "Seasonal", "VarRatio"],
        ],
        "fusion": [("max", 1), ("vote", 2), ("vote", 3)],
        "spot_q": [1e-4, 1e-5, 1e-6, 1e-7],
        "cooldown": [100, 300, 600],
        "merge_gap": [10, 50],
        "lag_h": [32, 64, 128],
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
    for info, df, wins in _corpus(categories):
        res, _ = run_file(factory, df, info.key)
        fs = scorer.score_run(res, wins, info.key, info.category)
        out.append(fs.__dict__)
    return out


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
    per_cfg = [[scorer.FileScore(**d) for d in r] for r in raw]

    full = [scorer.metrics(fss) for fss in per_cfg]
    order = sorted(range(len(cfgs)), key=lambda i: -(full[i]["f1"] or 0))
    best = order[0]
    lo = loco(cfgs, per_cfg)
    now = datetime.now()
    record = {
        "kind": "tune", "created": now.isoformat(timespec="seconds"), "tag": args.tag, "detector": args.detector,
        "git_sha": _git_sha(), "sentinel_source": sentinel_source() if args.detector == "sentinel" else None,
        "grid": GRIDS[args.grid], "base": base, "n_configs": len(cfgs),
        "best_config": cfgs[best], "best_full_corpus": full[best], "loco": lo,
        "top10": [{"config": cfgs[i], **{k: full[i][k] for k in ("f1", "precision", "recall", "tp", "fp", "fn")}}
                  for i in order[:10]],
    }
    RUNS_DIR.mkdir(parents=True, exist_ok=True)
    path = RUNS_DIR / f"{now:%Y%m%d-%H%M%S}_tune_{args.tag}.json"
    path.write_text(json.dumps(record, indent=1))
    (REPO_ROOT / "results" / "tuning.md").write_text(render_md(record))

    print(f"best (all categories): F1={full[best]['f1']:.3f}  {cfgs[best]}")
    print(f"LOCO mean held-out F1={lo['mean_held_out_f1']:.3f}  pooled held-out F1={lo['pooled_held_out']['f1']:.3f}")
    for c, v in lo["per_category"].items():
        print(f"  {c:24s} held-out F1={v['held_out_f1'] if v['held_out_f1'] is not None else float('nan'):.3f}  "
              f"(train {v['train_f1']:.3f})  fp={v['fp']}")
    print("saved", path.relative_to(REPO_ROOT), "and results/tuning.md")
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
             f"- Best config on all categories (in-sample F1 {_f(r['best_full_corpus']['f1'])}): `{json.dumps(r['best_config'])}`",
             "", "| Held-out category | Selected config (tuned on the other 6) | Train F1 | Held-out F1 | TP | FP | FN |",
             "|---|---|---|---|---|---|---|"]
    for c, v in lo["per_category"].items():
        lines.append(f"| {c} | `{json.dumps(v['config'])}` | {_f(v['train_f1'])} | **{_f(v['held_out_f1'])}** | "
                     f"{v['tp']} | {v['fp']} | {v['fn']} |")
    lines += ["", "## Top 10 configs (full corpus, in-sample)", "", "| F1 | P | R | TP | FP | FN | Config |",
              "|---|---|---|---|---|---|---|"]
    for t in r["top10"]:
        lines.append(f"| {_f(t['f1'])} | {_f(t['precision'])} | {_f(t['recall'])} | {t['tp']} | {t['fp']} | {t['fn']} | "
                     f"`{json.dumps(t['config'])}` |")
    lines.append("")
    return "\n".join(lines)


if __name__ == "__main__":
    main()
