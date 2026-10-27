"""Ablation: baseline -> single detector -> +SPOT -> +events -> each detector alone / removed -> fusion.

    python -m eval.ablation [--tag name] [--categories ...]

Variants use sentinel/config.py keys (detectors, use_spot, fusion, vote_k, debounce, events.*) layered on --base.
Writes results/runs/<ts>_ablation_<tag>.json, results/ablation.md and results/figures/ablation.png.
"""
from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

from eval.run import REPO_ROOT, RUNS_DIR, _git_sha, evaluate, make_factory, sentinel_source

ALL = ["robust_z", "last_w", "lagged_residual", "var_ratio", "seasonal", "cusum"]
NAMES = {"robust_z": "RobustZ", "last_w": "LastW", "lagged_residual": "LaggedResidual", "var_ratio": "VarRatio",
         "seasonal": "Seasonal", "cusum": "CUSUM", "isolation_forest": "IsolationForest"}
# Sentinel always runs its event manager; this makes it a plain rising-edge detector
NO_EVENTS = {"debounce": 1, "events.cooldown": 0, "events.merge_gap": 1}
FIG_DIR = REPO_ROOT / "results" / "figures"


# phase-2 state (before the recall mechanisms of ADR-010): used so the build-up runs step by step
P2 = {"spatial": {"enabled": False}, "events.cooldown_frac": 0.0, "extreme_ratio": 1.5, "debounce": 2,
      "spot.q_per_detector": {"LastW": 1e-4, "LaggedResidual": 1e-4},
      "lagged_residual": {"w": 16, "h": 64}}
SPATIAL = {"spatial": {"enabled": True, "tol": 0.05}}
PREVIOUS_DEFAULT = {"lagged_residual": {"w": 16, "h": 64}, **SPATIAL}


def configs() -> List[Tuple[str, str, str, dict]]:
    """(group, name, detector, config). Sentinel configs use sentinel/config.py keys (dotted = nested), layered
    on --base (default: engine defaults = final tuned config). Sentinel always includes RobustZ."""
    mx = {"fusion": "max", "vote_k": 1}
    z = {"detectors": ["robust_z"]}
    q2 = {"spot.q_per_detector": {}}  # single global SPOT risk (before ADR-009)
    out = [("baseline", "windowedGaussian (thr 1.0)", "windowed_gaussian", {"threshold": 1.0}),
           ("build-up", "RobustZ, static thr, no event logic", "sentinel",
            {**z, "use_spot": False, **mx, **P2, **NO_EVENTS}),
           ("build-up", "RobustZ + SPOT", "sentinel", {**z, "use_spot": True, **mx, **P2, **NO_EVENTS}),
           ("build-up", "RobustZ + SPOT + events", "sentinel", {**z, "use_spot": True, **mx, **P2}),
           ("build-up", "all 6 + SPOT + events (MAX)", "sentinel", {"detectors": ALL, **mx, **P2, **q2}),
           ("build-up", "all 6, vote k=2", "sentinel", {"detectors": ALL, "fusion": "vote", "vote_k": 2, **P2, **q2}),
           ("build-up", "per-detector SPOT risk: RobustZ + LaggedResidual (ADR-009)", "sentinel", {**P2}),
           ("build-up", "+ spatial-extreme flag", "sentinel", {**P2, **SPATIAL}),
           ("build-up", "+ debounce 3, extreme ratio 2.0", "sentinel", {**P2, **SPATIAL, "debounce": 3,
                                                                      "extreme_ratio": 2.0}),
           ("build-up", "+ cooldown 0.5 x probation", "sentinel", {**PREVIOUS_DEFAULT,
                                                                   "spot.q_per_detector": {"LaggedResidual": 1e-4}}),
           ("tuned", "+ LaggedResidual risk 1e-5 (previous default)", "sentinel", PREVIOUS_DEFAULT),
           ("tuned", "+ lag 8/256 and spatial margin 5.75% (current default)", "sentinel", {})]
    out += [("removed", "default − spatial flag", "sentinel", {"spatial": {"enabled": False}}),
            ("removed", "default − debounce 3 (→ 2)", "sentinel", {"debounce": 2}),
            ("removed", "default − extreme ratio 2.0 (→ 1.5)", "sentinel", {"extreme_ratio": 1.5}),
            ("removed", "default − proportional cooldown (→ 600 pts)", "sentinel", {"events.cooldown_frac": 0.0}),
            ("removed", "default − per-detector SPOT risk", "sentinel", {**q2})]
    # "alone" (phase-2 settings, MAX fusion): under voting a lone detector could only fire via the extreme bypass
    out += [("alone", f"RobustZ + {NAMES[d]} (MAX)" if d != "robust_z" else "RobustZ only (MAX)", "sentinel",
             {"detectors": [d], **mx, **P2}) for d in ALL]
    out += [("added", "default + Seasonal", "sentinel", {"detectors": ["robust_z", "lagged_residual", "seasonal"]}),
            ("added", "default + Seasonal (auto period, thr 40)", "sentinel",
             {"detectors": ["robust_z", "lagged_residual", "seasonal"], "seasonal": {"auto_period": True},
              "detector_thresholds": {"Seasonal": 40.0}}),
            ("added", "default + VarRatio two-sided (flatlines, thr x2)", "sentinel",
             {"detectors": ["robust_z", "lagged_residual", "var_ratio"], "var_ratio": {"two_sided": True},
              "detector_thresholds": {"VarRatio": 3.0}}),
            ("added", "default + IsolationForest (D7)", "sentinel",
             {"detectors": ["robust_z", "lagged_residual", "isolation_forest"]})]
    return out


def run_ablation(categories: Optional[Sequence[str]] = None, base: Optional[dict] = None,
                 verbose: bool = True) -> List[Dict]:
    """base: config every sentinel variant starts from (e.g. the tuned config); variants override it."""
    rows = []
    for group, name, det, cfg in configs():
        if det == "sentinel":
            cfg = {**(base or {}), **cfg}
        r = evaluate(make_factory(det, cfg), categories=categories)
        o = r["overall"]
        row = {"group": group, "name": name, "detector": det, "config": cfg,
               "f1": o["f1"], "precision": o["precision"], "recall": o["recall"],
               "tp": o["tp"], "fp": o["fp"], "fn": o["fn"],
               "delay_points_median": o["delay_points_median"], "delay_fraction_median": o["delay_fraction_median"],
               "nab": r["nab"]["standard"]["score"], "nab_events": r["nab_events"]["score"],
               "us_per_point": r["runtime"]["us_per_point"],
               "per_category_f1": {c: m["f1"] for c, m in r["per_category"].items()}}
        rows.append(row)
        if verbose:
            print(f"{group:9s} {name:34s} F1={row['f1']:.3f} P={row['precision'] or 0:.3f} R={row['recall']:.3f} "
                  f"FP={row['fp']:4d} NAB(starts)={row['nab_events']:.1f}")
    return rows


def _f(v, nd=3):
    return "—" if v is None else f"{v:.{nd}f}"


def render_md(rows: List[Dict], meta: Dict) -> str:
    lines = ["# Ablation", "",
             f"Generated by `python -m eval.ablation` ({meta['created']}, sha {meta['git_sha']}, "
             f"sentinel: {meta['sentinel_source']}). Full corpus, event metrics per AGENTS.md §5. "
             f"Base config for all Sentinel variants: `{json.dumps(meta.get('base') or {})}`.", "",
             "| Group | Variant | Event-F1 | P | R | TP | FP | FN | Delay med (frac) | NAB starts | µs/pt |",
             "|---|---|---|---|---|---|---|---|---|---|---|"]
    for r in rows:
        lines.append(f"| {r['group']} | {r['name']} | **{_f(r['f1'])}** | {_f(r['precision'])} | {_f(r['recall'])} | "
                     f"{r['tp']} | {r['fp']} | {r['fn']} | {_f(r['delay_fraction_median'], 2)} | "
                     f"{_f(r['nab_events'], 1)} | {_f(r['us_per_point'], 1)} |")
    lines.append("")
    return "\n".join(lines)


def plot(rows: List[Dict], path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    colors = {"baseline": "#9aa0a6", "build-up": "#1a73e8", "fusion": "#7e57c2", "tuned": "#d93025",
              "alone": "#12a4af", "removed": "#e8710a", "added": "#34a853"}
    fig, ax = plt.subplots(figsize=(9, 0.32 * len(rows) + 1.2))
    names = [r["name"] for r in rows][::-1]
    vals = [r["f1"] or 0 for r in rows][::-1]
    cols = [colors.get(r["group"], "#555") for r in rows][::-1]
    bars = ax.barh(names, vals, color=cols)
    for b, v in zip(bars, vals):
        ax.text(v + 0.005, b.get_y() + b.get_height() / 2, f"{v:.3f}", va="center", fontsize=8)
    ax.set_xlabel("Event-F1 (full NAB corpus)")
    ax.set_xlim(0, max(vals) * 1.15 if vals else 1)
    ax.spines[["top", "right"]].set_visible(False)
    handles = [plt.Rectangle((0, 0), 1, 1, color=c) for c in colors.values()]
    ax.legend(handles, list(colors), loc="lower right", fontsize=8, frameon=False)
    ax.set_title("Sentinel ablation")
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main(argv: Optional[Sequence[str]] = None) -> List[Dict]:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tag", default="default")
    ap.add_argument("--categories", nargs="+")
    ap.add_argument("--base", help="JSON config (or path) all sentinel variants start from, e.g. tuned config")
    args = ap.parse_args(argv)
    base = {}
    if args.base:
        p = Path(args.base)
        base = json.loads(p.read_text() if p.is_file() else args.base)
    rows = run_ablation(args.categories, base)
    now = datetime.now()
    meta = {"kind": "ablation", "created": now.isoformat(timespec="seconds"), "tag": args.tag,
            "git_sha": _git_sha(), "sentinel_source": sentinel_source(), "categories": args.categories, "base": base}
    RUNS_DIR.mkdir(parents=True, exist_ok=True)
    (RUNS_DIR / f"{now:%Y%m%d-%H%M%S}_ablation_{args.tag}.json").write_text(json.dumps({**meta, "rows": rows}, indent=1))
    (REPO_ROOT / "results" / "ablation.md").write_text(render_md(rows, meta))
    plot(rows, FIG_DIR / "ablation.png")
    print("wrote results/ablation.md, results/figures/ablation.png")
    return rows


if __name__ == "__main__":
    main()
