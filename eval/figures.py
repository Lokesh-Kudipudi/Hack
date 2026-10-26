"""Slide figures -> results/figures/.

    python -m eval.figures [--config tuned.json]

- f1_comparison.png     event-F1 / precision / recall: baselines vs Sentinel (latest full-corpus runs)
- per_category_f1.png   event-F1 per category: best baseline vs Sentinel
- runtime.png           µs per point from the latest benchmark
- example_<name>.png    series + GT windows + Sentinel event starts with evidence type
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from eval.data import REPO_ROOT, list_files, load_series, load_windows, probation_length, window_indices, windows_path  # noqa: E402
from eval.leaderboard import load_runs  # noqa: E402
from eval.run import make_factory, run_file  # noqa: E402

FIG = REPO_ROOT / "results" / "figures"
BLUE, GREY, ORANGE, RED, GREEN = "#1a73e8", "#9aa0a6", "#e8710a", "#d93025", "#34a853"
EXAMPLES = ["realKnownCause/nyc_taxi.csv", "realKnownCause/machine_temperature_system_failure.csv",
            "realAWSCloudwatch/ec2_cpu_utilization_825cc2.csv", "artificialWithAnomaly/art_daily_jumpsup.csv"]
TYPE_MARK = {"spike": "^", "level_shift": "s", "variance_change": "D", "pattern_break": "P", "sequence": "o"}


def _style(ax):
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", alpha=0.25)


def _latest(runs: List[Dict]) -> Dict[str, Dict]:
    """Latest full-corpus run per (detector, tag)."""
    out = {}
    for r in runs:
        if "overall" in r and not r.get("categories"):
            out[f"{r['detector']} [{r.get('tag', '')}]"] = r
    return out


def f1_comparison(runs: Dict[str, Dict], path: Path) -> None:
    items = sorted(runs.items(), key=lambda kv: kv[1]["overall"]["f1"] or 0)
    labels = [k.replace("windowed_gaussian", "windowedGaussian") for k, _ in items]
    fig, ax = plt.subplots(figsize=(9, 0.5 * len(items) + 1.5))
    y = range(len(items))
    for off, key, col, lab in ((-0.25, "f1", BLUE, "Event-F1"), (0, "precision", GREY, "Precision"),
                               (0.25, "recall", ORANGE, "Recall")):
        vals = [r["overall"][key] or 0 for _, r in items]
        ax.barh([i + off for i in y], vals, height=0.25, color=col, label=lab)
        if key == "f1":
            for i, v in zip(y, vals):
                ax.text(v + 0.01, i + off, f"{v:.3f}", va="center", fontsize=8, color=BLUE, fontweight="bold")
    ax.set_yticks(list(y), labels, fontsize=8)
    ax.set_xlim(0, 1)
    ax.legend(frameon=False, fontsize=8, loc="lower right")
    ax.set_title("Full NAB corpus (58 files, 116 windows)")
    _style(ax)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def per_category(base: Dict, ours: Dict, path: Path) -> None:
    cats = [c for c in ours["per_category"] if ours["per_category"][c]["tp"] + ours["per_category"][c]["fn"] > 0]
    fig, ax = plt.subplots(figsize=(9, 3.6))
    x = range(len(cats))
    for off, r, col, lab in ((-0.2, base, GREY, "windowedGaussian"), (0.2, ours, BLUE, "Sentinel")):
        vals = [r["per_category"][c]["f1"] or 0 for c in cats]
        ax.bar([i + off for i in x], vals, width=0.4, color=col, label=lab)
        for i, v in zip(x, vals):
            ax.text(i + off, v + 0.01, f"{v:.2f}", ha="center", fontsize=7)
    ax.set_xticks(list(x), [c.replace("real", "").replace("artificial", "art.") for c in cats], fontsize=8)
    ax.set_ylabel("Event-F1")
    ax.set_ylim(0, 1)
    ax.legend(frameon=False, fontsize=8)
    _style(ax)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def runtime(bench: Dict, path: Path) -> None:
    rows = sorted(bench["results"], key=lambda r: r["us_per_point_mean"])
    fig, ax = plt.subplots(figsize=(7, 2.6))
    names = [r["detector"].replace("windowed_gaussian", "windowedGaussian") for r in rows]
    vals = [r["us_per_point_mean"] for r in rows]
    ax.barh(names, vals, color=[BLUE if n == "sentinel" else GREY for n in names])
    for i, (v, r) in enumerate(zip(vals, rows)):
        ax.text(v * 1.02, i, f"{v:.1f} µs  ({r['points_per_sec'] / 1000:,.0f}k pts/s)", va="center", fontsize=8)
    ax.set_xlabel("µs per streaming update (mean, single core)")
    ax.set_xlim(0, max(vals) * 1.6)
    _style(ax)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def example(key: str, config: dict, windows: Dict, path: Path) -> None:
    paths = {f.key: f.path for f in list_files()}
    if key not in paths:
        print(f"skip example {key}: not in corpus")
        return
    df = load_series(paths[key])
    out, _ = run_file(make_factory("sentinel", config), df, key)
    fig, ax = plt.subplots(figsize=(11, 3.2))
    ts = out["timestamp"]
    p = probation_length(len(df))
    ax.axvspan(ts.iloc[0], ts.iloc[p], color=GREY, alpha=0.15, lw=0, label="probation")
    for n, (i, j) in enumerate(window_indices(ts, windows.get(key, []))):
        ax.axvspan(ts.iloc[i], ts.iloc[j], color=GREEN, alpha=0.18, lw=0, label="GT window" if n == 0 else None)
    ax.plot(ts, out["value"], color="#202124", lw=0.6)
    seen = set()
    for _, r in out[out["is_event_start"]].iterrows():
        t = (r["evidence"] or {}).get("type", "spike")
        ax.scatter(r["timestamp"], r["value"], marker=TYPE_MARK.get(t, "o"), s=60, color=RED, edgecolor="white",
                   zorder=5, label=t if t not in seen else None)
        seen.add(t)
    ax.set_title(key, fontsize=10)
    ax.legend(frameon=False, fontsize=7, ncol=6, loc="upper left")
    _style(ax)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main(argv: Optional[Sequence[str]] = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", help="Sentinel config JSON (or path) for example plots; default: latest tuned")
    args = ap.parse_args(argv)
    FIG.mkdir(parents=True, exist_ok=True)
    runs = load_runs()
    if args.config:
        p = Path(args.config)
        config = json.loads(p.read_text() if p.is_file() else args.config)
    else:
        tunes = [r for r in runs if r.get("kind") == "tune"]
        config = tunes[-1]["best_config"] if tunes else {}

    latest = _latest(runs)
    if latest:
        f1_comparison(latest, FIG / "f1_comparison.png")
    sent = [r for r in latest.values() if r["detector"] == "sentinel"]
    wg = [r for r in latest.values() if r["detector"] == "windowed_gaussian"]
    if sent and wg:
        per_category(max(wg, key=lambda r: r["overall"]["f1"]), max(sent, key=lambda r: r["overall"]["f1"]),
                     FIG / "per_category_f1.png")
    benches = [r for r in runs if r.get("kind") == "benchmark"]
    if benches:
        runtime(benches[-1], FIG / "runtime.png")
    windows = load_windows(windows_path())
    for key in EXAMPLES:
        example(key, config, windows, FIG / f"example_{Path(key).stem}.png")
    print("wrote", ", ".join(sorted(p.name for p in FIG.glob("*.png"))))


if __name__ == "__main__":
    main()
