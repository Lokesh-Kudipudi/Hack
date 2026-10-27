"""Compare first-alert delay with NAB's stored detector predictions.

Run: python -m eval.delay_compare
Delay is conditional on a detected ground-truth window. Published models use
their NAB standard thresholds; Sentinel uses its saved current-default run.
"""
from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

from eval.data import REPO_ROOT
from eval.leaderboard import load_runs
from eval.nab_compare import NAMES, published_event_metrics


def comparison_rows() -> list[dict]:
    candidates = [run for run in load_runs() if run.get("detector") == "sentinel"
                  and run.get("tag") == "engine default (final)"
                  and not run.get("categories") and "overall" in run]
    if not candidates:
        raise FileNotFoundError("No saved current-default Sentinel run")
    current = candidates[-1]
    models = [("Sentinel current default", current["overall"], "saved current-default run")]
    models += [(NAMES.get(name, name), metrics, "NAB standard threshold")
               for name, metrics in published_event_metrics().items()]
    rows = []
    for name, metric, source in models:
        rows.append({"model": name, "source": source, "f1": metric["f1"],
                     "windows_found": metric["tp"],
                     "median_delay_points": metric["delay_points_median"],
                     "median_window_fraction": metric["delay_fraction_median"],
                     "mean_window_fraction": metric["delay_fraction_mean"]})
    return sorted(rows, key=lambda row: -(row["f1"] or 0))


def report(rows: list[dict]) -> str:
    lines = ["# Detection delay comparison", "",
             "Full NAB corpus: 58 files and 116 labelled anomaly windows. Delay is measured from "
             "the start of a labelled window to the first predicted event start inside it. "
             "Medians and means use **detected windows only**; a missed window has no delay value. "
             "The window-position percentage makes files with different sampling rates and window widths "
             "easier to compare. Lower delay can still come with lower recall, so read it beside "
             "the number of windows found.", "",
             "Published models use NAB's stored per-point scores at its standard-profile threshold, "
             "with each new threshold crossing counted as an event. Sentinel uses the latest saved "
             "full-corpus run tagged `engine default (final)`. "
             "These event rules differ between systems. NAB windows usually centre the marked anomaly, "
             "which is why most medians are near 50% of the window width.", "",
             "| Model | Event-F1 | Windows found / 116 | Median delay (readings) | Median position | Mean position |",
             "|---|---:|---:|---:|---:|---:|"]
    for row in rows:
        points = row["median_delay_points"]
        median = row["median_window_fraction"]
        mean = row["mean_window_fraction"]
        lines.append(f"| {row['model']} | {row['f1']:.3f} | {row['windows_found']} | "
                     f"{'—' if points is None else f'{points:.1f}'} | "
                     f"{'—' if median is None else f'{median * 100:.1f}%'} | "
                     f"{'—' if mean is None else f'{mean * 100:.1f}%'} |")
    lines.append("")
    return "\n".join(lines)


def main() -> Path:
    rows = comparison_rows()
    path = REPO_ROOT / "results" / "delay_comparison.md"
    path.write_text(report(rows))
    run_path = REPO_ROOT / "results" / "runs" / f"{datetime.now():%Y%m%d-%H%M%S}_delay_comparison.json"
    run_path.write_text(json.dumps({"kind": "delay_comparison", "created": datetime.now().isoformat(),
                                    "protocol": "AGENTS.md §5; NAB standard thresholds for published models",
                                    "rows": rows}, indent=1))
    print(f"Compared {len(rows)} models; wrote {path.relative_to(REPO_ROOT)} and {run_path.relative_to(REPO_ROOT)}")
    return path


if __name__ == "__main__":
    main()
