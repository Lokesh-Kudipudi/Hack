# AGENTS.md — Sentinel (Hackathon Challenge 2)

This is the single source of truth for any AI agent (Claude Code, Codex, etc.) working in this repo.
Read it fully before writing code. Requirements checklist: [target.md](docs/target.md). Work split & status: [TASKS.md](TASKS.md).

---

## 0. Session start protocol (DO THIS FIRST)

The user opens a chat and states their name: **Lokesh** or **Likhit**.

1. **Before any change, switch to a branch named after the person working** (see §9): `git checkout main && git pull --rebase && git checkout -b <name>/<short-topic>` (e.g. `lokesh/l10-stretch`, `likhit/k9-detector-fix`). Never edit or commit on `main` directly.
2. Read `AGENTS.md` (this file), `docs/target.md`, `TASKS.md`, and the tail of your own log `logs/<name>.md`.
3. In `TASKS.md`, find that person's section. Pick the **first task that is not `Done`** whose dependencies are `Done`.
   - If it is blocked on the other person, say so and pick the next unblocked task (or a stretch task).
4. Mark it `In progress` in `TASKS.md`, implement it, write tests, run them.
5. When finished: mark `Done` in `TASKS.md`, append a session entry to `logs/<name>.md`, commit on your branch, push the branch and open a PR into `main` (see §9).
6. Continue to the next task unless the user says stop.

Rules:
- Only edit files **owned by that person** (see §8). Shared files: `TASKS.md` (edit only your own section + the "Status so far" line about your work), `logs/<your name>.md`, `results/runs/`.
- If you need a change in the other person's file, do NOT edit it — add a note under "Requests" in `TASKS.md` for them.
- Never commit data (see §10).

---

## 1. Problem (short)

Build a **lightweight, streaming** anomaly detector for univariate time series (NAB benchmark subset, hidden test windows).
For every point/flagged event output:
- `score` — anomaly score in [0, 1]
- `label` — `NORMAL` / `ANOMALOUS`
- `evidence` — interpretable explanation of why it was flagged

Scored on: **Event-based F1** (primary), Precision, Recall, Detection Delay, Inference Runtime.

Judging: Technical 40% (hidden data score) · Approach & Innovation 20% (algorithmic intelligence, **threshold design**) · Efficiency 15% (runtime) · Presentation 15% · Code & Demo 10%.

---

## 2. Key decisions (and why) — do not reverse without discussing

| Decision | Why |
|---|---|
| **Unsupervised, streaming, statistical detectors** (NumPy only in the core) | Test windows are hidden → no labels at test time. Runtime is scored. NAB leaderboard is won by simple adaptive methods, not deep nets. |
| **No TensorFlow / PyTorch in `sentinel/`** | Runtime + "lightweight". DL only allowed in an optional stretch comparison under `eval/`. |
| **Ensemble of complementary one-liners + EVT thresholds** | ICLR 2026 paper ("When Foundation Models are One-Liners") shows time-series foundation models ≈ moving-variance / squared-difference baselines. We build those adaptively and cheaply. |
| **Short + long horizon residuals, fused with MAX** | Paper §6.1: short horizon catches point anomalies, long (lagged) horizon catches sequence anomalies, because the short context gets "contaminated" by the anomaly. |
| **No per-window z-normalization** | Paper §5.1: it makes scores ≈ window variance. Learn scale (median/MAD) in probation, update slowly. |
| **Event post-processing (merge + cooldown)** | NAB/event-F1 counts only the first detection per window; repeated flags only add FPs. |
| **Global hyperparameters only; never tune per file on labels** | Hidden test windows. Validate with leave-one-category-out. |
| **First 15% of each file (max 750 points) = probation** | Same as NAB. Learn baseline there; never flag during probation; not scored. |

Rejected: the TinyEdge supervised CNN notebook (label leakage, shuffled split, wrong labels file, non-streaming normalization, no evidence).

---

## 3. Data

- NAB repo: https://github.com/numenta/NAB — 58 CSVs (`timestamp,value`), ~365k rows, 7 categories:
  `artificialNoAnomaly(5)`, `artificialWithAnomaly(6)`, `realAWSCloudwatch(17)`, `realAdExchange(6)`, `realKnownCause(7)`, `realTraffic(7)`, `realTweets(10)`.
- Ground truth: **`labels/combined_windows.json`** (anomaly windows `[start, end]`). Do NOT use `combined_labels.json` (single points) for scoring.
- Organizer starter pack: put it in `data/starter/`. Loader must accept any folder of `timestamp,value` CSVs + a windows JSON in NAB format. Adapt the loader when the starter pack format is known.
- Location: `data/NAB/` (each person clones it themselves — see §10). Override with env var `SENTINEL_DATA`.

NAB reference leaderboard (standard profile): ARTime 74.9 · Numenta HTM 70.1 · contextOSE 69.9 · … · randomCutForest 51.7 · **windowedGaussian 39.6 (our baseline)** · random 16.8.

---

## 4. Architecture

```
(timestamp, value) stream
   │
   ▼
Sentinel.update(t, x)
   ├─ probation? → learn baseline stats, return NORMAL
   ├─ detectors D1..D6 → raw scores
   ├─ per-detector SPOT threshold → calibrated score in [0,1] + exceed flag
   ├─ fusion (MAX / k-of-n vote) → fused score
   ├─ event manager → label, event start, cooldown, re-baseline
   └─ explain → evidence dict (only for ANOMALOUS)
   │
   ▼
Result(timestamp, value, score, label, is_event_start, evidence)
```

### 4.1 Interface contract (`sentinel/types.py`) — both people code against this

```python
from dataclasses import dataclass, field
from typing import Optional
import pandas as pd

@dataclass
class Result:
    timestamp: pd.Timestamp
    value: float
    score: float                 # fused anomaly score in [0, 1]
    label: str                   # "NORMAL" | "ANOMALOUS"
    is_event_start: bool         # True only on the first point of a new anomaly event
    evidence: Optional[dict] = None   # filled when label == "ANOMALOUS"
    detector_scores: dict = field(default_factory=dict)  # name -> calibrated score in [0,1]

class Detector:                  # base class for D1..D6
    name: str
    def update(self, t: pd.Timestamp, x: float) -> float: ...   # raw score (higher = more anomalous), O(1)
    def expected(self) -> Optional[float]: ...                   # expected value (for evidence/plots), may be None
    def reset_baseline(self) -> None: ...                        # called by event manager after confirmed level shift

class Sentinel:
    def __init__(self, n_total: Optional[int] = None, config: Optional[dict] = None): ...
        # n_total lets probation = min(0.15*n_total, 750); if None, use config["probation"]
    def update(self, t: pd.Timestamp, x: float) -> Result: ...
```

Convenience used by eval: `run_series(df) -> pd.DataFrame` with columns `timestamp,value,score,label,is_event_start,evidence`.

Changing this contract requires a note in `TASKS.md → Requests` and agreement from both.

### 4.2 Detectors (`sentinel/detectors.py`)

| ID | Name | Raw score | Catches |
|---|---|---|---|
| D1 | `RobustZ` | \|x − median\| / (1.4826·MAD), stats from probation, slow EW update | spikes |
| D2 | `LastW` (short horizon) | (mean(last w) − x)², w≈3 | point anomalies (ICLR baseline) |
| D3 | `LaggedResidual` (long horizon) | (x − mean of window that ends h steps ago)², h≈64 | sequence anomalies (ICLR §6.1) |
| D4 | `VarRatio` | var(last w) / baseline var | volatility changes (ICLR Var-w) |
| D5 | `Seasonal` | residual vs expected value for (hour-of-day[, weekday]) bucket, EW-updated | broken daily/weekly patterns |
| D6 | `CUSUM` | two-sided CUSUM statistic on standardized residual | level shifts / drift |

Every detector must be O(1) or O(w) per point using `collections.deque` / running sums. No look-ahead (no `Centered-w`).

### 4.3 Thresholding (`sentinel/spot.py`)

SPOT / DSPOT (Siffer et al., KDD 2017): Peaks-Over-Threshold with a Generalized Pareto fit.
- Init on probation scores: initial threshold `t` = high quantile (e.g. 98%); fit GPD to excesses (Grimshaw or method of moments).
- Final threshold `z_q` for risk `q` (e.g. 1e-4). Streaming: excesses above `t` but below `z_q` update the fit; above `z_q` = alarm.
- Calibrated score in [0,1] = 1 − tail probability of the raw score under the fitted GPD.
- DSPOT variant (on residual from moving average) for drifting series.

### 4.4 Fusion (`sentinel/fusion.py`)
Default: fused score = MAX of calibrated detector scores; label ANOMALOUS when any detector exceeds its SPOT threshold. Option: k-of-n vote. Ablation decides.

### 4.5 Event manager (`sentinel/events.py`)
- Consecutive / close (gap ≤ `merge_gap`) ANOMALOUS points = one event; only first point gets `is_event_start=True`.
- `cooldown` points after an event start: no new event starts.
- If an event persists > `rebaseline_after` points with a stable new level (CUSUM-confirmed), call `reset_baseline()` on detectors → "new normal".

### 4.6 Evidence (`sentinel/explain.py`)
```json
{
  "type": "spike | level_shift | variance_change | pattern_break | sequence",
  "expected": 74.2, "observed": 91.8, "deviation_sigma": 4.1,
  "fired": ["RobustZ", "CUSUM"],
  "top_detector": "RobustZ",
  "text": "Value 91.8 is 4.1σ above expected 74.2 (Tue 14:00 profile); CUSUM confirms upward level shift."
}
```
Type comes from the top detector (D1/D2→spike, D3→sequence, D4→variance_change, D5→pattern_break, D6→level_shift).

---

## 5. Evaluation protocol (`eval/`) — fixed definitions

Per file, ignore the probation part (first `min(floor(0.15·n), 750)` points) for both GT windows and predictions.

- **GT events** = windows from `combined_windows.json`.
- **Predicted events** = points with `is_event_start=True`.
- **TP window** = GT window containing ≥1 predicted event start (inclusive bounds). Extra predictions inside the same window are ignored (not TP, not FP).
- **FP** = predicted event start outside every window.
- **FN** = GT window with no predicted event start.
- Corpus-level (micro): `P = TP / (TP + FP)`, `R = TP / (TP + FN)`, `F1 = 2PR/(P+R)`. Also report per-category and per-file.
- **Detection delay** (TP windows only): first predicted start − window start, in points and in minutes; also as fraction of window width. Report mean and median.
- **Runtime**: mean µs per `Sentinel.update` call over the whole corpus (excluding I/O), plus points/sec. Single core.
- **NAB score**: also compute the official NAB standard-profile score via `data/NAB/nab/sweeper.py` so we can compare with the published leaderboard. Needs per-point `score` in [0,1].

Tuning: global params only. Use leave-one-category-out: tune on 6 categories, evaluate on the 7th, rotate. Report the held-out mean.

---

## 6. Repo layout

```
hack/
├── AGENTS.md  CLAUDE.md  TASKS.md  README.md
├── docs/                    # project documentation, including target.md
├── requirements.txt  .gitignore
├── data/                    # GITIGNORED — local only
│   ├── NAB/                 # git clone of numenta/NAB
│   └── starter/             # organizer starter pack
├── sentinel/                # detection engine  (owner: Likhit)
│   ├── __init__.py  types.py  config.py
│   ├── detectors.py  spot.py  fusion.py  events.py  explain.py
│   └── pipeline.py          # Sentinel class + run_series()
├── eval/                    # evaluation & experiments  (owner: Lokesh)
│   ├── __init__.py  data.py  scorer.py  nab_score.py
│   ├── baselines.py         # windowedGaussian, null, random
│   ├── run.py               # CLI: run a detector over the corpus
│   ├── benchmark.py  ablation.py  tune.py  leaderboard.py
├── app/dashboard.py         # Streamlit replay  (owner: Lokesh)
├── tests/
│   ├── test_scorer.py test_data.py ...   (Lokesh)
│   └── test_detectors.py test_spot.py test_events.py test_pipeline.py ...  (Likhit)
├── results/
│   ├── runs/                # one JSON per run (committed, small)
│   ├── leaderboard.md       # generated by eval/leaderboard.py
│   └── figures/             # PNGs for slides
└── logs/
    ├── lokesh.md
    └── likhit.md
```

---

## 7. Environment

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
git clone --depth 1 https://github.com/numenta/NAB data/NAB   # once per machine
pytest -q
```
`requirements.txt`: numpy, pandas, scipy, scikit-learn, streamlit, plotly, matplotlib, pytest. Python ≥ 3.10.

Run a full evaluation (once `eval/run.py` exists):
```bash
python -m eval.run --detector sentinel --tag "<short description>"
python -m eval.run --detector windowed_gaussian --tag baseline
python -m eval.leaderboard
```

---

## 8. Ownership

| Area | Owner |
|---|---|
| `sentinel/**`, `tests/test_detectors.py`, `tests/test_spot.py`, `tests/test_fusion.py`, `tests/test_events.py`, `tests/test_explain.py`, `tests/test_pipeline.py` | **Likhit** |
| `eval/**`, `app/**`, `tests/test_scorer.py`, `tests/test_data.py`, `requirements.txt`, `.gitignore`, `README.md`, `results/figures/` | **Lokesh** |
| `AGENTS.md`, `docs/target.md` | Both — change only with agreement |
| `TASKS.md` | Each edits only their own section / their requests |
| `logs/<name>.md` | That person |
| `results/runs/` | Both (append-only, unique filenames) |

---

## 9. Git & logging rules

- **Branch first — always.** The first thing in every session, before editing any file, is to create (or reuse) a branch named after the person working: `<name>/<short-topic>`, where `<name>` is `lokesh` or `likhit` (e.g. `lokesh/l10-stretch`). Never commit directly on `main`.
- Start the branch from an up-to-date `main` (`git checkout main && git pull --rebase`). Before pushing, `git fetch origin && git rebase origin/main`.
- Commit after each finished task (or meaningful sub-step). Message format:
  `[<name>] <TASK-ID>: <what>` + key metric if any, e.g. `[likhit] K3: SPOT thresholds — event-F1 0.52 → 0.61`.
- Push your branch (`git push -u origin <name>/<short-topic>`) after each task and merge it into `main` via a pull request (the other person may review). Pushing your own branch is **authorized**. Never push directly to `main`, never force-push `main`.
- After every task append to `logs/<name>.md`:
  ```
  ## YYYY-MM-DD HH:MM — <TASK-ID> <title>
  - Changed: <files>
  - Result: <metrics / tests passed>
  - Notes / next: <...>
  ```
- Every evaluation run writes `results/runs/<YYYYmmdd-HHMMSS>_<detector>_<tag>.json` (config, git sha, metrics overall + per category, runtime). Commit these. Regenerate `results/leaderboard.md` with `python -m eval.leaderboard`.

## 10. Data rule — NEVER push data

- `data/` is gitignored. Also ignored: `*.csv` outside `results/`, `*.parquet`, `*.pkl`, model files, `.venv/`.
- Each person keeps their own local copy (`data/NAB`, `data/starter`).
- Before committing, check `git status` — no CSV/data files may be staged.

---

## 11. Coding conventions

- Python 3.10+, type hints, small functions, docstrings only where non-obvious.
- Core engine: NumPy + stdlib only (scipy allowed in `spot.py` if needed). No pandas inside the per-point hot path.
- No look-ahead anywhere in `sentinel/`. Every detector is causal.
- Determinism: fixed seeds where randomness exists.
- Tests with `pytest`; synthetic series for detector tests (spike, level shift, variance change, seasonal break) — assert the right detector fires.
- Target runtime: < 50 µs per point for full `Sentinel.update`.

## 12. References

- PS checklist: `docs/target.md`
- NAB: https://github.com/numenta/NAB (scorer: `nab/sweeper.py`, baseline: `nab/detectors/gaussian/windowedGaussian_detector.py`)
- Zhu, Carpentier, Verbeke — *When Foundation Models are One-Liners*, ICLR 2026 (TSFMs ≈ Var-w / Last-w; multi-horizon MAX ensemble; avoid per-window z-norm; visualize per series)
- Siffer et al. — *Anomaly Detection in Streams with Extreme Value Theory* (SPOT/DSPOT), KDD 2017
