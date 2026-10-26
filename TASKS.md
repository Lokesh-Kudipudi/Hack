# TASKS — Work split & status

Agents: read [AGENTS.md](AGENTS.md) first. Pick the first non-`Done` task in your section whose dependencies are `Done`.
Status values: `Not started` · `In progress` · `Done` · `Blocked (<reason>)`

---

## Status so far

- 2026-10-26 — Planning complete: PS analysed, NAB repo explored, ICLR 2026 one-liners paper reviewed, `target.md` / `AGENTS.md` / `TASKS.md` written. TinyEdge CNN notebook rejected (see AGENTS.md §2).
- **No code implemented yet.** Local git repo initialised; GitHub remote not yet added.

---

## Lokesh — Evaluation, experiments, dashboard

Owns: `eval/**`, `app/**`, `tests/test_scorer.py`, `tests/test_data.py`, `requirements.txt`, `.gitignore`, `README.md`, `results/figures/`.

| ID | Task | Depends on | Done when | Status |
|---|---|---|---|---|
| L0 | Create GitHub repo, add remote, first push of docs. Share with Likhit. | — | Likhit can clone | Not started |
| L1 | `requirements.txt`; `eval/data.py`: list NAB files, load series (`timestamp,value`), load windows from `combined_windows.json`, probation length = `min(floor(0.15n), 750)`; `SENTINEL_DATA` env var; starter-pack-ready loader. `tests/test_data.py`. | L0 | Loads all 58 files + windows; tests pass | Not started |
| L2 | `eval/scorer.py`: event-F1, P, R, delay (points, minutes, fraction of window), per file / category / corpus — exactly per AGENTS.md §5. `tests/test_scorer.py` with hand-made toy cases (TP, FP, FN, duplicate-in-window, probation exclusion, no-anomaly file). | L1 | All toy cases pass | Not started |
| L3 | `eval/nab_score.py`: wrapper around `data/NAB/nab/sweeper.py` to get official NAB standard score from per-point scores. Validate: null detector → 0, compare windowedGaussian vs published 39.6. | L1 | Reproduces published baseline within a few points | Not started |
| L4 | `eval/baselines.py` (windowedGaussian, null, random) + `eval/run.py` CLI (runs any detector over the corpus, writes `results/runs/*.json`) + `eval/leaderboard.py` → `results/leaderboard.md`. | L2, L3 | **First baseline numbers on the leaderboard** | Not started |
| L5 | `eval/benchmark.py`: µs/point and points/sec for any detector (single core, no I/O). | L4, K1 | Numbers in leaderboard | Not started |
| L6 | `eval/ablation.py`: baseline → +each detector → +SPOT → +fusion → +events; table + bar chart to `results/figures/`. | L4, K2–K5 | Ablation table committed | Not started |
| L7 | `eval/tune.py`: global hyperparameter search with leave-one-category-out validation; writes best config proposal to `results/runs/` and a Request to Likhit to adopt it in `sentinel/config.py`. | L6 | Held-out F1 reported | Not started |
| L8 | `app/dashboard.py` (Streamlit): pick file → replay stream, plot value + expected band + GT windows + flags, per-detector score lanes, evidence card per event, live metrics. | K1, K6 | Demo runs end-to-end | Not started |
| L9 | `README.md` (problem, approach, how to run, results, ablation) + slide figures in `results/figures/`. | L6 | README complete | Not started |
| L10 | *Stretch*: Chronos-Bolt-small vs `Last-3` on ~10 NAB files (F1 + runtime) for the "foundation model = one-liner" slide. Isolated in `eval/stretch_tsfm.py`, heavy deps not in main requirements. | L4 | Comparison table | Not started |

---

## Likhit — Detection engine (`sentinel/`)

Owns: `sentinel/**`, `tests/test_detectors.py`, `tests/test_spot.py`, `tests/test_fusion.py`, `tests/test_events.py`, `tests/test_explain.py`, `tests/test_pipeline.py`.

| ID | Task | Depends on | Done when | Status |
|---|---|---|---|---|
| K1 | **Contract first (push within the first hour):** `sentinel/types.py` (Result, Detector) exactly per AGENTS.md §4.1; `sentinel/config.py` defaults; `sentinel/pipeline.py` with `Sentinel` + `run_series()` using D1 only and a simple fixed threshold. `tests/test_pipeline.py`. | L0 | Lokesh can import and run `Sentinel` | Not started |
| K2 | `sentinel/detectors.py`: D1 RobustZ, D2 LastW, D3 LaggedResidual, D4 VarRatio, D5 Seasonal, D6 CUSUM — causal, O(1)/O(w), with `expected()` and `reset_baseline()`. `tests/test_detectors.py` on synthetic series (each fires on its anomaly type). | K1 | Tests pass; each detector registered in pipeline | Not started |
| K3 | `sentinel/spot.py`: SPOT + DSPOT (GPD fit, streaming update, calibrated score in [0,1]). Init on probation. `tests/test_spot.py` (false-alarm rate ≈ q on Gaussian noise). | K1 | Tests pass | Not started |
| K4 | `sentinel/fusion.py`: MAX fusion (default) + k-of-n vote option, configurable. `tests/test_fusion.py`. | K2, K3 | Tests pass | Not started |
| K5 | `sentinel/events.py`: merge gap, cooldown, `is_event_start`, re-baseline after confirmed level shift. `tests/test_events.py`. | K4 | Tests pass | Not started |
| K6 | `sentinel/explain.py`: evidence dict + text per AGENTS.md §4.6 for every ANOMALOUS point. `tests/test_explain.py`. | K5 | Every ANOMALOUS Result has evidence | Not started |
| K7 | Performance pass: profile `Sentinel.update`, remove hot-path allocations, target < 50 µs/point. Adopt tuned config from L7 into `sentinel/config.py`. | K6, L5 | Benchmark meets target | Not started |
| K8 | *Stretch*: D7 Isolation Forest on rolling window features (fit on probation, periodic refit), kept only if ablation shows gain. | K5 | Ablation decides | Not started |

---

## Hand-off points

- **K1 → L4/L5/L8**: Lokesh needs `Sentinel` importable. Until K1 lands, Lokesh works on L1–L3 and baselines.
- **K2–K5 → L6**: ablation needs components with config flags to switch each on/off (`config["detectors"]`, `config["use_spot"]`, `config["fusion"]`, `config["use_events"]`).
- **L4 → Likhit**: once `eval/run.py` exists, Likhit runs `python -m eval.run --detector sentinel --tag <desc>` after each task and quotes the event-F1 in the commit message.
- **L7 → K7**: tuned global config handed over via Requests.

---

## Requests (cross-owner change requests)

<!-- Format: - [from → to] YYYY-MM-DD: request — status -->
- (none)
