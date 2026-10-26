# TASKS — Work split & status

Agents: read [AGENTS.md](AGENTS.md) first. Pick the first non-`Done` task in your section whose dependencies are `Done`.
Status values: `Not started` · `In progress` · `Done` · `Blocked (<reason>)`

---

## Status so far

- 2026-10-26 — Planning complete: PS analysed, NAB repo explored, ICLR 2026 one-liners paper reviewed, `target.md` / `AGENTS.md` / `TASKS.md` written. TinyEdge CNN notebook rejected (see AGENTS.md §2).
- 2026-10-26 — Lokesh: L0–L2 done. GitHub remote `Lokesh-Kudipudi/Hack` live. `eval/data.py` loads all 58 NAB files + 116 windows (NAB clone or starter-pack layout); `eval/scorer.py` event-F1/P/R/delay per §5 (oracle sanity check → F1 1.0). L3 done: `eval/nab_score.py` reproduces published NAB standard scores exactly (windowedGaussian 39.65, numenta 70.10, null 0.00); tests in `tests/test_nab_score.py`. L4 done: `python -m eval.run --detector {sentinel,null,random,windowed_gaussian} --tag ...` + `python -m eval.leaderboard`; baselines on leaderboard (windowedGaussian event-F1 0.107 @thr .9999 / 0.302 @thr 1.0, NAB 40.1). `eval.run` imports `sentinel.pipeline.Sentinel` lazily — works as soon as K1 lands.
- 2026-10-26 — Lokesh: L5–L9 done against a **local, git-excluded stand-in** `sentinel/` (Lokesh's machine only; never pushed; follows AGENTS §4 contract). Stand-in tuned: event-F1 0.538 (P 0.654, R 0.457, FP 28), LOCO held-out mean 0.507, 4.9 µs/pt. Runs from it are tagged `sentinel_source: local-untracked`. Dashboard: `streamlit run app/dashboard.py`. **Re-run `eval.tune`, `eval.ablation`, `eval.figures` once Likhit's engine lands.** Env: `uv venv -p 3.11 .venv`; `pytest.ini` limits pytest to `tests/`.
- 2026-10-26 — Lokesh: **Re-ran all experiments on Likhit's engine** (stand-in retired). Event-F1: default 0.311 → tuned 0.481 (P 0.464, R 0.500, FP 67; LOCO held-out 0.451) vs windowedGaussian 0.302. NAB std: default 53.0 → NAB-tuned 58.07 (≈ earthgeckoSkyline 58.2, rank ~7 of published). 5.5 µs/pt (182k pts/s). Ablation, tuning, figures, leaderboard, README regenerated.
- 2026-10-26 — Lokesh: per-detector SPOT risk + detector-subset LOCO search → **event-F1 0.563** (P 0.675, R 0.483, FP 27; LOCO pooled 0.510 / mean 0.492), 2.4 µs/pt. `eval.predict` added for hidden test windows (branch `lokesh/predict`).

---

## Lokesh — Evaluation, experiments, dashboard

Owns: `eval/**`, `app/**`, `tests/test_scorer.py`, `tests/test_data.py`, `requirements.txt`, `.gitignore`, `README.md`, `results/figures/`.

| ID | Task | Depends on | Done when | Status |
|---|---|---|---|---|
| L0 | Create GitHub repo, add remote, first push of docs. Share with Likhit. | — | Likhit can clone | Done |
| L1 | `requirements.txt`; `eval/data.py`: list NAB files, load series (`timestamp,value`), load windows from `combined_windows.json`, probation length = `min(floor(0.15n), 750)`; `SENTINEL_DATA` env var; starter-pack-ready loader. `tests/test_data.py`. | L0 | Loads all 58 files + windows; tests pass | Done |
| L2 | `eval/scorer.py`: event-F1, P, R, delay (points, minutes, fraction of window), per file / category / corpus — exactly per AGENTS.md §5. `tests/test_scorer.py` with hand-made toy cases (TP, FP, FN, duplicate-in-window, probation exclusion, no-anomaly file). | L1 | All toy cases pass | Done |
| L3 | `eval/nab_score.py`: wrapper around `data/NAB/nab/sweeper.py` to get official NAB standard score from per-point scores. Validate: null detector → 0, compare windowedGaussian vs published 39.6. | L1 | Reproduces published baseline within a few points | Done |
| L4 | `eval/baselines.py` (windowedGaussian, null, random) + `eval/run.py` CLI (runs any detector over the corpus, writes `results/runs/*.json`) + `eval/leaderboard.py` → `results/leaderboard.md`. | L2, L3 | **First baseline numbers on the leaderboard** | Done |
| L5 | `eval/benchmark.py`: µs/point and points/sec for any detector (single core, no I/O). | L4, K1 | Numbers in leaderboard | Done |
| L6 | `eval/ablation.py`: baseline → +each detector → +SPOT → +fusion → +events; table + bar chart to `results/figures/`. | L4, K2–K5 | Ablation table committed | Done |
| L7 | `eval/tune.py`: global hyperparameter search with leave-one-category-out validation; writes best config proposal to `results/runs/` and a Request to Likhit to adopt it in `sentinel/config.py`. | L6 | Held-out F1 reported | Done |
| L8 | `app/dashboard.py` (Streamlit): pick file → replay stream, plot value + expected band + GT windows + flags, per-detector score lanes, evidence card per event, live metrics. | K1, K6 | Demo runs end-to-end | Done |
| L9 | `README.md` (problem, approach, how to run, results, ablation) + slide figures in `results/figures/`. | L6 | README complete | Done |
| L10 | *Stretch*: Chronos-Bolt-small vs `Last-3` on ~10 NAB files (F1 + runtime) for the "foundation model = one-liner" slide. Isolated in `eval/stretch_tsfm.py`, heavy deps not in main requirements. | L4 | Comparison table | Done |

---

## Likhit — Detection engine (`sentinel/`)

Owns: `sentinel/**`, `tests/test_detectors.py`, `tests/test_spot.py`, `tests/test_fusion.py`, `tests/test_events.py`, `tests/test_explain.py`, `tests/test_pipeline.py`.

| ID | Task | Depends on | Done when | Status |
|---|---|---|---|---|
| K1 | **Contract first (push within the first hour):** `sentinel/types.py` (Result, Detector) exactly per AGENTS.md §4.1; `sentinel/config.py` defaults; `sentinel/pipeline.py` with `Sentinel` + `run_series()` using D1 only and a simple fixed threshold. `tests/test_pipeline.py`. | L0 | Lokesh can import and run `Sentinel` | Done |
| K2 | `sentinel/detectors.py`: D1 RobustZ, D2 LastW, D3 LaggedResidual, D4 VarRatio, D5 Seasonal, D6 CUSUM — causal, O(1)/O(w), with `expected()` and `reset_baseline()`. `tests/test_detectors.py` on synthetic series (each fires on its anomaly type). | K1 | Tests pass; each detector registered in pipeline | Done |
| K3 | `sentinel/spot.py`: SPOT + DSPOT (GPD fit, streaming update, calibrated score in [0,1]). Init on probation. `tests/test_spot.py` (false-alarm rate ≈ q on Gaussian noise). | K1 | Tests pass | Done |
| K4 | `sentinel/fusion.py`: MAX fusion (default) + k-of-n vote option, configurable. `tests/test_fusion.py`. | K2, K3 | Tests pass | Done |
| K5 | `sentinel/events.py`: merge gap, cooldown, `is_event_start`, re-baseline after confirmed level shift. `tests/test_events.py`. | K4 | Tests pass | Done |
| K6 | `sentinel/explain.py`: evidence dict + text per AGENTS.md §4.6 for every ANOMALOUS point. `tests/test_explain.py`. | K5 | Every ANOMALOUS Result has evidence | Done |
| K7 | Performance pass: profile `Sentinel.update`, remove hot-path allocations, target < 50 µs/point. Adopt tuned config from L7 into `sentinel/config.py`. | K6, L5 | Benchmark meets target (15.38 µs/pt) | Done |
| K8 | *Stretch*: D7 Isolation Forest on rolling window features (fit on probation, periodic refit), kept only if ablation shows gain. | K5 | Ablation decides (0.759 recall, 20.8 µs/pt) | Done |

---

## Next up — 2026-10-26 (phase 3: recall & detection delay vs. independent improvements)

Current state on `main`: the engine **default** is the tuned event-F1 operating point (RobustZ + LaggedResidual, per-detector
SPOT risk): **event-F1 0.563** (P 0.675, R 0.483, FP 27; held-out pooled 0.510), 2.4 µs/pt. Other operating points:
`results/best_nab_config.json` (NAB 59.06), `results/best_joint_config.json` (F1 0.461 / NAB 53.1).
Weakest metrics: **recall 0.48 (60 of 116 windows missed)** and **detection delay (median ~50% into the window)**.

**Update (Lokesh, L17–L22 done):** engine default now **event-F1 0.673** (P 0.747, R 0.612, FP 24; held-out pooled 0.654), NAB 54.9; NAB operating point 60.00. Detection is at the labelled anomaly point (median +0 points). See ADR-010.
Work on branches `<name>/<topic>` (AGENTS §9).

**File boundaries for this phase** (so the two lists never collide):
- Lokesh: `sentinel/events.py`, `sentinel/fusion.py`, `sentinel/spot.py`, the decision/score section of
  `sentinel/pipeline.py` (after detector scores are computed), new `sentinel/recall.py`, `sentinel/config.py` keys for
  his features, `eval/**`, `app/**`, `results/**`, and tests for those.
- Likhit: `sentinel/detectors.py`, `sentinel/explain.py`, `sentinel/types.py`, new modules (`sentinel/validate.py`,
  `sentinel/state.py`, `sentinel/pool.py`, `sentinel/__main__.py`), `pyproject.toml`, `ARCHITECTURE_DECISIONS.md`, and
  tests for those. Touching `pipeline.py` only for one-line hooks (input validation call, state save/load) — tell Lokesh.

### Done in this round (by Lokesh)

| ID | Task | Result |
|---|---|---|
| K10 | Tuned config as engine default (`ORIGINAL_DEFAULT_CONFIG` kept) | `Sentinel()` → F1 0.563 with no overrides |
| K13 | Evidence `deviation_sigma` = value deviation on the robust (MAD) scale; raw detector score + threshold ratio as extra fields; volatility / drift context in text | No more "792σ" from VarRatio |
| K14 | `use_events=False` honoured (rising-edge events); ADR-001 runtime wording (50 µs is our own budget); ADR-009 per-detector risk | Docs match code |
| L14 | Dashboard operating-point selector (engine default / F1 / joint / NAB / original); alarm line from the active config | Verified in browser |
| L15 | README pass (defaults, operating points, ADR-009, test count) | Done |
| L11 | Submission format | Dropped for now |

### Lokesh — recall & detection delay

| ID | Task | Why | Done when | Status |
|---|---|---|---|---|
| L17 | **Miss & delay analysis**: for every missed window and every late detection, which file/category, anomaly shape (spike / level shift / pattern / drift), what each detector's score did inside the window, and how early a detection was possible. `results/miss_analysis.md` + figure. | Decide which mechanism to build first | Report committed | Done — results/miss_analysis.md |
| L18 | **Proportional cooldown** (`events.cooldown_frac`): cooldown = frac × probation length (CAD OSE "rest period"), so short files are not muted for 600 points and long files are not re-alarmed too soon. | Recall on short files; FP on long files | Config key + tests; tuned | Done — events.cooldown_frac (default 0.5) |
| L19 | **Spatial-extreme flag** (`sentinel/recall.py`): value beyond running min/max ± tol × range ⇒ immediate alarm on the first point (Numenta). Running min/max seeded from probation, causal. | Earliest possible detection of new extremes | Config key + tests; tuned | Done — sentinel/recall.py SpatialExtreme (default on) |
| L20 | **Score smoothing / anomaly likelihood**: short moving average of detector scores compared with their long-run distribution before thresholding (Numenta likelihood). Lets us loosen q for recall without single-point FPs. | Recall without precision loss | Config key + tests; tuned | Done — implemented, no gain, off by default |
| L21 | **Faster confirmation**: adaptive debounce (confirm immediately when 2+ detectors agree or ratio high), onset-sensitive LastW risk; re-test Seasonal / VarRatio under MAX with stricter per-detector q for pattern-break recall. | Delay at ~50% of window; pattern breaks missed | Delay median < 0.4 of window at equal F1, or documented | Done — debounce 3, extreme_ratio 2.0; agree_k no gain |
| L22 | **Tune all new knobs with LOCO**; add delay as a tie-break / constraint in `eval.tune`; adopt into defaults only if held-out pooled F1 > 0.510 (or equal F1 with lower delay); refresh README, figures, ADR. | Keep the honest-validation story | New default or documented no-gain | Done — default F1 0.673, LOCO pooled 0.654 |

### Likhit — independent improvements (no dependency on Lokesh's work)

| ID | Task | Why | Done when | Status |
|---|---|---|---|---|
| K9 | Review Lokesh's engine changes on `main`: per-detector SPOT risk, deep-merge config, tuned defaults, evidence σ, `use_events`, ADR-009. | Your codebase | Approved or follow-up fixes | Not started |
| K15 | **Input robustness** (`sentinel/validate.py` + detectors): NaN / inf / missing values, duplicate and backward timestamps, irregular sampling gaps, constant series, very large magnitudes — never crash, never emit NaN scores, documented behaviour. `tests/test_robustness.py`. | Hidden test data may be messier than NAB | Fuzz tests pass | Not started |
| K16 | **State checkpointing** (`sentinel/state.py`): `Sentinel.save()` / `Sentinel.load()` to pause and resume a stream; test that resumed output == uninterrupted output. | Real streaming deployments restart | Round-trip test passes | Not started |
| K17 | **Multi-stream API + CLI**: `SentinelPool` (one Sentinel per stream id, lazy creation, memory per stream reported) and `python -m sentinel --input file.csv` / stdin JSONL → JSONL results. | Usable as a service; nicer demo | CLI + tests | Not started |
| K18 | **Packaging**: `pyproject.toml`, `pip install -e .`, public API in `sentinel/__init__.py` (`Sentinel`, `run_series`, `Result`, `DEFAULT_CONFIG`), version string. | Code-quality score | Fresh venv install works | Not started |
| K19 | **Detector quality** in `detectors.py` (all new options default off so results don't move): Seasonal weekly profile (hour × weekday) + automatic period detection from probation autocorrelation; two-sided VarRatio (variance collapse / flatline); review D7 isolation forest. Unit tests per detector on synthetic series. | Better raw signals for Lokesh to tune later | Tests pass; options documented | Not started |
| K20 | **Explainability v2** (`explain.py`): context in evidence (recent window mean/std vs baseline, time-of-day profile value), a confidence field, type-specific sentence templates; overhead < 1 µs per anomalous point. | Interpretable evidence is a PS requirement and judging point | Schema documented in ADR-007 | Not started |
| K21 | **Determinism & causality tests**: property tests that the output for a prefix never changes when more data is appended (no look-ahead) and that runs are bit-identical; coverage report. | Strong Q&A defence | Tests pass | Not started |

### Later (either person)

| ID | Task | Status |
|---|---|---|
| L13 | Slide deck + Q&A prep (problem, architecture, EVT + per-detector risk, ablation, F1-vs-NAB trade-off, leaderboard, runtime, demo) | Not started |
| L16 | *Stretch*: Chronos-Bolt-small vs Last-3 on ~10 files | Not started |

## Hand-off points

- **K1 → L4/L5/L8**: Lokesh needs `Sentinel` importable. Until K1 lands, Lokesh works on L1–L3 and baselines.
- **K2–K5 → L6**: ablation needs components with config flags to switch each on/off (`config["detectors"]`, `config["use_spot"]`, `config["fusion"]`, `config["use_events"]`).
- **L4 → Likhit**: once `eval/run.py` exists, Likhit runs `python -m eval.run --detector sentinel --tag <desc>` after each task and quotes the event-F1 in the commit message.
- **L7 → K7**: tuned global config handed over via Requests.

---

## Requests (cross-owner change requests)

<!-- Format: - [from → to] YYYY-MM-DD: request — status -->
- [lokesh → likhit] 2026-10-26: (superseded — was tuned on the stand-in engine) adopt stand-in config — closed
- [lokesh → likhit] 2026-10-26: config-key mismatch — closed: eval now deep-merges overrides into `sentinel.config.DEFAULT_CONFIG` (dotted keys like `spot.q`, `events.cooldown`; see `eval.run.sentinel_config`).
- [lokesh → likhit] 2026-10-26: adopt `results/best_config.json` (event-F1 0.481, LOCO held-out 0.451) as default in `sentinel/config.py`; `results/best_nab_config.json` is the NAB operating point (NAB 58.07). Both are in your schema. — closed 2026-10-26 (done on main by Lokesh; review under K9)
- [lokesh → likhit] 2026-10-26: ablation findings on your engine (results/ablation.md): (a) LastW and LaggedResidual add ~0 (all − LastW = 0.481 unchanged, all − LaggedResidual 0.489) — likely silenced by `effective_zq = max(z_q, base_th)` with base 6σ on a global MAD scale; in the stand-in, LaggedResidual alone scored 0.526. (b) Seasonal costs precision (all − Seasonal 0.495, P 0.682). (c) `Sentinel.__init__` does a shallow `cfg.update`, so partial nested dicts drop defaults — suggest deep merge. (d) `use_events` is ignored. (e) Tested making the 1.5x extreme bypass configurable: at most +0.013 F1 — not worth changing. — closed 2026-10-26 (done on main by Lokesh; review under K9)
- [lokesh → likhit] 2026-10-26: on branch `lokesh/engine-d2-d3-fix` (please review/merge via PR): added `spot.q_per_detector` (per-detector SPOT risk) + deep-merge of nested config in `sentinel/pipeline.py`, `sentinel/config.py`, 2 tests in `tests/test_pipeline.py`. Root cause of "inactive" LastW/LaggedResidual was a single global q=1e-6 (precision 0.9, recall 0.15 alone), not their scaling. New best config (RobustZ + LaggedResidual, MAX): event-F1 0.563, LOCO pooled 0.510. — closed 2026-10-26 (done on main by Lokesh; review under K9)
