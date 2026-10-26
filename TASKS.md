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

## Next up — 2026-10-26 (after per-detector q + joint tuning)

Current best (global configs, full NAB corpus): **event-F1 0.563** (P 0.675, R 0.483; held-out pooled 0.510) with
`results/best_config.json`; **NAB 59.06** with `results/best_nab_config.json`; joint (F1 0.461, NAB 53.1) with
`results/best_joint_config.json`. 2.4–5.7 µs/pt. F1 and NAB trade off (see `results/figures/pareto_f1_nab.png`).
Work on branches `<name>/<topic>` (AGENTS §9).

### Lokesh

| ID | Task | Why | Done when | Status |
|---|---|---|---|---|
| L11 | Ask organisers for the hidden-test **submission format** (file layout, columns, one file vs many, labels format, is `n_total` known per stream); adapt `eval.predict` output to it. | 40% of the score is hidden data; we can't submit without it | `eval.predict` writes the exact required file(s) | Not started |
| L12 | Experiment harness for recall/delay ideas from NAB leaders: proportional cooldown, score smoothing, spatial-extreme flag — tune each via `eval.tune` with LOCO once Likhit exposes the config keys (K11–K12). | Recall 0.48 and detection at ~50% of the window are our weakest metrics | Held-out pooled F1 > 0.510 or documented as no gain | Not started |
| L13 | Slide deck: problem, architecture, threshold design (EVT + per-detector risk), ablation story (+0.16 from per-detector q), F1-vs-NAB trade-off, leaderboard, runtime, demo screenshots; Q&A prep sheet. | Presentation is 15% | Deck reviewed by both | Not started |
| L14 | Dashboard: re-verify on final engine; add an operating-point selector (F1 / joint / NAB config) and show the active config. | Demo is 10% | Demo run-through without errors | Not started |
| L15 | Final README / figures / leaderboard pass after K10 lands (badges, tables, ADR links). | Numbers must match the submitted engine | README numbers = latest runs | Not started |
| L16 | *Stretch* (old L10): Chronos-Bolt-small vs Last-3 on ~10 files (F1 + runtime) for the "foundation model = one-liner" slide. | Story, not score | Comparison table | Not started |

### Likhit

| ID | Task | Why | Done when | Status |
|---|---|---|---|---|
| K9 | Review Lokesh's engine change now on `main` (`spot.q_per_detector`, deep-merge config in `sentinel/pipeline.py`, 2 tests). | Your files; root cause of "inactive" LastW/LaggedResidual | Approved or follow-up fix | Not started |
| K10 | Make `results/best_config.json` the built-in default in `sentinel/config.py` (RobustZ + LaggedResidual, MAX, q 1e-6 / 1e-4 for LaggedResidual, debounce 2, cooldown 600, merge gap 64) so `Sentinel()` gives F1 0.563 with no overrides; update tests. | Hidden-test runs and judges' demo use defaults | `python -m eval.run --detector sentinel` → F1 0.563 | Not started |
| K11 | Proportional cooldown option (`events.cooldown_frac`: cooldown = frac × probation length or × n_total, as CAD OSE's rest period) + spatial-extreme flag (value beyond running min/max ± 5% of range ⇒ alarm, as Numenta) — both configurable, default off. | Earlier detection / recall on short vs long files | Config keys + tests; Lokesh tunes (L12) | Not started |
| K12 | Score smoothing before thresholding (short moving average of detector scores, Numenta anomaly-likelihood style), configurable. | Fewer single-point false alarms → room to loosen q for recall | Config key + tests | Not started |
| K13 | Evidence quality: `deviation_sigma` uses the top detector's raw score (e.g. "792σ" for VarRatio); compute σ-deviation vs RobustZ scale/expected value instead, and keep `text` readable. | Interpretable evidence is a PS requirement and a judging point | No absurd σ values on the corpus | Not started |
| K14 | Honour or remove ignored config keys (`use_events`); update ADR-001 wording (50 µs is our own budget, ≈ windowedGaussian cost, not an industry requirement); add an ADR for per-detector SPOT risk. | Correctness of docs/claims in Q&A | ADRs consistent with code and results | Not started |

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
- [lokesh → likhit] 2026-10-26: adopt `results/best_config.json` (event-F1 0.481, LOCO held-out 0.451) as default in `sentinel/config.py`; `results/best_nab_config.json` is the NAB operating point (NAB 58.07). Both are in your schema. — open
- [lokesh → likhit] 2026-10-26: ablation findings on your engine (results/ablation.md): (a) LastW and LaggedResidual add ~0 (all − LastW = 0.481 unchanged, all − LaggedResidual 0.489) — likely silenced by `effective_zq = max(z_q, base_th)` with base 6σ on a global MAD scale; in the stand-in, LaggedResidual alone scored 0.526. (b) Seasonal costs precision (all − Seasonal 0.495, P 0.682). (c) `Sentinel.__init__` does a shallow `cfg.update`, so partial nested dicts drop defaults — suggest deep merge. (d) `use_events` is ignored. (e) Tested making the 1.5x extreme bypass configurable: at most +0.013 F1 — not worth changing. — open
- [lokesh → likhit] 2026-10-26: on branch `lokesh/engine-d2-d3-fix` (please review/merge via PR): added `spot.q_per_detector` (per-detector SPOT risk) + deep-merge of nested config in `sentinel/pipeline.py`, `sentinel/config.py`, 2 tests in `tests/test_pipeline.py`. Root cause of "inactive" LastW/LaggedResidual was a single global q=1e-6 (precision 0.9, recall 0.15 alone), not their scaling. New best config (RobustZ + LaggedResidual, MAX): event-F1 0.563, LOCO pooled 0.510. — open
