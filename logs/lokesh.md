# Lokesh — session log

## 2026-10-26 — Planning
- Changed: target.md, AGENTS.md, TASKS.md, CLAUDE.md, .gitignore
- Result: PS analysed, NAB explored, ICLR 2026 one-liners paper reviewed, work split agreed.
- Notes / next: L0 — create GitHub repo and push.

## 2026-10-26 — L0 GitHub repo
- Remote `origin` = github.com/Lokesh-Kudipudi/Hack, docs already pushed. Done.

## 2026-10-26 — L1 Data loader
- Changed: requirements.txt, eval/__init__.py, eval/data.py, tests/test_data.py, pytest.ini
- Result: 58 files / 7 categories / 116 windows / 365,558 points load in ~0.1 s; works on NAB clone (`data/NAB`) and starter layout (`data/<cat>/<cat>/*.csv`, skips __MACOSX). 12 tests pass.
- Notes: Python 3.9 system → use `uv venv -p 3.11 .venv`. `SENTINEL_DATA` / `SENTINEL_WINDOWS` env overrides.

## 2026-10-26 — L2 Scorer
- Changed: eval/scorer.py, tests/test_scorer.py
- Result: 10 toy-case tests pass (TP, FP, FN, inclusive bounds, duplicates, probation clip/drop, no-anomaly file, micro aggregation). Oracle detector → F1 1.0 on full corpus.
- Notes: windows straddling probation are clipped; delay measured from clipped start. Next: L3 NAB score wrapper.

## 2026-10-26 — L3 NAB score wrapper
- Changed: eval/nab_score.py, tests/test_nab_score.py
- Result: reproduces published standard-profile scores to 0.01 from NAB's stored result CSVs — null 0.00, windowedGaussian 39.65, numenta 70.10. Corpus scoring ~1 s. 28 tests pass.
- Notes: mirrors NAB runner (corpus-wide threshold optimisation + null/perfect normalisation); also supports fixed threshold and all 3 profiles. CLI: `python -m eval.nab_score --nab-results <det>...`. Next: L4 baselines + run CLI + leaderboard.

## 2026-10-26 — L4 Baselines, run CLI, leaderboard
- Changed: eval/baselines.py, eval/run.py, eval/leaderboard.py, tests/test_run.py, eval/data.py (+test), results/runs/*.json, results/leaderboard.md
- Result (event-F1 / NAB std / µs per pt): null 0.000 / 0.0 / 0.5 · random 0.058 / 12.9 / 0.6 · windowedGaussian thr .9999 0.107 / 40.1 / 49 · windowedGaussian thr 1.0 0.302 / 40.1 / 49. 35 tests pass.
- Bug fixed: `load_series` used to sort by timestamp; machine_temperature has a backward time jump, so now file (stream) order is kept and `window_indices` is robust to non-monotonic time.
- Notes: our windowedGaussian port matches NAB's per-point scores (max diff 5e-13); NAB 40.13 vs published 39.65 is purely NAB's stored CSVs rounding scores to 12 digits (rounding ours gives 39.65 exactly). Random differs from published 16.8 only by RNG. Baselines use naive rising-edge events (no merge/cooldown) → huge FP counts; that's the gap Sentinel's events/SPOT should close. Next: L5 needs K1.

## 2026-10-26 — L5–L9 (against local stand-in engine)
- Context: K1–K6 not pushed yet; per user instruction wrote a stand-in `sentinel/` (+ tests/test_sentinel_local.py) excluded via `.git/info/exclude` — local only, never committed, so Likhit's files are untouched.
- Changed: eval/benchmark.py, eval/ablation.py, eval/tune.py, eval/figures.py, eval/run.py (NAB on event starts, sentinel_source), eval/leaderboard.py, app/dashboard.py, .streamlit/config.toml, README.md, results/*, tests/test_run.py
- Result: L5 sentinel 4.9 µs/pt (203k pts/s) vs windowedGaussian 49 µs. L7 648-config grid, LOCO mean held-out F1 0.507 (pooled 0.479), best in-sample 0.538 → results/best_config.json (vote k=2, q=1e-5, cooldown 600, merge 50, lag_h 32). L6: RobustZ static 0.185 → +SPOT 0.332 → +events 0.362; all-6 MAX 0.370; vote k=2 0.538; removing LaggedResidual hurts most. L8 dashboard verified in browser (replay, GT windows, events, evidence, live F1). 42 tests pass.
- Notes: per-point NAB score is 0 for Sentinel (sustained alarms keep score≈1 → NAB penalises every point); NAB on event starts 40–48. Requests to Likhit added in TASKS. Next: re-run tune/ablation/figures on Likhit's real engine; L10 stretch.

## 2026-10-26 — Re-run of all experiments on Likhit's engine
- Changed: eval/run.py (sentinel_config deep merge, dotted keys, thr_scale), eval/tune.py (grid on Likhit's schema, NAB per config, best_nab), eval/ablation.py (Likhit's schema), eval/figures.py (headline runs only, real engine only), app/dashboard.py (sigma fallback), README.md (numbers corrected), results/*
- Result: default 0.311 F1 / NAB 53.0 → tuned-for-F1 0.481 (LOCO held-out 0.451, pooled 0.412) and tuned-for-NAB 58.07 (F1 0.353, R 0.759). 5.5 µs/pt. 648-config grid. Extreme-bypass what-if: ≤ +0.013 F1. Ablation: RobustZ+VarRatio alone 0.488; LastW/LaggedResidual ~0 contribution on Likhit's engine. 70 tests pass.
- Notes: README previously showed stand-in numbers (0.538, 15.38 µs) as Likhit's — corrected. Requests to Likhit updated in TASKS. Stand-in removed (`.git/info/exclude` entries deleted).

## 2026-10-26 — Re-activate LastW/LaggedResidual (branch lokesh/engine-d2-d3-fix) + eval.predict (branch lokesh/predict)
- Diagnosis: not the 6σ base floor (SPOT z_q dominates in most files) and not scaling (adaptive-scale prototype 0.453). Alone (others masked) LastW/LaggedResidual had P≈0.9, R≈0.15 → global q=1e-6 too strict for clean detectors.
- Changed: sentinel/pipeline.py + sentinel/config.py (`spot.q_per_detector`, deep-merge config), tests/test_pipeline.py (+2), eval/tune.py (perdet, subsets grids), eval/ablation.py (alone rows MAX), eval/figures.py, README.md, results/*
- Result: perdet grid 0.502 (LOCO mean 0.491); subsets grid → RobustZ + LaggedResidual, MAX, q_LR=1e-4: **F1 0.563, P 0.675, R 0.483, FP 27**, LOCO pooled 0.510 / mean 0.492, 2.4 µs/pt. Per-detector q worth +0.16 in that config. NAB best stays 58.07 (results/best_nab_config.json).
- eval.predict: any CSV folder → per-point predictions + events.csv + summary.json; on starter pack reproduces F1 0.481 with the then-best config. Fixed windows lookup for roots with "..".
- Next: merge both branches via PR; L10/slides; try NAB-leader ideas (proportional cooldown, score smoothing, spatial extreme).

## 2026-10-26 — One config for both metrics? (branch lokesh/joint-config)
- Changed: eval/tune.py (`--objective f1|nab|joint`, `--nab-min`, joint grid, all configs stored), eval/figures.py (pareto_f1_nab.png), README.md (3 operating points, trade-off, table fix), TASKS.md (Next up: L11–L16, K9–K14), results/*
- Result: 576 configs. Joint (max F1 s.t. NAB ≥ 52): F1 0.461 / NAB 53.1 (held-out pooled 0.399). New NAB best 59.06 (> earthgeckoSkyline 58.2). F1 best unchanged 0.563 / NAB 42.8.
- Notes: no single config wins both — NAB rewards recall (FP weight 0.11), event-F1 counts every FP. Decision: submit the event-F1 config; quote NAB config for leaderboard.

## 2026-10-26 — Non-recall improvements + phase-3 task split (branch lokesh/non-recall-improvements)
- Changed: sentinel/config.py (tuned DEFAULT_CONFIG, ORIGINAL_DEFAULT_CONFIG), sentinel/pipeline.py (use_events honoured, evidence σ on MAD scale + detector_score/threshold_ratio), sentinel/explain.py (optional extras, volatility/drift context), tests/test_pipeline.py (+2), eval/tune.py (pin ALL6 in old grids), eval/figures.py (examples use engine defaults), app/dashboard.py (operating-point selector; fixed: it loaded the joint config as "tuned"; alarm line key), ARCHITECTURE_DECISIONS.md (ADR-001 wording, ADR-009), README.md, TASKS.md (phase 3 lists, file boundaries), results/*
- Result: `Sentinel()` defaults → event-F1 0.563 / 2.35 µs/pt; evidence σ median 27 (was raw detector scores, e.g. 792σ); 77 tests pass; dashboard verified (NAB operating point on nyc_taxi: 5/5 windows, F1 0.83).
- Notes: TASKS phase 3 — Lokesh L17–L22 (recall & delay), Likhit K9, K15–K21 (independent), file boundaries listed. L11 dropped per user.

## 2026-10-26 — L17–L22 recall & detection delay (branch lokesh/recall-delay)
- Changed: eval/miss_analysis.py (new), sentinel/recall.py (new: SpatialExtreme, ScoreSmoother), sentinel/pipeline.py (cooldown_frac, spatial flag, smoothing, configurable extreme_ratio, agree_k pass-through; empty dict replaces in deep merge), sentinel/events.py (agree_k), sentinel/config.py (new defaults), tests/test_recall.py (new, 8), tests/test_pipeline.py, eval/tune.py (recall, recall2 grids; delay tie-break), eval/ablation.py (phase build-up), eval/figures.py (multi-phase Pareto), README.md, ARCHITECTURE_DECISIONS.md (ADR-010), TASKS.md, results/*
- L17: misses 60 = 16 suppressed, 31 other-detector (28 Seasonal), 13 weak; detections at the labelled point (median +0) → lever is recall.
- Single-factor: spatial flag +0.03, debounce 3 / agree +0.01–0.02, cooldown_frac alone ≈0, smoothing hurts.
- L22: recall grid (480) → 0.640 / LOCO pooled 0.623; recall2 grid (432, per-detector risk re-opened) → **0.673 / LOCO pooled 0.654 / mean 0.593**, NAB 54.9; NAB point 60.00. Adopted as default. 2.6 µs/pt. 85 tests pass.
- Bug fixed: deep merge could not clear a nested dict ({} now replaces) — had made an ablation row silently equal to default.

## 2026-10-27 — Final: review Likhit's phase 3, finalise leaderboard & figures (branch lokesh/finalize)
- Likhit's findings (PR #1: validation, state, pool, CLI, packaging, K19 detector options, explain v2, causality tests): 107 tests pass. His "event-F1 0.563" benchmark was on sha b6343e1 (pre-merge); on main the default is 0.673 — no accuracy regression.
- Fixed: input-validation overhead (fast path in validate.py + lazy slow path in pipeline: 4.74 → 2.7 µs/pt, identical output); pandas `dayofweek` deprecation (6,401 warnings → 0); K19 options were unreachable from config → `var_ratio.two_sided`, `seasonal.weekly`, `seasonal.auto_period` (default off).
- K19 evaluation: on the default they cost precision (F1 0.51–0.60) → default unchanged; two-sided VarRatio in the NAB point → NAB 60.0 → **60.58**.
- Final: engine default F1 0.673 / P 0.747 / R 0.612 / FP 24, LOCO pooled 0.654, NAB 54.90, 2.7 µs/pt; NAB point 60.58. Leaderboard "Final results" section; regenerated all figures (published_comparison and ablation_progression rebuilt from real data — the old ones had stale/incorrect numbers); README + JUDGES_COMPREHENSIVE_REPORT.md updated.
