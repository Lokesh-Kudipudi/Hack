# Likhit — session log

<!-- Append one entry per finished task:
## YYYY-MM-DD HH:MM — <TASK-ID> <title>
- Changed: <files>
- Result: <metrics / tests passed>
- Notes / next: <...>
-->

## 2026-10-26 19:46 — K1 Contract & Initial Pipeline
- Changed: `sentinel/__init__.py`, `sentinel/types.py`, `sentinel/config.py`, `sentinel/pipeline.py`, `tests/test_pipeline.py`
- Result: All 40 unit tests pass. Full-corpus benchmark: `sentinel [k1 d1 robust-z]` Event-F1 = 0.176 (Recall 0.664, TP=77/116), latency median = 108 pts, **inference runtime = 4.4 µs/pt** (11x faster than windowed_gaussian 49.0 µs/pt).
- Notes / next: D1 alone catches 77 of 116 anomaly windows. High FP count (683) needs suppression.

## 2026-10-26 19:54 — K1 Tuning & ADRs (ADR-001 to ADR-005)
- Changed: `ARCHITECTURE_DECISIONS.md`, `sentinel/config.py`, `sentinel/pipeline.py`, `tests/test_pipeline.py`
- Result: Formal ADR cards established. Debounce (K=2 confirmation / 1.5x spike bypass) and calibrated threshold (6.0σ) cut False Positives nearly in half: **683 → 389 FPs**!
- Event-F1 jumped from **0.176 → 0.240** (+36.4% gain), `realAdExchange` F1 hit **0.560**, `realTraffic` F1 hit **0.562**, `realKnownCause` hit **0.341**, runtime improved to **3.20 µs/pt**.
- Notes / next: Moving to K2 to build the multi-horizon detectors.

## 2026-10-26 19:59 — K2 Detectors Implementation (D1 to D6)
- Changed: `sentinel/detectors.py`, `tests/test_detectors.py`, `sentinel/pipeline.py`, `sentinel/config.py`
- Result: All 6 causal O(1) detectors implemented (D1 RobustZ, D2 LastW, D3 LaggedResidual, D4 VarRatio, D5 Seasonal, D6 CUSUM). All 47 tests pass. Each detector fires specifically on synthetic spike, jump, sequence, variance burst, and seasonal break.
- Observation: Applying a single static 6.0 threshold across raw scores of different mathematical distributions (Z-scores vs variance ratios) shows why per-detector threshold calibration (Task K3: SPOT) is strictly needed to normalize exceedances into uniform tail probabilities.
- Notes / next: Task K3 (SPOT / DSPOT with GPD fit on probation and streaming tail update).

## 2026-10-26 20:25 — K3 Dynamic Thresholding via SPOT & DSPOT
- Changed: `sentinel/spot.py`, `tests/test_spot.py`, `sentinel/pipeline.py`, `sentinel/config.py`, `tests/test_pipeline.py`
- Result: Implemented EVT streaming thresholding via SPOT (Method of Moments GPD fit on tail exceedances) and DSPOT (moving average residual drift removal).
- Benchmark: **Event-F1 surged to 0.324** (NEW #1 ON REPO LEADERBOARD, beating windowed_gaussian 0.302).
  - Recall: **0.724** (84/116 anomaly windows detected across NAB corpus).
  - False Positives: **Reduced from 860 down to 318** (>63% drop).
  - `artificialNoAnomaly`: 0 false positives.
  - `realTraffic`: Event-F1 = 0.611.
  - `realAdExchange`: Event-F1 = 0.519.
  - Runtime: **16.37 µs/pt** (well within < 50 µs/pt budget).
- Notes / next: Task K4 (Extract modular fusion engine to `sentinel/fusion.py` with k-of-n vote and tests).

## 2026-10-26 20:28 — K4 Multi-Detector Fusion Engine
- Changed: `sentinel/fusion.py`, `tests/test_fusion.py`, `sentinel/pipeline.py`
- Result: Modular `FusionEngine` with MAX fusion, k-of-n consensus voting, and extreme outlier bypass. All 58 tests in full test suite pass.
- Notes / next: Task K5 (Event lifecycle manager in `sentinel/events.py`).

## 2026-10-26 20:33 — K5 Event Lifecycle Manager
- Changed: `sentinel/events.py`, `tests/test_events.py`, `sentinel/pipeline.py`
- Result: Modular `EventManager` with debounce, gap merge, cooldown, and sustained level-shift rebaselining. All 62 tests pass.
- Benchmark: Recall reached **0.733** (85/116 windows), Event-F1 = 0.311, runtime = **37.42 µs/pt**.
- Notes / next: Task K6 (Explainability engine in `sentinel/explain.py`).

## 2026-10-26 20:36 — K6 Explainability Engine
- Changed: `sentinel/explain.py`, `tests/test_explain.py`, `sentinel/pipeline.py`
- Result: Modular `ExplainEngine` generating structured evidence dictionaries and contextual narrative text for all flagged anomalies conforming to AGENTS.md §4.6. All 65 tests pass.
- Benchmark: Event-F1 = 0.311, Recall = **0.733** (85/116 windows), inference runtime = **23.04 µs/pt** (well within < 50 µs/pt budget).
- Notes / next: Task K7 (Performance profiling and hot-path optimization).

## 2026-10-26 20:42 — K7 Performance & Hot-Path Optimization
- Changed: `sentinel/detectors.py`, `sentinel/pipeline.py`
- Result: Converted `LaggedResidual` to strict $O(1)$ dual-buffer running sum with 0 iterator overhead. Pre-allocated and reused per-point dictionaries in `Sentinel.update`.
- Benchmark: Inference runtime sped up from **23.04 µs/pt → 15.38 µs/pt** (33% faster, executing at 65,000 points/sec). All 65 unit tests pass.
- Notes / next: Task K8 (*Stretch* D7 Streaming Trees / Isolation Forest).

## 2026-10-26 20:53 — K8 Stretch Detector (D7 Streaming Isolation Forest) & Ablation
- Changed: `sentinel/detectors.py`, `tests/test_detectors.py`, `sentinel/config.py`, `sentinel/explain.py`, `sentinel/pipeline.py`
- Result: Implemented `StreamingIsolationForest` using fast rolling random cut trees in strictly $O(1)$ time ($< 4$ µs per point overhead). All 66 tests pass.
- Benchmark: Recall jumped to **0.759** (88/116 windows detected — all-time high, catching 3 additional anomaly windows). Event-F1 = **0.311**, inference runtime = **20.80 µs/pt** (well within < 50 µs/pt budget).
## 2026-10-26 21:08 — ADR Cards, Pitch README, Ablation Engine & Figures
- Changed: `ARCHITECTURE_DECISIONS.md`, `README.md`, `eval/ablation.py`, `results/ablation.md`, `results/figures/*.png`, `TASKS.md`
- Result:
  - Formally documented all architecture decision records: ADR-001 (Streaming Invariant), ADR-002 (Ensemble), ADR-003 (Debounce), ADR-004 (Fusion), ADR-005 (SPOT/EVT), ADR-006 (Lifecycle), ADR-007 (Explainability), ADR-008 (NAB Continuous Calibration).
  - Generated slide-ready figures: `results/figures/ablation_progression.png` and `results/figures/published_comparison.png`.
  - Created complete submission pitch `README.md` with benchmark tables, architecture diagrams, and quickstart commands.
  - Tasks L6, L8, L9 complete.

## 2026-10-26 23:48 — Phase 3 Independent Improvements (K9, K15, K16, K17, K18, K19, K20, K21)
- Changed: `sentinel/validate.py`, `sentinel/state.py`, `sentinel/pool.py`, `sentinel/__main__.py`, `sentinel/detectors.py`, `sentinel/explain.py`, `sentinel/pipeline.py`, `sentinel/__init__.py`, `pyproject.toml`, `tests/test_robustness.py`, `tests/test_state.py`, `tests/test_pool_and_cli.py`, `tests/test_causality.py`, `tests/test_detectors.py`, `tests/test_explain.py`, `TASKS.md`
- Results:
  - **K9 Review**: Engine changes on `main` verified and approved.
  - **K15 Input Robustness**: `sentinel/validate.py` handles NaN, Inf, missing/duplicate/backward timestamps, constant series, and extreme magnitudes. 8 robustness tests pass.
  - **K18 Packaging**: Modern `pyproject.toml` created, package `sentinel-ad-0.3.0` installed in editable mode (`pip install -e .`), clean exports in `sentinel/__init__.py`.
  - **K16 State Checkpointing**: `sentinel/state.py` adds `Sentinel.save()` and `Sentinel.load()`. Verified bit-identical stream resumption post-probation and mid-probation across all 7 detectors (3 tests pass).
  - **K17 Multi-Stream API & CLI**: `SentinelPool` in `sentinel/pool.py` for lazy multiplexing of thousands of streams + memory accounting. CLI in `sentinel/__main__.py` (`python -m sentinel`) supporting stdin/stdout JSONL and CSV (4 tests pass).
  - **K19 Detector Quality**: Added `two_sided=True` to `VarRatio` for sensor flatline / collapse detection; added `weekly=True` (168 buckets) and probation autocorrelation period detection to `Seasonal` (10 tests pass).
  - **K20 Explainability v2**: Structured `confidence` and `context` fields in `ExplainEngine`, fast tuple weekday lookup (< 1.5 µs overhead, 4 tests pass).
  - **K21 Determinism & Zero Lookahead**: Formal property tests verifying prefix invariance (output at $t$ is strictly invariant to future points $t+1..T$) and bit-identical repeated runs (4 tests pass).
  - **Master Suite**: **99 passed in 31.6s** (up from 77 passed).
