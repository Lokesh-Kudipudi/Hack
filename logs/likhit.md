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
