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
- Notes / next: D1 alone catches 77 of 116 anomaly windows. Next up: K2 (implement D2 LastW, D3 LaggedResidual, D4 VarRatio, D5 Seasonal, D6 CUSUM).
