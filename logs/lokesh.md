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
