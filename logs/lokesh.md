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
