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
