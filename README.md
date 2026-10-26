# Sentinel — lightweight streaming anomaly detection

Hackathon Challenge 2: adaptive time-series anomaly detection on the **Numenta Anomaly Benchmark (NAB)**.
Sentinel reads one `(timestamp, value)` point at a time and returns, for every point:

| field | meaning |
|---|---|
| `score` | fused anomaly score in [0, 1] (1 − tail probability under an EVT model) |
| `label` | `NORMAL` / `ANOMALOUS` |
| `is_event_start` | first point of a new anomaly event (what event-F1 counts) |
| `evidence` | why it was flagged — type, expected vs observed, σ-deviation, detectors that fired, a sentence |

```json
{"type": "spike", "expected": 14093.6, "observed": 39197.0, "deviation_sigma": 7.9,
 "fired": ["LastW", "LaggedResidual", "Seasonal"], "top_detector": "LaggedResidual",
 "text": "Value 3.92e+04 is 7.9σ above expected 1.409e+04 at Sun 01:00: a deviation from the longer-horizon trend (LaggedResidual; also LastW, Seasonal)."}
```

## Results (full NAB corpus: 58 series, 365k points, 116 anomaly windows)

<!-- RESULTS:START -->
> ⚠️ Sentinel numbers below were produced with a provisional engine implementation (`sentinel_source: local-untracked` in the run JSONs). They will be regenerated once the final `sentinel/` engine is merged.

| Detector | Event-F1 | Precision | Recall | FP events | Median delay (fraction of window) | µs / point |
|---|---|---|---|---|---|---|
| null | 0.000 | — | 0.000 | 0 | — | 0.3 |
| random | 0.058 | 0.030 | 0.793 | 2981 | 0.18 | 0.4 |
| windowedGaussian (NAB baseline, best threshold 1.0) | 0.302 | 0.211 | 0.534 | 232 | 0.49 | 49.0 |
| Sentinel, default config | 0.333 | 0.225 | 0.647 | 259 | 0.49 | 4.9 |
| **Sentinel, tuned global config** | **0.538** | **0.654** | 0.457 | **28** | 0.50 | **4.9** |

- **Generalisation (leave-one-category-out):** tuning only on the other 6 categories and testing on the held-out one gives a
  mean held-out event-F1 of **0.507** (pooled 0.479), close to the in-sample 0.538 — the global config is not over-fitted.
- **Runtime:** 4.9 µs per streaming update on one core (≈ 200k points/s), 10× faster than windowedGaussian;
  the whole 365k-point corpus streams in under 2 s.
- **Ablation** ([results/ablation.md](results/ablation.md)): SPOT thresholds + event merging take a single robust z-score
  from F1 0.185 to 0.362; MAX fusion of all 6 detectors reaches 0.370, and **2-of-6 voting reaches 0.538** (FP 104 → 28).
  Removing the long-horizon LaggedResidual costs the most (−0.08), confirming the short + long horizon design.

![ablation](results/figures/ablation.png)
<!-- RESULTS:END -->

Full tables: [results/leaderboard.md](results/leaderboard.md) · [results/ablation.md](results/ablation.md) ·
[results/tuning.md](results/tuning.md). Figures for slides: [results/figures/](results/figures/).

![per-category F1](results/figures/per_category_f1.png)
![example](results/figures/example_nyc_taxi.png)

## Approach

```
(timestamp, value) ─► probation (first 15%, ≤750 pts): learn baselines, never flag
                   ─► 6 causal detectors (O(1)/O(w) each) ─► raw scores
                   ─► per-detector SPOT (EVT / GPD tail)   ─► calibrated score + alarm
                   ─► fusion (k-of-n vote)                  ─► label
                   ─► event manager (merge gap, cooldown, re-baseline after level shift)
                   ─► evidence (only for ANOMALOUS points)
```

| Detector | Raw score | Catches |
|---|---|---|
| RobustZ | \|x − median\| / MAD, slow clipped EW update | spikes |
| LastW (short horizon) | (mean(last 3) − x)² | point anomalies |
| LaggedResidual (long horizon) | (x − mean of a window that ended h steps ago)² | sequence anomalies — the reference window isn't yet contaminated by the anomaly |
| VarRatio | \|log(var(last w) / baseline var)\| | volatility bursts *and* flatlines |
| Seasonal | residual vs hour-of-day bucket profile | broken daily patterns |
| CUSUM | two-sided CUSUM on robust z | level shifts / drift |

Design choices and why:

- **Simple, adaptive one-liners instead of deep models.** The ICLR 2026 paper *When Foundation Models are One-Liners* shows
  time-series foundation models behave like moving-variance / squared-difference baselines; the NAB leaderboard is won by
  simple adaptive methods. We get the same signal at ~5 µs/point.
- **Short + long horizon residuals.** A short context gets contaminated by a long anomaly; a lagged context does not.
- **Threshold design = Extreme Value Theory (SPOT, Siffer et al. KDD 2017).** Each detector's raw score gets its own
  Peaks-Over-Threshold GPD tail fitted on probation and updated online. The threshold is a *risk level* `q`
  (false-alarm probability), which is comparable across detectors and series — no per-file magic numbers.
- **Voting fusion.** Six detectors each at risk `q` would multiply false alarms under MAX fusion; requiring 2 of 6 to agree
  cuts FPs sharply with little recall loss (see ablation).
- **Event post-processing.** Event-F1 and NAB only reward the first detection per window; merge + cooldown removes
  duplicate alarms; a confirmed level shift re-baselines the detectors ("new normal").
- **Honest validation.** Only global hyperparameters, chosen with leave-one-category-out cross-validation.

## Evaluation protocol

Implemented in [eval/scorer.py](eval/scorer.py), exactly as specified in [AGENTS.md §5](AGENTS.md):
probation (first `min(⌊0.15·n⌋, 750)` points) excluded; GT events = windows from `labels/combined_windows.json`;
TP window = contains ≥ 1 predicted event start; FP = event start outside all windows; FN = window with no start;
micro-averaged P / R / F1; detection delay in points, minutes and fraction of window width.

We also report the **official NAB score** ([eval/nab_score.py](eval/nab_score.py), wraps NAB's own `sweeper.py`;
reproduces the published leaderboard: windowedGaussian 39.65, numenta 70.10, null 0.00) in two forms: on the per-point
`score` (threshold optimised like NAB) and on event starts only.

## How to run

```bash
uv venv -p 3.11 .venv && source .venv/bin/activate     # or: python3.11 -m venv .venv
pip install -r requirements.txt
git clone --depth 1 https://github.com/numenta/NAB data/NAB   # data never committed
pytest -q
```

```bash
python -m eval.run --detector sentinel --tag "my change"          # score the corpus -> results/runs/*.json
python -m eval.run --detector windowed_gaussian --tag baseline
python -m eval.leaderboard                                         # -> results/leaderboard.md
python -m eval.benchmark                                           # µs/point, single core
python -m eval.ablation --base results/best_config.json            # -> results/ablation.md + figure
python -m eval.tune                                                # LOCO grid search -> results/tuning.md
python -m eval.figures                                             # slide figures
streamlit run app/dashboard.py                                     # live replay demo
```

The loader accepts any folder of `timestamp,value` CSVs plus a NAB-format windows JSON (`SENTINEL_DATA`,
`SENTINEL_WINDOWS` env vars), including the organiser starter pack layout (`data/<category>/<category>/*.csv`).

Use from Python:

```python
from sentinel.pipeline import Sentinel
s = Sentinel(n_total=len(df))                 # or config={...}
for t, x in zip(df.timestamp, df.value):
    r = s.update(t, x)
    if r.is_event_start:
        print(r.timestamp, r.score, r.evidence["text"])
```

## Demo

`streamlit run app/dashboard.py` — pick any series, press **Replay** and watch the stream: value vs expected ±kσ band,
ground-truth windows, flagged events, fused score on a −log₁₀(p) scale with the SPOT alarm line, per-detector score
lanes, running event-F1, and an evidence card for every event.

## Repo layout

```
sentinel/   detection engine (detectors, SPOT, fusion, events, explain, pipeline)
eval/       data loader, scorer, NAB wrapper, baselines, run/benchmark/ablation/tune/figures CLIs
app/        Streamlit dashboard
tests/      pytest suite
results/    runs/*.json (one per evaluation), leaderboard.md, ablation.md, tuning.md, figures/
```

## References

- Numenta Anomaly Benchmark — https://github.com/numenta/NAB
- Siffer et al., *Anomaly Detection in Streams with Extreme Value Theory*, KDD 2017
- Zhu, Carpentier, Verbeke, *When Foundation Models are One-Liners*, ICLR 2026
