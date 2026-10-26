# Sentinel — Adaptive Streaming Time-Series Anomaly Detection

[![Tests](https://img.shields.io/badge/pytest-107%20passed-brightgreen.svg)]()
[![NAB Standard Score](https://img.shields.io/badge/NAB%20Score-60.58-orange.svg)]()
[![Event F1](https://img.shields.io/badge/Event--F1-0.673%20(LOCO%200.654)-blue.svg)]()
[![Throughput](https://img.shields.io/badge/Throughput-367k%20pts%2Fsec-success.svg)]()
[![Latency](https://img.shields.io/badge/Latency-2.7%20%C2%B5s%2Fpt-purple.svg)]()

> **Hackathon Challenge 2:** Online, lookahead-free streaming anomaly detection on the 58-file Numenta Anomaly Benchmark (NAB). Built with a causal multi-horizon detector ensemble, Extreme Value Theory (SPOT/EVT) dynamic thresholding, event lifecycle debounce & re-baselining, and structured operational explainability.

Sentinel reads one `(timestamp, value)` point at a time and returns, for every point:

| Field | Meaning |
|---|---|
| `score` | Fused anomaly score in [0, 1] (1 − tail probability under an EVT model) |
| `label` | `NORMAL` / `ANOMALOUS` |
| `is_event_start` | First point of a new anomaly event (what event-F1 and alarms count) |
| `evidence` | Structured diagnostic evidence — type, expected vs observed, deviation $\sigma$, fired detectors, natural-language explanation |

```json
{
  "type": "spike",
  "expected": 14093.6,
  "observed": 39197.0,
  "deviation_sigma": 7.9,
  "fired": ["LastW", "LaggedResidual", "Seasonal"],
  "top_detector": "LaggedResidual",
  "text": "Value 3.92e+04 is 7.9σ above expected 1.409e+04 at Sun 01:00: a deviation from the longer-horizon trend (LaggedResidual; also LastW, Seasonal)."
}
```

---

## Key Highlights & Achievements

1. **Beats Major Published Industry Standards on NAB**:
   - **Sentinel (NAB operating point): 60.58 Standard / 54.02 Reward Low FP / 64.75 Reward Low FN**; the engine default scores 54.90 (original K1–K8 defaults: 53.00)
   - Above **earthgecko Skyline (58.2)** and **KNN CAD (58.0)**; outperforms **Relative Entropy (54.6)**, **Random Cut Forest (51.7)**, **Twitter ADVec (47.1)**, **Windowed Gaussian (39.6)**, **Etsy Skyline (35.7)**, **Bayesian Changepoint (17.7)**, and **EXPoSE (16.4)**.
2. **High Generalization & Event-F1**:
   - Engine default (`Sentinel()`, no config) reaches **Event-F1 = 0.673** (Precision 0.747, Recall 0.612, 24 FP events) vs **0.302** for the windowedGaussian baseline and 0.311 for the original K1–K8 defaults.
   - **Leave-One-Category-Out (LOCO) held-out F1: pooled 0.654, mean 0.593** — tuned on 6 categories, tested on the unseen 7th.
   - Two engine changes made the difference: a **per-detector SPOT risk level** (ADR-009: 0.311 → 0.563) and **recall mechanisms**
     (ADR-010: spatial-extreme flag, proportional cooldown, debounce 3, extreme ratio 2.0: 0.563 → 0.673, recall 0.48 → 0.61).
   - Detections land **at the labelled anomaly point** (median +0 points; 62 of 71 at or before it).
3. **Ultra-Low Latency & High Throughput**:
   - Executes in **2.7 µs/point** (~367,000 points/second on a single CPU core; p99 4.4 µs), including input validation.
   - **~19x faster than our < 50 µs/point budget** and than windowedGaussian (48.7 µs).
4. **Strictly Causal & $O(1)$**:
   - Zero lookahead, no future centering, fixed-size rolling circular deques with entry/exit running sums.
5. **Human-in-the-Loop Explainability**:
   - Generates structured JSON evidence cards and natural-language root-cause diagnostic text for every detected incident.

---

## Benchmark Leaderboard

### Official Published NAB Leaderboard Comparison

Evaluated on the full 58-file corpus using NAB's official sweeper (`data/NAB/nab/sweeper.py`):

| Detector | Standard Profile | Reward Low FP | Reward Low FN | Inference Latency | Theoretical Complexity |
|---|---|---|---|---|---|
| **Perfect Detector** | 100.0 | 100.0 | 100.0 | — | — |
| ARTime | 74.9 | 65.1 | 80.4 | Heavy batch | Non-streaming |
| Numenta HTM | 70.1 | 63.1 | 74.3 | ~500 µs/pt | Heavy Cortical Network |
| CAD OSE | 69.9 | 67.0 | 73.2 | High memory | Conformal Predictor |
| **Sentinel (Ours, NAB operating point)** | **60.58** | **54.02** | **64.75** | **4.8 µs/pt** | **Strict $O(1)$ Causal** |
| earthgecko Skyline | 58.2 | 46.2 | 63.9 | Heavy ensemble | Multi-process |
| KNN CAD | 58.0 | 43.4 | 64.8 | High memory | $O(N)$ Nearest Neighbors |
| Relative Entropy | 54.6 | 47.6 | 58.8 | High memory | Kernel Density |
| Sentinel (Ours, engine default) | 54.90 | 53.78 | 57.00 | 2.7 µs/pt | Strict $O(1)$ Causal |
| Sentinel (Ours, original K1–K8 defaults) | 53.00 | 37.27 | 59.76 | 5.5 µs/pt | Strict $O(1)$ Causal |
| Random Cut Forest | 51.7 | 38.4 | 59.7 | ~200 µs/pt | Random Trees |
| Twitter ADVec v1.0.0 | 47.1 | 33.6 | 53.5 | >1,000 µs/pt | Seasonal ESD |
| Windowed Gaussian | 39.6 | 20.9 | 47.4 | 49.0 µs/pt | Sliding Window |
| Etsy Skyline | 35.7 | 27.1 | 44.5 | Batch | Heuristics |
| Bayesian Changepoint | 17.7 | 3.2 | 32.2 | Quadratic | $O(N^2)$ Exact Bayesian |
| EXPoSE | 16.4 | 3.2 | 26.9 | Batch | Kernel Embedding |
| Random Detector | 16.8 | 5.8 | 25.9 | 0.4 µs/pt | Uniform Random |
| Null Detector | 0.0 | 0.0 | 0.0 | 0.5 µs/pt | Constant 0.5 |

### Event-based metrics (PS primary metric) — full corpus, 116 anomaly windows

Global operating points, all found by searching global configs only with leave-one-category-out validation
([results/tuning.md](results/tuning.md)):

| Detector | Event-F1 | Precision | Recall | FP events | Missed windows | NAB std | µs/pt |
|---|---|---|---|---|---|---|---|
| windowedGaussian (NAB baseline, best thr) | 0.302 | 0.211 | 0.534 | 232 | 54 | 39.6 | 48.7 |
| Sentinel — original K1–K8 defaults (`ORIGINAL_DEFAULT_CONFIG`) | 0.311 | 0.197 | 0.733 | 346 | 31 | 53.0 | 5.5 |
| Sentinel — phase 2: per-detector risk (ADR-009) | 0.563 | 0.675 | 0.483 | 27 | 60 | 42.8 | 2.4 |
| **Sentinel — engine default** (phase 3, ADR-010; `results/best_config.json`) | **0.673** | **0.747** | 0.612 | **24** | 45 | 54.9 | **2.7** |
| **Sentinel — NAB operating point** (`results/best_nab_config.json`) | 0.489 | 0.366 | **0.733** | 147 | **31** | **60.6** | 4.8 |

- Engine default: **RobustZ + LaggedResidual**, MAX fusion, SPOT q = 1e-6 (1e-5 for LaggedResidual), **spatial-extreme
  flag** (new running min/max beyond 5% of the range), debounce 3, extreme ratio 2.0, cooldown = 0.5 × probation length.
- NAB operating point: adds Seasonal (threshold 15), two-sided VarRatio (flatline / variance-collapse detection, K19) and a
  2-detector agreement bypass, LaggedResidual risk 1e-4.
- **Event-F1 and NAB still trade off** (figure below, 1488 evaluated configs): NAB rewards catching every window and charges
  an extra alarm only 0.11, event-F1 counts every false alarm in full. Phase 3 moved the whole front up and right: the
  default now scores above Random Cut Forest on NAB *and* 0.673 F1, so it also serves as the joint config
  (`results/best_joint_config.json`). **We submit the engine default** and quote the NAB point for leaderboard comparison.
- **Delay** ([results/miss_analysis.md](results/miss_analysis.md)): NAB windows start well before the labelled anomaly, so a
  "50% of the window" delay means detection at the labelled point itself (median +0 points). Of the 45 remaining misses,
  30 are pattern breaks only Seasonal sees (adding it costs too many false alarms), 12 have no detector signal at all.

![Event-F1 vs NAB trade-off](results/figures/pareto_f1_nab.png)
- **Ablation** ([results/ablation.md](results/ablation.md)): SPOT turns a static robust-z (F1 0.040, 4519 FPs) into 0.376;
  events → 0.392; all 6 detectors with MAX fusion 0.419, voting 0.456; per-detector risk with RobustZ + LaggedResidual
  0.563; + spatial-extreme flag 0.587; + debounce 3 / extreme ratio 2.0 0.610; + proportional cooldown 0.640;
  + LaggedResidual risk 1e-5 **0.673**. Removing the spatial flag from the final default costs the most (→ 0.428).

---

## Architectural Components

```
                   Streaming Observation (t, x_t)
                                 │
                 ┌───────────────┴───────────────┐
                 │                               │
       Probation (Warmup)               Post-Probation (Streaming)
                 │                               │
                 ▼                               ▼
       EVT / GPD Tail Fit               Multi-Detector Ensemble
     (Peaks-Over-Threshold)       ┌──────────────┴──────────────┐
                 │                │ D1: RobustZ (Spikes)        │
                 ▼                │ D2: LastW (Step Jumps)      │
       Dynamic Thresholds (z_q)   │ D3: LaggedResidual (Seq)    │
                 │                │ D4: VarRatio (Volatility)   │
                 │                │ D5: Seasonal (Diurnal)      │
                 │                │ D6: CUSUM (Level Shift)     │
                 │                │ D7: Streaming IForest       │
                 │                └──────────────┬──────────────┘
                 │                               │ Raw Deviations
                 ▼                               ▼
         ┌───────────────────────────────────────────────┐
         │ Dynamic Tail Probability Calibration (SPOT)   │
         └───────────────────────┬───────────────────────┘
                                 │ Calibrated Scores in [0, 1]
                                 ▼
         ┌───────────────────────────────────────────────┐
         │ MAX-Score & Consensus Fusion Engine           │
         └───────────────────────┬───────────────────────┘
                                 │ Fused Score & Alarms
                                 ▼
         ┌───────────────────────────────────────────────┐
         │ Event Lifecycle Manager                       │
         │ - Debounce Window (K=2 / Extreme Spike Bypass)│
         │ - Gap Merging (W=16 points)                   │
         │ - Post-Alarm Cooldown (W=200 points)          │
         │ - Sustained Level-Shift Re-Baselining (W=80)  │
         └───────────────────────┬───────────────────────┘
                                 │ Event Start Flags & Decision
                                 ▼
         ┌───────────────────────────────────────────────┐
         │ Explainability & Causal Evidence Engine       │
         │ - Primary Detector Attribution                │
         │ - Expected vs Observed & Deviation Sigma      │
         │ - Structured Human-Readable Narrative Text   │
         └───────────────────────┬───────────────────────┘
                                 ▼
                     Result(timestamp, value,
                            score, label,
                            is_event_start, evidence)
```

### 1. Multi-Horizon Causal Detectors (`sentinel/detectors.py`)
- **D1 (`RobustZ`):** Tracks online rolling median and Median Absolute Deviation (MAD). Slow exponential-weighted anti-pollution baseline update.
- **D2 (`LastW`):** Immediate squared deviation from rolling mean ($w=3$) to catch sudden step jumps.
- **D3 (`LaggedResidual`):** Dual-buffer rolling sum residual against a historical reference window ($w=16, h=64$) for sequence anomalies.
- **D4 (`VarRatio`):** Volatility detector computing the ratio of rolling short-term variance ($w=24$) to long-term baseline variance.
- **D5 (`Seasonal`):** 24-hour diurnal profile buckets with online mean and standard deviation updates to flag contextual pattern breaks.
- **D6 (`CUSUM`):** Two-sided Cumulative Sum tracking persistent mean drift.
- **D7 (`StreamingIsolationForest`):** Random cut tree isolation depth estimator evaluating multi-dimensional boundary exceedances in $O(1)$ time.

### 2. Extreme Value Theory Dynamic Thresholding (`sentinel/spot.py`)
- Fits a **Generalized Pareto Distribution (GPD)** to upper tail excesses during probation via Method of Moments.
- Computes dynamic extreme quantile thresholds $z_q$ for target risk level $q = 10^{-4}$.
- Updates streaming tail statistics continuously while preventing alarm contamination.

### 3. Event Lifecycle Manager (`sentinel/events.py`)
- **Debounce Confirmation:** Requires 2 consecutive anomaly points (or 1 extreme outlier $> 1.5\times$ threshold) before firing.
- **Gap Merging ($W = 16$):** Groups clustered anomaly alarms into a single coherent incident.
- **Cooldown Suppression ($W = 200$):** Eliminates redundant re-triggering during post-anomaly recovery.
- **Level-Shift Adaptation ($W = 80$):** Detects when a level shift becomes the "new normal" and calls `reset_baseline()` across all detectors.

---

## Visualizations & Figures

NAB leaderboard position and measured latency:

![published comparison](results/figures/published_comparison.png)

Event-F1 along the build-up (baseline → final engine default):

![progression](results/figures/ablation_progression.png)

Per-category event-F1 vs the windowedGaussian baseline, and why the remaining windows are missed:

![per-category F1](results/figures/per_category_f1.png)

![miss analysis](results/figures/miss_analysis.png)

Example stream with ground-truth windows and detections:

![example](results/figures/example_nyc_taxi.png)

Full tables: [results/leaderboard.md](results/leaderboard.md) · [results/ablation.md](results/ablation.md) ·
[results/tuning.md](results/tuning.md) · [results/miss_analysis.md](results/miss_analysis.md). Full ablation chart:
[results/figures/ablation.png](results/figures/ablation.png).

---

## Interactive Streamlit Dashboard

Launch the live replay demo:
```bash
streamlit run app/dashboard.py
```

Pick any series and an operating point (engine default / event-F1 / joint / NAB / original defaults), press **▶ Replay** and watch the stream: value vs expected $\pm k\sigma$ band, ground-truth windows, flagged events, fused score on a $-\log_{10}(p)$ scale with the SPOT alarm line, per-detector score lanes, running event-F1, and an evidence card for every event.

---

## Quickstart & Reproduction

### 1. Setup Environment
```bash
python3.11 -m venv .venv   # Python >= 3.10
source .venv/bin/activate  # Or on Windows: .venv\Scripts\activate
pip install -r requirements.txt
git clone --depth 1 https://github.com/numenta/NAB data/NAB
```

### 2. Run Test Suite
```bash
pytest -q
# All 107 tests pass in ~5 seconds (incl. robustness fuzzing, checkpoint round-trips, no-look-ahead property tests)
```

### 3. Run Benchmark & Evaluation
```bash
python -m eval.run --detector sentinel --tag "sentinel-eval"
python -m eval.leaderboard
python -m eval.benchmark
python -m eval.ablation            # build-up from baseline to the engine default
python -m eval.tune --grid recall2  # leave-one-category-out search
python -m eval.miss_analysis        # why windows are missed / detection delay
python -m eval.figures
```

### 4. Predict on new data (starter pack / hidden test windows)
```bash
python -m eval.predict --input path/to/csv_folder --out outputs/predictions [--combined] [--windows labels.json]
```
Streams every `timestamp,value` CSV (any folder nesting, or a single file) through a fresh Sentinel using the tuned
`results/best_config.json`, and writes per-point `score,label,is_event_start,evidence` CSVs, an `events.csv` with one row
per detected event (type, top detector, explanation), and `summary.json` (runtime; event-F1/P/R if `--windows` is given).

### 5. Install as a package / command-line streaming
```bash
pip install -e .                                           # package sentinel-ad
python -m sentinel --input data.csv --output results.jsonl # one stream, JSONL results
cat stream.jsonl | python -m sentinel > flagged.jsonl      # stdin -> stdout
python -m sentinel --input multi.csv --stream-col stream_id  # many streams (SentinelPool)
```
`Sentinel.save()` / `Sentinel.load()` checkpoint a stream mid-way; inputs with NaN/inf, missing values or
duplicate/backward timestamps are sanitised (`sentinel/validate.py`) instead of crashing.

Use from Python:
```python
from sentinel.pipeline import Sentinel
s = Sentinel(n_total=len(df))   # defaults = event-F1 operating point; config={...} overrides (nested keys merge)
for t, x in zip(df.timestamp, df.value):
    r = s.update(t, x)
    if r.is_event_start:
        print(r.timestamp, r.score, r.evidence["text"])
```

---

## Architecture Decision Records (ADRs)

All foundational architectural decisions, invariants, and empirical milestones are formally documented in [ARCHITECTURE_DECISIONS.md](ARCHITECTURE_DECISIONS.md):
- `ADR-001`: Strict Streaming & Causal Invariant ($O(1)$ memory & time, $< 50\ \mu\text{s/pt}$)
- `ADR-002`: Causal Multi-Horizon Detector Ensemble (D1 to D7)
- `ADR-003`: Event Debounce and Extreme Outlier Bypass Protocol
- `ADR-004`: Multi-Detector Fusion Engine
- `ADR-005`: Dynamic Tail Thresholding via Extreme Value Theory (SPOT/DSPOT)
- `ADR-006`: Event Lifecycle Management (Gap Merging & Re-Baselining)
- `ADR-007`: Structured Evidence & Explainability Schema
- `ADR-008`: Event-Aligned Continuous Scoring for Official NAB Calibration
- `ADR-009`: Per-Detector SPOT Risk Level and Tuned Defaults
- `ADR-010`: Recall Mechanisms (Spatial-Extreme Flag, Proportional Cooldown, Confirmation)
