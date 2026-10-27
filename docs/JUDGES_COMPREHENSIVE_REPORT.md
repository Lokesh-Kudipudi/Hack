# Sentinel — Comprehensive Technical Architecture & Engineering Report
**Challenge 2:** Adaptive Streaming Time-Series Anomaly Detection on the Numenta Anomaly Benchmark (NAB)  
**Authors:** Likhit & Lokesh  
**Target:** Hackathon Checkpoint & Technical Judging Review  
**Repository:** `Hack/` | **Core Package:** `sentinel/` | **Evaluation:** `eval/` | **Demo:** `app/dashboard.py`

---

## 1. Executive Summary

**Sentinel** is an ultra-lightweight, strictly causal, lookahead-free streaming anomaly detection engine designed for real-time univariate time series. It resolves the fundamental conflict between academic deep learning models (which suffer from lookahead data leakage, high compute latency, and uncalibrated false alarms) and naive statistical heuristics (which suffer from rigid thresholds and catastrophic false-positive storms).

### Key Performance Highlights (final, full NAB corpus):
- **Superior Event Detection (Primary Metric):** The engine default (`Sentinel()`, no config) achieves **Event-F1 = 0.691** (Precision 0.731, Recall 0.655, 28 FP events) versus **0.302** for the Windowed Gaussian baseline (232 FPs) and 0.311 for the original K1–K8 defaults.
- **LOCO grid check:** The 16-setting lag/margin selection scored **0.664 pooled / 0.590 mean** when each category was held out in turn. This is a selection-procedure result, not a held-out score for the one fixed default.
- **Beats Published Industry Models on NAB:** The NAB operating point scores **60.58** (standard profile), above earthgecko Skyline (58.2), KNN CAD (58.0), Relative Entropy (54.6), AWS Random Cut Forest (51.7), Twitter ADVec (47.1) and Windowed Gaussian (39.6). The engine default scores 58.80.
- **Detection at the anomaly:** median detection is at the labelled anomaly point (+0 points); NAB windows start well before it.
- **Ultra-Low Latency & High Throughput:** **2.86 µs/point** (~350,000 observations/sec on one CPU core) including input validation — ~18x under our own < 50 µs/pt budget (≈ the cost of the Windowed Gaussian baseline, 49 µs).
- **Production-Grade Explainability:** Emits structured JSON diagnostic cards with expected normal ranges, standard deviation magnitudes, fired detectors, and human-readable root-cause explanations.

---

## 2. Theoretical Lineage & Academic Foundations (What is Taken from Papers)

Sentinel's architecture is grounded in three major peer-reviewed research foundations:

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                      ACADEMIC RESEARCH LINEAGE                                  │
├───────────────────────────────┬─────────────────────────────────────────────────┤
│ ICLR 2026 Paper               │ Multi-horizon causal decomposition (§6.1)       │
│ "When Foundation Models are   │ Pitfall of per-window z-normalization (§5.1)    │
│  One-Liners"                  │ Short Last-w & Volatility Var-w baselines       │
├───────────────────────────────┼─────────────────────────────────────────────────┤
│ KDD 2017 Paper                │ Dynamic extreme tail thresholding (SPOT)        │
│ Siffer et al.                 │ Generalized Pareto Distribution (GPD) tail fit  │
│ Extreme Value Theory          │ Drift-aware residual modeling (DSPOT)           │
├───────────────────────────────┼─────────────────────────────────────────────────┤
│ Numenta Benchmark (NAB)       │ Lookahead-free streaming protocol               │
│ Lavin & Ahmad (2015)          │ Event window scoring & detection delay penalties│
│ Ahmad et al. (2017)           │ Asymmetric application-profile scoring matrices │
└───────────────────────────────┴─────────────────────────────────────────────────┘
```

### 2.1 ICLR 2026: *"When Foundation Models are One-Liners: Anomaly Detection in Time Series"*
Recent literature has shown that heavy time-series foundation models (Transformers, LLM-based TS forecasters, and deep autoencoders) often fail to outperform elementary moving-window statistics when evaluated rigorously without data leakage. Sentinel incorporates four core principles directly from this work:

1. **The Moving-Baseline Equivalence (Definitions 4 & 5):**
   - The paper proves that time-series foundation models primarily measure deviations from short-term local averages or short-term variance shifts.
   - *Sentinel Implementation:* Rather than deploying bloated neural network weights in the hot path, Sentinel implements causal $O(1)$ building blocks: `LastW` (Definition 5: immediate preceding step deviation) and `VarRatio` (Definition 4: local variance relative to baseline variance).
2. **The Pitfall of Per-Window Z-Normalization (§5.1):**
   - The paper identifies a widespread flaw in academic benchmark pipelines: standardizing time-series windows independently via $(x_t - \mu_{window}) / \sigma_{window}$ mathematically forces the anomaly score to approximate the inverse of local window variance. In quiet periods, sensor quantization noise gets amplified into false alarms.
   - *Sentinel Implementation:* Sentinel strictly rejects per-window z-normalization. Instead, scale ($\text{MAD}$) and baseline location ($\text{median}$) are learned during an initial probation warmup window (first 15% of the series, max 750 points) and updated with slow exponential anti-pollution smoothing ($\alpha = 0.01$).
3. **Multi-Horizon Residual Decomposition (§6.1):**
   - A critical discovery in §6.1 is that short-horizon baselines ($w=3$) excel at capturing isolated spikes, but fail catastrophically on sequence anomalies (e.g., persistent level shifts or multi-point drops). Why? Because after 3 data points, the anomaly enters the moving window and "contaminates" the reference mean, causing the anomaly score to collapse back to zero.
   - *Sentinel Implementation:* Sentinel includes complementary short and long reference methods:
     - **Short/reference level (`RobustZ`):** Catches values far from the robust normal level. `LastW` ($w=3$) is available as an optional short-horizon detector.
     - **Long Horizon (`LaggedResidual`, $w=8, h=256$):** Measures the deviation of $x_t$ against a reference window that ended $h=256$ steps in the past. Even if an anomaly lasts 30 points, the reference window remains uncontaminated in the pre-anomaly normal regime.
4. **Volatility / Noise Burst Detection (`VarRatio`, $w=24$):**
   - Captures second-order regime shifts where the mean remains unchanged but variance explodes (e.g., mechanical vibration chatter or network packet jitter).

---

### 2.2 KDD 2017: *"Anomaly Detection in Streams with Extreme Value Theory"* (Siffer et al.)
In real-world streams, fixed heuristic thresholds (e.g., $3\sigma$ or $6\sigma$) fail completely: Gaussian assumptions grossly underestimate the probability of tail events in heavy-tailed distributions (e.g., cloud server CPU spikes, web traffic, financial transactions).

1. **Streaming Peak-Over-Threshold (SPOT):**
   - Extreme Value Theory (EVT), specifically the **Pickands–Balkema–de Haan Theorem**, states that regardless of the underlying parent distribution, the excesses $Y = (X - t)$ over a sufficiently high threshold $t$ asymptotically converge to a **Generalized Pareto Distribution (GPD)**:
     $$F_{(t)}(y) = P(X - t \le y \mid X > t) \approx 1 - \left(1 + \frac{\gamma y}{\sigma}\right)^{-1/\gamma}$$
     where $\gamma$ is the shape parameter (tail heaviness) and $\sigma$ is the scale parameter.
2. **Method of Moments (MoM) Estimation:**
   - Instead of running iterative, computationally prohibitive maximum likelihood optimization (like Grimshaw's trick) at every streaming point, Sentinel implements **Method of Moments** estimation:
     $$\mu = \frac{1}{N_{ex}} \sum y_i, \quad v = \frac{1}{N_{ex}} \sum y_i^2 - \mu^2$$
     $$\gamma = \frac{1}{2}\left(1 - \frac{\mu^2}{v}\right), \quad \sigma = \frac{1}{2}\mu\left(\frac{\mu^2}{v} + 1\right)$$
   - This executes in strictly $O(1)$ operations per point using running accumulators $\sum y_i$ and $\sum y_i^2$.
3. **Dynamic Alarm Quantile ($z_q$):**
   - For a user-specified tail risk level $q$ (e.g., $q = 10^{-4}$ or $10^{-6}$), the threshold $z_q$ is dynamically calculated:
     $$z_q = t + \frac{\sigma}{\gamma}\left[\left(\frac{q \cdot n}{N_{ex}}\right)^{-\gamma} - 1\right]$$
4. **Anti-Pollution Streaming Tail Updating:**
   - Observations between $t$ and $z_q$ are incorporated into the tail distribution statistics to adapt to natural drift. Observations exceeding $z_q$ are flagged as anomalies and strictly prevented from polluting the GPD statistics.
5. **DSPOT (Drifting SPOT):**
   - For non-stationary streams with slow continuous drift, DSPOT computes the residual against a trailing moving median or mean before feeding the residual into the SPOT GPD tail estimator.

---

### 2.3 Numenta Anomaly Benchmark (NAB) Protocol & Scoring Rules
Sentinel adheres strictly to the evaluation methodology defined by Lavin & Ahmad (2015) and Ahmad et al. (2017):

1. **Strict Lookahead-Free Streaming:**
   - The detector processes tuples $(t, x_t)$ sequentially in historical order. No forward-looking windows, no bidirectional smoothing, and no global scaling using future timestamps.
2. **Probation Warmup Period:**
   - In accordance with NAB rules, the first $15\%$ of each time series (capped at a maximum of 750 points) is designated as probation. Sentinel uses this period solely to fit initial baseline statistics and GPD tail parameters. No anomaly flags are permitted during probation, and this period is strictly excluded from scoring.
3. **Event Window Scoring vs Point-Wise Metrics:**
   - Real-world anomalies are intervals, not isolated points. NAB provides ground-truth windows $[W_{start}, W_{end}]$.
   - An algorithm is awarded a True Positive (TP) if it emits an alarm within the window. Emitting multiple alarms inside the same window receives no additional credit. Emitting alarms outside any window is penalized as a False Positive (FP).
4. **Asymmetric Application Profiles:**
   - Evaluated under the official NAB metric matrix:
     - **Standard Profile:** Balanced weights between early detection and low false alarms.
     - **Reward Low FP:** Heavily penalizes false alarms (critical for operator alarm-fatigue prevention).
     - **Reward Low FN:** Heavily penalizes missed anomalies (critical for safety and high-availability systems).

---

## 3. Every Architecture Decision Made (ADR Logs)

The development of Sentinel was guided by 10 formal Architecture Decision Records (full text in `ARCHITECTURE_DECISIONS.md`):

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                      SENTINEL ARCHITECTURAL DECISIONS                           │
├─────────┬─────────────────────────────────────┬─────────────────────────────────┤
│ ADR-001 │ Strict Streaming & Causal Invariant │ O(1) time & O(w) bounded memory │
│ ADR-002 │ Multi-Horizon Specialized Ensemble  │ D1 through D7 complementary     │
│ ADR-003 │ Event Debounce & Extreme Bypass     │ K=2 confirmation / 1.5x bypass  │
│ ADR-004 │ Multi-Detector Fusion Engine        │ MAX vs k-of-n consensus voting  │
│ ADR-005 │ Dynamic Tail Thresholding via EVT   │ SPOT / DSPOT GPD estimation     │
│ ADR-006 │ Event Lifecycle Management          │ Gap merge, cooldown, rebaseline │
│ ADR-007 │ Operational Explainability Engine   │ Structured JSON evidence card   │
│ ADR-008 │ Continuous Scoring for NAB Sweeper  │ Event-aligned probability curve │
│ ADR-009 │ Per-Detector SPOT Risk + Defaults   │ 0.311 -> 0.563 event-F1         │
│ ADR-010 │ Recall Mechanisms                   │ Spatial flag, cooldown -> 0.673 │
└─────────┴─────────────────────────────────────┴─────────────────────────────────┘
```

### ADR-001: Strict Streaming & Causal Invariant
- **Context:** Many published models cheat by centering windows ($[t - w/2, t + w/2]$) or requiring offline batch normalization. Furthermore, high-throughput cloud streaming systems require microsecond execution speeds.
- **Decision:** Every operation must be strictly causal (depending only on $\tau \le t$) and execute in $O(1)$ time complexity using circular buffers (`collections.deque`) with running sum maintenance. Deep neural network dependencies (PyTorch/TensorFlow) were rejected for the runtime core.
- **Empirical Impact:** **2.76 µs/point** for the current engine default (~362,000 points/sec, incl. input validation), consuming less than 15 MB of RAM across all 58 concurrent series.

### ADR-002: Multi-Horizon Specialized Ensemble
- **Context:** Anomalies have diverse mathematical topologies: single-point spikes, step jumps, variance bursts, diurnal pattern violations, and slow drift. No single scalar distance can detect all types simultaneously without massive false alarms.
- **Decision:** Decompose detection into an ensemble of 7 specialized detectors (D1 to D7) operating on complementary time horizons ($w \in \{1, 3, 8, 24, \text{diurnal}\}$).
- **Empirical Impact:** Recall increased from $0.595 \to 0.759$ (88 of 116 ground-truth windows caught across the 58 streams).

### ADR-003: Event Debounce and Confirmation Protocol
- **Context:** Industrial sensors frequently experience high-frequency single-sample noise (e.g., bitflips or transient transmission spikes) that are non-actionable. Emitting immediate alarms inflates False Positives and destroys Precision.
- **Decision:** Dual-track alarm protocol:
  - **Standard Outliers ($1.0\times \le \text{ratio} < 1.5\times$ threshold):** Require $K=2$ consecutive anomalous points before confirming an alarm.
  - **Extreme Outlier Bypass ($\text{ratio} \ge 1.5\times$ threshold):** Immediately confirm on point 1 if the deviation is severe enough ($> 1.5\times$ threshold), preventing delay on critical spikes.
- **Empirical Impact:** Slashed False Positives by **43%** ($683 \to 389$) on single detector D1, boosting Event-F1 from $0.176 \to 0.240$.

### ADR-004: Multi-Detector Fusion Engine
- **Context:** Combining 6+ heterogeneous detector scores requires a unified decision rule that balances recall against false-alarm accumulation.
- **Decision:** Implement a modular `FusionEngine` supporting two distinct modes:
  1. **MAX Fusion:** Triggers if any detector exceeds its calibrated threshold (optimal for maximum recall; used for official NAB profile).
  2. **Consensus Voting ($k$-of-$n$):** Requires at least $k=2$ simultaneous detector alarms (or an extreme outlier bypass) to confirm an event (optimal for Event-F1).
- **Empirical Impact:** Consensus voting ($k=2$) dropped False Positives from **115 down to 67**, driving Event-F1 to **0.481** (phase 2; the final default uses MAX fusion over two well-calibrated detectors, see ADR-009/010).

### ADR-005: Dynamic Tail Thresholding via Extreme Value Theory (SPOT/DSPOT)
- **Context:** Static thresholds (e.g., $3\sigma$ or $6\sigma$) fail catastrophically across varied datasets (Gaussian server traffic vs heavy-tailed Twitter volume).
- **Decision:** Replace static thresholds with online Extreme Value Theory (GPD tail fit via Method of Moments). Calibrate detector exceedances to uniform tail probabilities $P(X > x) \le q$.
- **Empirical Impact:** Surged Event-F1 to **0.324** on initial deployment (surpassing Windowed Gaussian 0.302) and slashed False Positives from **860 down to 318** (a 63% reduction).

### ADR-006: Event Lifecycle Management (Gap Merging & Re-Baselining)
- **Context:** Multi-point anomaly windows produce oscillatory alarms. Under event-based metrics, each duplicate alarm outside a window start is counted as a false positive. Furthermore, permanent level shifts (e.g., a server cluster permanently handling double traffic) must eventually be recognized as the "new normal".
- **Decision:** Implement `EventManager`:
  - **Gap Merging ($W_{merge} = 16..64$):** Alarms separated by $\le W_{merge}$ normal points are merged into a single incident. Only the initial trigger point receives `is_event_start = True`.
  - **Post-Alarm Cooldown ($W_{cooldown} = 200..600$):** Suppresses new event starts during the immediate wake of an alarm.
  - **Sustained Re-Baselining ($W_{rebaseline} = 80$):** If an anomaly persists for $> 80$ points with a stable new level (confirmed by CUSUM), trigger `reset_baseline()` across all detectors.
- **Empirical Impact:** Increased Recall to **0.733** (85/116 windows) while suppressing alert storms.

### ADR-007: Structured Evidence & Explainability Schema
- **Context:** Black-box anomaly scores are non-actionable for Site Reliability Engineers (SREs).
- **Decision:** Generate an explanatory evidence dictionary for every anomalous point conforming to the contract:
  ```json
  {
    "type": "spike | level_shift | variance_change | pattern_break | sequence",
    "expected": 14093.6,
    "observed": 39197.0,
    "deviation_sigma": 7.9,
    "fired": ["LastW", "LaggedResidual", "Seasonal"],
    "top_detector": "LaggedResidual",
    "text": "Value 3.92e+04 is 7.9σ above expected 1.409e+04 at Sun 01:00: a deviation from the longer-horizon trend."
  }
  ```
- **Empirical Impact:** Seamlessly powers the live Streamlit replay dashboard and incident cards.

### ADR-008: Event-Aligned Continuous Scoring for Official NAB Calibration
- **Context:** The official NAB sweeper (`data/NAB/nab/sweeper.py`) optimizes a corpus-wide threshold on continuous floating-point scores. Emitting sustained high scores ($0.95$) across multi-point windows caused the NAB sweeper to penalize every subsequent point as a false positive, pushing net scores negative and causing the optimizer to select threshold 1.1 (yielding a null score of 0.00).
- **Decision:** Align the continuous scalar score strictly with event initiation:
  - **Event Starts (`is_event_start = True`):** Assigned peak calibrated score in $[0.60, 1.0]$.
  - **Sustained / Debounced Points:** Mapped strictly to $[0.30, 0.48]$.
  - **Normal Points:** Maintained strictly $\le 0.30$.
- **Empirical Impact:** NAB Standard Score went from **0.00 to 53.00** with the original defaults; the final NAB operating point reaches **60.58**, above Random Cut Forest (51.7), KNN CAD (58.0) and earthgecko Skyline (58.2).

---

## 4. Detailed Breakdown of Every Model & Detector

Sentinel's detection core consists of 7 specialized, lookahead-free detectors:

```
┌───────────────────────────────────────────────────────────────────────────────────────┐
│                              DETECTOR REGISTRY OVERVIEW                               │
├────┬──────────────────────────┬──────────────┬─────────────┬──────────────────────────────┤
│ ID │ Detector Name            │ Window Size  │ Complexity  │ Target Anomaly Topology      │
├────┼──────────────────────────┼──────────────┼─────────────┼──────────────────────────────┤
│ D1 │ RobustZ                  │ Probation/EW │ O(1)        │ Single-point isolated spikes │
│ D2 │ LastW                    │ w = 3        │ O(1)        │ Abrupt step discontinuities  │
│ D3 │ LaggedResidual           │ w=8, h=256   │ O(1)        │ Multi-point sequence shifts  │
│ D4 │ VarRatio                 │ w = 24       │ O(1)        │ Volatility & noise bursts    │
│ D5 │ Seasonal                 │ 24 buckets   │ O(1)        │ Contextual diurnal breaks    │
│ D6 │ CUSUM                    │ k = 0.5      │ O(1)        │ Persistent drift & level     │
│ D7 │ StreamingIsolationForest │ 10 trees     │ O(1)        │ Multi-feature cut isolation  │
└────┴──────────────────────────┴──────────────┴─────────────┴──────────────────────────────┘
```

---

### D1: `RobustZ` — Median Absolute Deviation Spike Detector
- **Purpose:** Detects high-magnitude isolated pulses and impulse spikes while being impervious to extreme outliers.
- **Mathematical Formulation:**
  $$\text{Raw Score} = \frac{|x_t - \text{median}|}{\text{scale} + \epsilon}, \quad \text{scale} = 1.4826 \times \text{MAD}$$
  where $\text{MAD} = \text{median}(|x_{warmup} - \text{median}(x_{warmup})|)$.
- **State & Data Structures:**
  - `median` (float), `scale` (float), `alpha` (float = 0.01).
- **Online Adaptation:**
  - On non-anomalous points, the baseline slowly adapts via exponential smoothing:
    $$\text{median} \leftarrow \text{median} + \alpha (x_t - \text{median})$$
  - On anomalous points, the update is bypassed to prevent baseline poisoning.
- **Computational Complexity:** Strictly $O(1)$ time, $O(1)$ memory.

---

### D2: `LastW` — Short-Horizon Step Jump Detector
- **Purpose:** Catches instantaneous step jumps and sudden discontinuities (ICLR 2026 Definition 5).
- **Mathematical Formulation:**
  $$\text{Raw Score} = \frac{|x_t - \bar{x}_{w}|}{\text{scale} + \epsilon}, \quad \bar{x}_w = \frac{1}{w}\sum_{i=1}^{w} x_{t-i}, \quad w = 3$$
- **State & Data Structures:**
  - Circular buffer `collections.deque(maxlen=3)` and scalar accumulator `running_sum`.
- **Online Adaptation:**
  - Upon arrival of $x_t$, subtract the exiting point from `running_sum` and add $x_t$.
- **Computational Complexity:** Strictly $O(1)$ time, $O(w)$ space (3 floats).

---

### D3: `LaggedResidual` — Multi-Horizon Sequence Anomaly Detector
- **Purpose:** Catches sustained sequence anomalies and prolonged drops that contaminate short-horizon windows (ICLR 2026 Section 6.1).
- **Mathematical Formulation:**
  $$\text{Raw Score} = \frac{|x_t - \bar{x}_{ref}|}{\text{scale} + \epsilon}, \quad \bar{x}_{ref} = \frac{1}{w}\sum_{i=h}^{h+w-1} x_{t-i}, \quad w = 16, \ h = 64$$
- **State & Data Structures:**
  - Two decoupled circular buffers: `delay_buffer = deque(maxlen=64)` and `ref_buffer = deque(maxlen=16)`, with scalar accumulator `ref_sum`.
- **Online Adaptation:**
  - Observations spend 64 steps passing through `delay_buffer`. The point exiting `delay_buffer` enters `ref_buffer`, maintaining an uncontaminated pre-anomaly reference frame.
- **Computational Complexity:** Strictly $O(1)$ time (zero list iterations), $O(w + h)$ space (80 floats).

---

### D4: `VarRatio` — Volatility & Chatter Detector
- **Purpose:** Detects sudden bursts in volatility, variance explosions, or sensor fluttering without requiring a change in mean (ICLR 2026 Definition 4).
- **Mathematical Formulation:**
  $$\sigma^2_{local} = \frac{1}{w}\sum x_i^2 - \left(\frac{1}{w}\sum x_i\right)^2, \quad \text{Raw Score} = \max\left(0, \sqrt{\frac{\sigma^2_{local} + \epsilon}{\sigma^2_{baseline} + \epsilon}} - 1\right), \quad w = 24$$
- **State & Data Structures:**
  - Circular buffer `deque(maxlen=24)`, `running_sum`, and `running_sumsq`.
- **Computational Complexity:** Strictly $O(1)$ time, $O(w)$ space (24 floats).

---

### D5: `Seasonal` — Contextual Diurnal Profile Detector
- **Purpose:** Catches contextual pattern violations (e.g., peak daytime traffic occurring at 3:00 AM).
- **Mathematical Formulation:**
  - 24 hourly buckets: $h \in \{0, 1, \dots, 23\}$, each tracking $(\mu_h, \sigma_h, n_h)$.
  $$\text{Raw Score} = \frac{|x_t - \mu_{t.hour}|}{\sigma_{t.hour} + \epsilon}$$
- **State & Data Structures:**
  - Array of 24 bucket structs initialized during probation.
- **Online Adaptation:**
  - Non-anomalous points update hourly profiles via online Welford exponential smoothing ($\alpha = 0.05$):
    $$\mu_h \leftarrow \mu_h + \alpha (x_t - \mu_h), \quad \sigma_h \leftarrow \sigma_h + \alpha (|x_t - \mu_h| - \sigma_h)$$
- **Computational Complexity:** Strictly $O(1)$ time, $O(1)$ space (24 buckets).

---

### D6: `CUSUM` — Two-Sided Cumulative Sum Drift Detector
- **Purpose:** Detects subtle persistent mean shifts and gradual trend drift.
- **Mathematical Formulation:**
  $$z_t = \frac{x_t - \text{median}}{\text{scale}}, \quad k = 0.5 \ (\text{slack factor})$$
  $$S^+_t = \begin{cases} \min(25.0, S^+_{t-1} + (z_t - k)), & \text{if } z_t > k \\ \max(0.0, S^+_{t-1} \cdot 0.85 + (z_t - k)), & \text{otherwise} \end{cases}$$
  $$S^-_t = \begin{cases} \min(25.0, S^-_{t-1} + (-z_t - k)), & \text{if } -z_t > k \\ \max(0.0, S^-_{t-1} \cdot 0.85 + (-z_t - k)), & \text{otherwise} \end{cases}$$
  $$\text{Raw Score} = \max(S^+_t, S^-_t)$$
- **State & Data Structures:**
  - Leaky accumulators $S^+$ and $S^-$ with bounded decay factors.
- **Computational Complexity:** Strictly $O(1)$ time, $O(1)$ space.

---

### D7: `StreamingIsolationForest` — Random Cut Tree Isolation Depth
- **Purpose:** Multi-dimensional isolation boundary detector evaluating boundary exceedances without batch tree re-fitting.
- **Mathematical Formulation:**
  - An ensemble of $T=10$ random cut trees of depth $D=5$ sampled across the probation standardized range $[z_{min}, z_{max}]$.
  - For streaming observation $z_t$, path length $d_i(z_t)$ to the isolating leaf is computed:
    $$\text{Raw Score} = 2.0 \times \left(D - \frac{1}{T}\sum_{i=1}^T d_i(z_t)\right) + \text{boundary\_excess}$$
- **Computational Complexity:** Strictly $O(1)$ time ($< 4\ \mu\text{s/pt}$), $O(T \cdot D)$ space (50 floats).

---

## 5. Empirical Benchmark & Ablation Study

Sentinel was extensively evaluated across all 58 time-series files (365,558 observations, 116 ground-truth anomaly windows) in the Numenta Anomaly Benchmark.

### 5.1 Component Ablation Progression
Final build-up (`python -m eval.ablation`, `docs/results/detector_ablation_results.md`; latency = mean µs/pt of that run):

| Phase / Configuration | Event-F1 | Precision | Recall | FP Events | Key Takeaway |
|---|---|---|---|---|---|
| **Windowed Gaussian (Baseline)** | 0.302 | 0.211 | 0.534 | 232 | Static sliding window baseline |
| Single RobustZ (static threshold, no event logic) | 0.040 | 0.020 | 0.810 | 4519 | Severe false-positive explosion |
| RobustZ + **SPOT (EVT)** | 0.376 | 0.373 | 0.379 | 74 | GPD tail fit replaces the heuristic cutoff |
| RobustZ + SPOT + **Event lifecycle** | 0.392 | 0.811 | 0.259 | 7 | Debounce / merge / cooldown remove duplicates |
| All 6 detectors, MAX fusion | 0.419 | 0.335 | 0.560 | 129 | More recall, many more FPs |
| All 6 detectors, vote k=2 | 0.456 | 0.397 | 0.534 | 94 | Voting trims FPs |
| **Per-detector SPOT risk**, RobustZ + LaggedResidual (ADR-009) | 0.563 | 0.675 | 0.483 | 27 | One global risk level had silenced the clean detectors |
| + Spatial-extreme flag (ADR-010) | 0.587 | 0.627 | 0.552 | 38 | New running extremes flagged on the first point |
| + Debounce 3, extreme ratio 2.0 | 0.610 | 0.681 | 0.552 | 30 | Fewer one-point false alarms |
| + Cooldown 0.5 × probation | 0.640 | 0.652 | 0.629 | 39 | Short streams no longer muted for 600 points |
| **+ LaggedResidual risk 1e-5 = previous default** | 0.673 | 0.747 | 0.612 | 24 | Prior setting |
| **+ lag 8/256 and margin 5.75% = current default** | **0.691** | **0.731** | **0.655** | **28** | **2.86 µs/pt** |
| NAB operating point (+ Seasonal, two-sided VarRatio, agreement bypass) | 0.489 | 0.366 | 0.733 | 147 | Trades F1 for recall; NAB 60.58 |

For the previous default, removing the spatial flag lowered F1 from 0.673 to 0.428. The current-default ablation is in `docs/results/detector_ablation_results.md`. Adding Seasonal, two-sided VarRatio
(K19), the auto-period Seasonal (K19) or the isolation forest (D7) to the default raises recall but lowers event-F1
(0.54–0.62), so they are not part of the event-F1 default. The NAB operating point uses Seasonal and VarRatio.

---

### 5.2 Official Published NAB Benchmark Leaderboard Comparison

Evaluated using the official Numenta scoring sweeper (published scores from `data/NAB/results/final_results.json`;
latencies measured on our machine with `python -m eval.nab_runtime` — NAB's own implementations, same streams; see
`docs/results/inference_runtime_comparison.md`):

| Model | NAB Standard | Reward Low FP | Reward Low FN | Execution Latency | Architecture Type |
|---|---|---|---|---|---|
| Perfect Detector | 100.0 | 100.0 | 100.0 | — | Oracle |
| ARTime | 74.9 | 65.1 | 80.4 | not runnable here (Julia) | Non-streaming Auto-regressive |
| Numenta HTM | 70.1 | 63.1 | 74.3 | not runnable here (nupic, Py2) | Hierarchical Temporal Memory |
| CAD OSE | 69.9 | 67.0 | 73.2 | 231.1 µs/pt (measured) | Conformal Anomaly Detector |
| **Sentinel (Ours — NAB operating point)** | **60.58** | **54.02** | **64.75** | **5.2 µs/pt (full-corpus run)** | **Strict $O(1)$ Causal Streaming** |
| earthgecko Skyline | 58.2 | 46.2 | 63.9 | 316.9 µs/pt (measured) | Multi-process heuristics |
| KNN CAD | 58.0 | 43.4 | 64.8 | 773.7 µs/pt (measured) | $O(N)$ Nearest Neighbors |
| **Sentinel (Ours — engine default)** | **58.80** | **57.53** | **61.04** | **2.7 µs/pt (same-stream timing)** | **Strict $O(1)$ Causal Streaming** |
| Relative Entropy | 54.6 | 47.6 | 58.8 | 585.3 µs/pt (measured) | Kernel Density Estimator |
| Random Cut Forest (AWS) | 51.7 | 38.4 | 59.7 | not runnable here (AWS service) | Batch Random Cut Trees |
| Twitter ADVec v1.0.0 | 47.1 | 33.6 | 53.5 | not runnable here (R) | Seasonal ESD |
| Windowed Gaussian (NAB Baseline) | 39.6 | 20.9 | 47.4 | 57.8 µs/pt (measured, NAB impl.) | Sliding Gaussian Window |
| Etsy Skyline | 35.7 | 27.1 | 44.5 | 2612.1 µs/pt (measured) | Heuristic voting |
| Bayesian Changepoint | 17.7 | 3.2 | 32.2 | 112.7 µs/pt (measured) | $O(N^2)$ Exact Bayesian |
| EXPoSE | 16.4 | 3.2 | 26.9 | 545.0 µs/pt (measured) | Kernel Embedding |

---

## 6. Verification & Test Suite

Sentinel maintains an exhaustive test suite guaranteeing mathematical and structural correctness:
- **Unit & Integration Tests:** 107 passed tests covering:
  - Synthetic point spikes, step jumps, variance bursts, sequence shifts, and diurnal breaks.
  - SPOT GPD tail probability accuracy and false alarm probability calibration.
  - Event debounce confirmation, gap merging, cooldown suppression, and re-baselining.
  - Strict $O(1)$ memory bound assertions (zero memory growth over 100,000 iterations).
  - Exact reproduction of official published NAB scoring numbers.
  - Input robustness fuzzing (NaN/inf, missing values, duplicate/backward timestamps, constant series).
  - Checkpoint round-trips (`Sentinel.save/load` resumes bit-identically) and the multi-stream pool / CLI.
  - Formal no-look-ahead property tests (output for a prefix never changes when data is appended) and determinism.
  - Recall mechanisms (spatial-extreme flag, proportional cooldown, agreement bypass, score smoothing).

---

## 7. Status of Earlier Next Steps

1. **Config default synchronization — done:** the tuned operating point is the built-in default (`sentinel/config.py`, ADR-009/010).
2. **Deep-merge configuration — done:** nested overrides merge; an empty dict clears a default.
3. **Horizon rescaling — tested, not adopted:** adaptive local scaling of LaggedResidual lowered F1 (0.453); the real fix was a per-detector SPOT risk level (ADR-009).
4. **Starter-pack validation — tooling ready:** `python -m eval.predict --input <folder>` runs any folder of CSVs and writes per-point predictions, events and a summary (reproduces F1 0.691 on the starter pack).

---
*Generated autonomously for the Hackathon Technical Checkpoint Review.*
