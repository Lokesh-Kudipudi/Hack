# Architecture Decision Records (ADRs) — Sentinel

This document captures the architectural rationale, technical trade-offs, and empirical benchmarks guiding the development of **Sentinel** for Hackathon Challenge 2: Adaptive Streaming Time-Series Anomaly Detection on the Numenta Anomaly Benchmark (NAB).

---

## ADR-001: Strict Streaming & Causal Invariant

- **Status:** Accepted & Implemented
- **Date:** 2026-10-26
- **Context:**
  Many academic time-series anomaly detectors (including Centered-W windowing, bidirectional RNNs, and batch Transformers) leak future information by centering windows around point $t$ or calculating stats using $[t - w/2, t + w/2]$. In real-time production pipelines, only data up to point $t$ is causally available. Runtime is a judged metric, so we set ourselves a budget of $< 50\ \mu\text{s/pt}$ — roughly the cost of NAB's windowedGaussian baseline on our machine (48.7 µs/pt). It is our own target, not an external requirement.
- **Decision:**
  1. Sentinel is strictly **causal**: all statistical estimators, rolling statistics, and anomaly scores at time $t$ depend solely on observations $x_\tau$ for $\tau \le t$.
  2. Every operation must execute in strictly **$O(1)$ time and $O(w)$ space** using fixed-size deques and entry/exit rolling sum buffers. No memory leaks, no unbounded lists, and no generator allocations in the inner loop.
- **Consequences:**
  - **Positive:** Zero data leakage; deterministic $O(1)$ memory footprint; fast inference: 2.4 µs/pt for the default (tuned) config, 5.5 µs/pt with all 6 detectors (180k–420k pts/sec on a single core; `python -m eval.benchmark`).
  - **Negative:** Lookahead-free detection makes sequence lag anomalies harder to confirm without historical buffering.

---

## ADR-002: Causal Multi-Horizon Detector Ensemble (D1 to D7)

- **Status:** Accepted & Implemented
- **Date:** 2026-10-26
- **Context:**
  Time-series anomalies manifest in fundamentally diverse mathematical topologies: single-point spikes, variance bursts, diurnal pattern violations, trend drift, and persistent level shifts. A single mathematical model (e.g., Gaussian Z-score) cannot detect all types without incurring massive false-positive inflation.
- **Decision:**
  Implement a decomposed ensemble of causal, specialized detectors:
  - **D1 (`RobustZ`):** Median Absolute Deviation (MAD) scaled Z-score for isolated pulse spikes.
  - **D2 (`LastW`):** Short-horizon ($w=3$) squared deviation for abrupt step jumps (ICLR 2026 baseline).
  - **D3 (`LaggedResidual`):** Long-horizon ($w=16, h=64$) residual against past historical window for sequence anomalies.
  - **D4 (`VarRatio`):** Rolling variance ratio ($w=24$) against baseline for volatility regime shifts.
  - **D5 (`Seasonal`):** 24-hour diurnal profile buckets with exponential smoothing for contextual pattern breaks.
  - **D6 (`CUSUM`):** Two-sided Cumulative Sum statistic on standardized residuals for persistent mean drift.
  - **D7 (`StreamingIsolationForest`):** Fast rolling random cut trees evaluating boundary isolation depth in $O(1)$ time ($< 4\ \mu\text{s/pt}$).
- **Consequences:**
  - **Positive:** Diverse coverage across all NAB anomaly categories; recall jumped from $0.595 \to 0.759$ (88 of 116 windows caught across 58 streams).
  - **Negative:** Requires threshold calibration across heterogeneous distribution types.

---

## ADR-003: Event Debounce and Confirmation Protocol

- **Status:** Accepted & Implemented
- **Date:** 2026-10-26
- **Context:**
  High-frequency noisy sensors often emit transient single-sample outliers that are non-actionable. Triggering immediate alarms on single outliers inflates False Positives (FPs) and severely degrades Event Precision.
- **Decision:**
  Implement a dual-track confirmation protocol:
  1. **Standard Outliers ($1.0\times \le \text{ratio} < 1.5\times$ threshold):** Require $K=2$ consecutive anomaly points before confirming an alarm.
  2. **Extreme Outlier Bypass ($\text{ratio} \ge 1.5\times$ threshold):** Immediately confirm on point 1 if the deviation is severe enough ($> 1.5\times$ threshold), preventing critical delay on massive sudden spikes.
- **Consequences:**
  - **Empirical Impact:** Dropped False Positives nearly in half ($683 \to 389$), while Event-F1 jumped from $0.176 \to 0.240$ (+36.4% gain) on single detector D1.

---

## ADR-004: Multi-Detector Fusion Engine (`sentinel/fusion.py`)

- **Status:** Accepted & Implemented
- **Date:** 2026-10-26
- **Context:**
  With 6+ detectors running concurrently, the pipeline needs a unified decision rule to combine normalized scores and alarms without suffering from voter paralysis or false-alarm accumulation.
- **Decision:**
  1. Default to **MAX-Score Fusion** with Extreme Value Theory calibration: the ensemble score equals the maximum of the calibrated detector probabilities.
  2. Support configurable **$k$-of-$n$ consensus voting** for high-noise streams, requiring at least $k$ simultaneous alarms unless an extreme outlier bypass triggers.
- **Consequences:**
  - **Positive:** High recall on diverse anomaly types; preserves sensitivity to rare specialized anomalies (e.g. diurnal breaks detected only by D5).

---

## ADR-005: Dynamic Tail Thresholding via Extreme Value Theory (SPOT/DSPOT)

- **Status:** Accepted & Implemented
- **Date:** 2026-10-26
- **Context:**
  Fixed static thresholds (e.g. $3\sigma$ or $6\sigma$) fail catastrophically across real-world datasets: a $6\sigma$ threshold that works on Gaussian traffic is completely overwhelmed on heavy-tailed Twitter volume series.
- **Decision:**
  Implement **Streaming Peak-Over-Threshold (SPOT)** based on the Generalized Pareto Distribution (GPD):
  1. **Probation Calibration:** Fit GPD parameters $(\gamma, \sigma)$ via Method of Moments on the upper tail excesses above high quantile $t$ (e.g., 98th percentile).
  2. **Dynamic Extreme Quantile:** Compute tail threshold $z_q$ for target risk level $q = 10^{-4}$.
  3. **Streaming Adaptation:** Excesses between $t$ and $z_q$ update GPD tail statistics, while alarms ($x > z_q$) bypass to avoid baseline pollution.
  4. **DSPOT Extension:** Subtract local moving median to eliminate local trend drift prior to tail evaluation.
- **Consequences:**
  - **Empirical Impact:** **Surged Event-F1 to 0.324** (securing **#1 on the Repo Leaderboard**, beating `windowed_gaussian` 0.302). False Positives plunged from **860 down to 318** (63% reduction) with $16.37\ \mu\text{s/pt}$ latency.

---

## ADR-006: Event Lifecycle Management (Gap Merging & Re-Baselining)

- **Status:** Accepted & Implemented
- **Date:** 2026-10-26
- **Context:**
  When an anomaly occurs, it often spans multiple successive points or produces oscillatory alarms. Emitting multiple alarms for the same physical event overwhelms downstream operators and incurs multiple false-alarm penalties under NAB scoring. Furthermore, persistent structural level shifts (e.g., server load doubling permanently) must eventually be accepted as a "new normal".
- **Decision:**
  Implement `EventManager`:
  1. **Debounce Confirmation:** As established in ADR-003.
  2. **Merge Gap ($W_{merge} = 16$):** Group alarms separated by $\le 16$ normal points into a single continuous event. Only the initial trigger point receives `is_event_start = True`.
  3. **Cooldown ($W_{cooldown} = 200$):** Suppress new event alarms during the immediate cooldown window post-event.
  4. **Level-Shift Adaptation ($W_{rebaseline} = 80$):** If an anomaly persists for $> 80$ points with stable new mean confirmed by CUSUM, call `reset_baseline()` across all detectors to adapt to the new regime.
- **Consequences:**
  - **Empirical Impact:** Recall increased to **0.733** (85/116 windows), while preventing alert storms.

---

## ADR-007: Structured Evidence & Explainability Schema

- **Status:** Accepted & Implemented
- **Date:** 2026-10-26
- **Context:**
  In mission-critical operations, black-box anomaly flags ("Anomaly detected at index 4210") are useless to site reliability engineers (SREs). Operators require causal attribution, expected normal values, and natural-language diagnostics.
- **Decision:**
  Implement `ExplainEngine` adhering to the AGENTS.md §4.6 schema for every anomalous point:
  ```json
  {
    "type": "spike | level_shift | variance_change | pattern_break | sequence | multivariate_outlier",
    "expected": 74.2,
    "observed": 91.8,
    "deviation_sigma": 4.1,
    "fired": ["RobustZ", "CUSUM"],
    "top_detector": "RobustZ",
    "text": "Value 91.8 is 4.1σ above expected 74.2; CUSUM confirms persistent upward level shift."
  }
  ```
- **Consequences:**
  - **Positive:** Full operational transparency; feeds directly into the Streamlit interactive diagnostic incident cards.

---

## ADR-008: Event-Aligned Continuous Scoring for Official NAB Calibration

- **Status:** Accepted & Implemented
- **Date:** 2026-10-26
- **Context:**
  The official Numenta Anomaly Benchmark sweeper (`data/NAB/nab/sweeper.py`) evaluates continuous floating-point scores across the entire corpus. In earlier iterations, continuous plateau scores on multi-point anomaly windows caused NAB's threshold optimizer to count each subsequent point as an individual false positive, pushing the net score negative and forcing the optimizer to pick threshold 1.1 (yielding a null score of 0.00).
- **Decision:**
  Align the continuous scalar score strictly with event initiation:
  - **Event Starts (`is_event_start = True`):** Assigned the peak calibrated score in $[0.60, 1.0]$ based on GPD tail probability.
  - **Sustained / Debounced Points:** Mapped strictly below the 0.50 threshold ($[0.30, 0.48]$).
  - **Normal Points:** Maintained strictly $\le 0.30$.
- **Consequences:**
  - **Empirical Impact:** **Official NAB Standard Score skyrocketed from 0.00 to 53.00** (beating Random Cut Forest 51.7, Twitter ADVec 47.1, Windowed Gaussian 39.6, and Etsy Skyline 35.7).

---

## ADR-009: Per-Detector SPOT Risk Level and Tuned Defaults

- **Status:** Accepted & Implemented
- **Date:** 2026-10-26
- **Context:**
  With one global SPOT risk level (`spot.q`), the short- and long-horizon residual detectors (LastW, LaggedResidual) almost
  never fired: alone they reached precision ≈ 0.9 but recall ≈ 0.15. The noisy detectors (VarRatio, Seasonal) need a strict
  q, the clean residual detectors a looser one. Rescaling the detectors (adaptive local scale) did not help (F1 0.453).
- **Decision:**
  1. `spot.q_per_detector` sets SPOT's risk level per detector; missing names fall back to `spot.q`.
  2. Nested config overrides deep-merge with the defaults (a partial `{"spot": {"q": ...}}` keeps the other keys).
  3. The default config is the event-F1 operating point chosen by leave-one-category-out search over 720 global configs:
     RobustZ + LaggedResidual, MAX fusion, q = 1e-6 (1e-4 for LaggedResidual), debounce 2, cooldown 600, merge gap 64.
     The original K1–K8 defaults are kept as `ORIGINAL_DEFAULT_CONFIG`.
- **Consequences:**
  - **Empirical Impact:** event-F1 0.311 (original defaults) → **0.563** (P 0.675, R 0.483, FP 27); held-out pooled 0.510.
    Removing per-detector q from that config drops it to 0.400.
  - Event-F1 and the NAB score favour different settings (`results/figures/pareto_f1_nab.png`); the NAB operating point
    (`results/best_nab_config.json`, NAB 59.06) uses all 6 detectors with 2-of-n voting.
  - Recall (0.48) and detection delay (~50% into the window) remain the main weaknesses.

---

## ADR-010: Recall Mechanisms (Spatial-Extreme Flag, Proportional Cooldown, Confirmation)

- **Status:** Accepted & Implemented
- **Date:** 2026-10-26
- **Context:**
  After ADR-009 the engine missed 60 of 116 windows (recall 0.48). The miss analysis (`results/miss_analysis.md`) showed
  27% of misses were *suppressed* (flagged, but cooldown / an earlier event / debounce blocked a new event) and most others
  had only weak or pattern-only signal. Detection "delay" turned out to be detection at the labelled anomaly point
  (median +0 points) — NAB windows start well before the label — so the lever is recall, not earlier firing.
- **Decision** (`sentinel/recall.py`, `sentinel/events.py`, config keys; each validated with leave-one-category-out search):
  1. **Spatial-extreme flag** (Numenta): a value beyond the running min/max by more than 5% of the range is anomalous on its
     first point, confirmed without debounce. Seeded from probation, active only after 100 observed values.
  2. **Proportional cooldown** (CAD OSE "rest period"): `events.cooldown_frac` = 0.5 × probation length instead of a fixed
     600 points, so short streams are not muted for most of their length.
  3. **Confirmation**: debounce 3, single-detector bypass at 2.0× threshold (`extreme_ratio`), LaggedResidual risk 1e-5.
  4. Also implemented and evaluated, **off by default** because they did not help: score smoothing before SPOT
     (`smoothing.window`), 2-detector agreement bypass (`events.agree_k`).
- **Consequences:**
  - **Empirical Impact:** event-F1 0.563 → **0.673** (P 0.747, R 0.612, FP 24), NAB 42.8 → 54.9, held-out pooled F1
    0.510 → **0.654**; 2.6 µs/pt. Removing the spatial flag costs the most (→ 0.428).
  - NAB operating point (`results/best_nab_config.json`): NAB **60.00** (adds Seasonal + agreement bypass), F1 0.541.
  - Remaining misses: 30 pattern breaks only Seasonal detects (adding it costs ~90 FPs), 12 with no detector signal.

