# How Sentinel works, end to end

This document is for a technical reader who has not seen the code. It follows one point through `Sentinel.update()`
([sentinel/pipeline.py](../sentinel/pipeline.py)) and uses the parameter values in
[sentinel/config.py](../sentinel/config.py) (`DEFAULT_CONFIG`). Design decisions are linked to the ADRs in
[ARCHITECTURE_DECISIONS.md](ARCHITECTURE_DECISIONS.md). Terms are defined in the [glossary](#glossary). All numbers
were re-measured for this document (commands in [README.md](../README.md#reproducibility)).

## 1. Problem framing

- **Streaming.** Points arrive one at a time as `(timestamp, value)`. `Sentinel.update(t, x)` returns a result for
  that point immediately. The engine never sees future points.
- **Univariate.** Every series is processed alone, by a fresh `Sentinel`, in file order.
- **Per-point output** (`sentinel/types.py: Result`):
  - `score`: anomaly score in [0, 1]
  - `label`: `NORMAL` or `ANOMALOUS`
  - `is_event_start`: True only on the first point of a new anomaly event. Event-F1 is scored on these points.
  - `evidence`: a dict that explains the flag (filled when `label == "ANOMALOUS"`)
  - `detector_scores`: the calibrated score of each active detector
- **Causality** ([ADR-001](ARCHITECTURE_DECISIONS.md#adr-001-strict-streaming--causal-invariant)). Every statistic is
  computed from points that have already been seen. The only information about the future is the stream length
  `n_total`, which sets the probation length exactly as NAB does (§2). `tests/test_causality.py` checks this:
  results on a prefix of a stream are identical to the first part of the results on the full stream.

## 2. Data

- **Corpus:** the [Numenta Anomaly Benchmark](https://github.com/numenta/NAB) (NAB). It has 58 CSV files with
  `timestamp,value` columns, 365,558 points in total, across 7 categories:

  | Category | Files | Windows | Points |
  |---|---|---|---|
  | artificialNoAnomaly | 5 | 0 | 20,160 |
  | artificialWithAnomaly | 6 | 6 | 24,192 |
  | realAWSCloudwatch | 17 | 30 | 67,740 |
  | realAdExchange | 6 | 14 | 9,610 |
  | realKnownCause | 7 | 19 | 69,561 |
  | realTraffic | 7 | 14 | 15,664 |
  | realTweets | 10 | 33 | 158,631 |
  | **Total** | **58** | **116** | **365,558** |

- **Labels:** `labels/combined_windows.json` gives each file a list of anomaly windows `[start, end]` (timestamps).
  NAB centres each window on a labelled anomaly. The window widths add up to 10% of the file, split evenly between
  its windows. Widths range from 29 to 1,499 points (median 212) and from 2.3 h to 535 h (median 33 h).
  `combined_labels.json` (single points) is not used for scoring.
- **Sampling:** 48 files have a 5-minute interval, 7 hourly, 2 have 10 minutes and 1 has 30 minutes. Eight files have
  duplicate timestamps. `realKnownCause/machine_temperature_system_failure.csv` has one backward timestamp. All files
  are streamed in file order, as NAB does, never sorted.
- **Probation:** the first `min(floor(0.15·n), 750)` points of each file. Sentinel only learns during this part and
  never flags inside it. The scorer ignores it for both labels and predictions. When `n_total` is unknown (a live
  stream), probation is `config["probation"] = 300` points.

## 3. The pipeline, stage by stage

```
(t, x) ─► validate ─► probation? ──yes──► store; at the last probation point: fit baselines + SPOT ─► NORMAL, score 0
                          │ no
                          ▼
              detectors D1..D7 (raw scores)
                          ▼
              per-detector SPOT: z_q, effective threshold, calibrated score, alarm
                          ▼
              fusion (MAX or k-of-n)  +  spatial-extreme flag
                          ▼
              event manager (debounce, merge gap, cooldown, rebaseline)
                          ▼
              baseline update on NORMAL points · evidence on ANOMALOUS points · event-aligned score
                          ▼
              Result(timestamp, value, score, label, is_event_start, evidence, detector_scores)
```

### 3.1 Input validation ([sentinel/validate.py](../sentinel/validate.py))

The fast path covers clean input: a `pd.Timestamp` later than the previous one, and a finite Python float with
|x| ≤ 1e12. That input goes straight through with no extra work. Anything else goes through
`sanitize_observation`:

| Input | Handling |
|---|---|
| NaN, None, non-numeric value | replaced by the previous value (or the baseline median, or 0.0) |
| ±inf or \|x\| > 1e12 | clipped to ±1e12 (keeps the running sums of squares from overflowing) |
| missing or unparseable timestamp | previous timestamp + 60 s |
| duplicate or backward timestamp | kept as is (flagged, not reordered) |
| tz-aware timestamp mixed with tz-naive ones | aligned to the tz-ness of the previous timestamp |

The fast path and the slow path give identical results. This was checked on the whole realKnownCause category by
feeding the same data as strings and `numpy.float64`.

### 3.2 Probation and initialisation (`Sentinel._finish_probation`)

During probation each point is stored and the result is `NORMAL` with score 0. At the last probation point the
engine does the following:

1. **Baselines from the probation values**
   - RobustZ: median *m* and robust scale *s* = 1.4826 · MAD. If MAD < 1e-6 it falls back to the standard deviation,
     then to 1.0. LastW, LaggedResidual and CUSUM reuse *m* and *s*.
   - VarRatio: baseline variance = variance of the probation values.
   - Seasonal: a mean and standard deviation (floored at 0.1) for each hour-of-day bucket that has at least 2 values.
2. **Warm-up.** The probation values are replayed through every detector. This fills the sliding buffers and gives
   each detector a sample of its raw scores on normal data. The CUSUM sums are then reset to 0.
3. **SPOT initialisation.** Each detector's SPOT model is fitted on that detector's probation raw scores (§3.4).
4. **Spatial range.** The spatial-extreme flag is seeded with the probation min and max.

### 3.3 Detectors D1–D7 ([sentinel/detectors.py](../sentinel/detectors.py), [ADR-002](ARCHITECTURE_DECISIONS.md#adr-002-causal-multi-horizon-detector-ensemble-d1-to-d7))

All detectors are O(1) or O(w) per point and use running sums over `deque` buffers. *m* and *s* are the probation
median and robust scale.

| ID | Name | Raw score | Catches | Default |
|---|---|---|---|---|
| D1 | RobustZ | \|x − m\| / s. *m* follows slow drift on NORMAL points only: m ← m + 0.01·(x − m) | spikes, large level jumps | **on** |
| D2 | LastW | \|x − mean(last w)\| / s, w = 3 | sudden point jumps (ICLR "Last-w") | off |
| D3 | LaggedResidual | \|x − mean of the w = 16 values that ended h = 64 steps ago\| / s | sequence anomalies and shifts that would contaminate a short window (ICLR multi-horizon) | **on** |
| D4 | VarRatio | √(var_w / var_probation) − 1, w = 24 (needs ≥ 8 values). With `two_sided`, also 1/√ratio − 1 when variance collapses | volatility bursts, flatlines (ICLR "Var-w") | off |
| D5 | Seasonal | \|x − μ_hour\| / σ_hour per hour-of-day bucket (168 hour-of-week buckets if `weekly`). Profiles are EW-updated with α = 0.05 on NORMAL points; a bucket needs ≥ 3 values | broken daily patterns | off |
| D6 | CUSUM | z = (x − m)/s; s⁺ ← min(25, s⁺ + z − k) if z > k, else max(0, 0.85·s⁺ + z − k) (leaky); s⁻ is the mirror image; score = max(s⁺, s⁻), k = 0.5 | slow drift, persistent shifts | off |
| D7 | StreamingIsolationForest | 10 random "trees", each 5 sorted cut points over the probation z-range ± 20%. Score = 2·(5 − mean depth) + mean distance outside the outer cuts | boundary outliers (stretch) | off |

Base thresholds (`detector_thresholds`): RobustZ 6.0, LastW 2.0, LaggedResidual 2.0, VarRatio 1.5, Seasonal 5.0,
CUSUM 8.0, IsolationForest 6.0.

The default ensemble is **RobustZ + LaggedResidual**. Leave-one-category-out search picked it over all six
detectors. Adding Seasonal or VarRatio raises recall but costs more false positives than it gains (see
`docs/results/detector_ablation_results.md`, "added" rows).

LastW and LaggedResidual score absolute residuals, not the squared differences of the ICLR one-liners. A squared
residual is a monotone function of the absolute one, so thresholding gives the same decisions. The absolute value
keeps the scale linear for evidence.

`seasonal.auto_period` estimates an autocorrelation period during probation, but the scoring never uses it. The
option has no effect, so the ablation row "Seasonal (auto period, thr 40)" is really Seasonal at threshold 40.

RobustZ is always instantiated, even when it is not listed in `detectors`, because its median and scale are the
baseline for the other detectors and for evidence.

### 3.4 SPOT / EVT thresholds ([sentinel/spot.py](../sentinel/spot.py), [ADR-005](ARCHITECTURE_DECISIONS.md#adr-005-dynamic-tail-thresholding-via-extreme-value-theory-spotdspot), [ADR-009](ARCHITECTURE_DECISIONS.md#adr-009-per-detector-spot-risk-level-and-tuned-defaults))

Each detector gets its own SPOT model. SPOT stands for Streaming Peaks-Over-Threshold (Siffer et al., KDD 2017).

1. **Initial threshold t.** The 98% quantile of the detector's probation scores (95% if there are fewer than 100).
   The excesses are y = score − t for scores above t.
2. **GPD fit by method of moments.** Let m and v be the mean and variance of the excesses. Then
   γ = ½·(1 − m²/v) and σ = ½·m·(m²/v + 1). γ is clamped to [−0.5, 0.5] and σ to at least 1e-6.
3. **Alarm threshold for risk q.** With n scores seen and N_t excesses,
   **z_q = t + (σ/γ)·((q·n/N_t)^(−γ) − 1)**. When γ ≈ 0 this becomes z_q = t − σ·ln(q·n/N_t).
   z_q is the score that the fitted tail exceeds with probability q.
4. **Fallbacks.** If there are fewer than 10 probation scores or all are ~0: z_q = max(1.25·max, 3). If there are
   fewer than 10 excesses: z_q = max(99.9% quantile, 1.15·max).
5. **Risk q.** The global value is `spot.q` = 1e-6. Per-detector overrides (`spot.q_per_detector`):
   LaggedResidual 1e-5 (and LastW 1e-4 when it is enabled). Clean residual detectors need a looser q than noisy
   ones to fire at all. When it was introduced, per-detector risk took the RobustZ + LaggedResidual ensemble from event-F1 0.438 to 0.563 (re-measured; ADR-009).
6. **Interaction with the base threshold.** The alarm fires when **raw > max(z_q, base threshold)**. The
   *threshold ratio* raw / max(z_q, base) is used for fusion, the extreme bypass and the evidence. The base
   threshold works as a floor: RobustZ never fires below 6 robust σ, even when the probation tail is very light.
7. **Online updates.** Every post-probation score increments n. A score between t and z_q is added to the
   excesses, and the GPD and z_q are refit after every 5 new excesses (once there are at least 10). Scores above
   z_q are alarms and are *not* added, so anomalies do not inflate the tail.
8. **Calibrated score in [0, 1].** If raw ≤ t: 0.5·raw/t. Above t, take the tail probability
   p = (N_t/n)·(1 + γ·y/σ)^(−1/γ); the score is 0.5 + 0.5·min(1, ln p / ln q). This gives 0.5 at t and 1.0 at z_q.
   The score is relative to z_q, not to the effective threshold. A detector can therefore show 1.0 in
   `detector_scores` without alarming, when z_q is below its base threshold.

DSPOT (SPOT on the residual from a moving average) is implemented as `spot.dspot_depth > 0` and is off by default.

### 3.5 Fusion ([sentinel/fusion.py](../sentinel/fusion.py), [ADR-004](ARCHITECTURE_DECISIONS.md#adr-004-multi-detector-fusion-engine-sentinelfusionpy))

- The fused score is the **MAX** of the calibrated detector scores.
- **Decision** (default `fusion = "max"`, `vote_k = 1`): the point is ANOMALOUS if any detector alarms.
- **k-of-n vote** (`fusion = "vote"`): ANOMALOUS when at least k detectors alarm, or when the top threshold ratio is
  ≥ `extreme_ratio`. A single detector far past its threshold still counts.
- The **top detector** is the one with the highest threshold ratio. It decides the evidence type.

### 3.6 Spatial-extreme flag ([sentinel/recall.py](../sentinel/recall.py), [ADR-010](ARCHITECTURE_DECISIONS.md#adr-010-recall-mechanisms-spatial-extreme-flag-proportional-cooldown-confirmation))

This flag comes from Numenta's "spatial anomaly". The engine keeps the running min `lo` and max `hi` of all values
seen, seeded from probation. Once at least 100 values have been seen, a value with
**x > hi + 0.0575·(hi − lo)** or **x < lo − 0.0575·(hi − lo)** is flagged. The range then extends to include x.

A flagged point is ANOMALOUS and confirmed immediately: its ratio is raised to `extreme_ratio`, which bypasses
debounce. If no detector alarmed, the evidence names `SpatialExtreme` as the top detector and reports the previous
range. In the previous configuration (5% margin), removing this flag dropped event-F1 from 0.673 to 0.428.

### 3.7 Event manager ([sentinel/events.py](../sentinel/events.py), [ADR-003](ARCHITECTURE_DECISIONS.md#adr-003-event-debounce-and-confirmation-protocol), [ADR-006](ARCHITECTURE_DECISIONS.md#adr-006-event-lifecycle-management-gap-merging--re-baselining))

Event-F1 counts only the first event start in each window, and every start outside a window is a false positive. The
event manager turns per-point decisions into few, well-placed starts. Default parameters:

| Parameter | Value | Meaning |
|---|---|---|
| `debounce` | 3 | a new event needs 3 consecutive ANOMALOUS points ... |
| `extreme_ratio` | 2.0 | ... unless the top ratio is ≥ 2.0 (or the spatial flag fired): confirm on the first point |
| `events.agree_k` | 0 (off) | optional: confirm on the first point when ≥ k detectors alarm together |
| `events.merge_gap` | 64 | an open event ends after 64 consecutive NORMAL points; shorter gaps merge into the same event |
| `events.cooldown_frac` | 0.5 | after an event start, no new start for 0.5 × probation points (375 points on files with ≥ 5,000 points; 150 when `n_total` is unknown). It replaces the fixed `events.cooldown` = 600 |
| `events.rebaseline_after` | 80 | after 80 ANOMALOUS points in one event, accept a "new normal" |

**Rebaseline.** Every detector's `reset_baseline()` runs and the event is closed. LastW and LaggedResidual clear
their buffers, VarRatio takes the current window variance as its baseline, CUSUM resets its sums and the RobustZ
median jumps to the current value. AGENTS.md §4.5 asks for a CUSUM-confirmed shift. The implemented rule is simpler:
80 anomalous points inside one event.

**Baseline hygiene.** RobustZ's median and the Seasonal profiles update only on NORMAL points, so an ongoing anomaly
does not become the baseline.

### 3.8 Evidence ([sentinel/explain.py](../sentinel/explain.py), [ADR-007](ARCHITECTURE_DECISIONS.md#adr-007-structured-evidence--explainability-schema))

Every ANOMALOUS point carries the following evidence:

| Field | Meaning |
|---|---|
| `type` | from the top detector: RobustZ/LastW/SpatialExtreme → `spike`, LaggedResidual → `sequence`, VarRatio → `variance_change`, Seasonal → `pattern_break`, CUSUM → `level_shift`, IsolationForest → `multivariate_outlier` |
| `expected` | the top detector's expected value: RobustZ median, LastW window mean, LaggedResidual reference mean. Otherwise the RobustZ median |
| `observed` | the value |
| `deviation_sigma` | \|observed − expected\| / s, on the robust MAD scale from probation. It is a deviation of the value itself, even when the top detector measures variance or drift |
| `fired`, `top_detector` | detectors that alarmed (plus `SpatialExtreme`), and the one with the highest ratio |
| `detector_score`, `threshold_ratio` | top detector's raw score and raw / max(z_q, base). For a spatial-only flag: distance beyond the range / (0.0575·range) |
| `confidence` | 0.50 + 0.35·clip((ratio − 1)/2, 0, 1) + 0.08·min(3, #fired − 1), clipped to [0.50, 0.99] |
| `context` | spatial-only flags: the range `[range_low, range_high]` it broke |
| `text` | one human-readable sentence built from the fields above, with weekday and time |

### 3.9 Event-aligned output score ([ADR-008](ARCHITECTURE_DECISIONS.md#adr-008-event-aligned-continuous-scoring-for-official-nab-calibration))

The NAB scorer thresholds the per-point `score`, and every point above the threshold counts as a detection. A plateau
of high scores across an event would add a false positive for every extra point. So the score is aligned with event
starts:

- event start: max(0.6, min(1.0, fused score))
- ANOMALOUS but not a start: min(0.48, 0.45 · fused score)
- NORMAL: min(0.30, 0.30 · fused score)

Any threshold in (0.48, 0.6] therefore reproduces exactly the event starts. The NAB score of the per-point scores
(58.80) equals the NAB score of the 0/1 event starts (58.80).

## 4. Historical worked example: realKnownCause/nyc_taxi.csv

NYC taxi passenger counts in 30-minute buckets: 10,320 points and 5 labelled windows. Probation is
min(floor(0.15 · 10,320), 750) = **750** points, so the cooldown is 0.5 × 750 = **375** points. Output from the
previous default (`w=16, h=64`, 5% margin; run with the explicit config in §4.1):

**After probation.** RobustZ median 16,104 and scale 5,504.9. The spatial range is [1,769, 29,985]. SPOT fits:

| Detector | t (98% quantile) | γ | σ | excesses | q | z_q |
|---|---|---|---|---|---|---|
| RobustZ | 2.522 | −0.500 | 0.092 | 15 | 1e-6 | 2.704 |
| LaggedResidual | 3.080 | −0.082 | 0.114 | 15 | 1e-5 | 3.726 |

**The anomaly.** The first labelled window runs from 2014-10-30 15:30 to 2014-11-03 22:30 (points 5,839–6,045). It
covers the NYC Marathon on 2 Nov 2014. At point 5,954 (Sun 2014-11-02 01:00) the value jumps to 39,197 passengers
after 23,109 at 00:30. This is also the night clocks went back an hour, so that bucket likely holds two hours of
rides. Stage by stage:

| Stage | What happens at point 5,954 |
|---|---|
| validate | clean Timestamp and float: fast path |
| RobustZ | raw = \|39,197 − 18,768\| / 5,504.9 = **3.71**. z_q has adapted online to 2.716, but the effective threshold is max(2.716, 6.0) = **6.0**: no alarm |
| LaggedResidual | raw = **3.79**. Effective threshold max(4.063, 2.0) = **4.063**: no alarm |
| fusion | no detector fired: not anomalous by the detectors alone |
| spatial flag | the running range is [1,431, 30,373], so the tolerance is 0.05 × 28,942 = 1,447 and the bound is 31,820. 39,197 > 31,820: **flag**, ratio set to 2.0 |
| event manager | not in an event, cooldown 0, ratio 2.0 ≥ `extreme_ratio`: **event start** (no debounce wait) |
| score | max(0.6, fused) = **1.0** |
| scorer | the start lies inside window 1: **TP**, delay 115 points into a 207-point window (the labelled point is the window centre) |

The evidence JSON exactly as returned:

```json
{
 "type": "spike",
 "expected": 18768.361,
 "observed": 39197.0,
 "deviation_sigma": 3.71,
 "fired": ["SpatialExtreme"],
 "top_detector": "SpatialExtreme",
 "text": "Value 39197.00 is a new high (Sun 01:00): above the range seen so far [1431.00, 30373.00] by more than the tolerated margin (+8824.00; 3.7σ above expected 18768.36).",
 "threshold_ratio": 6.098,
 "confidence": 0.85,
 "context": {"range_low": 1431.0, "range_high": 30373.0, "exceedance_ratio": 6.098}
}
```

On this file the previous default makes exactly one event, a TP, and misses the other 4 windows (Thanksgiving,
Christmas, New Year, the January blizzard). Those are pattern breaks inside the normal range. The **NAB operating
point** (`results/best_nab_config.json`: adds Seasonal and two-sided VarRatio, agreement bypass) makes 9 events on
nyc_taxi. It finds the marathon at the same point (LaggedResidual + SpatialExtreme) and the blizzard window at
2015-01-28 19:00, but also 7 false positives at 03:30–04:00 night-time lows. This is the event-F1 vs NAB trade-off in
miniature.

With the **current** `w=8, h=256`, 5.75% default, the same file has four event starts, all inside labelled windows:
four TP, zero FP, one FN. The first event is at 2014-11-01 13:30 and is attributed to LaggedResidual. See the
regenerated [example plot](../results/figures/example_nyc_taxi.png).

### 4.1 Reproduce the example

```bash
.venv/bin/python -m eval.predict --input data/NAB/data/realKnownCause/nyc_taxi.csv --config '{"lagged_residual":{"w":16,"h":64},"spatial":{"tol":0.05}}' --out outputs/nyc-previous
.venv/bin/python -m eval.predict --input data/NAB/data/realKnownCause/nyc_taxi.csv --config default --out outputs/nyc-current
```

## 5. Glossary

| Term | Definition |
|---|---|
| **Event** | A run of ANOMALOUS points treated as one incident. Only its first point has `is_event_start = True` |
| **TP / FP / FN window** (AGENTS.md §5) | Per file, after removing probation. **TP**: a label window that contains ≥ 1 predicted event start (inclusive bounds; extra starts in the same window are ignored). **FP**: a predicted start outside every window. **FN**: a window with no predicted start |
| **Precision / recall / event-F1** | Micro-averaged over the corpus: P = TP/(TP+FP), R = TP/(TP+FN), F1 = 2PR/(P+R) = 2TP/(2TP+FP+FN). The primary metric |
| **Detection delay** | For TP windows: first start − window start, in points, minutes and as a fraction of the window width. NAB windows are centred on the labelled point, so a fraction of ~0.5 means detection at the label itself |
| **Probation** | The first min(floor(0.15n), 750) points: learning only, never flagged, not scored |
| **NAB score** | Numenta's benchmark score. Inside a window, a detection is rewarded by a sigmoid of its position (earlier is better). FPs and FNs are penalised by profile weights, the threshold is optimised over the corpus, and the result is normalised so null = 0 and perfect = 100. Profiles (`config/profiles.json`): **standard** (FP weight 0.11, FN 1.0), **reward_low_FP_rate** (FP 0.22), **reward_low_FN_rate** (FN 2.0) |
| **SPOT / DSPOT** | Streaming Peaks-Over-Threshold (Siffer et al., KDD 2017): fit a GPD to the tail of scores and set the alarm threshold for a target risk q. DSPOT does the same on residuals from a moving average |
| **GPD** | Generalized Pareto Distribution, the limit distribution of excesses over a high threshold (Pickands–Balkema–de Haan). Shape γ, scale σ |
| **Method of moments** | Estimating γ and σ from the sample mean and variance of the excesses (closed form, O(1) to update) |
| **Risk q** | The target probability that a normal score exceeds z_q. Smaller q gives a higher threshold and fewer alarms |
| **MAD** | Median absolute deviation, median(\|x − median(x)\|). 1.4826·MAD estimates σ for Gaussian data and is robust to outliers |
| **Robust z-score** | (x − median) / (1.4826·MAD) |
| **LaggedResidual / one-liner** | Zhu, Carpentier and Verbeke, *When Foundation Models are One-Liners* (ICLR 2026) show that time-series foundation models behave like simple baselines (Var-w, Last-w). Their §6.1 multi-horizon idea: a long-horizon reference (a window that ended h steps ago) is not contaminated by the anomaly the way a short window is. D2–D4 are streaming versions of these one-liners |
| **CUSUM** | Cumulative sum control chart. It accumulates standardised deviations beyond a slack k and catches small persistent shifts. Sentinel's version leaks (×0.85) when in control and caps at 25 |
| **Operating point** | One global config, i.e. one point on the precision/recall or F1/NAB trade-off. Sentinel ships two: the engine default (event-F1) and `results/best_nab_config.json` (NAB) |
| **LOCO validation** | Leave-one-category-out: for each of the 7 categories, choose the best config on the other 6 and score it on the held-out one. Reported as the pooled (TP/FP/FN summed) and mean held-out F1 |
| **Ablation** | Adding or removing one component at a time and measuring the change (`docs/results/detector_ablation_results.md`) |
| **Pareto front** | Configs that no other config beats on both event-F1 and NAB (`results/figures/pareto_f1_nab.png`) |
| **Spatial extreme** | A value outside the running [min, max] by more than 5.75% of the range (§3.6) |
| **Cooldown** | The number of points after an event start during which no new event may start |
| **Debounce** | The number of consecutive ANOMALOUS points needed before an event starts (unless bypassed) |
| **Extreme-ratio bypass** | A single detector at ≥ `extreme_ratio` × its threshold (or the spatial flag) confirms an event on the first point |
| **Look-ahead** | Using information from after time t to decide about t, e.g. normalising by full-file min/max or using centred windows. Sentinel uses none. The only exception is the probation length, which depends on `n_total`, exactly as in NAB |
