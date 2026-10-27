# Sentinel — 10-slide presentation outline

Keep the slides simple: white background, dark text, one muted accent colour, and a plain sans-serif font. Show only Sentinel's current default when discussing our model. Compare it with other models where useful.

---

## Slide 1 — Sentinel

**Team April 28** · Lokesh K · Likhit G
**Track:** Challenge 2 — Adaptive Time-Series Anomaly Detection

**One-line solution:** Sentinel flags unusual events in a live stream using two simple statistical checks, adaptive thresholds, and one clear alert per event.

- Event-F1 **0.691** on the full NAB corpus.
- **2.8 µs** per point, with an explanation for each alert.

> Speaker notes: This result uses the current default. It is measured on the public NAB corpus; hidden-test performance is not yet known.

---

## Slide 2 — Problem and motivation

- A time series is a sequence of readings, such as CPU usage, traffic, or sales.
- We must flag unusual behaviour **as each reading arrives**, using only what came before it.
- Missed incidents matter; too many false alerts waste people's attention.
- Each result includes a score, a `NORMAL` or `ANOMALOUS` label, and a short explanation.

> Speaker notes: The main challenge is to catch genuine incidents quickly while keeping false alarms low. We score events rather than treating every unusual point as a separate incident.

---

## Slide 3 — Data and preparation

| NAB category | Files | Anomaly windows |
|---|---:|---:|
| Artificial | 11 | 6 |
| AWS CloudWatch | 17 | 30 |
| Ad Exchange | 6 | 14 |
| Known Cause | 7 | 19 |
| Traffic | 7 | 14 |
| Tweets | 10 | 33 |
| **Total** | **58** | **116** |

- **365,558 readings** in total. Tweet files contain the most anomaly windows (33); the artificial no-anomaly files contain none.
- We process each file in its original order. A few files have repeated or out-of-order timestamps.
- The first 15% of each file, up to 750 readings, is a learning period. No alarms are raised or scored there.

> Speaker notes: The labelled anomaly windows come from NAB and are used to evaluate results. Sentinel does not need those labels when it processes a stream. The learning period sets the initial usual level and thresholds.

---

## Slide 4 — System design

```text
Reading (time, value)
          ↓
Check and clean input
          ↓
Learning period complete?
          ↓
RobustZ + LaggedResidual → adaptive thresholds
          ↓
New-high / new-low check
          ↓
Event manager → score, label, explanation
```

- Every decision uses the current and earlier readings only.
- The event manager confirms unusual readings, joins nearby alarms, and spaces out repeated alerts.

> Speaker notes: Each detector creates a score. Its threshold decides whether that score is unusually high for the current series. The spatial check catches a value that breaks the range seen so far. The event manager returns one event start for an incident.

---

## Slide 5 — Core algorithm

| Part | Simple explanation |
|---|---|
| **RobustZ** | Checks how far the current value is from its usual level, while allowing for normal variation. |
| **LaggedResidual** | Compares the current value with the **average of 8 readings from roughly 256 points earlier**. |
| **Adaptive thresholds** | Each detector sets its own alarm level from the high scores seen in the learning period and updates it as normal data arrives. |
| **Spatial-extreme check** | Flags a new high or low beyond the range seen so far. |

- RobustZ starts with the **median** of the learning-period readings. The typical spread is measured separately, so a value is not called unusual merely because it is above the median.
- LaggedResidual helps detect changes that continue for a while.

> Speaker notes: The exact older window is 257–264 points back once full. The thresholds use Extreme Value Theory (SPOT). Keeping separate thresholds matters because the two detectors produce different kinds of scores.

---

## Slide 6 — Experiment history and tuning

**We measured each change before choosing the current default.**

1. Compared simple baselines and detector signals on NAB.
2. Searched detector combinations and tested lag and window lengths.
3. Tested the spatial-extreme margin and event grouping rules.
4. Selected one global default: **RobustZ + LaggedResidual**, an **8-point** older average, a **256-point** lag, and a **5.75%** spatial margin.

- Settings are global; we do not choose a different configuration for each file using its labels.
- During tuning, we rotated the category left out of selection to check whether settings transferred to other data types.

> Speaker notes: This slide describes the search process without presenting earlier Sentinel settings as separate products or comparing their scores. The final 0.691 result was measured on the public corpus after selection, so it is not a hidden-test result.

---

## Slide 7 — Evaluation method

- **True positive:** an alert starts inside a labelled anomaly window.
- **False positive:** an alert starts outside every labelled window.
- **False negative:** a labelled window has no alert start.
- On all 58 files: **76 true positives**, **28 false positives**, **40 false negatives**.

| Metric | Current default |
|---|---:|
| Precision | **0.731** |
| Recall | **0.655** |
| Event-F1 | **0.691** |
| Official NAB score | **58.80** |

> Speaker notes: Precision is the share of our alerts that are correct. Recall is the share of labelled windows we find. Event-F1 combines both. The official NAB score follows a different scoring rule. The current default scores 58.80 on that rule. Runtime is measured on one core without file-reading time.

---

## Slide 8 — Comparison with other models

| Model | Event-F1 | False events | Time per point |
|---|---:|---:|---:|
| **Sentinel current default** | **0.691** | **28** | **2.8 µs** |
| CAD OSE | 0.676 | 64 | 231 µs |
| Numenta HTM | 0.527 | 152 | — |
| Windowed Gaussian baseline | 0.302 | 232 | 49 µs |

![Current default compared with Windowed Gaussian by category](results/figures/per_category_f1.png)

> Speaker notes: Published models' event results come from stored NAB scores at NAB's threshold and do not use Sentinel's event post-processing. Runtimes for runnable models were measured on the same machine; Numenta HTM could not be run locally. Windowed Gaussian's F1 is our re-implemented baseline, while its runtime comes from the NAB implementation. The chart compares only the current Sentinel default with Windowed Gaussian.

---

## Slide 9 — Example alert and evidence

![Current default on NYC taxi data](results/figures/example_nyc_taxi.png)

**Example alert:** NYC taxi readings on 1 November 2014 rose to **25,115**, compared with an older-window average of about **2,602**. LaggedResidual flagged the difference.

- Green bands: labelled anomaly windows.
- Red dots: Sentinel event starts.
- On this file, the current default finds **4 of 5** windows with **0 false events**.

> Speaker notes: The example shows what a user sees: an event marker plus a plain explanation of the observed and expected values. This is one file; Slide 7 reports the full-corpus result.

---

## Slide 10 — Limits and next steps

- **40 of 116** labelled windows are still missed. Changes in daily patterns can be hard to spot when values remain inside the usual range.
- The default was selected and measured on public NAB data. Performance on hidden data is unknown.
- Future work: improve detection of pattern changes while keeping false alerts and runtime low.

**Conclusion:** A small, causal detector can produce clear alerts and reach **0.691 event-F1** at about **2.8 µs per point**.

> Speaker notes: Avoid claiming that this score guarantees hidden-test performance. The current system's strengths are simple operation, speed, and clear evidence; pattern changes are the main remaining weakness.
