# Sentinel — presentation outline

Seven simple slides. Use a plain sans-serif font, a white background, dark text, and one muted accent colour. Keep figures large and text brief. Speaker notes are below each slide.

---

## Slide 1 — Sentinel

**A fast, explainable anomaly detector for streaming time series**

Lokesh K · Likhit G · Challenge 2

- Event-F1 **0.691** on 58 NAB files and 116 anomaly windows.
- Precision **0.731** · Recall **0.655** · **2.8 µs** per point.
- Official NAB score **58.80**.

> Speaker notes: All the numbers on this slide refer to the current default. Event-F1 counts alert events in labelled windows; the official NAB score uses NAB's own scoring rules.

---

## Slide 2 — The problem

- A stream gives us one value at a time. We must decide whether it signals an unusual event using only past values.
- Each alarm needs a score, a `NORMAL` or `ANOMALOUS` label, and a short explanation.
- The benchmark has **58 files**, **365,558 readings**, and **116 labelled anomaly windows**.
- The first 15% of each file, up to 750 readings, is used to learn what is usual. We do not raise or score alarms during this period.

> Speaker notes: NAB supplies labelled windows for evaluation. Sentinel does not use those labels while processing a stream. A correct alarm starts inside a labelled window; an alarm outside every window is a false alarm.

---

## Slide 3 — How Sentinel works

```text
New reading
   ↓
RobustZ + LaggedResidual
   ↓
Each score gets its own adaptive threshold
   ↓
Spatial-extreme check
   ↓
Event manager: confirm, merge, and space out alerts
   ↓
Score + label + explanation
```

- **RobustZ** checks whether the reading is far from the usual level. It starts with the median of the initial readings and also measures how much those readings vary.
- **LaggedResidual** compares the current reading with the **average of 8 readings from roughly 256 points earlier**. This helps spot changes that last for a while.
- **Spatial-extreme check** catches a new high or low beyond the range seen so far.
- **Event manager** turns repeated unusual points into one alert for an incident.

> Speaker notes: The LaggedResidual reference window is 257–264 points back once it is full. Each detector uses a threshold adapted from its own early scores. RobustZ's usual level moves slowly after ordinary readings, while flagged readings do not pull it towards an anomaly.

---

## Slide 4 — What the default detects

- **Isolated spike:** a value suddenly rises or falls far beyond its usual level.
- **Sustained change:** values remain different from what an older part of the stream would suggest.
- **New extreme:** a value moves beyond the previously seen range.
- The explanation names the signal that fired and shows the observed and expected values where available.

![Current default on NYC taxi data](results/figures/example_nyc_taxi.png)

> Speaker notes: The figure shows the current default on NYC taxi data. Green bands are labelled anomaly windows; red markers are event starts. It finds four of the five windows with no false event on this file. This is one example, so use the corpus result on the next slide for the overall comparison.

---

## Slide 5 — Results against other models

| Model | Event-F1 | False events | Runtime per point |
|---|---:|---:|---:|
| **Sentinel current default** | **0.691** | **28** | **2.8 µs** |
| CAD OSE | 0.676 | 64 | 231 µs |
| Numenta HTM | 0.527 | 152 | — |
| Windowed Gaussian baseline | 0.302 | 232 | 49 µs |

- On the full NAB corpus, Sentinel finds **76** anomaly windows, misses **40**, and creates **28** false events.
- Its event-F1 is above the compared models while using much less time per reading than the runnable ones shown here.

> Speaker notes: The other published detectors' event metrics come from their stored NAB scores at NAB's threshold; they did not use Sentinel's event post-processing. Runtime for runnable models was measured on the same machine. Numenta HTM could not be run locally, so no runtime is shown. The Windowed Gaussian row uses our event-F1 baseline and the locally measured NAB implementation runtime, so its two columns come from different implementations.

---

## Slide 6 — Where results vary

- Sentinel's F1 differs by data category. We show this rather than relying only on one corpus-wide number.
- Some changes in daily pattern remain hard to detect when values stay within the usual range.

![Event-F1 by category for the current default and Windowed Gaussian](results/figures/per_category_f1.png)

> Speaker notes: The chart compares the current default with Windowed Gaussian. Sentinel improves F1 in each category with anomaly windows shown here. The no-anomaly category is omitted from the chart; Sentinel has zero false events there.

---

## Slide 7 — Takeaway and limits

- **Two simple detectors**, adaptive thresholds, and careful alert grouping reach **0.691 event-F1** at about **2.8 µs per point**.
- Every alert has an explanation, so a person can check why it fired.
- The default was selected using public NAB data. Its performance on hidden data is still unknown.
- Pattern changes that stay within the normal value range are the clearest area for improvement.

> Speaker notes: Do not claim that public-corpus performance guarantees hidden-test performance. The result shows that a small causal detector can compete well against published NAB models while remaining fast and interpretable.
