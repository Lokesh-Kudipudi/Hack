# Sentinel configuration tuning

Updated 2026-10-27. The best combined lag and spatial-margin setting is now the engine default in `sentinel/config.py`.

## Data and scoring

All recent sweeps used the full public NAB corpus: 58 files, 365,558 readings, seven categories, and 116 labelled anomaly windows. Event-F1, precision, and recall are pooled across files using the event definitions in [AGENTS.md](../AGENTS.md#5-evaluation-protocol-eval--fixed-definitions). Probation points are excluded from scoring. Each sweep changed global settings only; no detector was given ground-truth windows while processing a series.

The full-corpus scores below are **in-sample for configuration selection** because the settings were compared on this public corpus. Leave-one-category-out (LOCO) numbers select a setting on six categories and score the seventh, rotating through all seven. A LOCO score for a sweep is a score for that **selection procedure**, not a direct hidden-test score for one fixed setting. The hidden evaluation has not been run.

## Current default and previous setting

| Setting | Event-F1 | Precision | Recall | TP | FP | FN | NAB standard |
|---|---:|---:|---:|---:|---:|---:|---:|
| Previous engine default (`w=16, h=64`, margin 5%) | 0.673 | 0.747 | 0.612 | 71 | **24** | 45 | 54.90 |
| Best lag change only: `w=8, h=256` | 0.682 | 0.710 | **0.655** | **76** | 31 | **40** | 58.65 |
| Best spatial margin change only: 5.75% | 0.683 | **0.772** | 0.612 | 71 | **21** | 45 | 55.04 |
| **Current engine default: `w=8, h=256`, margin 5.75%** | **0.691** | 0.731 | **0.655** | **76** | 28 | **40** | **58.80** |

The new default gains 0.018 event-F1 and finds five more windows than the previous setting, with four more false-positive events. A fresh no-override run confirmed F1 0.691, precision 0.731, recall 0.655 and NAB 58.80 at 2.86 µs per point. The 16-setting LOCO **selection procedure** scored **0.664 pooled / 0.590 mean**, while the older 432-setting procedure scored **0.654 pooled / 0.593 mean**. These are results of different searches, not held-out scores for either fixed configuration.

The current default is also expressible as an override on the previous engine revision:

```json
{
  "lagged_residual": {"w": 8, "h": 256},
  "spatial": {"enabled": true, "tol": 0.0575}
}
```

Every other key is unchanged. Reproduce the result with the current defaults:

```bash
python -m eval.run --detector sentinel --tag "engine default (final)"
```

Saved full evaluations: [previous default](../results/runs/20261027-105601_sentinel_engine-default-final.json), [margin 5.75% alone](../results/runs/20261027-121105_sentinel_spatial-margin5p75-w16-h64.json), [best combined setting before adoption](../results/runs/20261027-121028_sentinel_joint-lag-spatial-w8-h256-margin5p75.json), and [new no-override default](../results/runs/20261027-122246_sentinel_engine-default-final.json). The separate margin-only run recorded an unusually high update time under transient machine load; its event metrics reproduced the sweep exactly, but that latency is not used in the comparison above.

### What the current default actually runs

- **Detectors:** `RobustZ` and `LaggedResidual`; fusion is MAX. The separate spatial-extreme flag is enabled.
- **LaggedResidual:** mean of 8 older points (`w=8`) with a 256-point delay (`h=256`).
- **RobustZ baseline:** probation median and 1.4826 × median absolute deviation (MAD); its center then moves slowly on non-anomalous points.
- **Thresholds:** SPOT enabled; `q=1e-6` generally and `q=1e-5` for LaggedResidual. The fixed floors are 6 for RobustZ and 2 for LaggedResidual.
- **Spatial extreme:** an immediate flag beyond the running minimum or maximum by 5.75% of the observed range, after at least 100 seed points.
- **Events:** debounce 3, immediate extreme bypass at a 2× threshold ratio, merge gap 64, and cooldown of half the probation length.

The prior 432-configuration `recall2` grid selected the previous default: **event-F1 0.673**, LOCO **0.654 pooled / 0.593 mean**. See [hyperparameter_tuning.md](results/hyperparameter_tuning.md) and [the saved tuning run](../results/runs/20261027-104941_tune_loco-recall2.json). The separate NAB-focused operating point remains **NAB 60.58, event-F1 0.489**; it optimizes a different metric and is not the event-F1 default.

## Tuning history before the recent sweeps

The project evaluated 3,288 grid points across six searches, on successive code revisions. Counts include repeated parameter combinations across grids, so they are not 3,288 distinct detector designs.

| Grid | Configurations | Main purpose |
|---|---:|---|
| `default` | 648 | Detector sets, fusion, SPOT risk, thresholds, and event settings |
| `perdet` | 432 | Separate SPOT risks and floors for individual detectors |
| `subsets` | 720 | Selected smaller detector groups with per-detector risk |
| `joint` | 576 | Event-F1 subject to a NAB-score floor |
| `recall` | 480 | Spatial flag, debounce, cooldown, and recall controls |
| `recall2` | 432 | Refinement that produced the previous default |

These grids and their options are defined in [eval/tune.py](../eval/tune.py); the final default's ablations are in [docs/results/detector_ablation_results.md](results/detector_ablation_results.md).

## Recent sweep 1: all detector subsets

We evaluated all **127 non-empty named subsets** of seven available detector options with every other current setting fixed. Sentinel automatically inserts RobustZ when omitted, so these represent **64 distinct effective combinations**. The current RobustZ + LaggedResidual pair remained the highest event-F1 configuration.

| Effective detectors | Event-F1 | Precision | Recall | TP | FP |
|---|---:|---:|---:|---:|---:|
| **RobustZ + LaggedResidual** | **0.673** | 0.747 | 0.612 | 71 | 24 |
| RobustZ only | 0.667 | **0.758** | 0.595 | 69 | **22** |
| RobustZ + LastW + LaggedResidual | 0.649 | 0.670 | **0.629** | **73** | 36 |
| RobustZ + LastW | 0.640 | 0.670 | 0.612 | 71 | 35 |
| RobustZ + LaggedResidual + CUSUM | 0.600 | 0.605 | 0.595 | 69 | 45 |

The 127-subset LOCO selection scored **0.642 pooled / 0.584 mean**. All 127 requested rows, including the duplicate effective configurations, are in [the detector-subset sweep](../results/runs/20261027-121431_tune_detector_subsets.json).

## Recent sweep 2: LaggedResidual window and lag

The first grid tested 25 combinations: `w = 4, 8, 16, 32, 64` and `h = 16, 32, 64, 128, 256`. A wider grid then tested **132 combinations**:

- `w = 2, 4, 6, 8, 10, 12, 16, 24, 32, 48, 64`
- `h = 8, 16, 24, 32, 48, 64, 96, 128, 192, 256, 384, 512`

All other default settings, including the 5% spatial margin, stayed fixed.

| Window `w` | Lag `h` | Event-F1 | Precision | Recall | TP | FP | NAB standard |
|---:|---:|---:|---:|---:|---:|---:|---:|
| **8** | **256** | **0.682** | 0.710 | **0.655** | **76** | 31 | **58.65** |
| **16** | **64** (previous default) | 0.673 | 0.747 | 0.612 | 71 | **24** | 54.90 |
| 16 | 96 | 0.670 | **0.753** | 0.603 | 70 | **23** | 54.21 |
| 10 | 96 | 0.667 | 0.732 | 0.612 | 71 | 26 | 54.88 |
| 8 | 128 | 0.667 | 0.745 | 0.603 | 70 | 24 | 54.56 |
| 4 | 256 | 0.664 | 0.702 | 0.629 | 73 | 31 | 56.41 |

The 25-setting LOCO search scored **0.648 pooled / 0.577 mean**. The 132-setting search scored **0.651 pooled / 0.579 mean**. A longer lag alone improved full-corpus recall but did not clearly improve the held-out result. Every setting in the wider grid is in [the lag sweep](../results/runs/20261027-121431_tune_lagged_residual.json).

## Recent sweep 3: RobustZ center and spread

These were **experimental in-memory variants**, not selectable `sentinel/config.py` options. Only RobustZ's probation center and spread changed. LaggedResidual's scale was held at the original median/MAD value, and the online update rule stayed the same. Variance describes spread rather than a normal level, so the variance-based option uses standard deviation. For continuous readings, the mode trial used 20 bins over the 1st–99th percentile range.

| Probation center + spread | Event-F1 | Precision | Recall | TP | FP | NAB standard |
|---|---:|---:|---:|---:|---:|---:|
| **Median + MAD (previous lag/margin setting)** | 0.673 | 0.747 | 0.612 | 71 | 24 | 54.90 |
| Median + standard deviation | **0.676** | **0.755** | 0.612 | 71 | **23** | **54.92** |
| Mean + standard deviation | 0.639 | 0.680 | 0.603 | 70 | 33 | 53.66 |
| Mean + MAD | 0.633 | 0.676 | 0.595 | 69 | 33 | 52.87 |
| 10% trimmed mean + MAD | 0.651 | 0.696 | 0.612 | 71 | 31 | 54.59 |
| Binned mode + MAD | 0.634 | 0.657 | 0.612 | 71 | 37 | 54.31 |

The median + standard deviation variant removed **one false positive** with no recall change. That small gain has not been implemented as an engine option or tested jointly with the later lag/margin candidates. The six-variant LOCO selection scored **0.673 pooled / 0.608 mean**. See [the RobustZ statistics sweep](../results/runs/20261027-121511_tune_robustz_statistics.json).

## Recent sweep 4: spatial-extreme margin

The spatial-extreme flag uses the running minimum and maximum, **not a fixed time window**. We tested 39 distinct settings: disabled, 0%–100% margins on a broad grid, and 0.25-percentage-point steps from 3% through 8%. Only the margin changed; LaggedResidual stayed at `w=16, h=64` (the previous default).

| Margin | Event-F1 | Precision | Recall | TP | FP | NAB standard |
|---:|---:|---:|---:|---:|---:|---:|
| Off | 0.428 | 0.791 | 0.293 | 34 | **9** | 29.38 |
| 0% | 0.547 | 0.469 | **0.655** | 76 | 86 | 56.48 |
| 1% | 0.644 | 0.633 | **0.655** | 76 | 44 | 58.06 |
| 2% | 0.649 | 0.670 | 0.629 | 73 | 36 | 56.07 |
| 2.5% | 0.661 | 0.695 | 0.629 | 73 | 32 | 56.21 |
| 5% (previous default) | 0.673 | 0.747 | 0.612 | 71 | 24 | 54.90 |
| **5.75%** | **0.683** | 0.772 | 0.612 | 71 | 21 | 55.04 |
| 7.5% | 0.667 | 0.788 | 0.578 | 67 | 18 | 51.97 |
| 10% | 0.657 | 0.793 | 0.560 | 65 | 17 | 50.74 |
| 20% | 0.624 | **0.829** | 0.500 | 58 | 12 | 45.44 |
| 100% | 0.454 | 0.787 | 0.319 | 37 | 10 | 29.38 |

At 5.75%, the same 71 windows were found as at 5%, with **three fewer false-positive events**. The 39-setting LOCO selection scored **0.673 pooled / 0.605 mean**. Every tested margin and its per-category metrics are in [the spatial-margin sweep](../results/runs/20261027-121431_tune_spatial_margin.json).

## Recent sweep 5: combine the strongest lag and margin settings

We tested **16 combinations** from `(w,h) = (16,64), (8,256), (16,96), (8,128)` and spatial margins `5%, 5.5%, 5.75%, 6%`. All other defaults stayed fixed.

| Lag window and delay | Margin | Event-F1 | Precision | Recall | TP | FP | NAB standard |
|---|---:|---:|---:|---:|---:|---:|---:|
| **8 / 256** | **5.75%** | **0.691** | 0.731 | **0.655** | **76** | 28 | **58.80** |
| 8 / 256 | 6% | 0.685 | 0.728 | 0.647 | 75 | 28 | 57.99 |
| 16 / 64 | 5.75% | 0.683 | 0.772 | 0.612 | 71 | 21 | 55.04 |
| 8 / 256 | 5% | 0.682 | 0.710 | **0.655** | **76** | 31 | 58.65 |
| 16 / 96 | 5.75% | 0.680 | **0.778** | 0.603 | 70 | **20** | 54.36 |

The 16-setting LOCO selection scored **0.664 pooled / 0.590 mean**. This was the highest full-corpus event-F1 configuration in the lag/margin grids and is now the current default. All 16 rows are in [the joint lag/margin sweep](../results/runs/20261027-121431_tune_joint_lag_spatial.json).

## Interpretation

- The **previous default** scored event-F1 **0.673**, precision **0.747**, recall **0.612**, with **24** false-positive events.
- The **current default** is `w=8, h=256` plus a **5.75% spatial margin**: event-F1 **0.691**, precision **0.731**, recall **0.655**, with **28** false-positive events.
- The **best standalone margin** is 5.75%; it improves precision without losing recall relative to the previous 5% setting.
- The RobustZ median + standard deviation prototype is a much smaller change: one fewer false positive. It is not part of the best combined configuration.
- A new single-setting ablation after adoption scored **0.697** (P 0.745, R 0.655, FP 26) when per-detector SPOT risk was removed from the 0.691 default. It has not gone through the joint-grid validation; it is recorded in [docs/results/detector_ablation_results.md](results/detector_ablation_results.md) as a follow-up candidate.
- These results are on the same public corpus used for tuning. LOCO reduces optimism, but an independent hidden set is needed to know whether the 0.691 default generalizes.
