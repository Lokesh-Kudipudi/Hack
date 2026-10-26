# Sentinel — Incremental Ablation Study & Milestone History

This table traces the empirical impact of each architectural component from initial baseline to final production pipeline on the 58-file Numenta Anomaly Benchmark corpus.

| Pipeline Step | Event-F1 | Precision | Recall | TP / 116 | FP | NAB Standard | Latency (µs/pt) | Key Impact |
|---|---|---|---|---|---|---|---|---|
| Baseline (Windowed Gaussian) | 0.302 | 0.211 | 0.534 | 62 | 232 | 39.6 | 49.0 | Static window, no tail calibration |
| K1: RobustZ (D1 alone) | 0.176 | 0.101 | 0.664 | 77 | 683 | 0.0 | **4.4** | High raw recall, FP inflation |
| K1-v2: + Debounce (K=2) | 0.240 | 0.151 | 0.595 | 69 | 389 | 0.0 | **3.2** | FPs cut 43% (683 -> 389) |
| K2: + 6 Causal Detectors | 0.153 | 0.086 | 0.698 | 81 | 860 | 0.0 | **13.3** | Multi-horizon coverage, raw uncalibrated |
| K3: + Dynamic SPOT / EVT | **0.324** | 0.209 | 0.724 | 84 | 318 | 0.0 | 16.4 | Leaderboard #1; FPs cut 63% (860 -> 318) |
| K5: + Event Lifecycle Manager | 0.311 | 0.197 | 0.733 | 85 | 346 | 0.0 | 37.4 | Gap merge, cooldown, re-baselining |
| K7: + Hot-Path O(1) Optimization | 0.311 | 0.197 | 0.733 | 85 | 346 | 0.0 | **15.4** | 33% speedup (65,000 pts/sec) |
| K8: + Streaming Isolation Forest | 0.311 | 0.196 | **0.759** | 88 | 362 | 0.0 | 20.8 | Max Recall: 88/116 windows detected |
| Sentinel (Official NAB Calibrated) | 0.311 | 0.197 | 0.733 | 85 | 346 | **53.0** | 35.1 | Official NAB Standard Score: 53.00 |