# Sentinel — Targets

Challenge 2: Adaptive Time-Series Anomaly Detection

## Objective
- [ ] Lightweight anomaly-detection system
- [ ] Learns baseline behavior from **streaming** time-series data
- [ ] Flags observations or windows that deviate significantly

## Output (for every flagged event)
- [ ] Anomaly score
- [ ] Binary label: `NORMAL` / `ANOMALOUS`
- [ ] Interpretable evidence explaining the flag

## Dataset
- [ ] Numenta Anomaly Benchmark (NAB) subset from the starter pack
- [ ] Must generalize to hidden test windows

## Metrics
- [ ] **Event-Based F1** (primary): detection accuracy across anomaly windows
- [ ] Precision & Recall: true flags vs. total flagged windows and ground-truth events
- [ ] Detection Delay: time from anomaly onset to model trigger
- [ ] Inference Runtime: execution speed per streaming window

## Permitted Approaches
- Statistical: rolling z-scores, Isolation Forest, LOF, One-Class SVM
- Advanced: autoencoders, temporal clustering, sliding-window adaptive thresholds, hybrid ensembles

## Judging
| Criteria | Weight | Focus |
|---|---|---|
| Technical Performance | 40% | Primary score on hidden evaluation data |
| Approach & Innovation | 20% | Algorithmic intelligence, creative engineering, threshold design |
| Efficiency & Constraints | 15% | Runtime speed |
| Presentation Quality | 15% | Clarity, structure, visualizations, Q&A defense |
| Code & Demo Quality | 10% | Code organization, documentation, working demo |
