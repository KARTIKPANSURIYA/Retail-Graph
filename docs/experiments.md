# Experiment protocol

## Required order

First pin data/code/config and reproduce an available reference baseline and confirmed metric.
Only then compare a proposed model; do not describe an unverified difference as improvement.
Every run reports split/group policy, per-class/sample counts, macro metrics, confidence intervals
when statistically appropriate (group bootstrap preferred), abstentions/skips, runtime, hardware,
seed(s), software versions, and exact dataset revision.

## Matrix

| Track | Axis | Planned comparisons |
|---|---|---|
| RetailAction | views | single-view vs synchronized multi-view |
| RetailAction | modality | video-only vs video + provided pose |
| RetailAction | time | frame-level vs temporal features |
| RetailAction | imbalance | unweighted vs weighted/sampling/focal approaches |
| RetailAction | uncertainty | forced prediction vs calibrated confidence/abstention |
| Retail Gaze | context | head-only vs scene-aware |
| Retail Gaze | target | continuous point vs shelf region |
| Retail Gaze | grouping | subject/session-aware evaluation only |

RetailAction requires precision/recall and counts for `take`, `put`, `touch`; later official
spatio-temporal mAP is reported only after its implementation is confirmed against source code or
paper. Current matching greedily pairs same-class events by temporal IoU ≥ 0.5 once each and is
not that metric. Retail Gaze currently reports normalized Euclidean point distance and region
accuracy only when a region label/prediction exists, plus evaluated/skipped counts.

Calibration requires probabilistic predictions and held-out calibration data; scalar confidence
alone is not silently treated as a spatial probability distribution. Pre-register thresholds and
ablation hypotheses. Placement optimization and purchase conversion are out of scope until a
future intervention joined to consented POS outcomes supports causal analysis.
