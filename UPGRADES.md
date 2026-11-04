# DigitalTwin upgrade status

## Current model

The dashboard now uses the **multirate TCN** (12,558 parameters) on real measured sequences plus 140 summary features. Its full-data checkpoint is `training_outputs/external_validation/tcn_full.pt`, fitted on all 2,205 UCI cycles. The earlier 1,974-parameter linear classifier remains in `training_outputs/model.pt` as a historical baseline. Its original 328-cycle held-out test scored **0.950** mean component macro F1 and **80.49%** all-four-correct; that protocol differs from the TCN comparison. The original checkpoint and raw UCI measurements remain unchanged.

Underperforming MLP, nearest-cycle, GRU, boosted-tree and CNN integrations, experiment scripts, checkpoints, tests and generated experiment folders were removed. The table below retains only the conclusions needed to explain that choice. Its rows use different evaluation partitions and must not be compared across rows.

| Candidate | Matched comparison | Candidate | Linear baseline |
|---|---|---:|---:|
| Small MLP | Original held-out test mean macro F1 | 0.901 | 0.950 |
| Nearest-cycle accumulator classifier | Separate validation accumulator F1 | 0.207 | 0.674 |
| One-layer GRU | Separate validation mean macro F1 | 0.549 | 0.801 |
| Boosted trees | Four-fold mean macro F1 | 0.743 | 0.772 |
| Two-block CNN | Four-fold mean macro F1 | 0.7717 | 0.7719 |

The CNN was close overall but had lower out-of-fold accumulator F1 than linear (0.453 versus 0.515). None of these candidates established an improvement. Old test partitions have been inspected, and all measurements come from the same physical rig. A new internal comparison remains exploratory until independent measurements are available.

## Completed software fixes

- [x] Invalid or incomplete sensor readings become JSON `null` and chart gaps. The dashboard withholds estimates for an incomplete cycle; CSV exports leave unavailable values empty.
- [x] Browser tabs keep separate replay selections. API requests carry the selected partition and record position.
- [x] The sidebar says `MODEL UNAVAILABLE` when there is no valid checkpoint.
- [x] The dashboard uses the full-data TCN for predictions and labels its three-fold scores as research evidence. The earlier linear test and sensor-removal diagnostic remain historical.

## Multirate TCN experiment completed

A **small temporal convolutional network (TCN)** was tested against the retained linear architecture on the same three grouped folds. The [UCI dataset](https://archive.ics.uci.edu/dataset/447/condition+monitoring+of+hydraulic+systems) has 60-second cycles from sensors sampled at 100, 10 and 1 Hz. Unlike the removed CNN's one-second averages, this experiment preserves pressure/power block means and flow at 10 Hz, alongside native 1 Hz temperature/vibration and the 140 measured-sample summaries.

The TCN has two residual temporal blocks and **12,558 parameters**. It ran on CPU with two threads. Training-only channel normalization and class weights were used. Inner grouped validation chose training epochs; outer validation groups were scored afterward. The frozen protocol, exact predictions, fold metrics and [report](training_outputs/tcn_experiment/report.md) are retained.

| Same three-fold protocol | Linear, refitted | Multirate TCN |
|---|---:|---:|
| Mean of fold macro F1 | 0.8957 | **0.9449** |
| Out-of-fold accumulator macro F1 | 0.7266 | **0.8739** |
| All four conditions correct | 68.39% | **82.22%** |

TCN led on all three outer folds. Most of the gain was on fold 3, where inner validation selected only 24 epochs for linear. A **post-hoc stronger-baseline check** refitted linear for its original 250-epoch maximum on each fold and reached **0.9156** fold-mean macro F1. TCN still led, but this check is exploratory because it was added after viewing the first outer-fold results. It is saved as `linear_250_diagnostic.json` and exact predictions beside the TCN report.

The dashboard now applies a TCN fitted on all 2,205 UCI cycles. The three-fold score describes separate fold-fitted models, not an independent test of this final checkpoint. The original linear 0.950 test score uses a different split and must not be directly ranked against 0.9449. All cycles come from the previously studied rig. The TCN result is promising internal evidence, **not independent-machine validation**. TCNs are a plausible sequence-model family based on [Bai, Kolter and Koltun's study](https://arxiv.org/abs/1803.01271); that paper does not establish performance on this particular rig.

Before another architecture, inspect accumulator misclassifications and the original measured channels. This may identify a simpler feature improvement. Keep labels out of inputs and preserve condition-combination grouping.

## Upgrades that require additional data

- [x] **External evaluation path prepared.** `scripts/external_validation.py` fits frozen linear and TCN candidates on the same full UCI development data; the TCN candidate powers the dashboard. The evaluator scores a new labeled collection without refitting. It requires provenance, complete sensor cycles, all condition classes and no exact UCI cycle overlap. See README for the input format.
- [ ] **Independent-machine validation result.** Obtain measurements from another hydraulic machine or independently documented later run, with machine/run IDs and source/license. Confirm physical quantities, sensor placement, units, sampling rates, cycle length and condition-code mapping before testing. Another shuffle of the current UCI rig is not independent evidence.
- [ ] **Future-condition prediction or remaining useful life.** Obtain ordered, timestamped runs with future condition or failure events and a defined prediction horizon. Use only readings available before the horizon and hold out complete runs or machines. Current full-cycle labels support condition classification only.

No external-machine, advance-warning or remaining-life performance is claimed with the current data.
