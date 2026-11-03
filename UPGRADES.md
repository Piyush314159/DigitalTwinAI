# DigitalTwin upgrade status

## Current model

The dashboard and routine training use **one linear classifier** on 140 measured-sensor summary features. Its 1,974-parameter checkpoint is `training_outputs/model.pt`. The original 328-cycle held-out test has mean component macro F1 **0.950** and all four conditions correct together **80.49%**. The checkpoint hash and raw UCI measurements were unchanged during cleanup.

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
- [x] The dashboard validation page and evaluation export show only the retained linear model and its sensor-removal diagnostic.

## Next model experiment

A **small temporal convolutional network (TCN)** is the most relevant deep-learning candidate to test next. This is a proposal, not a proven improvement. The [UCI dataset](https://archive.ics.uci.edu/dataset/447/condition+monitoring+of+hydraulic+systems) has 60-second cycles from sensors sampled at 100, 10 and 1 Hz. The removed CNN used only one-second averages plus summary features; a new model could preserve more of the measured pressure, power and flow dynamics.

Keep the design understandable and CPU-sized: two or three short residual convolution blocks with 16–32 channels, separate handling of sensor rates, pooled outputs and four component-condition heads. Compare it against a newly fitted linear baseline on identical grouped folds. Choose settings inside each training fold; report per-component macro F1, accumulator class recalls, all-four-correct rate, training time and parameter count. Do not deploy it solely because it is deeper. TCNs are a plausible sequence-model family based on [Bai, Kolter and Koltun's study](https://arxiv.org/abs/1803.01271); that study does not establish performance on this hydraulic rig.

Before another architecture, inspect accumulator misclassifications and the original measured channels. This may identify a simpler feature improvement. Keep labels out of inputs and preserve condition-combination grouping.

## Upgrades that require additional data

- [ ] **Independent-machine validation.** Obtain measurements from another hydraulic machine or independently documented run, with machine/run IDs and source/license. Check physical quantities, units, sampling rates, cycle length and condition-code mapping before testing. Another shuffle of the current UCI rig is not independent evidence.
- [ ] **Future-condition prediction or remaining useful life.** Obtain ordered, timestamped runs with future condition or failure events and a defined prediction horizon. Use only readings available before the horizon and hold out complete runs or machines. Current full-cycle labels support condition classification only.

No external-machine, advance-warning or remaining-life performance is claimed with the current data.
