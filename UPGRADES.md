# Upgrade status

## Completed software upgrades

- [x] **Handle invalid readings throughout the API and dashboard.** Missing or non-finite statistics and trace samples become standard JSON `null`. Incomplete cycles return `insufficient_data` and no estimates. Charts show gaps rather than zero values; entirely unavailable signals have an explicit message. CSV exports leave unavailable cells empty. Regression checks cover NaN, both infinities, trace-only invalidity and strict JSON parsing.
- [x] **Separate replay between tabs/users.** The server has no shared replay variables. Every request carries its partition and record position; the dashboard holds its selection within its own tab. Reference labels and CSV exports request that same selection explicitly. Two independent API clients and JavaScript request checks verify isolation.
- [x] **Keep unavailable-model labels accurate.** The sidebar now says `MODEL UNAVAILABLE` when no checkpoint is loaded.

## Accumulator investigation completed; improvement still open

- [x] **Run a lightweight improvement experiment using a separately reserved protocol.** The linear baseline, small neural model and nine nearest-cycle accumulator candidates were fitted from scratch. Previously inspected test combinations were excluded. Training-only references and normalization, validation-only selection, exact test predictions and a frozen split manifest are retained in `training_outputs/accumulator_experiment/`.
- [ ] **Establish an accumulator model improvement.** The nearest-cycle candidate did not beat its linear baseline on validation. Its accumulator validation macro F1 was 0.207 versus 0.674 for the baseline; reserved-test F1 was 0.254 versus 0.489. The original deployed model and its evidence are retained. The original test has 328 different cycles and accumulator F1 0.869; these protocols are not directly comparable. Further experiments must not tune against either inspected test set.

Reproduce the experiment with `.venv/bin/python -m scripts.compare_accumulator`. It saves only to its own experiment folder and never deploys a model automatically. See [the report](training_outputs/accumulator_experiment/training_report.md) and [frozen protocol](training_outputs/accumulator_experiment/upgrade_protocol.json).

## Upgrades requiring additional data

- [ ] **Independent-machine validation.** Obtain measurements from another hydraulic machine or independently documented experimental run. Record machine/run identifiers and source/license. Check that the same 14 physical quantities, units, native sample rates, 60-second windows and condition codes can be mapped without guessing. Preserve these records as an external test; fit preprocessing on training data only and report external scores separately. Another copy or reshuffle of the current UCI data does not establish this capability.
- [ ] **Future-condition prediction / remaining useful life.** Obtain ordered, timestamped sequences with machine/run identifiers, future condition or failure events and a defined prediction horizon. Build inputs using only readings available before that horizon and hold out complete runs or machines. The current dataset's whole-cycle classification labels do not establish remaining life, causal diagnosis or intervention benefits.

No external-machine or forecasting performance is claimed without those measurements.

## GRU comparison completed

- [x] **Train the agreed one-layer GRU and compare it with the linear architecture.** Used 32 hidden units, 5,070 parameters and 60 ordered one-second averages for each of the 14 physical channels. Both architectures were fitted from scratch on the same 804 training cycles; 424 validation and 413 test cycles were reserved by whole condition combination. Both earlier test sets were excluded. Scaling and loss weights use training only, with no hidden memory carried between cycles.
- [x] **Apply the requested condition for adding layers.** The one-layer GRU validation macro F1 was **0.5490**, below the refitted linear model's **0.8012**. Extra layers were therefore **skipped**, as requested. Test macro F1 was **0.6483 / 0.7029** for GRU / linear. These figures use a different evaluation from the deployed model's 0.950 score and are not directly comparable with it.
- [x] **Retain the current deployment and save the evidence.** The active model and original reports are unchanged. Frozen settings, group indices, validation-only layer selection, checkpoints, histories and exact test predictions are saved under `training_outputs/gru_experiment/`. The dashboard shows this comparison separately. [Read the report](training_outputs/gru_experiment/report.md).

This tests one small GRU design using averaged traces, with one training seed and one grouped split. It does not establish that every recurrent architecture will perform worse. The remaining accumulator, independent-machine and forecasting upgrades above remain open.

## Boosted-tree comparison completed

- [x] **Test a compact tree model on the same 140 real-data features.** Four grouped cross-validation folds used 1,228 development cycles from 73 condition combinations. The three previously inspected test partitions, 977 cycles together, were excluded. The linear architecture selected its epoch on a separate inner grouped validation and was refitted on each outer training fold. Tree settings were fixed before evaluating folds.
- [x] **Compare with the freshly refitted linear baseline.** Fold-mean macro F1 was **0.7433 for boosted trees** versus **0.7719 for linear**. Out-of-fold macro F1 was **0.7839 / 0.8182**, and accumulator F1 was **0.3662 / 0.5154**, respectively. Trees trailed on all four folds. The active model remains linear; the original 0.950 test score uses a different protocol and cannot be directly ranked against these values.
- [x] **Keep the comparison reproducible and visible.** The frozen protocol, exact out-of-fold predictions, fold scores and report are under `training_outputs/tree_experiment/`; the dashboard's validation page displays them separately. Run `.venv/bin/python -m scripts.compare_trees` after installing `requirements-experiments.txt`. [Read the report](training_outputs/tree_experiment/report.md).

This is an internal same-rig comparison and does not measure generalization to a new machine. The tree result does not justify replacing the deployed model.

## Proposed next model comparisons

Keep upgrades sequential, understandable and small enough for local CPU training. Improving accumulator classification remains the first model priority; independent-machine validation and future-condition prediction remain planned as described above.

1. **Consider a shallow 1D convolutional neural network as the next separate model experiment.** Use the existing 14-channel, 60-point measured traces, with training-only channel scaling, two small convolution blocks and four condition outputs. This would test whether within-cycle signal patterns add value beyond the current summaries. These traces are one-second averages, not the native high-frequency waveforms. Keep training bounded and CPU friendly; better performance is not guaranteed. Reference: [PyTorch Conv1d documentation](https://docs.pytorch.org/docs/stable/generated/torch.nn.Conv1d.html).
2. **Compare evidence before replacing the current model.** Predeclare the evaluation, refit every candidate from scratch, keep entire condition combinations together and select settings using training-only inner validation. Three same-rig test partitions have now been inspected. Nested grouped cross-validation can support an internal comparison; independent measurements are still needed for external validation. Record accumulator macro F1, class recalls, four-component exact accuracy, runtime and model size.

The CNN comparison remains future work. Retain the current deployed linear model until a replacement has adequate evidence. Larger architectures are not currently a priority under the data and machine constraints.

## Retained deployed results

- Dataset: UCI hydraulic condition monitoring, 2,205 measured cycles.
- Deployed model: validation-selected linear classifier; 1,974 parameters, CPU only.
- Original held-out test: 328 cycles with unseen joint condition combinations.
- Mean component macro F1: 0.950; all four components correct together: 80.49%.
- Exact deployment evidence remains in `training_outputs/metrics.json`, `test_predictions.npz`, `split_manifest.json` and the sensor-removal reports.
