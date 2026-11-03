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

## Retained deployed results

- Dataset: UCI hydraulic condition monitoring, 2,205 measured cycles.
- Deployed model: validation-selected linear classifier; 1,974 parameters, CPU only.
- Original held-out test: 328 cycles with unseen joint condition combinations.
- Mean component macro F1: 0.950; all four components correct together: 80.49%.
- Exact deployment evidence remains in `training_outputs/metrics.json`, `test_predictions.npz`, `split_manifest.json` and the sensor-removal reports.
