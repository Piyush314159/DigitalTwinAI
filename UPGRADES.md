# Future upgrades

These items are recorded for later work. They have not been implemented as part of the file cleanup.

## Software fixes

- [ ] **Handle invalid readings throughout the API and dashboard.** The prediction function refuses non-finite input, but `/api/state` can still serialize a measured value as `NaN`. Browsers cannot parse that as standard JSON. Convert unavailable readings to `null`, keep estimates unavailable, and add a test of the complete API response using strict JSON parsing.
- [ ] **Separate replay state between browser tabs/users.** `current_partition` and `current_position` are currently shared process variables. Seeking in one tab affects another tab's current cycle and export. Store the selection per browser/session, or pass an explicit cycle/partition with each request. Verify isolation with two independent clients.

## Model and evaluation improvements

- [ ] **Improve accumulator condition classification.** Current held-out macro F1 is 0.869. Of 132 slightly reduced-pressure cycles, 21 were estimated as optimal. Investigate these errors using training/validation data and evaluate subsequent changes with a newly reserved protocol; do not repeatedly tune against the inspected test set.
- [ ] **Evaluate generalization beyond the current rig.** Current data are real measurements from one physical laboratory rig. Obtain compatible measurements from another machine or independent experimental run, document feature/label compatibility, and report external performance separately.
- [ ] **Investigate future-condition prediction as a separate task.** Current classification uses a complete 60-second cycle. Predictive lead time, remaining useful life, causal diagnosis and intervention benefits are not established. Such capabilities require suitable temporal targets and independent evidence.

## Retained reference results

- Dataset: UCI hydraulic condition monitoring, 2,205 measured cycles.
- Deployed model: validation-selected linear classifier.
- Held-out test: 328 cycles with unseen joint condition combinations.
- Mean component macro F1: 0.950; all four components correct together: 80.49%.
- Exact baseline evidence remains in `training_outputs/metrics.json`, `test_predictions.npz`, `split_manifest.json` and the sensor-removal reports.
