# DigitalTwin.ai

A simulated 40-station assembly line with a **trained, single-task station anomaly detector** and an earlier multi-head GNN prototype. Developed for the Accenture Innovation Challenge.

## First completed training milestone

`anomaly.py` reuses the existing GATv2 + GRU block in a small **10,369-parameter** detector. It learns whether a station's **current simulator disturbance severity is at least 0.30**. This is current-condition detection, not forecasting or measured advance warning.

The first CPU run used two threads and took approximately 20 seconds, including data generation, baseline training, evaluation and checkpoint reload verification. Actual runtime depends on the machine and settings.

| Held-out synthetic test result | Precision | Recall | F1 | Average precision |
|---|---:|---:|---:|---:|
| Compact GAT + GRU | 0.962 | 0.898 | 0.929 | 0.955 |
| Local linear baseline | 0.936 | 0.809 | 0.868 | 0.887 |

These are descriptive results from one fixed run. The comparison does not isolate the contribution of graph structure from recurrence or nonlinear model capacity.

### Data and evaluation

- 128 training, 32 validation and 32 test **complete independent simulated shifts**, each 30 steps with 40 stations. No overlapping windows across splits.
- Different seeds produce different fault locations, onsets, severities, combinations and additional sensor outages. Healthy shifts are included.
- Feature normalization and class weights use training observations only. Checkpoints are chosen by validation average precision and detection thresholds by validation F1.
- Test shifts are generated and evaluated only after model and threshold selection. `training_outputs/heldout_predictions.npz` records the evaluated outputs.
- Readings, availability masks and the fixed graph are the only model inputs. Simulator severity and scenario names are never input features; missing-sensor ground truth is excluded.
- Every split shares the same simulator, failure categories and graph. Station-step observations within a shift are correlated. This is **not real-factory validation**, unseen-fault-category evaluation, calibrated probability, causal diagnosis or proven intervention benefit.

Exact results, thresholds, epoch histories and artifact hashes are in [metrics.json](training_outputs/metrics.json). See [training_report.md](training_outputs/training_report.md) for a readable summary and [dataset_manifest.json](training_outputs/dataset_manifest.json) for all split seeds and scenarios.

## Train on your laptop

The existing `.venv` has the required dependencies. From this directory:

```bash
.venv/bin/python train_anomaly.py
```

This saves `training_outputs/anomaly_detector.pt`, a linear baseline checkpoint, metrics, the dataset manifest and held-out predictions. Checkpoints are local artifacts ignored by Git; retrain after cloning. For a smaller experiment in a separate directory:

```bash
.venv/bin/python train_anomaly.py --train-shifts 32 --validation-shifts 16 --test-shifts 16 --epochs 15 --output-dir /tmp/digitaltwin-small-run
```

If setting up a fresh environment, install `requirements.txt` first. No LLM or GPU is required for this milestone. Avoid tuning against this already inspected test split; use a newly reserved evaluation set for a subsequent final comparison.

## Run the monitoring dashboard

```bash
.venv/bin/python app.py
```

Open **http://localhost:5001**. Restart the app after retraining so it loads the new checkpoint.

- **Overview:** a selectable assembly graph, trained anomaly alerts, observed sensor coverage and queue readings, shift replay and a detection-activity chart. Map colors, alerts and counts use the same model scores and fixed threshold.
- **Stations:** search and filter by zone or detection status, sort detection scores, inspect observed readings and export a 40-station CSV snapshot. Missing sensors remain unavailable. Throughput is expressed relative to target, not as a measured units-per-hour value.
- **Model validation:** original held-out metrics, a linear-model comparison, confusion counts and the six-condition robustness evaluation. Export the exact saved results as JSON.
- **Methodology:** data source, evaluation protocol and explicit availability of each capability.
- Replay supports stepping, play/pause, reset and seeking in either direction. Model recurrence replays only the observed prefix; repeated reads do not change memory, and future chart points remain unobserved.
- Without the checkpoint, detection is explicitly unavailable. It is never replaced by random model output or an all-clear status.
- The dashboard runs locally with Flask and plain HTML/CSS/JavaScript. It has no additional frontend dependencies, external chart service, or LLM requirement.

### Monitoring endpoints

| Endpoint | Purpose |
|---|---|
| `GET /api/state` | Observed readings, trained scores, flags and observed history |
| `POST /api/step`, `/api/reset`, `/api/seek` | Serialized shift replay; seek accepts an integer step 0–29 |
| `GET /api/anomaly` | Trained scores for the current observed prefix |
| `GET /api/validation`, `/api/anomaly/validation` | Frozen metrics for the exact loaded checkpoint |
| `GET /api/snapshot.csv` | Current observed readings and trained flags, without simulator answers |
| `GET /api/evaluation.json` | Saved evaluation and robustness report |

## Frozen-model robustness evaluation

```bash
.venv/bin/python evaluate_robustness.py
```

This evaluates the saved detector and linear baseline **without training or selecting a new threshold**. The extra conditions use seeds separate from the original train/validation/test sets. Sensor challenges reuse the fresh reference shifts for paired comparisons. Each condition has 32 shifts, with 38,400 station-time observations.

| Condition | Detector F1 | Recall | False-positive station-steps | Missed station-steps |
|---|---:|---:|---:|---:|
| Fresh simulated shifts | 0.929 | 0.896 | 23 | 71 |
| 20% additional sensor outages | 0.768 | 0.666 | 46 | 228 |
| Noisy measurements | 0.916 | 0.890 | 36 | 75 |
| Weaker disturbances | 0.928 | 0.920 | 37 | 46 |
| 4–6 concurrent disturbances | 0.935 | 0.917 | 86 | 159 |
| Healthy shifts | — | — | 12 | 0 |

Additional sensor outages expose the largest measured degradation. F1 and recall are undefined in the all-healthy case; its useful measure is false alarms (12 of 38,400 station-steps). Station-time observations are correlated, and counts do not represent independent factory incidents. No condition establishes real-factory generalization.

See [the robustness report](training_outputs/robustness/report.md), [exact metrics](training_outputs/robustness/metrics.json) and `training_outputs/robustness/predictions.npz`. Model and original training artifacts are preserved. These challenge sets are now inspected; reserve new evaluation data before further tuning.

## Original prototype status

The original `DigitalTwinGNN` in `model.py` remains an untrained research architecture. Operational endpoints and the dashboard no longer execute its random heads. Queue forecasting, defect-risk prediction, virtual sensing, confidence calibration and intervention estimates are marked unavailable in the capability view. `/api/validation` now reports the trained anomaly detector's saved results; `/api/whatif` remains unavailable (HTTP 409).

The graph has 40 stations and 56 edges (42 forward, 14 feedback/rework), with 25 fully instrumented, 9 partially instrumented and 6 manual stations. Telemetry is simulated. No sensor savings, predictive lead time, causal root-cause accuracy or financial savings have been established. The monitoring dashboard contains no illustrative savings or rollout claims.

The command-line `demo.py` still demonstrates the original untrained multi-head architecture. PDF/PPTX exports and proposal documents are historical design artifacts, not evidence of trained-model performance. Attention weights alone do not establish causal attribution.

## Checks

```bash
.venv/bin/python -m unittest test_integrity test_anomaly_training test_monitoring -v
```

23 checks cover missing readings, simulator-answer independence, healthy-shift generation, reproducible original data, isolated graph batches, sequence replay, checkpoint reload, tied-score metrics, invalid seeks, unavailable-model behavior, CSV units and reproducible frozen-model challenge metrics. Browser verification also covers desktop/mobile layouts, graph selection, feedback paths, replay boundaries, station filters and the downloaded CSV.

## Next DigitalTwin milestones

Improve temporal queue dynamics before forecasting; train and evaluate the other heads separately; evaluate on real telemetry if available; calibrate scores; and validate interventions before enabling what-if benefits. Gen AI explanations can follow once the relevant outputs have measured support.
