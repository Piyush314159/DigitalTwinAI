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

## Run the dashboard

```bash
.venv/bin/python app.py
```

Open **http://localhost:5001**. Restart the app after retraining so it loads the new checkpoint.

- The **Trained Station Anomaly Detector** panel shows current detected stations and frozen held-out metrics for the exact loaded checkpoint.
- `GET /api/anomaly` uses only the observed shift prefix and replays temporal state from the beginning. Repeated reads do not advance the trained model's memory.
- `GET /api/anomaly/validation` serves saved test metrics only when their checkpoint hash matches the loaded artifact. It never calculates validation results from the current demo shift.
- Without the checkpoint, the trained panel explicitly reports `not_trained`.

## Original prototype status

The original `DigitalTwinGNN` in `model.py` and its queue, defect-risk, virtual-sensor and confidence views still use untrained weights. Their `/api/validation` response remains `not_trained`. The new anomaly detector has its own checkpoint, endpoints and clearly labeled panel; its results do not validate the original heads. `/api/whatif` remains unavailable (HTTP 409).

The graph has 40 stations and 56 edges (42 forward, 14 feedback/rework), with 25 fully instrumented, 9 partially instrumented and 6 manual stations. Telemetry is simulated. No sensor savings, predictive lead time, causal root-cause accuracy or financial savings have been established. Dashboard business figures are illustrative assumptions.

The command-line `demo.py` still demonstrates the original untrained multi-head architecture. PDF/PPTX exports and proposal documents are historical design artifacts, not evidence of trained-model performance. Attention weights alone do not establish causal attribution.

## Checks

```bash
.venv/bin/python -m unittest test_integrity test_anomaly_training -v
```

Checks cover missing readings, simulator-answer independence, healthy-shift generation, reproducible split data, isolated graph batches, sequence replay, checkpoint reload, tied-score metrics and API availability states.

## Next DigitalTwin milestones

Improve temporal queue dynamics before forecasting; train and evaluate the other heads separately; evaluate on real telemetry if available; calibrate scores; and validate interventions before enabling what-if benefits. Gen AI explanations can follow once the relevant outputs have measured support.
