# DigitalTwin — first anomaly training run

A compact GAT + GRU detects current simulated station disturbances. Only this capability is trained; the original multi-head dashboard remains a prototype.

CPU run: 19.7 seconds, 2 threads, 10,369 parameters.

Complete shifts: 128 train / 32 validation / 32 test, each 30 steps with 40 stations.

Held-out results with validation-selected thresholds:

| Model | Precision | Recall | F1 | Average precision |
|---|---:|---:|---:|---:|
| GAT + GRU | 0.962 | 0.898 | 0.929 | 0.955 |
| Local linear | 0.936 | 0.809 | 0.868 | 0.887 |

Average precision is a ranking metric, not classification accuracy. A graph model advantage is established only if the measured comparison supports it.

- All splits use the same synthetic generator and fixed assembly graph.
- Failure categories are shared across splits; only shifts/configurations are unseen.
- Station-step observations within a shift are correlated; they are not independent trials.
- Scores are not calibrated probabilities. No measured predictive lead time.
- Queue forecasting, defect risk, virtual sensing and confidence heads remain untrained.
- No real-factory validation, causal attribution, intervention benefit or measured savings.

See metrics.json for thresholds, confusion counts and epoch history; dataset_manifest.json for seeds/scenario configurations; heldout_predictions.npz for the exact evaluated outputs.
