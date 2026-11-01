# DigitalTwin.ai

An untrained spatio-temporal GNN architecture and simulated assembly-line dashboard, developed as an Accenture Innovation Challenge prototype.

## Current status

- Defines GATv2 + GRU blocks and queue, defect, virtual-sensor, confidence and anomaly heads.
- The bundled `load_model()` initializes seeded random weights. No trained checkpoint or training/validation evidence is supplied.
- The default assembly graph has 40 stations and 56 edges (42 forward, 14 feedback/rework). Its instrumentation counts are 25 full, 9 partial and 6 manual stations.
- Telemetry and anomaly scenarios come from a simulator. Model output decoding reads only the model heads; simulator ground-truth anomalies and current queue observations are not blended into predictions.
- Missing sensor NaNs are replaced before masking so model outputs remain finite.
- Risk and confidence outputs demonstrate the interface and are not validated or calibrated.
- `/api/validation` explicitly reports `not_trained` with no metrics. `/api/whatif` reports `not_trained` (HTTP 409); the former hard-coded intervention benefit is removed.
- No predictive lead time, causal root-cause accuracy, sensor savings or annual financial savings have been measured. Business values shown in the dashboard are illustrative assumptions.

## Run

```bash
pip install -r requirements.txt
python app.py
```

Open http://localhost:5001. The dashboard displays the prototype status. The command-line demonstration is `python demo.py`; it marks model outputs as untrained and replaces the former validation chart with a status notice.

## Checks

```bash
python -m unittest test_integrity -v
```

The checks verify finite outputs under sensor masking, prediction independence from simulator labels, and unavailable-state API responses.

## Remaining work

Train on appropriately separated sequences, validate on unseen scenarios, compare simple baselines, calibrate uncertainty, and validate intervention estimates before enabling performance claims. Attention weights alone do not establish causal attribution.

PDF/PPTX exports and proposal documents are historical design artifacts. They are not evidence of trained-model results or realized operational savings. This README and current code define the implementation status.
