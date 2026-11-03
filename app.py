"""
DigitalTwin.ai — Web Dashboard Backend
========================================
Flask application serving the interactive multi-stakeholder dashboard.

Endpoints:
    GET  /             — Dashboard SPA
    GET  /api/state    — Current twin state (all stations + predictions)
    GET  /api/timeline — Temporal history (all time steps so far)
    POST /api/whatif   — Submit what-if intervention
    GET  /api/validation — Validation metrics
    POST /api/step     — Advance simulation by one time step
    POST /api/reset    — Reset simulation to step 0

Author : Piyush Maji
Project: Accenture Innovation Challenge 2026
"""

import os
import json
import time
import hashlib
import numpy as np
import torch
import torch.nn.functional as F
from flask import Flask, render_template, jsonify, request

from assembly_line import (
    build_assembly_graph, get_sensor_mask_tensor, get_station_positions,
    get_zone_assignment, Zone, InstrumentationLevel, FEATURE_NAMES, NUM_FEATURES,
    RISK_LABELS, RISK_COLORS, ZONE_COLORS, print_assembly_stats,
)
from simulator import ProductionSimulator, DEFAULT_SCENARIOS, to_torch_tensors
from model import DigitalTwinGNN, load_model, prediction_arrays
from anomaly import DEFAULT_CHECKPOINT, load_detector

# ═══════════════════════════════════════════════════════════════════════════════
# APP SETUP
# ═══════════════════════════════════════════════════════════════════════════════

app = Flask(__name__)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ── Initialize components ──
print("\n🏭  DigitalTwin.ai — Initializing Assembly Line Digital Twin...\n")

stations, edge_index, G = build_assembly_graph()
print_assembly_stats(stations, G)

model, total_params, trainable_params = load_model(DEVICE)
print(f"  ✓ Model loaded: {total_params:,} parameters on {DEVICE}")

# The trained single-task detector is separate from the original untrained heads.
anomaly_detector = None
anomaly_checkpoint = None
anomaly_report = None
if DEFAULT_CHECKPOINT.is_file():
    anomaly_detector, anomaly_checkpoint = load_detector()
    report_path = DEFAULT_CHECKPOINT.with_name('metrics.json')
    if report_path.is_file():
        candidate = json.loads(report_path.read_text())
        checkpoint_hash = hashlib.sha256(DEFAULT_CHECKPOINT.read_bytes()).hexdigest()
        if candidate.get('checkpoint_sha256') == checkpoint_hash:
            anomaly_report = candidate
    print("  ✓ Trained synthetic-data anomaly detector loaded")

simulator = ProductionSimulator(stations, shift_duration_steps=30)
shift_data = simulator.generate_shift()
print(f"  ✓ Shift data generated: {len(shift_data)} time steps\n")

edge_index_device = edge_index.to(DEVICE)
sensor_mask = get_sensor_mask_tensor(stations).float().to(DEVICE)
positions = get_station_positions(stations)

# ── Simulation state ──
current_step = 0
timeline = []
h_states = None


# ═══════════════════════════════════════════════════════════════════════════════
# INFERENCE ENGINE
# ═══════════════════════════════════════════════════════════════════════════════

def run_inference(step_data, use_mask=True):
    """Decode untrained model heads without simulator outcomes."""
    global h_states

    tensors = to_torch_tensors(step_data, DEVICE)
    x = tensors['masked_features'] if use_mask else tensors['features']

    with torch.no_grad():
        output = model(x, edge_index_device, h_states=h_states, mask=sensor_mask if use_mask else None)
        h_states = output['h_states']

    arrays = prediction_arrays(output)
    return {
        **{key: value.tolist() for key, value in arrays.items()},
        'model_status': 'untrained_demo',
        'features': step_data['features'].tolist(),
        'active_scenarios': step_data['active_scenarios'],
    }


def run_trained_anomaly(shift_prefix):
    """Replay observed readings once per step; repeated reads are idempotent.

    Simulator severity, scenario names and true missing-sensor readings never
    enter this function's inputs to the model. Recurrence starts fresh per shift.
    """
    if anomaly_detector is None:
        return {'status': 'not_trained', 'message': 'Run train_anomaly.py to train the anomaly detector.'}
    readings = torch.from_numpy(np.stack([s['masked_features'] for s in shift_prefix])).float()
    masks = torch.from_numpy(np.stack([s['sensor_mask'] for s in shift_prefix])).float()
    with torch.no_grad():
        scores = anomaly_detector(readings[None], edge_index.cpu(), masks[None]).sigmoid()[0, -1]
    threshold = anomaly_checkpoint['threshold']
    return {
        'status': 'trained_synthetic', 'task': 'current_station_anomaly',
        'scores': scores.tolist(), 'flagged': (scores >= threshold).tolist(),
        'threshold': threshold, 'steps_observed': len(shift_prefix),
        'score_note': 'Detection scores, not calibrated probabilities or future warnings.',
    }


# ═══════════════════════════════════════════════════════════════════════════════
# ROUTES
# ═══════════════════════════════════════════════════════════════════════════════

@app.route('/')
def index():
    return render_template('index.html')


@app.route('/api/config')
def api_config():
    """Return static configuration (station list, graph topology, etc.)."""
    station_list = []
    for s in stations:
        station_list.append({
            'id': s.id,
            'name': s.name,
            'short': s.short_name,
            'zone': s.zone.value,
            'instrumentation': s.instrumentation.value,
            'vintage': s.vintage.value,
            'takt': s.takt_time_sec,
            'maintWindows': s.maint_windows_per_year,
            'sensorMask': s.sensor_mask,
            'pos': positions.get(s.id, [0, 0]),
        })

    edges = []
    for u, v, d in G.edges(data=True):
        edges.append({'source': u, 'target': v, 'type': d.get('type', 'forward')})

    return jsonify({
        'stations': station_list,
        'edges': edges,
        'featureNames': FEATURE_NAMES,
        'riskLabels': RISK_LABELS,
        'riskColors': RISK_COLORS,
        'zoneColors': {z.value: c for z, c in ZONE_COLORS.items()},
        'totalParams': total_params,
        'modelStatus': 'untrained_demo',
        'anomalyDetectorStatus': 'trained_synthetic' if anomaly_detector is not None else 'not_trained',
        'scenarios': [
            {'name': s.name, 'category': s.category, 'onset': s.onset_step,
             'stations': s.affected_stations, 'description': s.description}
            for s in DEFAULT_SCENARIOS
        ],
    })


@app.route('/api/state')
def api_state():
    """Return current twin state with predictions."""
    global current_step

    if current_step >= len(shift_data):
        current_step = len(shift_data) - 1

    step_data = shift_data[current_step]
    predictions = run_inference(step_data)

    return jsonify({
        'step': current_step,
        'totalSteps': len(shift_data),
        'predictions': predictions,
        'timestamp': f"Shift T+{current_step * 16}min",
    })


@app.route('/api/timeline')
def api_timeline():
    """Return the full timeline of predictions accumulated so far."""
    return jsonify({
        'timeline': timeline,
        'currentStep': current_step,
    })


@app.route('/api/anomaly')
def api_anomaly():
    """Trained current-station detection using the observed shift prefix only."""
    return jsonify(run_trained_anomaly(shift_data[:current_step + 1]))


@app.route('/api/anomaly/validation')
def api_anomaly_validation():
    """Frozen held-out metrics tied to the exact loaded checkpoint."""
    if anomaly_detector is None:
        return jsonify({'status': 'not_trained', 'metrics': None})
    if anomaly_report is None:
        return jsonify({'status': 'report_unavailable', 'metrics': None})
    return jsonify({'status': 'trained_synthetic', 'metrics': anomaly_report['detector']['test'],
                    'baseline': anomaly_report['linear_baseline']['test'],
                    'split_counts': anomaly_report['split_counts'],
                    'parameters': anomaly_report['detector']['parameters'],
                    'limitations': anomaly_report['limitations']})


@app.route('/api/step', methods=['POST'])
def api_step():
    """Advance simulation by one time step."""
    global current_step, h_states

    if current_step < len(shift_data) - 1:
        current_step += 1
        step_data = shift_data[current_step]
        predictions = run_inference(step_data)

        timeline.append({
            'step': current_step,
            'queue': predictions['queue'],
            'risk_class': predictions['risk_class'],
            'anomaly_score': predictions['anomaly_score'],
            'confidence': predictions['confidence'],
            'active_scenarios': predictions['active_scenarios'],
        })

        return jsonify({'status': 'ok', 'step': current_step, 'predictions': predictions})
    else:
        return jsonify({'status': 'end_of_shift', 'step': current_step})


@app.route('/api/reset', methods=['POST'])
def api_reset():
    """Reset simulation to step 0."""
    global current_step, h_states, timeline
    current_step = 0
    h_states = None
    timeline.clear()
    return jsonify({'status': 'ok', 'step': 0})


@app.route('/api/whatif', methods=['POST'])
def api_whatif():
    """
    Run a what-if simulation.
    Body: {"interventions": {station_id: {feature_idx: new_value, ...}, ...}}
    """
    return jsonify({
        'status': 'not_trained',
        'message': 'Intervention estimates require a trained and independently validated model.',
    }), 409


@app.route('/api/validation')
def api_validation():
    """
    Report that validation is unavailable without a trained checkpoint.
    """
    return jsonify({
        'status': 'not_trained',
        'message': 'No trained checkpoint is supplied. Validation metrics are unavailable.',
        'metrics': None,
    })


# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == '__main__':
    print("\n" + "=" * 60)
    print("  DigitalTwin.ai Dashboard")
    print("  Open http://localhost:5001 in your browser")
    print("=" * 60 + "\n")
    app.run(host='0.0.0.0', port=5001, debug=False)
