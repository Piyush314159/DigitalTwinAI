"""DigitalTwin monitoring API: observed telemetry and the frozen anomaly detector.

Operational routes never run the original untrained prediction heads. Simulator
answers remain offline evaluation targets and are not returned as station status.
"""

import csv
import hashlib
import io
import json
from threading import RLock

import numpy as np
import torch
from flask import Flask, Response, jsonify, render_template, request

from anomaly import DEFAULT_CHECKPOINT, load_detector
from assembly_line import FEATURE_NAMES, build_assembly_graph
from simulator import ProductionSimulator


app = Flask(__name__)
torch.set_num_threads(2)
stations, edge_index, graph = build_assembly_graph()
state_lock = RLock()
current_step = 0
shift_data = ProductionSimulator(stations, seed=42, shift_duration_steps=30).generate_shift()

anomaly_detector = None
anomaly_checkpoint = None
anomaly_report = None
robustness_report = None
checkpoint_hash = None
if DEFAULT_CHECKPOINT.is_file():
    anomaly_detector, anomaly_checkpoint = load_detector()
    checkpoint_hash = hashlib.sha256(DEFAULT_CHECKPOINT.read_bytes()).hexdigest()
    for path, name in ((DEFAULT_CHECKPOINT.with_name('metrics.json'), 'training'),
                       (DEFAULT_CHECKPOINT.parent / 'robustness' / 'metrics.json', 'robustness')):
        if path.is_file():
            candidate = json.loads(path.read_text())
            if candidate.get('checkpoint_sha256') == checkpoint_hash:
                if name == 'training':
                    anomaly_report = candidate
                else:
                    robustness_report = candidate

FEATURES = [
    ('Queue length', 'units', 1), ('Cycle time', 's', 1),
    ('Defect rate', '%', 2), ('Temperature', '°C', 1),
    ('Vibration', 'mm/s', 2), ('Torque deviation', '%', 2),
    ('Utilization', '%', 1), ('Throughput / target', '%', 1),
    ('Humidity', '%', 1), ('Power draw', 'kW', 1),
    ('Since maintenance', 'h', 1), ('Operator skill index', '', 2),
]
CAPABILITIES = [
    {'name': 'Station anomaly detection', 'status': 'trained_synthetic' if anomaly_detector is not None else 'not_trained'},
    {'name': 'Queue forecasting', 'status': 'not_trained'},
    {'name': 'Defect-risk prediction', 'status': 'not_trained'},
    {'name': 'Virtual sensor reconstruction', 'status': 'not_trained'},
    {'name': 'Confidence calibration', 'status': 'not_validated'},
    {'name': 'Intervention estimates', 'status': 'not_validated'},
]


def run_trained_anomaly(shift_prefix, include_history=False):
    """Replay only available observations; no persistent inference side effects."""
    if anomaly_detector is None:
        return {'status': 'not_trained', 'message': 'The anomaly model is unavailable. Train or restore its checkpoint.'}
    if not shift_prefix:
        raise ValueError('At least one observed step is required.')
    readings = torch.from_numpy(np.stack([s['masked_features'] for s in shift_prefix])).float()
    masks = torch.from_numpy(np.stack([s['sensor_mask'] for s in shift_prefix])).float()
    with torch.inference_mode():
        all_scores = anomaly_detector(readings[None], edge_index, masks[None]).sigmoid()[0].numpy()
    threshold = float(anomaly_checkpoint['threshold'])
    result = {
        'status': 'trained_synthetic', 'task': 'current_station_anomaly',
        'scores': all_scores[-1].tolist(),
        'flagged': (all_scores[-1] >= threshold).tolist(),
        'threshold': threshold, 'steps_observed': len(shift_prefix),
        'score_note': 'Detection scores are uncalibrated and describe current simulated conditions.',
    }
    if include_history:
        result['history_scores'] = all_scores.tolist()
    return result


def observed_values(step_data):
    """JSON-safe measured readings only; unobserved/invalid readings are null."""
    values = np.asarray(step_data['masked_features'])
    mask = np.asarray(step_data['sensor_mask']) > 0
    return [[float(v) if present and np.isfinite(v) else None
             for v, present in zip(row, row_mask)]
            for row, row_mask in zip(values, mask)]


def build_state(step):
    prefix = shift_data[:step + 1]
    detection = run_trained_anomaly(prefix, include_history=True)
    ready = detection['status'] == 'trained_synthetic'
    observations = observed_values(prefix[-1])
    measured = sum(value is not None for row in observations for value in row)
    queues = [row[0] for row in observations if row[0] is not None]
    history = []
    for index, sample in enumerate(prefix):
        scores = detection['history_scores'][index] if ready else None
        history.append({
            'step': index, 'elapsedMinutes': index * 16,
            'flaggedCount': sum(s >= detection['threshold'] for s in scores) if ready else None,
            'scores': scores,
        })
    hour, minute = divmod(8 * 60 + step * 16, 60)
    return {
        'status': 'ok', 'source': 'synthetic', 'step': step, 'totalSteps': len(shift_data),
        'elapsedMinutes': step * 16, 'timestamp': f'{hour:02d}:{minute:02d}',
        'observations': observations, 'modelStatus': detection['status'],
        'threshold': detection.get('threshold'),
        'scores': detection.get('scores'), 'flagged': detection.get('flagged'),
        'summary': {
            'observedStations': sum(any(v is not None for v in row) for row in observations),
            'flaggedStations': sum(detection['flagged']) if ready else None,
            'sensorCoverage': measured / (len(stations) * len(FEATURE_NAMES)),
            'meanObservedQueue': float(np.mean(queues)) if queues else None,
        },
        'history': history,
        'predictions': {'model_status': detection['status'],
                        'anomaly_score': detection.get('scores')},
    }


def validation_payload():
    if anomaly_detector is None:
        return {'status': 'not_trained', 'metrics': None, 'robustness': None}
    if anomaly_report is None:
        return {'status': 'report_unavailable', 'metrics': None, 'robustness': robustness_report}
    return {
        'status': 'trained_synthetic', 'task': 'current_station_anomaly',
        'metrics': anomaly_report['detector']['test'],
        'baseline': anomaly_report['linear_baseline']['test'],
        'split_counts': anomaly_report['split_counts'],
        'parameters': anomaly_report['detector']['parameters'],
        'limitations': anomaly_report['limitations'], 'checkpoint_sha256': checkpoint_hash,
        'robustness': robustness_report,
    }


@app.get('/')
def index():
    return render_template('index.html')


@app.get('/api/config')
def api_config():
    return jsonify({
        'source': 'synthetic', 'session': 'SIM-042', 'stepMinutes': 16,
        'stations': [{'id': s.id, 'name': s.name, 'short': s.short_name,
                      'zone': s.zone.value, 'instrumentation': s.instrumentation.value,
                      'vintage': s.vintage.value, 'takt': s.takt_time_sec,
                      'sensorMask': s.sensor_mask} for s in stations],
        'edges': [{'source': u, 'target': v, 'type': d['type']} for u, v, d in graph.edges(data=True)],
        'features': [{'key': 'throughput_target_fraction' if i == 7 else key,
                      'name': name, 'unit': unit, 'decimals': decimals,
                      'multiplier': 100 if i == 7 else 1}
                     for i, (key, (name, unit, decimals)) in enumerate(zip(FEATURE_NAMES, FEATURES))],
        'modelStatus': 'trained_synthetic' if anomaly_detector is not None else 'not_trained',
        'threshold': float(anomaly_checkpoint['threshold']) if anomaly_checkpoint else None,
        'capabilities': [{**c, 'status': ('trained_synthetic' if anomaly_detector is not None else 'not_trained')}
                         if i == 0 else c for i, c in enumerate(CAPABILITIES)],
    })


@app.get('/api/state')
def api_state():
    with state_lock:
        return jsonify(build_state(current_step))


@app.get('/api/timeline')
def api_timeline():
    with state_lock:
        return jsonify({'timeline': build_state(current_step)['history'], 'currentStep': current_step})


@app.get('/api/anomaly')
def api_anomaly():
    with state_lock:
        return jsonify(run_trained_anomaly(shift_data[:current_step + 1]))


@app.get('/api/validation')
@app.get('/api/anomaly/validation')
def api_validation():
    return jsonify(validation_payload())


@app.post('/api/step')
def api_step():
    global current_step
    with state_lock:
        ended = current_step == len(shift_data) - 1
        current_step = min(current_step + 1, len(shift_data) - 1)
        payload = build_state(current_step)
        if ended:
            payload['status'] = 'end_of_shift'
        return jsonify(payload)


@app.post('/api/seek')
def api_seek():
    global current_step
    body = request.get_json(silent=True)
    step = body.get('step') if isinstance(body, dict) else None
    if type(step) is not int or not 0 <= step < len(shift_data):
        return jsonify({'error': 'Choose a whole step between 0 and 29.'}), 400
    with state_lock:
        current_step = step
        return jsonify(build_state(current_step))


@app.post('/api/reset')
def api_reset():
    global current_step
    with state_lock:
        current_step = 0
        return jsonify(build_state(current_step))


@app.post('/api/whatif')
def api_whatif():
    return jsonify({'status': 'not_trained',
                    'message': 'Intervention effects require separate training and validation.'}), 409


@app.get('/api/snapshot.csv')
def api_snapshot():
    with state_lock:
        state = build_state(current_step)
    output = io.StringIO()
    writer = csv.writer(output)
    writer.writerow(['session', 'step', 'simulated_time', 'station_id', 'station', 'zone',
                     'anomaly_score', 'flagged',
                     *['throughput_target_fraction' if i == 7 else key
                       for i, key in enumerate(FEATURE_NAMES)]])
    for station, values in zip(stations, state['observations']):
        score = state['scores'][station.id] if state['scores'] is not None else ''
        flag = state['flagged'][station.id] if state['flagged'] is not None else ''
        writer.writerow(['SIM-042', state['step'], state['timestamp'], station.id,
                         station.name, station.zone.value, score, flag,
                         *['' if value is None else value for value in values]])
    return Response(output.getvalue(), mimetype='text/csv',
                    headers={'Content-Disposition': f'attachment; filename=digitaltwin-step-{state["step"]:02d}.csv'})


@app.get('/api/evaluation.json')
def api_evaluation_export():
    return Response(json.dumps(validation_payload(), indent=2), mimetype='application/json',
                    headers={'Content-Disposition': 'attachment; filename=digitaltwin-evaluation.json'})


if __name__ == '__main__':
    print('DigitalTwin · local monitoring dashboard at http://127.0.0.1:5001')
    app.run(host='127.0.0.1', port=5001, debug=False)
