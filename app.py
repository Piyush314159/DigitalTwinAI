"""DigitalTwin: recorded UCI hydraulic measurements and component estimates."""
import csv
import io
import json

import numpy as np
import torch
from flask import Flask, Response, jsonify, render_template, request
from flask.json.provider import DefaultJSONProvider
from werkzeug.exceptions import BadRequest

from hydraulics import OUTPUT, SENSORS, COMPONENTS, SOURCE_URL, prepare_data, file_hash
from model import load_model, predict

class StrictJSONProvider(DefaultJSONProvider):
    def dumps(self, obj, **kwargs):
        kwargs['allow_nan'] = False
        return super().dumps(obj, **kwargs)


app = Flask(__name__)
app.json = StrictJSONProvider(app)
torch.set_num_threads(2)
data = prepare_data()
model = checkpoint = report = robustness = None
accumulator_experiment = None
gru_experiment = None
tree_experiment = None
if (OUTPUT/'model.pt').exists():
    model, checkpoint = load_model(OUTPUT/'model.pt')
    if checkpoint['dataset_fingerprint'] != str(data['fingerprint']):
        raise ValueError('Dataset changed. Retrain the hydraulic model before using these measurements.')
    if (OUTPUT/'metrics.json').exists():
        candidate = json.loads((OUTPUT/'metrics.json').read_text())
        if candidate.get('checkpoint_sha256') == file_hash(OUTPUT/'model.pt') and candidate.get('dataset_fingerprint') == str(data['fingerprint']):
            report = candidate
    if (OUTPUT/'robustness/metrics.json').exists():
        candidate = json.loads((OUTPUT/'robustness/metrics.json').read_text())
        if candidate.get('checkpoint_sha256') == file_hash(OUTPUT/'model.pt'):
            robustness = candidate
split_manifest = json.loads((OUTPUT/'split_manifest.json').read_text()) if report else None
experiment_path = OUTPUT/'accumulator_experiment/metrics.json'
if experiment_path.exists():
    candidate = json.loads(experiment_path.read_text())
    experiment_checkpoint = experiment_path.parent/'model.pt'
    if (candidate.get('dataset_fingerprint') == str(data['fingerprint']) and experiment_checkpoint.exists()
            and candidate.get('checkpoint_sha256') == file_hash(experiment_checkpoint)):
        accumulator_experiment = candidate
gru_path = OUTPUT/'gru_experiment/metrics.json'
if gru_path.exists():
    candidate = json.loads(gru_path.read_text())
    hashes = candidate.get('checkpoint_sha256',{})
    protocol_path = gru_path.parent/'protocol.json'
    if (candidate.get('dataset_fingerprint') == str(data['fingerprint']) and hashes
            and protocol_path.exists() and file_hash(protocol_path) == candidate.get('protocol_sha256')
            and all((gru_path.parent/f'{kind}.pt').exists()
                    and file_hash(gru_path.parent/f'{kind}.pt') == sha for kind,sha in hashes.items())):
        gru_experiment = candidate
tree_path = OUTPUT/'tree_experiment/metrics.json'
if tree_path.exists():
    candidate = json.loads(tree_path.read_text())
    protocol_path = tree_path.parent/'protocol.json'
    predictions_path = tree_path.parent/'out_of_fold_predictions.npz'
    if (candidate.get('dataset_fingerprint') == str(data['fingerprint'])
            and protocol_path.exists() and predictions_path.exists()
            and file_hash(protocol_path) == candidate.get('protocol_sha256')
            and file_hash(predictions_path) == candidate.get('predictions_sha256')):
        tree_experiment = candidate
partitions = {'all':np.arange(len(data['features']))}
if split_manifest:
    for name in ('train','validation','test'):
        indices = np.array(split_manifest[name]['cycle_indices'], dtype=int)
        if len(np.unique(indices)) != len(indices) or np.any(indices < 0) or np.any(indices >= len(data['features'])):
            raise ValueError('Invalid split manifest.')
        partitions[name] = indices
DEFAULT_PARTITION = 'test' if 'test' in partitions else 'all'


def replay_selection(values=None, reset=False):
    """Selections travel with requests; tabs never share mutable replay state."""
    if values is None:
        values = request.args if request.method == 'GET' else request.get_json(silent=True)
    if values is None:
        values = {}
    if not hasattr(values, 'get'):
        raise BadRequest('Choose a valid record selection.')
    partition = values.get('partition', DEFAULT_PARTITION)
    if not isinstance(partition, str) or partition not in partitions:
        raise BadRequest('Unknown data partition.')
    position = 0 if reset else values.get('position', 0)
    if request.method == 'GET' and isinstance(position, str) and position.isascii() and position.isdecimal():
        position = int(position)
    if type(position) is not int or not 0 <= position < len(partitions[partition]):
        raise BadRequest('Choose a valid record position.')
    return position, partition


@app.errorhandler(400)
def bad_request(error):
    return jsonify({'error':error.description}), 400


def measured_value(value):
    return float(value) if np.isfinite(value) else None


def estimate_conditions(features):
    """Accept sensor features only. Condition annotations cannot enter inference."""
    if model is None:
        return {'status':'not_trained', 'conditions':[]}
    if not np.isfinite(features).all():
        return {'status':'insufficient_data', 'conditions':[]}
    predicted, scores = predict(model, np.asarray(features, dtype=np.float32).reshape(1,-1))
    conditions = []
    for i,c in enumerate(COMPONENTS):
        label = int(predicted[0,i])
        conditions.append({'id':c['id'], 'name':c['name'], 'class_index':label,
                           'label':c['labels'][label], 'code':c['codes'][label],
                           'flagged':label != 0, 'support':float(scores[i][0,label]),
                           'distribution':scores[i][0].tolist()})
    return {'status':'trained_real_data', 'conditions':conditions}


def build_state(position, partition):
    cycle = int(partitions[partition][position])
    features = data['features'][cycle]
    traces = data['traces'][cycle]
    invalid_channels = [s['id'] for i,s in enumerate(SENSORS)
                        if not np.isfinite(features[i*10:(i+1)*10]).all()
                        or not np.isfinite(traces[i]).all()]
    prediction = ({'status':'insufficient_data', 'conditions':[]} if invalid_channels
                  else estimate_conditions(features))
    observations = []
    for i,s in enumerate(SENSORS):
        values = features[i*10:i*10+4]
        observations.append({'id':s['id'], 'mean':measured_value(values[0]), 'std':measured_value(values[1]),
                             'min':measured_value(values[2]), 'max':measured_value(values[3]),
                             'trace':[measured_value(v) for v in traces[i]]})
    return {'status':'ok','source':'real_test_rig','cycle':cycle+1, 'position':position,
            'partition':partition,'totalRecords':len(partitions[partition]), 'totalCycles':len(data['features']),
            'cycleSeconds':60, 'modelStatus':prediction['status'],
            'conditions':prediction['conditions'], 'observations':observations,
            'dataQuality':{'complete':not invalid_channels, 'invalidChannels':invalid_channels},
            'summary':{'flaggedComponents':sum(c['flagged'] for c in prediction['conditions']) if prediction['conditions'] else None,
                       'sensorChannels':len(SENSORS),'components':4}}


def evaluation_payload():
    return {'status':'trained_real_data' if report else 'report_unavailable',
            'report':report, 'robustness':robustness, 'accumulatorExperiment':accumulator_experiment,
            'gruExperiment':gru_experiment, 'treeExperiment':tree_experiment}


@app.get('/')
def index():
    return render_template('index.html')


@app.get('/api/config')
def api_config():
    return jsonify({'source':'real_test_rig','sourceUrl':SOURCE_URL, 'license':'CC BY 4.0',
                    'sensors':SENSORS,'components':COMPONENTS,'totalCycles':len(data['features']),
                    'partitions':{k:len(v) for k,v in partitions.items()},
                    'defaultPartition':DEFAULT_PARTITION,
                    'modelKind':checkpoint['kind'] if checkpoint else None,
                    'modelStatus':'trained_real_data' if model else 'not_trained',
                    'excludedChannels':['CE','CP','SE'], 'cycleSeconds':60})


@app.get('/api/state')
def api_state():
    return jsonify(build_state(*replay_selection()))


@app.post('/api/step')
def api_step():
    position, partition = replay_selection()
    last = len(partitions[partition])-1
    result = build_state(min(position+1,last),partition)
    if position == last:
        result['status'] = 'end_of_records'
    return jsonify(result)


@app.post('/api/seek')
def api_seek():
    body=request.get_json(silent=True)
    if not isinstance(body,dict) or 'position' not in body:
        raise BadRequest('Choose a valid record position.')
    return jsonify(build_state(*replay_selection(body)))


@app.post('/api/partition')
def api_partition():
    body=request.get_json(silent=True)
    if not isinstance(body,dict) or 'partition' not in body:
        raise BadRequest('Unknown data partition.')
    return jsonify(build_state(*replay_selection(body, reset=True)))


@app.post('/api/reset')
def api_reset():
    return jsonify(build_state(*replay_selection(reset=True)))


@app.get('/api/reference')
def api_reference():
    """Explicit evaluation annotations, separate from input data and predictions."""
    position, partition = replay_selection()
    cycle=int(partitions[partition][position])
    labels=data['targets'][cycle]
    return jsonify({'cycle':cycle+1, 'annotationOnly':True,
                        'conditions':[{'id':c['id'],'label':c['labels'][int(labels[i])],
                                       'code':c['codes'][int(labels[i])]} for i,c in enumerate(COMPONENTS)],
                        'steadyStateReached':bool(data['profile'][cycle,4] == 0)})


@app.get('/api/validation')
def api_validation():
    return jsonify(evaluation_payload())


@app.get('/api/snapshot.csv')
def api_snapshot():
    state=build_state(*replay_selection())
    buffer=io.StringIO(); writer=csv.writer(buffer)
    writer.writerow(['source','cycle','partition','record_type','id','unit','mean','std','min','max',
                     'predicted_code','predicted_condition','class_support_uncalibrated'])
    for s,o in zip(SENSORS,state['observations']):
        writer.writerow(['UCI447 measured test rig',state['cycle'],state['partition'],'sensor',s['id'],s['unit'],
                         o['mean'],o['std'],o['min'],o['max'],'','',''])
    for c in state['conditions']:
        writer.writerow(['UCI447 measured test rig',state['cycle'],state['partition'],'prediction',c['id'],'',
                         '','','','',c['code'],c['label'],c['support']])
    return Response(buffer.getvalue(),mimetype='text/csv',headers={
        'Content-Disposition':f'attachment; filename=hydraulic-cycle-{state["cycle"]}.csv'})


@app.get('/api/evaluation.json')
def api_evaluation_export():
    return Response(json.dumps(evaluation_payload(),indent=2,allow_nan=False),mimetype='application/json',
                    headers={'Content-Disposition':'attachment; filename=hydraulic-evaluation.json'})


if __name__=='__main__':
    app.run(host='127.0.0.1',port=5001,debug=False)
