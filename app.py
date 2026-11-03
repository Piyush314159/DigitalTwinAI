"""DigitalTwin: recorded UCI hydraulic measurements and component estimates."""
import csv
import io
import json
from threading import RLock

import numpy as np
import torch
from flask import Flask, Response, jsonify, render_template, request

from hydraulics import OUTPUT, SENSORS, COMPONENTS, SOURCE_URL, prepare_data, file_hash
from model import load_model, predict

app = Flask(__name__)
torch.set_num_threads(2)
state_lock = RLock()
data = prepare_data()
model = checkpoint = report = robustness = None
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
partitions = {'all':np.arange(len(data['features']))}
if split_manifest:
    for name in ('train','validation','test'):
        indices = np.array(split_manifest[name]['cycle_indices'], dtype=int)
        if len(np.unique(indices)) != len(indices) or np.any(indices < 0) or np.any(indices >= len(data['features'])):
            raise ValueError('Invalid split manifest.')
        partitions[name] = indices
current_partition = 'test' if 'test' in partitions else 'all'
current_position = 0


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
    prediction = estimate_conditions(features)
    observations = []
    for i,s in enumerate(SENSORS):
        values = features[i*10:i*10+4]
        observations.append({'id':s['id'], 'mean':float(values[0]), 'std':float(values[1]),
                             'min':float(values[2]), 'max':float(values[3]),
                             'trace':data['traces'][cycle,i].tolist()})
    return {'status':'ok','source':'real_test_rig','cycle':cycle+1, 'position':position,
            'partition':partition,'totalRecords':len(partitions[partition]), 'totalCycles':len(data['features']),
            'cycleSeconds':60, 'modelStatus':prediction['status'],
            'conditions':prediction['conditions'], 'observations':observations,
            'summary':{'flaggedComponents':sum(c['flagged'] for c in prediction['conditions']) if prediction['conditions'] else None,
                       'sensorChannels':len(SENSORS),'components':4}}


def evaluation_payload():
    return {'status':'trained_real_data' if report else 'report_unavailable',
            'report':report, 'robustness':robustness}


@app.get('/')
def index():
    return render_template('index.html')


@app.get('/api/config')
def api_config():
    return jsonify({'source':'real_test_rig','sourceUrl':SOURCE_URL, 'license':'CC BY 4.0',
                    'sensors':SENSORS,'components':COMPONENTS,'totalCycles':len(data['features']),
                    'partitions':{k:len(v) for k,v in partitions.items()},
                    'defaultPartition':current_partition,
                    'modelKind':checkpoint['kind'] if checkpoint else None,
                    'modelStatus':'trained_real_data' if model else 'not_trained',
                    'excludedChannels':['CE','CP','SE'], 'cycleSeconds':60})


@app.get('/api/state')
def api_state():
    with state_lock:
        return jsonify(build_state(current_position,current_partition))


@app.post('/api/step')
def api_step():
    global current_position
    with state_lock:
        ended = current_position == len(partitions[current_partition])-1
        current_position = min(current_position+1,len(partitions[current_partition])-1)
        result = build_state(current_position,current_partition)
        if ended:
            result['status'] = 'end_of_records'
        return jsonify(result)


@app.post('/api/seek')
def api_seek():
    global current_position
    body=request.get_json(silent=True)
    position=body.get('position') if isinstance(body,dict) else None
    with state_lock:
        if type(position) is not int or not 0 <= position < len(partitions[current_partition]):
            return jsonify({'error':'Choose a valid record position.'}),400
        current_position=position
        return jsonify(build_state(current_position,current_partition))


@app.post('/api/partition')
def api_partition():
    global current_position,current_partition
    body=request.get_json(silent=True)
    name=body.get('partition') if isinstance(body,dict) else None
    if not isinstance(name,str) or name not in partitions:
        return jsonify({'error':'Unknown data partition.'}),400
    with state_lock:
        current_partition,current_position=name,0
        return jsonify(build_state(current_position,current_partition))


@app.post('/api/reset')
def api_reset():
    global current_position
    with state_lock:
        current_position=0
        return jsonify(build_state(0,current_partition))


@app.get('/api/reference')
def api_reference():
    """Explicit evaluation annotations, separate from input data and predictions."""
    with state_lock:
        cycle=int(partitions[current_partition][current_position])
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
    with state_lock:
        state=build_state(current_position,current_partition)
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
    return Response(json.dumps(evaluation_payload(),indent=2),mimetype='application/json',
                    headers={'Content-Disposition':'attachment; filename=hydraulic-evaluation.json'})


if __name__=='__main__':
    app.run(host='127.0.0.1',port=5001,debug=False)
