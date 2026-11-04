"""Freeze UCI-trained candidates and score a separately collected hydraulic dataset.

Run ``python -m scripts.external_validation prepare`` once, then
``python -m scripts.external_validation evaluate PATH_TO_NEW_DATA``.
The evaluator never updates model weights or normalization statistics.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path

import numpy as np
import torch

from hydraulics import (COMPONENTS, FEATURE_NAMES, OUTPUT, SENSORS,
                        evaluation_metrics, file_hash, prepare_data, summarize_cycle)
from model import HydraulicClassifier, predict
from scripts.compare_tcn import (HIGH_SENSORS, LOW_SENSORS, SIGNAL_CACHE,
                                 HydraulicTCN, fit_linear_full,
                                 measured_multirate, new_tcn, predict_tcn,
                                 train_tcn)

DIRECTORY = OUTPUT / 'external_validation'
MANIFEST = DIRECTORY / 'candidates.json'
TRAINING_RIG = 'UCI-447-test-rig'
REQUIRED_METADATA = ('source_id', 'machine_id', 'collection_period',
                     'independence_basis', 'label_provenance', 'sensor_alignment')


def _fingerprints(features):
    return {hashlib.sha256(np.ascontiguousarray(row).tobytes()).hexdigest()
            for row in features}


def _read_candidates():
    if not MANIFEST.is_file():
        raise FileNotFoundError('Run `python -m scripts.external_validation prepare` first.')
    manifest = json.loads(MANIFEST.read_text())
    for kind in ('linear', 'tcn'):
        item = manifest['candidates'][kind]
        path = DIRECTORY / f'{kind}_full.pt'
        if not path.is_file() or file_hash(path) != item['sha256']:
            raise ValueError(f'The frozen {kind} checkpoint is missing or changed.')
    return manifest


def prepare_candidates():
    """Fit both architectures on all UCI cycles using frozen epoch choices."""
    torch.set_num_threads(2)
    torch.use_deterministic_algorithms(True)
    data = prepare_data()
    comparison_dir = OUTPUT / 'tcn_experiment'
    comparison = json.loads((comparison_dir / 'metrics.json').read_text())
    if comparison['dataset_fingerprint'] != str(data['fingerprint']):
        raise ValueError('The TCN comparison and current UCI source differ.')
    for filename, key in (('protocol.json', 'protocol_sha256'),
                          ('out_of_fold_predictions.npz', 'predictions_sha256')):
        if file_hash(comparison_dir / filename) != comparison[key]:
            raise ValueError(f'TCN comparison evidence changed: {filename}.')
    high, low = measured_multirate(data)
    if file_hash(SIGNAL_CACHE) != comparison['signal_cache_sha256']:
        raise ValueError('TCN sequence cache differs from the comparison.')
    if MANIFEST.exists():
        frozen = _read_candidates()
        if frozen['training_dataset_fingerprint'] != str(data['fingerprint']):
            raise ValueError('Existing candidates were trained on another source.')
        if frozen['comparison_metrics_sha256'] != file_hash(comparison_dir / 'metrics.json'):
            raise ValueError('Existing candidates reference different TCN comparison evidence.')
        return frozen

    # The TCN length is the median of the three already saved inner-fold choices.
    # The stronger linear diagnostic used 250 epochs. Neither choice uses new data.
    tcn_epochs = int(np.median([fold['tcn']['inner_selected_epoch']
                                for fold in comparison['folds']]))
    linear_epochs = 250
    indices = np.arange(len(data['features']))
    linear = fit_linear_full(data, indices, linear_epochs)
    tcn, tcn_fit = train_tcn(data, high, low, indices, epochs=tcn_epochs)
    DIRECTORY.mkdir(parents=True, exist_ok=True)
    checkpoints = {
        'linear': {'task': 'external_hydraulic_condition_evaluation_v1',
                   'kind': 'linear', 'feature_names': FEATURE_NAMES,
                   'training_dataset_fingerprint': str(data['fingerprint']),
                   'state_dict': linear.state_dict()},
        'tcn': {'task': 'external_hydraulic_condition_evaluation_v1',
                'kind': 'tcn', 'feature_names': FEATURE_NAMES,
                'high_sensor_ids': [sensor['id'] for sensor in HIGH_SENSORS],
                'low_sensor_ids': [sensor['id'] for sensor in LOW_SENSORS],
                'training_dataset_fingerprint': str(data['fingerprint']),
                'state_dict': tcn.state_dict()},
    }
    candidates = {}
    for kind, model in (('linear', linear), ('tcn', tcn)):
        path = DIRECTORY / f'{kind}_full.pt'
        temporary = path.with_suffix('.tmp.pt')
        torch.save(checkpoints[kind], temporary)
        os.replace(temporary, path)
        candidates[kind] = {'sha256': file_hash(path),
                            'epochs': linear_epochs if kind == 'linear' else tcn_epochs,
                            'parameters': sum(p.numel() for p in model.parameters())}
    restored_linear, restored_tcn = load_candidates({
        'training_dataset_fingerprint': str(data['fingerprint']),
        'candidates': candidates})
    np.testing.assert_array_equal(predict(linear, data['features'][:2])[0],
                                  predict(restored_linear, data['features'][:2])[0])
    sample = np.arange(2)
    np.testing.assert_array_equal(predict_tcn(tcn, data, high, low, sample),
                                  predict_tcn(restored_tcn, data, high, low, sample))
    manifest = {'task': 'independent_hydraulic_condition_evaluation',
                'training_source': 'UCI 447, one physical test rig',
                'training_rig_id': TRAINING_RIG,
                'training_dataset_fingerprint': str(data['fingerprint']),
                'training_cycles': len(indices),
                'comparison_metrics_sha256': file_hash(comparison_dir / 'metrics.json'),
                'selection': 'TCN epochs = median of three saved inner-fold choices; '
                             'linear = 250-epoch stronger baseline diagnostic. '
                             'Both refitted on all 2,205 UCI cycles before external evaluation.',
                'tcn_fit_parameters': tcn_fit['parameters'],
                'candidates': candidates}
    MANIFEST.write_text(json.dumps(manifest, indent=2) + '\n')
    return manifest


def load_candidates(manifest):
    models = []
    for kind in ('linear', 'tcn'):
        path = DIRECTORY / f'{kind}_full.pt'
        if file_hash(path) != manifest['candidates'][kind]['sha256']:
            raise ValueError(f'Frozen {kind} checkpoint changed.')
        checkpoint = torch.load(path, map_location='cpu', weights_only=True)
        if (checkpoint.get('task') != 'external_hydraulic_condition_evaluation_v1'
                or checkpoint.get('kind') != kind
                or checkpoint.get('feature_names') != FEATURE_NAMES
                or checkpoint.get('training_dataset_fingerprint') !=
                manifest['training_dataset_fingerprint']):
            raise ValueError(f'Frozen {kind} checkpoint schema changed.')
        state = checkpoint['state_dict']
        if kind == 'linear':
            model = HydraulicClassifier(np.zeros(len(FEATURE_NAMES)),
                                        np.ones(len(FEATURE_NAMES)))
        else:
            if (checkpoint.get('high_sensor_ids') != [s['id'] for s in HIGH_SENSORS]
                    or checkpoint.get('low_sensor_ids') != [s['id'] for s in LOW_SENSORS]):
                raise ValueError('Frozen TCN sensor order changed.')
            model = HydraulicTCN(*(state[name] for name in (
                'feature_mean', 'feature_scale', 'high_mean', 'high_scale',
                'low_mean', 'low_scale')))
        model.load_state_dict(state)
        model.eval()
        models.append(model)
    return models


def read_external(directory, training_features):
    """Read UCI-compatible physical sensor files without accepting UCI duplicates."""
    directory = Path(directory).expanduser().resolve()
    metadata_path = directory / 'metadata.json'
    if not metadata_path.is_file():
        raise ValueError('External data needs metadata.json with collection provenance.')
    metadata = json.loads(metadata_path.read_text())
    missing = [key for key in REQUIRED_METADATA
               if not isinstance(metadata.get(key), str) or not metadata[key].strip()]
    if missing:
        raise ValueError(f'Missing external provenance fields: {", ".join(missing)}.')
    if metadata['independence_basis'] not in ('different_machine', 'later_collection'):
        raise ValueError('independence_basis must be different_machine or later_collection.')
    if (metadata['independence_basis'] == 'different_machine'
            and metadata['machine_id'] == TRAINING_RIG):
        raise ValueError('The original UCI rig cannot be declared as a different machine.')
    profile_path = directory / 'profile.txt'
    if not profile_path.is_file():
        raise FileNotFoundError(profile_path)
    profile = np.loadtxt(profile_path, dtype=np.int64, ndmin=2)
    if profile.shape[1] not in (4, 5) or not len(profile):
        raise ValueError('profile.txt must have one row per cycle and four or five columns.')
    targets = np.empty((len(profile), 4), dtype=np.int64)
    for i, component in enumerate(COMPONENTS):
        unknown = set(profile[:, i]) - set(component['codes'])
        if unknown:
            raise ValueError(f"Unknown {component['name']} condition codes: {sorted(unknown)}.")
        targets[:, i] = [component['codes'].index(int(value)) for value in profile[:, i]]
        missing_classes = set(component['codes']) - set(profile[:, i])
        if missing_classes:
            raise ValueError(f"External {component['name']} labels omit classes "
                             f'{sorted(missing_classes)}; full macro F1 would be misleading.')
    if profile.shape[1] == 5 and not set(profile[:, 4]).issubset({0, 1}):
        raise ValueError('Stability annotations must be 0 or 1.')
    features = np.empty((len(profile), len(FEATURE_NAMES)), dtype=np.float32)
    high = np.empty((len(profile), len(HIGH_SENSORS), 600), dtype=np.float32)
    low = np.empty((len(profile), len(LOW_SENSORS), 60), dtype=np.float32)
    sensor_files = {}
    for i, sensor in enumerate(SENSORS):
        path = directory / f"{sensor['id']}.txt"
        if not path.is_file():
            raise FileNotFoundError(path)
        rows = 0
        with path.open() as stream:
            for cycle, line in enumerate(stream):
                if cycle >= len(profile):
                    raise ValueError(f"Extra {sensor['id']} cycle after profile.txt ends.")
                raw = np.fromstring(line, sep=' ', dtype=np.float64)
                if raw.size != 60 * sensor['rate'] or not np.isfinite(raw).all():
                    raise ValueError(f"Invalid {sensor['id']} cycle {cycle + 1}: expected "
                                     f"{60 * sensor['rate']} finite samples.")
                features[cycle, i*10:(i+1)*10] = summarize_cycle(raw)
                if sensor['rate'] >= 10:
                    channel = HIGH_SENSORS.index(sensor)
                    high[cycle, channel] = (raw.reshape(600, 10).mean(axis=1)
                                            if sensor['rate'] == 100 else raw)
                else:
                    low[cycle, LOW_SENSORS.index(sensor)] = raw
                rows += 1
        if rows != len(profile):
            raise ValueError(f"Missing {sensor['id']} cycles: {rows}/{len(profile)}.")
        sensor_files[path.name] = {'sha256': file_hash(path), 'bytes': path.stat().st_size}
    overlap = _fingerprints(features) & _fingerprints(training_features)
    if overlap:
        raise ValueError(f'{len(overlap)} cycle(s) exactly match UCI training features; '
                         'this is not an independent evaluation set.')
    inputs = {'features': features, 'high': high, 'low': low, 'targets': targets}
    sensor_files['profile.txt'] = {'sha256': file_hash(profile_path),
                                   'bytes': profile_path.stat().st_size}
    sensor_files['metadata.json'] = {'sha256': file_hash(metadata_path),
                                     'bytes': metadata_path.stat().st_size}
    source_hash = hashlib.sha256(json.dumps(sensor_files, sort_keys=True).encode()).hexdigest()
    return inputs, metadata, source_hash


def evaluate_external(directory):
    torch.set_num_threads(2)
    manifest = _read_candidates()
    training = prepare_data()
    if str(training['fingerprint']) != manifest['training_dataset_fingerprint']:
        raise ValueError('Current UCI source differs from the frozen candidate source.')
    external, metadata, source_hash = read_external(directory, training['features'])
    linear, tcn = load_candidates(manifest)
    indices = np.arange(len(external['targets']))
    linear_predictions = predict(linear, external['features'])[0]
    tcn_predictions = predict_tcn(tcn, external, external['high'], external['low'], indices)
    scores = {kind: evaluation_metrics(external['targets'], predictions)
              for kind, predictions in (('linear', linear_predictions),
                                        ('tcn', tcn_predictions))}
    report = {'evaluation_type': 'declared_independent_external_data',
              'source_fingerprint': source_hash, 'source_metadata': metadata,
              'training_dataset_fingerprint': manifest['training_dataset_fingerprint'],
              'candidate_manifest_sha256': file_hash(MANIFEST),
              'candidate_sha256': {kind: manifest['candidates'][kind]['sha256']
                                   for kind in ('linear', 'tcn')},
              'cycles': len(indices), 'scores': scores,
              'tcn_minus_linear_macro_f1': (scores['tcn']['mean_macro_f1'] -
                                            scores['linear']['mean_macro_f1']),
              'limitations': [
                  'Collection and label provenance are supplied by the data provider; '
                  'the software cannot independently verify the machine identity.',
                  'Both models classify complete 60-second cycles, not future faults.',
                  'Different sensor placement or operating regimes can invalidate a direct comparison.']}
    results = DIRECTORY / 'evaluations' / source_hash[:16]
    if results.exists():
        raise FileExistsError(f'Results already exist for this dataset: {results}.')
    results.mkdir(parents=True)
    np.savez_compressed(results / 'predictions.npz', targets=external['targets'],
                        linear_predictions=linear_predictions,
                        tcn_predictions=tcn_predictions)
    report['predictions_sha256'] = file_hash(results / 'predictions.npz')
    (results / 'report.json').write_text(json.dumps(report, indent=2) + '\n')
    rows = ['| Model | Cooler F1 | Valve F1 | Pump F1 | Accumulator F1 | Mean macro F1 | All four correct |',
            '|---|---:|---:|---:|---:|---:|---:|']
    for kind, label in (('linear', 'Linear'), ('tcn', 'Multirate TCN')):
        score = scores[kind]
        component_scores = ' | '.join(
            f"{score['components'][component['id']]['macro_f1']:.4f}"
            for component in COMPONENTS)
        rows.append(f"| {label} | {component_scores} | "
                    f"{score['mean_macro_f1']:.4f} | {score['all_components_correct']:.2%} |")
    text = '# External hydraulic evaluation\n\n'
    text += f"Source: {metadata['source_id']} · machine: {metadata['machine_id']} · "
    text += f"collection: {metadata['collection_period']} · {len(indices)} cycles.\n\n"
    text += '\n'.join(rows) + '\n\n'
    text += 'Both frozen candidates were fitted only on UCI data. No model or normalization was updated here.\n\n'
    text += '\n'.join('- ' + item for item in report['limitations']) + '\n'
    (results / 'report.md').write_text(text)
    return results, report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='command', required=True)
    commands.add_parser('prepare', help='Freeze fair full-UCI linear and TCN candidates.')
    evaluate = commands.add_parser('evaluate', help='Score separately collected labeled cycles.')
    evaluate.add_argument('data_dir', type=Path)
    arguments = parser.parse_args()
    if arguments.command == 'prepare':
        manifest = prepare_candidates()
        print(f"Frozen {manifest['training_cycles']} UCI-cycle candidates: {MANIFEST}")
        for kind, item in manifest['candidates'].items():
            print(f"  {kind}: {item['epochs']} epochs, {item['parameters']:,} parameters")
    else:
        location, report = evaluate_external(arguments.data_dir)
        print(f"External evaluation: {location / 'report.md'}")
        for kind, score in report['scores'].items():
            print(f"  {kind}: mean macro F1 {score['mean_macro_f1']:.4f}")


if __name__ == '__main__':
    main()
