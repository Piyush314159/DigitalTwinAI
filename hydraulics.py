"""Real UCI hydraulic cycles, compact features and provenance checks.

Rows are 60-second experimental cycles. Labels in profile.txt are evaluation
annotations, never input features. No synthetic measurements are generated.
"""
from pathlib import Path
import hashlib
import json
import os

import numpy as np

ROOT = Path(__file__).resolve().parent
OUTPUT = ROOT / 'training_outputs'
CACHE = ROOT / 'data' / 'processed' / 'hydraulics.npz'
SOURCE_URL = 'https://archive.ics.uci.edu/dataset/447/condition+monitoring+of+hydraulic+systems'
SENSORS = [
    *[{'id': f'PS{i}', 'name': f'Pressure {i}', 'unit': 'bar', 'rate': 100, 'group': 'Pressure'} for i in range(1, 7)],
    {'id': 'EPS1', 'name': 'Motor power', 'unit': 'W', 'rate': 100, 'group': 'Power'},
    *[{'id': f'FS{i}', 'name': f'Volume flow {i}', 'unit': 'L/min', 'rate': 10, 'group': 'Flow'} for i in range(1, 3)],
    *[{'id': f'TS{i}', 'name': f'Temperature {i}', 'unit': '°C', 'rate': 1, 'group': 'Temperature'} for i in range(1, 5)],
    {'id': 'VS1', 'name': 'Vibration', 'unit': 'mm/s', 'rate': 1, 'group': 'Vibration'},
]
COMPONENTS = [
    {'id': 'cooler', 'name': 'Cooler', 'codes': [100, 20, 3],
     'labels': ['Full efficiency', 'Reduced efficiency', 'Near-total failure'], 'unit': '% efficiency'},
    {'id': 'valve', 'name': 'Valve', 'codes': [100, 90, 80, 73],
     'labels': ['Optimal switching', 'Small switching lag', 'Severe switching lag', 'Near-total failure'], 'unit': '% condition'},
    {'id': 'pump', 'name': 'Pump', 'codes': [0, 1, 2],
     'labels': ['No internal leakage', 'Weak internal leakage', 'Severe internal leakage'], 'unit': 'leakage class'},
    {'id': 'accumulator', 'name': 'Accumulator', 'codes': [130, 115, 100, 90],
     'labels': ['Optimal pressure', 'Slightly reduced pressure', 'Severely reduced pressure', 'Near-total failure'], 'unit': 'bar'},
]
FEATURE_STATS = ['mean', 'std', 'min', 'max', *[f'bin{i+1}_mean' for i in range(6)]]
FEATURE_NAMES = [f"{s['id']}_{stat}" for s in SENSORS for stat in FEATURE_STATS]


def file_hash(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def find_raw_data():
    if os.environ.get('HYDRAULICS_DATA_DIR'):
        directory = Path(os.environ['HYDRAULICS_DATA_DIR']).expanduser().resolve()
    else:
        candidates = [p.parent for p in (ROOT / 'data').glob('**/profile.txt')
                      if all((p.parent / f"{s['id']}.txt").is_file() for s in SENSORS)]
        if len(candidates) != 1:
            raise FileNotFoundError('Place the UCI hydraulic dataset under data/, or set HYDRAULICS_DATA_DIR.')
        directory = candidates[0]
    for name in ['profile.txt', *[s['id'] + '.txt' for s in SENSORS]]:
        if not (directory / name).is_file():
            raise FileNotFoundError(directory / name)
    return directory


def source_manifest(directory):
    return {p.name: {'sha256': file_hash(p), 'bytes': p.stat().st_size}
            for p in sorted(Path(directory).glob('*.txt'))}


def summarize_cycle(values):
    """Raw-signal statistics and six consecutive 10-second means."""
    values = np.asarray(values, dtype=np.float64)
    if values.ndim != 1 or not np.isfinite(values).all() or len(values) % 60:
        raise ValueError('Each sensor row must contain 60 seconds of finite, regular samples.')
    return np.array([values.mean(), values.std(), values.min(), values.max(),
                     *values.reshape(6, -1).mean(axis=1)], dtype=np.float32)


def prepare_data(force=False):
    """Stream one row at a time; cache is invalidated by raw-file hashes."""
    directory = find_raw_data()
    manifest = source_manifest(directory)
    fingerprint = hashlib.sha256(json.dumps(manifest, sort_keys=True).encode()).hexdigest()
    if CACHE.exists() and not force:
        with np.load(CACHE, allow_pickle=False) as cache:
            if str(cache['fingerprint']) == fingerprint:
                return {k: cache[k] for k in cache.files}
    profile = np.loadtxt(directory / 'profile.txt', dtype=np.int64)
    if profile.shape != (2205, 5):
        raise ValueError(f'Expected the full UCI profile (2205, 5); found {profile.shape}.')
    targets = np.empty((len(profile), 4), dtype=np.int64)
    for i, component in enumerate(COMPONENTS):
        if not set(profile[:, i]).issubset(component['codes']):
            raise ValueError(f"Unexpected {component['name']} condition code.")
        targets[:, i] = [component['codes'].index(int(code)) for code in profile[:, i]]
    features = np.empty((len(profile), len(FEATURE_NAMES)), dtype=np.float32)
    traces = np.empty((len(profile), len(SENSORS), 60), dtype=np.float32)
    for i, sensor in enumerate(SENSORS):
        print(f"Reading measured channel {sensor['id']}...", flush=True)
        rows = 0
        with (directory / f"{sensor['id']}.txt").open() as stream:
            for row, line in enumerate(stream):
                if row >= len(profile):
                    raise ValueError('Sensor/profile row count mismatch.')
                raw = np.fromstring(line, sep='\t', dtype=np.float64)
                if raw.size != 60 * sensor['rate']:
                    raise ValueError(f"Wrong sample count: {sensor['id']} cycle {row+1}.")
                features[row, i*10:(i+1)*10] = summarize_cycle(raw)
                traces[row, i] = raw.reshape(60, sensor['rate']).mean(axis=1)
                rows += 1
        if rows != len(profile):
            raise ValueError(f"Missing rows in {sensor['id']}.")
    CACHE.parent.mkdir(parents=True, exist_ok=True)
    result = {'features': features, 'traces': traces, 'targets': targets,
              'profile': profile, 'fingerprint': np.array(fingerprint),
              'manifest_json': np.array(json.dumps(manifest, sort_keys=True))}
    np.savez_compressed(CACHE, **result)
    return result


def grouped_split(profile, seed=42):
    """Hold out whole joint condition combinations, including all repeated cycles.

    This is a same-rig, held-out condition-combination experiment. It is not
    chronological future prediction or an independent-machine test.
    """
    _, groups = np.unique(profile[:, :4], axis=0, return_inverse=True)
    order = np.random.default_rng(seed).permutation(int(groups.max()) + 1)
    a, b = int(len(order)*0.6), int(len(order)*0.8)
    return {name: np.flatnonzero(np.isin(groups, selected)) for name, selected in (
        ('train', order[:a]), ('validation', order[a:b]), ('test', order[b:]))}


def component_metrics(truth, predicted, classes):
    confusion = np.zeros((classes, classes), dtype=int)
    np.add.at(confusion, (np.asarray(truth, dtype=int), np.asarray(predicted, dtype=int)), 1)
    tp = confusion.diagonal()
    precision = np.divide(tp, confusion.sum(axis=0), out=np.zeros(classes, float), where=confusion.sum(axis=0)>0)
    recall = np.divide(tp, confusion.sum(axis=1), out=np.zeros(classes, float), where=confusion.sum(axis=1)>0)
    f1 = np.divide(2*precision*recall, precision+recall, out=np.zeros(classes, float), where=(precision+recall)>0)
    return {'accuracy': float(tp.sum()/max(confusion.sum(), 1)), 'macro_f1': float(f1.mean()),
            'per_class_precision': precision.tolist(), 'per_class_recall': recall.tolist(),
            'support': confusion.sum(axis=1).tolist(), 'confusion': confusion.tolist()}


def evaluation_metrics(targets, predictions):
    per_component = {c['id']: component_metrics(targets[:, i], predictions[:, i], len(c['codes']))
                     for i, c in enumerate(COMPONENTS)}
    return {'mean_macro_f1': float(np.mean([m['macro_f1'] for m in per_component.values()])),
            'mean_accuracy': float(np.mean([m['accuracy'] for m in per_component.values()])),
            'all_components_correct': float(np.mean(np.all(targets == predictions, axis=1))),
            'cycles': int(len(targets)), 'components': per_component}
