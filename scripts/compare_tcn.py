"""CPU-only comparison of a small multirate TCN and the retained linear model.

Run: python -m scripts.compare_tcn
This is exploratory grouped cross-validation on the already studied UCI rig.
The dashboard checkpoint and its original evaluation are never replaced.
"""
import json
import time

import numpy as np
import torch
from torch import nn

from hydraulics import (CACHE, COMPONENTS, FEATURE_NAMES, OUTPUT as ACTIVE_OUTPUT,
                        SENSORS, evaluation_metrics, file_hash, find_raw_data,
                        prepare_data)
from model import HydraulicClassifier, predict
from scripts.train_hydraulics import fit as fit_linear_with_validation

OUTPUT = ACTIVE_OUTPUT / 'tcn_experiment'
SIGNAL_CACHE = CACHE.parent / 'tcn_multirate.npz'
HIGH_SENSORS = [sensor for sensor in SENSORS if sensor['rate'] >= 10]
LOW_SENSORS = [sensor for sensor in SENSORS if sensor['rate'] == 1]
SETTINGS = {
    'version': 'multirate_tcn_grouped_cv_v1',
    'outer_folds': 3, 'split_seed': 20261106, 'training_seed': 42,
    'linear_max_epochs': 250, 'tcn_max_epochs': 80, 'patience': 16,
    'batch_size': 64, 'learning_rate': .002, 'weight_decay': .0001,
    'gradient_clip': 3., 'cpu_threads': 2,
    'high_channels': 16, 'low_channels': 8, 'fusion_units': 32,
    'high_dilations': [1, 2],
    'outer_score': 'mean component macro F1; report accumulator and all-four-correct separately',
    'deployment_policy': 'No automatic replacement of training_outputs/model.pt.',
}


def measured_multirate(data):
    """Stream original measurements into 10 Hz and 1 Hz branches.

    A 10-sample arithmetic mean reduces the 100 Hz pressure and power signals
    to 10 Hz. Native 10 Hz flow and native 1 Hz temperature/vibration are kept.
    No profile annotation or simulated signal enters either branch.
    """
    fingerprint = str(data['fingerprint'])
    high_ids = [sensor['id'] for sensor in HIGH_SENSORS]
    low_ids = [sensor['id'] for sensor in LOW_SENSORS]
    if SIGNAL_CACHE.exists():
        with np.load(SIGNAL_CACHE, allow_pickle=False) as saved:
            if (str(saved['fingerprint']) == fingerprint
                    and saved['high_ids'].tolist() == high_ids
                    and saved['low_ids'].tolist() == low_ids
                    and saved['high'].shape == (len(data['profile']), len(high_ids), 600)
                    and saved['low'].shape == (len(data['profile']), len(low_ids), 60)):
                return saved['high'], saved['low']
    directory = find_raw_data()
    cycles = len(data['profile'])
    high = np.empty((cycles, len(high_ids), 600), dtype=np.float32)
    low = np.empty((cycles, len(low_ids), 60), dtype=np.float32)
    for branch, sensors, output in (('high', HIGH_SENSORS, high), ('low', LOW_SENSORS, low)):
        for channel, sensor in enumerate(sensors):
            rows = 0
            with (directory / f"{sensor['id']}.txt").open() as stream:
                for cycle, line in enumerate(stream):
                    if cycle >= cycles:
                        raise ValueError(f"Too many {sensor['id']} cycles.")
                    raw = np.fromstring(line, sep='\t', dtype=np.float64)
                    if raw.size != 60 * sensor['rate'] or not np.isfinite(raw).all():
                        raise ValueError(f"Invalid {sensor['id']} cycle {cycle+1}.")
                    output[cycle, channel] = (raw.reshape(600, 10).mean(axis=1)
                                              if sensor['rate'] == 100 else raw)
                    rows += 1
            if rows != cycles:
                raise ValueError(f"Missing {sensor['id']} cycles.")
            print(f'Prepared {branch}-rate measured channel {sensor["id"]}.', flush=True)
    SIGNAL_CACHE.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(SIGNAL_CACHE, high=high, low=low,
                        high_ids=np.array(high_ids), low_ids=np.array(low_ids),
                        fingerprint=np.array(fingerprint))
    return high, low


def grouped_folds(profile, folds, seed):
    """Balance cycles across folds while keeping joint conditions intact."""
    _, groups = np.unique(profile[:, :4], axis=0, return_inverse=True)
    counts = np.bincount(groups)
    order = np.random.default_rng(seed).permutation(len(counts))
    order = sorted(order, key=lambda group: -counts[group])
    allocation = np.empty(len(counts), dtype=np.int64)
    sizes = np.zeros(folds, dtype=np.int64)
    for group in order:
        fold = int(np.argmin(sizes))
        allocation[group] = fold
        sizes[fold] += counts[group]
    return [np.flatnonzero(allocation[groups] == fold) for fold in range(folds)]


def frozen_protocol(data):
    outer = grouped_folds(data['profile'], SETTINGS['outer_folds'], SETTINGS['split_seed'])
    partitions = []
    for number, validation in enumerate(outer):
        train = np.setdiff1d(np.arange(len(data['profile'])), validation)
        inner_local = grouped_folds(data['profile'][train], 5,
                                    SETTINGS['split_seed'] + 1 + number)
        inner_validation = train[inner_local[0]]
        inner_train = np.setdiff1d(train, inner_validation)
        partitions.append({'train': train.tolist(), 'validation': validation.tolist(),
                           'inner_train': inner_train.tolist(),
                           'inner_validation': inner_validation.tolist()})
    plan = {
        'dataset_fingerprint': str(data['fingerprint']), 'settings': SETTINGS,
        'feature_names': FEATURE_NAMES,
        'high_sensor_ids': [sensor['id'] for sensor in HIGH_SENSORS],
        'low_sensor_ids': [sensor['id'] for sensor in LOW_SENSORS],
        'inputs': 'Measured 10 Hz pressure/power (100 Hz block means), native 10 Hz flow, '
                  'native 1 Hz temperature/vibration, plus 140 native-sample summary features.',
        'folds': partitions,
        'selection': 'Inner grouped validation chooses each model training epoch. Each model '
                     'then refits on the complete outer training fold. Outer folds are scored once.',
        'limitation': 'All 2,205 cycles belong to the previously explored same-rig dataset. '
                      'This is an internal exploratory comparison, not a new external test.'}
    OUTPUT.mkdir(parents=True, exist_ok=True)
    path = OUTPUT / 'protocol.json'
    if path.exists():
        if json.loads(path.read_text()) != plan:
            raise ValueError('Frozen TCN protocol differs; do not tune against inspected folds.')
    else:
        path.write_text(json.dumps(plan, indent=2) + '\n')
    return plan


class ResidualBlock(nn.Module):
    def __init__(self, channels, dilation):
        super().__init__()
        pad = 2 * dilation
        self.first = nn.Conv1d(channels, channels, 5, padding=pad, dilation=dilation)
        self.second = nn.Conv1d(channels, channels, 5, padding=pad, dilation=dilation)
        self.activation = nn.ELU()

    def forward(self, sequence):
        return self.activation(sequence + self.second(self.activation(self.first(sequence))))


class HydraulicTCN(nn.Module):
    def __init__(self, feature_mean, feature_scale, high_mean, high_scale,
                 low_mean, low_scale):
        super().__init__()
        for name, values in (('feature_mean', feature_mean), ('feature_scale', feature_scale),
                             ('high_mean', high_mean), ('high_scale', high_scale),
                             ('low_mean', low_mean), ('low_scale', low_scale)):
            self.register_buffer(name, torch.as_tensor(values, dtype=torch.float32).clone())
        width = SETTINGS['high_channels']
        self.high_stem = nn.Sequential(nn.Conv1d(len(HIGH_SENSORS), width, 5,
                                                 stride=2, padding=2), nn.ELU())
        self.high_blocks = nn.Sequential(*[ResidualBlock(width, dilation)
                                           for dilation in SETTINGS['high_dilations']])
        self.low_branch = nn.Sequential(nn.Conv1d(len(LOW_SENSORS),
                                                  SETTINGS['low_channels'], 3, padding=1),
                                         nn.ELU())
        pooled = 2 * (width + SETTINGS['low_channels']) + len(FEATURE_NAMES)
        self.output = nn.Sequential(nn.Linear(pooled, SETTINGS['fusion_units']), nn.ELU(),
                                    nn.Dropout(.1), nn.Linear(SETTINGS['fusion_units'],
                                                             sum(len(c['codes']) for c in COMPONENTS)))

    def forward(self, features, high, low):
        if (features.ndim != 2 or features.shape[1] != len(FEATURE_NAMES)
                or high.ndim != 3 or tuple(high.shape[1:]) != (len(HIGH_SENSORS), 600)
                or low.ndim != 3 or tuple(low.shape[1:]) != (len(LOW_SENSORS), 60)):
            raise ValueError('TCN input schema must be 140 features, 9×600 and 5×60 signals.')
        if not all(torch.isfinite(part).all() for part in (features, high, low)):
            raise ValueError('TCN cannot estimate incomplete measured cycles.')
        summary = ((features - self.feature_mean) / self.feature_scale).clamp(-8, 8)
        h = self.high_blocks(self.high_stem(((high - self.high_mean) / self.high_scale).clamp(-8, 8)))
        l = self.low_branch(((low - self.low_mean) / self.low_scale).clamp(-8, 8))
        pooled = torch.cat((h.mean(-1), h.amax(-1), l.mean(-1), l.amax(-1), summary), dim=1)
        logits = self.output(pooled)
        return logits.split([len(c['codes']) for c in COMPONENTS], dim=-1)


def new_tcn(data, high, low, train):
    features = data['features']
    return HydraulicTCN(
        features[train].mean(axis=0), features[train].std(axis=0).clip(1e-4),
        high[train].mean(axis=(0, 2))[:, None], high[train].std(axis=(0, 2)).clip(1e-4)[:, None],
        low[train].mean(axis=(0, 2))[:, None], low[train].std(axis=(0, 2)).clip(1e-4)[:, None])


@torch.inference_mode()
def predict_tcn(model, data, high, low, indices):
    model.eval()
    results = []
    for batch in np.array_split(indices, max(1, int(np.ceil(len(indices) / 128)))):
        heads = model(torch.from_numpy(data['features'][batch]),
                      torch.from_numpy(high[batch]), torch.from_numpy(low[batch]))
        results.append(np.column_stack([head.argmax(dim=1).numpy() for head in heads]))
    return np.concatenate(results)


def train_tcn(data, high, low, train, validation=None, epochs=None):
    torch.manual_seed(SETTINGS['training_seed'])
    model = new_tcn(data, high, low, train)
    optimizer = torch.optim.Adam(model.parameters(), lr=SETTINGS['learning_rate'],
                                 weight_decay=SETTINGS['weight_decay'])
    losses = []
    for head, component in enumerate(COMPONENTS):
        counts = np.bincount(data['targets'][train, head], minlength=len(component['codes']))
        if np.any(counts == 0):
            raise ValueError('A TCN training class is absent.')
        weights = len(train) / (len(counts) * counts)
        losses.append(nn.CrossEntropyLoss(weight=torch.tensor(weights, dtype=torch.float32)))
    rng = np.random.default_rng(SETTINGS['training_seed'])
    best_epoch, best_score, stale, history = 0, -1., 0, []
    limit = epochs or SETTINGS['tcn_max_epochs']
    for epoch in range(1, limit + 1):
        model.train()
        order = rng.permutation(train)
        for offset in range(0, len(order), SETTINGS['batch_size']):
            batch = order[offset:offset + SETTINGS['batch_size']]
            optimizer.zero_grad(set_to_none=True)
            heads = model(torch.from_numpy(data['features'][batch]),
                          torch.from_numpy(high[batch]), torch.from_numpy(low[batch]))
            loss = sum(losses[i](heads[i], torch.from_numpy(data['targets'][batch, i]))
                       for i in range(4)) / 4
            if not torch.isfinite(loss):
                raise RuntimeError('Non-finite TCN loss.')
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), SETTINGS['gradient_clip'])
            optimizer.step()
        if validation is not None:
            predictions = predict_tcn(model, data, high, low, validation)
            score = evaluation_metrics(data['targets'][validation], predictions)['mean_macro_f1']
            history.append({'epoch': epoch, 'inner_mean_macro_f1': score})
            if score > best_score + 1e-6:
                best_epoch, best_score, stale = epoch, score, 0
            else:
                stale += 1
            if epoch == 1 or epoch % 10 == 0:
                print(f'TCN inner epoch {epoch}: macro F1 {score:.3f}', flush=True)
            if stale >= SETTINGS['patience']:
                break
    model.eval()
    return model, {'selected_epoch': best_epoch if validation is not None else limit,
                   'inner_score': best_score if validation is not None else None,
                   'history': history, 'epochs_run': epoch,
                   'parameters': sum(parameter.numel() for parameter in model.parameters())}


def fit_linear_full(data, train, epochs):
    """Refit the existing optimizer from scratch at an inner-selected epoch."""
    torch.manual_seed(SETTINGS['training_seed'])
    x, y = data['features'], data['targets']
    model = HydraulicClassifier(x[train].mean(axis=0), x[train].std(axis=0).clip(1e-4))
    optimizer = torch.optim.Adam(model.parameters(), lr=.02, weight_decay=1e-4)
    losses = []
    for head, component in enumerate(COMPONENTS):
        counts = np.bincount(y[train, head], minlength=len(component['codes']))
        if np.any(counts == 0):
            raise ValueError('A linear training class is absent.')
        losses.append(nn.CrossEntropyLoss(weight=torch.tensor(len(train)/(len(counts)*counts),
                                                       dtype=torch.float32)))
    rng = np.random.default_rng(SETTINGS['training_seed'])
    for _ in range(epochs):
        model.train()
        order = rng.permutation(train)
        for offset in range(0, len(order), 128):
            batch = order[offset:offset+128]
            optimizer.zero_grad(set_to_none=True)
            heads = model(torch.from_numpy(x[batch]))
            loss = sum(losses[i](heads[i], torch.from_numpy(y[batch, i])) for i in range(4))/4
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 3)
            optimizer.step()
    model.eval()
    return model


def main():
    torch.set_num_threads(SETTINGS['cpu_threads'])
    torch.use_deterministic_algorithms(True)
    started = time.perf_counter()
    data = prepare_data()
    plan = frozen_protocol(data)
    active_hashes = {str(path.relative_to(ACTIVE_OUTPUT)): file_hash(path)
                     for path in ACTIVE_OUTPUT.glob('*') if path.is_file()}
    high, low = measured_multirate(data)
    cycle_indices = np.arange(len(data['profile']))
    oof = {'linear': np.full((len(cycle_indices), 4), -1, dtype=np.int64),
           'tcn': np.full((len(cycle_indices), 4), -1, dtype=np.int64)}
    folds = []
    for number, partition in enumerate(plan['folds'], 1):
        train = np.array(partition['train'])
        validation = np.array(partition['validation'])
        inner = {'train': np.array(partition['inner_train']),
                 'validation': np.array(partition['inner_validation'])}
        began = time.perf_counter()
        _, linear_selection = fit_linear_with_validation(
            data, inner, SETTINGS['linear_max_epochs'], SETTINGS['training_seed'])
        linear = fit_linear_full(data, train, linear_selection['best_epoch'])
        linear_predictions = predict(linear, data['features'][validation])[0]
        linear_time = round(time.perf_counter() - began, 3)
        began = time.perf_counter()
        _, tcn_selection = train_tcn(data, high, low, inner['train'], inner['validation'])
        tcn, _ = train_tcn(data, high, low, train, epochs=tcn_selection['selected_epoch'])
        tcn_predictions = predict_tcn(tcn, data, high, low, validation)
        tcn_time = round(time.perf_counter() - began, 3)
        oof['linear'][validation] = linear_predictions
        oof['tcn'][validation] = tcn_predictions
        fold = {'fold': number, 'train_cycles': len(train), 'validation_cycles': len(validation),
                'train_groups': len(np.unique(data['profile'][train, :4], axis=0)),
                'validation_groups': len(np.unique(data['profile'][validation, :4], axis=0)),
                'linear': {'inner_selected_epoch': linear_selection['best_epoch'],
                           'seconds': linear_time,
                           'validation': evaluation_metrics(data['targets'][validation], linear_predictions)},
                'tcn': {'inner_selected_epoch': tcn_selection['selected_epoch'],
                        'inner_best_macro_f1': tcn_selection['inner_score'],
                        'parameters': tcn_selection['parameters'], 'seconds': tcn_time,
                        'history': tcn_selection['history'],
                        'validation': evaluation_metrics(data['targets'][validation], tcn_predictions)}}
        folds.append(fold)
        print(f"Outer fold {number}: linear {fold['linear']['validation']['mean_macro_f1']:.3f}; "
              f"TCN {fold['tcn']['validation']['mean_macro_f1']:.3f}.", flush=True)
    if any(np.any(values < 0) for values in oof.values()):
        raise RuntimeError('Missing out-of-fold predictions.')
    predictions_path = OUTPUT / 'out_of_fold_predictions.npz'
    np.savez_compressed(predictions_path, cycle_indices=cycle_indices,
                        targets=data['targets'], linear_predictions=oof['linear'],
                        tcn_predictions=oof['tcn'])
    models = {}
    for kind, predicted in oof.items():
        scores = [fold[kind]['validation']['mean_macro_f1'] for fold in folds]
        models[kind] = {'fold_mean_macro_f1': float(np.mean(scores)),
                        'fold_macro_f1_std': float(np.std(scores)),
                        'out_of_fold': evaluation_metrics(data['targets'], predicted),
                        'total_fit_seconds': round(sum(fold[kind]['seconds'] for fold in folds), 3)}
    models['linear']['parameters'] = sum(parameter.numel() for parameter in linear.parameters())
    models['tcn']['parameters'] = folds[0]['tcn']['parameters']
    tcn_better_folds = sum(fold['tcn']['validation']['mean_macro_f1'] >
                           fold['linear']['validation']['mean_macro_f1'] for fold in folds)
    report = {'source': 'real_test_rig', 'purpose': 'Small multirate TCN comparison',
              'dataset_fingerprint': str(data['fingerprint']),
              'protocol_sha256': file_hash(OUTPUT/'protocol.json'),
              'predictions_sha256': file_hash(predictions_path),
              'signal_cache_sha256': file_hash(SIGNAL_CACHE),
              'folds': folds, 'models': models,
              'tcn_better_folds': tcn_better_folds,
              'decision': 'research_only_retain_deployment',
              'seconds': round(time.perf_counter()-started, 3),
              'limitations': [plan['limitation'],
                  'Pressure and power were reduced from 100 Hz to 10 Hz by block averaging; high-frequency transients may be lost.',
                  'The TCN consumes the complete 60-second cycle and cannot predict future failures or remaining useful life.',
                  'The original deployed 0.950 score uses a different split and cannot be directly compared with these cross-validation values.']}
    for relative, digest in active_hashes.items():
        if file_hash(ACTIVE_OUTPUT/relative) != digest:
            raise RuntimeError('The active model or its evidence changed during this experiment.')
    (OUTPUT/'metrics.json').write_text(json.dumps(report, indent=2)+'\n')
    rows = ['| Model | Parameters | Fold mean macro F1 | Out-of-fold macro F1 | Accumulator F1 | All four correct | Fit time |',
            '|---|---:|---:|---:|---:|---:|---:|']
    for kind, label in [('linear', 'Linear, refitted'), ('tcn', 'Two-block multirate TCN')]:
        result = models[kind]
        pooled = result['out_of_fold']
        rows.append(f"| {label} | {result['parameters']:,} | {result['fold_mean_macro_f1']:.4f} | "
                    f"{pooled['mean_macro_f1']:.4f} | {pooled['components']['accumulator']['macro_f1']:.4f} | "
                    f"{pooled['all_components_correct']:.2%} | {result['total_fit_seconds']:.1f}s |")
    text = '# Multirate TCN on measured hydraulic cycles\n\n'
    text += 'Three grouped outer folds across all 2,205 measured cycles. Every repeated joint condition combination stays in one fold. Both models use the same outer training and validation cycles. Training epochs are chosen on an inner grouped partition and refitted on the full outer training fold.\n\n'
    text += '\n'.join(rows)+'\n\n'
    text += f'TCN beats linear on {tcn_better_folds} of 3 folds. The deployed dashboard model remains unchanged.\n\n'
    text += 'Input: 9 physical high-rate channels at 10 Hz (pressure/power block averages and native flow), 5 native 1 Hz channels, and 140 measured-sample summary features. Two residual dilated temporal blocks, 16 high-rate channels, 8 low-rate channels and a 32-unit fusion head. No simulated signals or annotation features.\n\n'
    diagnostic_path = OUTPUT/'linear_250_diagnostic.json'
    if diagnostic_path.exists():
        diagnostic = json.loads(diagnostic_path.read_text())
        text += (f"Post-hoc baseline sensitivity: refitting linear for 250 epochs on each outer "
                 f"training fold yielded fold-mean macro F1 {diagnostic['fold_mean_macro_f1']:.4f}. "
                 'This check was added after the first outer-fold results were observed and is not '
                 'a new held-out evaluation. Exact predictions are saved beside this report.\n\n')
    text += '\n'.join('- '+item for item in report['limitations'])+'\n'
    (OUTPUT/'report.md').write_text(text)
    print(text, flush=True)


if __name__ == '__main__':
    main()
