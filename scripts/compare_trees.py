"""Compare small boosted trees with the current linear architecture on real cycles.

Run: python -m scripts.compare_trees
This is grouped cross-validation on development cycles, not a new external test.
The active dashboard model is never replaced by this experiment.
"""
import json
import time

import numpy as np
import torch
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import GroupKFold, GroupShuffleSplit
from threadpoolctl import threadpool_limits
from torch import nn

from hydraulics import (OUTPUT as ACTIVE_OUTPUT, COMPONENTS, FEATURE_NAMES,
                        evaluation_metrics, file_hash, prepare_data)
from model import HydraulicClassifier, predict
from scripts.compare_gru import experiment_split
from scripts.train_hydraulics import fit

OUTPUT = ACTIVE_OUTPUT / 'tree_experiment'
SETTINGS = {
    'version': 'grouped_tree_cv_v1', 'outer_folds': 4, 'inner_validation_fraction': .2,
    'split_seed': 20261105, 'training_seed': 42, 'linear_max_epochs': 250,
    'tree': {'learning_rate': .1, 'max_iter': 120, 'max_leaf_nodes': 15,
             'min_samples_leaf': 15, 'l2_regularization': 1.,
             'class_weight': 'balanced', 'early_stopping': False},
    'cpu_threads': 2,
}


def grouped_plan(data):
    """Use only cycles outside the three already inspected test partitions."""
    previous, _ = experiment_split(data['profile'])
    development = np.sort(np.r_[previous['train'], previous['validation']])
    excluded = np.setdiff1d(np.arange(len(data['profile'])), development)
    _, groups = np.unique(data['profile'][:, :4], axis=0, return_inverse=True)
    folds = []
    for outer_train, outer_validation in GroupKFold(SETTINGS['outer_folds']).split(
            development, groups=groups[development]):
        train, validation = development[outer_train], development[outer_validation]
        inner_train, inner_validation = next(GroupShuffleSplit(
            n_splits=1, test_size=SETTINGS['inner_validation_fraction'],
            random_state=SETTINGS['split_seed'] + len(folds)).split(
                train, groups=groups[train]))
        folds.append({'train': train.tolist(), 'validation': validation.tolist(),
                      'inner_train': train[inner_train].tolist(),
                      'inner_validation': train[inner_validation].tolist()})
    plan = {
        'dataset_fingerprint': str(data['fingerprint']), 'settings': SETTINGS,
        'feature_names': FEATURE_NAMES, 'development_indices': development.tolist(),
        'excluded_inspected_test_indices': excluded.tolist(), 'folds': folds,
        'comparison': 'Each outer fold fits both models from scratch using the same training cycles. '
                      'Linear epoch selection uses only an inner grouped validation; it is then '
                      'refitted on the complete outer training fold. Trees use fixed settings and '
                      'no internal random early-stopping split. Outer validation labels are scored once.',
        'deployment_policy': 'Research comparison only; never replace the active model automatically.',
        'limitation': 'All records come from the same previously explored test rig. Grouped cross-validation '
                      'is an internal estimate, not a fresh independent-machine test.',
    }
    OUTPUT.mkdir(parents=True, exist_ok=True)
    path = OUTPUT / 'protocol.json'
    if path.exists() and json.loads(path.read_text()) != plan:
        raise ValueError('Frozen tree protocol differs; do not silently change inspected folds.')
    if not path.exists():
        path.write_text(json.dumps(plan, indent=2) + '\n')
    return plan


def fit_linear_full(data, indices, epochs, seed):
    """Refit the existing linear architecture at an inner-selected epoch count."""
    torch.manual_seed(seed)
    x, y = data['features'], data['targets']
    mean = x[indices].mean(axis=0)
    scale = x[indices].std(axis=0).clip(1e-4)
    model = HydraulicClassifier(mean, scale, 'linear')
    optimizer = torch.optim.Adam(model.parameters(), lr=.02, weight_decay=1e-4)
    losses = []
    for component, meta in enumerate(COMPONENTS):
        counts = np.bincount(y[indices, component], minlength=len(meta['codes']))
        if np.any(counts == 0):
            raise ValueError('A training condition class is missing.')
        weight = len(indices) / (len(counts) * counts)
        losses.append(nn.CrossEntropyLoss(weight=torch.tensor(weight, dtype=torch.float32)))
    rng = np.random.default_rng(seed)
    for _ in range(epochs):
        model.train()
        for batch in np.array_split(rng.permutation(indices),
                                    max(1, int(np.ceil(len(indices) / 128)))):
            optimizer.zero_grad(set_to_none=True)
            heads = model(torch.from_numpy(x[batch]))
            loss = sum(losses[i](heads[i], torch.from_numpy(y[batch, i]))
                       for i in range(4)) / 4
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 3)
            optimizer.step()
    model.eval()
    return model


def fit_trees(data, indices):
    features, targets = data['features'], data['targets']
    trees = []
    for head in range(4):
        tree = HistGradientBoostingClassifier(**SETTINGS['tree'], random_state=SETTINGS['training_seed'])
        tree.fit(features[indices], targets[indices, head])
        trees.append(tree)
    return trees


def main():
    torch.set_num_threads(SETTINGS['cpu_threads'])
    torch.use_deterministic_algorithms(True)
    started = time.perf_counter()
    data = prepare_data()
    plan = grouped_plan(data)
    active_hashes = {str(path.relative_to(ACTIVE_OUTPUT)): file_hash(path)
                     for path in ACTIVE_OUTPUT.glob('*') if path.is_file()}
    development = np.array(plan['development_indices'])
    locations = {int(index): offset for offset, index in enumerate(development)}
    saved = {'cycle_indices': development, 'targets': data['targets'][development]}
    predictions = {kind: np.full((len(development), 4), -1, dtype=np.int64)
                   for kind in ('linear', 'boosted_trees')}
    fold_results = []
    with threadpool_limits(limits=SETTINGS['cpu_threads']):
        for number, fold in enumerate(plan['folds'], 1):
            train = np.array(fold['train'])
            validation = np.array(fold['validation'])
            inner = {'train': np.array(fold['inner_train']),
                     'validation': np.array(fold['inner_validation'])}
            began = time.perf_counter()
            _, inner_fit = fit('linear', data, inner, SETTINGS['linear_max_epochs'],
                               SETTINGS['training_seed'])
            epoch = inner_fit['best_epoch']
            linear = fit_linear_full(data, train, epoch, SETTINGS['training_seed'])
            linear_prediction = predict(linear, data['features'][validation])[0]
            linear_seconds = time.perf_counter() - began
            began = time.perf_counter()
            trees = fit_trees(data, train)
            tree_prediction = np.column_stack([
                tree.predict(data['features'][validation]) for tree in trees])
            tree_seconds = time.perf_counter() - began
            positions = [locations[int(index)] for index in validation]
            predictions['linear'][positions] = linear_prediction
            predictions['boosted_trees'][positions] = tree_prediction
            result = {
                'fold': number, 'train_cycles': len(train), 'validation_cycles': len(validation),
                'inner_selected_linear_epoch': epoch,
                'linear': {**evaluation_metrics(data['targets'][validation], linear_prediction),
                           'seconds': round(linear_seconds, 3)},
                'boosted_trees': {**evaluation_metrics(data['targets'][validation], tree_prediction),
                                  'seconds': round(tree_seconds, 3)},
            }
            fold_results.append(result)
            print(f"Fold {number}: linear {result['linear']['mean_macro_f1']:.3f}; "
                  f"boosted trees {result['boosted_trees']['mean_macro_f1']:.3f}", flush=True)
    if any(np.any(values < 0) for values in predictions.values()):
        raise RuntimeError('Missing out-of-fold predictions.')
    for kind, values in predictions.items():
        saved[f'{kind}_predictions'] = values
    np.savez_compressed(OUTPUT / 'out_of_fold_predictions.npz', **saved)
    summary = {}
    for kind, values in predictions.items():
        summary[kind] = {
            'fold_mean_macro_f1': float(np.mean([fold[kind]['mean_macro_f1'] for fold in fold_results])),
            'fold_macro_f1_std': float(np.std([fold[kind]['mean_macro_f1'] for fold in fold_results])),
            'fold_accumulator_macro_f1': float(np.mean([
                fold[kind]['components']['accumulator']['macro_f1'] for fold in fold_results])),
            'out_of_fold': evaluation_metrics(data['targets'][development], values),
            'total_fit_seconds': round(sum(fold[kind]['seconds'] for fold in fold_results), 3),
        }
    better_folds = sum(fold['boosted_trees']['mean_macro_f1'] > fold['linear']['mean_macro_f1']
                       for fold in fold_results)
    report = {
        'source': 'real_test_rig', 'purpose': 'Grouped boosted-tree comparison',
        'dataset_fingerprint': str(data['fingerprint']),
        'protocol_sha256': file_hash(OUTPUT / 'protocol.json'),
        'predictions_sha256': file_hash(OUTPUT / 'out_of_fold_predictions.npz'),
        'development_cycles': len(development),
        'development_groups': len(np.unique(data['profile'][development, :4], axis=0)),
        'excluded_test_cycles': len(plan['excluded_inspected_test_indices']),
        'models': summary, 'folds': fold_results, 'tree_better_folds': better_folds,
        'decision': 'research_only_retain_deployment',
        'seconds': round(time.perf_counter() - started, 3),
        'limitations': [plan['limitation'],
            'Four grouped folds measure variation across these condition combinations, not variation across machines or independent runs.',
            'All input features summarize the complete 60-second cycle; these scores are condition classification, not future failure prediction.',
            'The original deployed 0.950 test score uses a different protocol and cannot be ranked directly against these cross-validation scores.'],
    }
    for relative, sha in active_hashes.items():
        if file_hash(ACTIVE_OUTPUT / relative) != sha:
            raise RuntimeError('An active model artifact changed during the experiment.')
    (OUTPUT / 'metrics.json').write_text(json.dumps(report, indent=2) + '\n')
    lines = [
        '# Boosted-tree comparison on measured hydraulic cycles', '',
        f"Grouped 4-fold comparison on {report['development_cycles']} development cycles "
        f"from {report['development_groups']} condition combinations. "
        f"The {report['excluded_test_cycles']} cycles in three previously inspected test partitions were excluded.", '',
        '| Model | Fold mean macro F1 | Out-of-fold macro F1 | Out-of-fold accumulator F1 | All four correct | Total fitting time |',
        '|---|---:|---:|---:|---:|---:|',
    ]
    for kind, label in [('linear', 'Linear architecture, refitted'),
                        ('boosted_trees', 'Histogram gradient-boosted trees')]:
        item = summary[kind]
        oof = item['out_of_fold']
        lines.append(f"| {label} | {item['fold_mean_macro_f1']:.4f} | "
                     f"{oof['mean_macro_f1']:.4f} | "
                     f"{oof['components']['accumulator']['macro_f1']:.4f} | "
                     f"{oof['all_components_correct']:.2%} | {item['total_fit_seconds']:.1f}s |")
    lines += ['', f"Trees beat linear on {better_folds} of 4 grouped folds. "
              'The active dashboard model is unchanged.', '',
              'The linear model chooses its epoch on an inner grouped validation, then refits on the full outer training fold. '
              'The tree settings are fixed before evaluating any outer fold; there is no random internal early stopping split. '
              'Both models use the same 140 native-sample summary features.', '',
              *['- ' + limitation for limitation in report['limitations']], '']
    (OUTPUT / 'report.md').write_text('\n'.join(lines))
    print('\n'.join(lines), flush=True)


if __name__ == '__main__':
    main()
