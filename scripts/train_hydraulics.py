"""Train on downloaded UCI measurements: python -m scripts.train_hydraulics."""
import argparse
import copy
import json
import time

import numpy as np
import torch
from torch import nn

from hydraulics import OUTPUT, COMPONENTS, FEATURE_NAMES, SOURCE_URL, prepare_data, grouped_split, evaluation_metrics, file_hash
from model import HydraulicClassifier, predict, load_model


def fit(data, split, epochs, seed):
    torch.manual_seed(seed)
    x, y = data['features'], data['targets']
    train, validation = split['train'], split['validation']
    mean, scale = x[train].mean(axis=0), x[train].std(axis=0).clip(1e-4)
    model = HydraulicClassifier(mean, scale)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.02, weight_decay=1e-4)
    losses = []
    for i, c in enumerate(COMPONENTS):
        counts = np.bincount(y[train, i], minlength=len(c['codes']))
        if np.any(counts == 0):
            raise ValueError('A training class is missing.')
        losses.append(nn.CrossEntropyLoss(weight=torch.tensor(len(train)/(len(counts)*counts), dtype=torch.float32)))
    rng = np.random.default_rng(seed)
    best, best_score, stale, history, best_epoch = None, -1, 0, [], 0
    started = time.perf_counter()
    for epoch in range(1, epochs+1):
        model.train()
        total = 0
        order = rng.permutation(train)
        for offset in range(0, len(order), 128):
            batch = order[offset:offset+128]
            optimizer.zero_grad(set_to_none=True)
            heads = model(torch.from_numpy(x[batch]))
            loss = sum(loss_fn(heads[i], torch.from_numpy(y[batch, i])) for i, loss_fn in enumerate(losses))/4
            if not torch.isfinite(loss):
                raise RuntimeError('Non-finite loss.')
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 3)
            optimizer.step()
            total += float(loss.detach())*len(batch)
        val_predictions, _ = predict(model, x[validation])
        metric = evaluation_metrics(y[validation], val_predictions)
        score = metric['mean_macro_f1']
        history.append({'epoch':epoch, 'loss':total/len(train), 'validation_mean_macro_f1':score})
        if score > best_score + 1e-6:
            best, best_score, stale, best_epoch = copy.deepcopy(model.state_dict()), score, 0, epoch
        else:
            stale += 1
        if epoch == 1 or epoch % 25 == 0:
            print(f'linear epoch {epoch}: validation macro F1 {score:.3f}', flush=True)
        if stale >= 35:
            break
    model.load_state_dict(best)
    model.eval()
    return model, {'best_epoch':best_epoch, 'epochs_run':len(history), 'history':history,
                   'seconds':round(time.perf_counter()-started, 3),
                   'parameters':sum(p.numel() for p in model.parameters()),
                   'validation':evaluation_metrics(y[validation], predict(model, x[validation])[0])}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--epochs', type=int, default=250)
    args = parser.parse_args()
    if args.epochs < 1:
        parser.error('Epoch count must be positive.')
    torch.set_num_threads(2)
    torch.use_deterministic_algorithms(True)
    started = time.perf_counter()
    data = prepare_data()
    split = grouped_split(data['profile'])
    # All cycles sharing a joint condition combination stay in a single split.
    OUTPUT.mkdir(exist_ok=True)
    selected = 'linear'
    model, training = fit(data, split, args.epochs, 42)
    print('Deployment model:', selected, flush=True)
    # Final test labels are first scored after model selection.
    test = split['test']
    predictions, scores = predict(model, data['features'][test])
    reports = {selected: {**training, 'test':evaluation_metrics(data['targets'][test], predictions)}}
    saved = {'targets':data['targets'][test], 'cycle_indices':test,
             'linear_predictions':predictions}
    for i, c in enumerate(COMPONENTS):
        saved[f"linear_{c['id']}_scores"] = scores[i]
    torch.save({'task':'uci_hydraulic_component_conditions_v1', 'kind':selected,
                'feature_names':FEATURE_NAMES, 'dataset_fingerprint':str(data['fingerprint']),
                'state_dict':model.state_dict()}, OUTPUT/'model.pt')
    restored, _ = load_model(OUTPUT/'model.pt')
    np.testing.assert_array_equal(predict(restored, data['features'][test])[0], predictions)
    manifest = {name:{'cycle_indices':idx.tolist(), 'cycles':len(idx),
                      'condition_combinations':np.unique(data['profile'][idx,:4],axis=0).tolist()}
                for name,idx in split.items()}
    manifest['protocol'] = 'Seed 42; 60/20/20 split of all 144 joint condition combinations. Entire combinations, including repeated cycles, stay together.'
    (OUTPUT/'split_manifest.json').write_text(json.dumps(manifest, indent=2)+'\n')
    (OUTPUT/'source_manifest.json').write_text(str(data['manifest_json'])+'\n')
    np.savez_compressed(OUTPUT/'test_predictions.npz', **saved)
    report = {'task':'four_component_condition_classification', 'source':'real_test_rig',
              'source_url':SOURCE_URL, 'dataset_fingerprint':str(data['fingerprint']),
              'checkpoint_sha256':file_hash(OUTPUT/'model.pt'), 'selected_model':selected,
              'selection':'Linear deployment retained after earlier model comparisons. Normalization and class weights fitted on training only.',
              'split_protocol':manifest['protocol'],
              'split_counts':{k:len(v) for k,v in split.items()},
              'sensor_channels':14, 'features':len(FEATURE_NAMES), 'excluded_channels':['CE','CP','SE'],
              'models':reports, 'seconds':round(time.perf_counter()-started,3),
              'limitations':['Same physical test rig; not an independent-machine or production-factory validation.',
                             'Joint condition combinations are held out; this is not a chronological forecast.',
                             'Uses a complete 60-second cycle. No advance warning or remaining useful life is established.',
                             'Scores are uncalibrated class support. No causal root cause or intervention benefits are established.',
                             'No run identifiers are supplied; cycles within this experiment can be correlated.',
                             'Original records contain no missing values; sensor removal tests are controlled sensitivity experiments.']}
    (OUTPUT/'metrics.json').write_text(json.dumps(report,indent=2)+'\n')
    rows = []
    for c in COMPONENTS:
        b = reports['linear']['test']['components'][c['id']]
        deployed_accuracy = reports[selected]['test']['components'][c['id']]['accuracy']
        rows.append(f"| {c['name']} | {b['macro_f1']:.3f} | {deployed_accuracy:.3f} |")
    text = '# DigitalTwin — real hydraulic measurements\n\n'+f"Source: {SOURCE_URL}\n\n{manifest['protocol']}\n\n"
    text += f"Cycles: {report['split_counts']}. Deployed model: **{selected}**.\n\n"
    text += '| Component | Linear macro F1 | Deployed accuracy |\n|---|---:|---:|\n'+'\n'.join(rows)+'\n\n'
    deployed_test = reports[selected]['test']
    text += (f"Deployed mean component macro F1: {deployed_test['mean_macro_f1']:.3f}. "
             f"Mean component accuracy: {deployed_test['mean_accuracy']:.2%}. "
             f"All four components correct: {deployed_test['all_components_correct']:.2%}.\n\n")
    text += '\n'.join('- '+v for v in report['limitations'])+'\n'
    (OUTPUT/'training_report.md').write_text(text)
    print(text)


if __name__ == '__main__':
    main()
