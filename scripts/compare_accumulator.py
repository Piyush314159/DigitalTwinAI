"""Compare a small accumulator upgrade: python -m scripts.compare_accumulator.

This experiment never replaces the deployed model. The protocol is frozen;
re-running it reproduces an inspected evaluation, not new external evidence."""
import argparse
import json
import time

import numpy as np
import torch

from hydraulics import OUTPUT as ACTIVE_OUTPUT, COMPONENTS, FEATURE_NAMES, SOURCE_URL, prepare_data, grouped_split, upgrade_split, evaluation_metrics, file_hash
from model import predict, load_model, with_accumulator_neighbors
from scripts.train_hydraulics import fit

OUTPUT = ACTIVE_OUTPUT / 'accumulator_experiment'

NEIGHBOR_GRID = [3,7,15]
FEATURE_SETS = {'pressure':list(range(60)), 'pressure_power_flow':list(range(90)),
                'all_physical':list(range(len(FEATURE_NAMES)))}


def freeze_protocol(data, epochs=250):
    split = upgrade_split(data['profile'])
    legacy_test = grouped_split(data['profile'])['test']
    plan = {'version':'upgrade_v2', 'seed':20261103, 'fit_epochs':epochs, 'training_seed':42,
            'dataset_fingerprint':str(data['fingerprint']),
            'protocol':'Seed 20261103; exclude all 29 previously inspected test combinations. Split the remaining 115 combinations into 69 training, 23 validation and 23 new test groups. Fit all models from scratch.',
            'selection':'Validation mean component macro F1 only; test scored after all choices are frozen. Re-running is reproduction, not a fresh test.',
            'candidates':{'base_models':['linear','mlp'], 'accumulator_neighbors':NEIGHBOR_GRID,
                          'accumulator_feature_sets':FEATURE_SETS, 'distance':'Training-standardized Euclidean distance; inverse-distance weighted votes.'},
            'partitions':{name:idx.tolist() for name,idx in split.items()},
            'excluded_legacy_test_indices':legacy_test.tolist(),
            'limitation':'Re-reserved cycles from the same previously explored dataset, not independent-machine or previously unseen external data.'}
    path=OUTPUT/'upgrade_protocol.json'
    if path.exists() and json.loads(path.read_text()) != plan:
        raise ValueError('The frozen upgrade protocol differs. Do not silently change the reserved evaluation.')
    if not path.exists():
        path.write_text(json.dumps(plan,indent=2)+'\n')
    return split,plan


def save_checkpoint(model, data, path, split):
    checkpoint={'task':'uci_hydraulic_component_conditions_v1','kind':model.kind,
                'feature_names':FEATURE_NAMES,'dataset_fingerprint':str(data['fingerprint']),
                'state_dict':model.state_dict(),'protocol_version':'upgrade_v2'}
    if model.kind=='linear_knn':
        checkpoint.update(neighbors=model.neighbors,reference_cycle_indices=split['train'].tolist())
    torch.save(checkpoint,path)


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
    OUTPUT.mkdir(parents=True,exist_ok=True)
    split, plan = freeze_protocol(data,args.epochs)
    models, fits = {}, {}
    for kind in ('linear', 'mlp'):
        models[kind], fits[kind] = fit(kind, data, split, args.epochs, 42)
    # Only validation inputs/targets enter the small, predeclared neighbor search.
    best, trials = None, []
    for feature_set, columns in FEATURE_SETS.items():
        for neighbors in NEIGHBOR_GRID:
            candidate=with_accumulator_neighbors(models['linear'],data['features'][split['train']],
                                                data['targets'][split['train'],3],neighbors,columns)
            metric=evaluation_metrics(data['targets'][split['validation']],
                                      predict(candidate,data['features'][split['validation']])[0])
            trial={'neighbors':neighbors,'feature_set':feature_set,'mean_macro_f1':metric['mean_macro_f1'],
                   'accumulator_macro_f1':metric['components']['accumulator']['macro_f1']}
            trials.append(trial)
            if best is None or metric['mean_macro_f1'] > best[1]['mean_macro_f1']:
                best=(candidate,metric,trial)
    models['linear_knn']=best[0]
    fits['linear_knn']={'parameters':sum(p.numel() for p in best[0].parameters()),
                        'base_model':'linear','reference_cycles':len(split['train']),
                        'neighbors':best[2]['neighbors'],'feature_set':best[2]['feature_set'],
                        'reference_feature_count':len(best[0].reference_columns),
                        'validation_candidates':trials,'validation':best[1]}
    selected = max(models, key=lambda k:fits[k]['validation']['mean_macro_f1'])
    print('Validation-selected experimental model:', selected, flush=True)
    # Final test labels are first scored after model selection.
    test = split['test']
    reports, saved = {}, {'targets':data['targets'][test], 'cycle_indices':test}
    for kind, model in models.items():
        predictions, scores = predict(model, data['features'][test])
        reports[kind] = {**fits[kind], 'test':evaluation_metrics(data['targets'][test], predictions)}
        saved[f'{kind}_predictions'] = predictions
        for i, c in enumerate(COMPONENTS):
            saved[f"{kind}_{c['id']}_scores"] = scores[i]
        save_checkpoint(model,data,OUTPUT/f'{kind}.pt',split)
        restored, _ = load_model(OUTPUT/f'{kind}.pt')
        np.testing.assert_array_equal(predict(restored, data['features'][test])[0], predictions)
    save_checkpoint(models[selected],data,OUTPUT/'model.pt',split)
    manifest = {name:{'cycle_indices':idx.tolist(), 'cycles':len(idx),
                      'condition_combinations':np.unique(data['profile'][idx,:4],axis=0).tolist()}
                for name,idx in split.items()}
    manifest['protocol'] = plan['protocol']
    manifest['version'] = plan['version']
    manifest['excluded_legacy_test_indices'] = plan['excluded_legacy_test_indices']
    (OUTPUT/'split_manifest.json').write_text(json.dumps(manifest, indent=2)+'\n')
    (OUTPUT/'source_manifest.json').write_text(str(data['manifest_json'])+'\n')
    np.savez_compressed(OUTPUT/'test_predictions.npz', **saved)
    report = {'task':'four_component_condition_classification', 'source':'real_test_rig',
              'purpose':'Accumulator improvement experiment; never changes the deployed model.',
              'deployment_policy':'Experimental checkpoints are saved here only. No automatic deployment.',
              'decision':'retain_existing_deployment' if fits['linear_knn']['validation']['mean_macro_f1'] <= fits['linear']['validation']['mean_macro_f1'] else 'candidate_requires_review',
              'source_url':SOURCE_URL, 'dataset_fingerprint':str(data['fingerprint']),
              'checkpoint_sha256':file_hash(OUTPUT/'model.pt'), 'selected_model':selected,
              'selection':plan['selection']+' Normalization, class weights and neighbor reference bank fitted on training only.',
              'protocol_version':plan['version'],
              'split_protocol':manifest['protocol'],
              'split_counts':{k:len(v) for k,v in split.items()},
              'sensor_channels':14, 'features':len(FEATURE_NAMES), 'excluded_channels':['CE','CP','SE'],
              'models':reports, 'seconds':round(time.perf_counter()-started,3),
              'limitations':['Same physical test rig; not an independent-machine or production-factory validation.',
                             'Joint condition combinations are held out; this is not a chronological forecast.',
                             'Uses a complete 60-second cycle. No advance warning or remaining useful life is established.',
                             'Scores are uncalibrated class support (softmax or neighbor votes). No causal root cause or intervention benefits are established.',
                             plan['limitation'],
                             'No run identifiers are supplied; cycles within this experiment can be correlated.',
                             'Original records contain no missing values; sensor removal tests are controlled sensitivity experiments.']}
    (OUTPUT/'metrics.json').write_text(json.dumps(report,indent=2)+'\n')
    rows = []
    for c in COMPONENTS:
        a, b = reports['mlp']['test']['components'][c['id']], reports['linear']['test']['components'][c['id']]
        hybrid = reports['linear_knn']['test']['components'][c['id']]
        deployed_accuracy = reports[selected]['test']['components'][c['id']]['accuracy']
        rows.append(f"| {c['name']} | {a['macro_f1']:.3f} | {b['macro_f1']:.3f} | {hybrid['macro_f1']:.3f} | {deployed_accuracy:.3f} |")
    text = '# DigitalTwin — real hydraulic measurements\n\n'+f"Source: {SOURCE_URL}\n\n{manifest['protocol']}\n\n"
    text += f"Cycles: {report['split_counts']}. Experiment winner chosen on validation: **{selected}**.\n\n"
    text += '| Component | MLP macro F1 | Linear macro F1 | Linear + neighbors macro F1 | Experiment winner accuracy |\n|---|---:|---:|---:|---:|\n'+'\n'.join(rows)+'\n\n'
    text += f"Accumulator neighbor search: {len(trials)} validation-only candidates. Selected {best[2]['neighbors']} neighbors using {best[2]['feature_set']} features. References are training cycles only. The deployed model is unchanged by this experiment.\n\n"
    deployed_test = reports[selected]['test']
    text += (f"Experiment winner mean component macro F1: {deployed_test['mean_macro_f1']:.3f}. "
             f"Mean component accuracy: {deployed_test['mean_accuracy']:.2%}. "
             f"All four components correct: {deployed_test['all_components_correct']:.2%}.\n\n")
    text += '\n'.join('- '+v for v in report['limitations'])+'\n'
    text += '\nDecision: '+report['decision']+'. Experimental results never replace the active deployment. Compare models within this protocol; its test set differs from the original 328-cycle test.\n'
    (OUTPUT/'training_report.md').write_text(text)
    print(text)


if __name__ == '__main__':
    main()
