"""Check real-data provenance, split isolation, feature extraction and metrics."""
import itertools
import json
import unittest

import numpy as np

import app
from hydraulics import (SENSORS, COMPONENTS, FEATURE_NAMES, OUTPUT, find_raw_data,
                        summarize_cycle, evaluation_metrics, grouped_split, upgrade_split, file_hash)
from model import predict, load_model


class HydraulicDataTests(unittest.TestCase):
    def test_features_are_physical_channels_only(self):
        self.assertEqual(app.data['features'].shape, (2205,140))
        self.assertEqual(app.data['traces'].shape,(2205,14,60))
        self.assertEqual(len(FEATURE_NAMES),140)
        self.assertTrue(np.isfinite(app.data['features']).all())
        self.assertFalse(any(name.split('_')[0] in ('CE','CP','SE') for name in FEATURE_NAMES))

    def test_original_raw_measurements_reproduce_feature_cache(self):
        directory=find_raw_data()
        for index in (0,7,13):
            sensor=SENSORS[index]
            with (directory/(sensor['id']+'.txt')).open() as stream:
                line=next(itertools.islice(stream,7,8))
            raw=np.fromstring(line,sep='\t')
            np.testing.assert_allclose(summarize_cycle(raw),app.data['features'][7,index*10:(index+1)*10])
            np.testing.assert_allclose(raw.reshape(60,sensor['rate']).mean(axis=1),
                                       app.data['traces'][7,index],rtol=1e-6,atol=1e-6)

    def test_invalid_or_incomplete_raw_cycles_are_rejected(self):
        for raw in (np.zeros(59),np.full(60,np.nan)):
            with self.assertRaises(ValueError): summarize_cycle(raw)

    def test_condition_combination_groups_never_cross_splits(self):
        split=grouped_split(app.data['profile'])
        groups=[]
        for name,indices in split.items():
            self.assertEqual(indices.tolist(),app.split_manifest[name]['cycle_indices'])
            groups.append({tuple(row) for row in app.data['profile'][indices,:4]})
        self.assertFalse(groups[0]&groups[1] or groups[0]&groups[2] or groups[1]&groups[2])
        indices=np.concatenate(list(split.values()))
        np.testing.assert_array_equal(np.sort(indices),np.arange(2205))

    def test_normalizer_is_fit_only_to_training_cycles(self):
        indices=app.partitions['train'];x=app.data['features'][indices]
        np.testing.assert_allclose(app.model.mean.numpy(),x.mean(axis=0))
        np.testing.assert_allclose(app.model.scale.numpy(),x.std(axis=0).clip(1e-4))

    def test_condition_codes_match_original_annotations(self):
        for i,c in enumerate(COMPONENTS):
            decoded=np.array(c['codes'])[app.data['targets'][:,i]]
            np.testing.assert_array_equal(decoded,app.data['profile'][:,i])

    def test_checkpoint_reload_reproduces_predictions(self):
        restored,meta=load_model(OUTPUT/'model.pt')
        x=app.data['features'][app.partitions['test'][:8]]
        a,sa=predict(app.model,x);b,sb=predict(restored,x)
        np.testing.assert_array_equal(a,b)
        for p,q in zip(sa,sb):
            np.testing.assert_array_equal(p,q)
            np.testing.assert_allclose(p.sum(axis=1),np.ones(len(p)),rtol=1e-6)
        self.assertEqual(meta['dataset_fingerprint'],str(app.data['fingerprint']))

    def test_saved_report_recomputes_from_exact_test_outputs(self):
        with np.load(OUTPUT/'test_predictions.npz') as saved:
            idx=saved['cycle_indices']
            np.testing.assert_array_equal(app.data['targets'][idx],saved['targets'])
            for kind in ('linear','mlp'):
                self.assertEqual(evaluation_metrics(saved['targets'],saved[f'{kind}_predictions']),
                                  app.report['models'][kind]['test'])
        self.assertEqual(file_hash(OUTPUT/'model.pt'),app.report['checkpoint_sha256'])

    def test_source_manifest_matches_downloaded_measurements(self):
        manifest=json.loads((OUTPUT/'source_manifest.json').read_text())
        directory=find_raw_data()
        for name in ('profile.txt','VS1.txt','PS1.txt'):
            self.assertEqual(file_hash(directory/name),manifest[name]['sha256'])

    def test_deployment_selection_uses_validation_metric(self):
        expected=max(app.report['models'],key=lambda k:app.report['models'][k]['validation']['mean_macro_f1'])
        self.assertEqual(expected,app.report['selected_model'])

    @unittest.skipUnless((OUTPUT/'accumulator_experiment/metrics.json').exists(),'Optional accumulator experiment has not been run.')
    def test_accumulator_experiment_has_frozen_disjoint_groups(self):
        from scripts.compare_accumulator import freeze_protocol
        plan=json.loads((OUTPUT/'accumulator_experiment/upgrade_protocol.json').read_text())
        self.assertEqual(freeze_protocol(app.data)[1],plan)
        with self.assertRaisesRegex(ValueError,'frozen upgrade protocol differs'):
            freeze_protocol(app.data,epochs=1)
        split=upgrade_split(app.data['profile'])
        groups=[]
        for name,indices in split.items():
            self.assertEqual(indices.tolist(),plan['partitions'][name])
            groups.append({tuple(row) for row in app.data['profile'][indices,:4]})
        legacy=np.array(plan['excluded_legacy_test_indices'])
        groups.append({tuple(row) for row in app.data['profile'][legacy,:4]})
        for a,b in itertools.combinations(groups,2):
            self.assertFalse(a&b)
        np.testing.assert_array_equal(np.sort(np.concatenate([*split.values(),legacy])),np.arange(2205))

    @unittest.skipUnless((OUTPUT/'accumulator_experiment/metrics.json').exists(),'Optional accumulator experiment has not been run.')
    def test_neighbor_reference_bank_contains_only_experiment_training_measurements(self):
        model,checkpoint=load_model(OUTPUT/'accumulator_experiment/linear_knn.pt')
        indices=np.array(checkpoint['reference_cycle_indices'])
        plan=json.loads((OUTPUT/'accumulator_experiment/upgrade_protocol.json').read_text())
        np.testing.assert_array_equal(indices,plan['partitions']['train'])
        x=app.data['features'][indices]
        np.testing.assert_allclose(model.mean.numpy(),x.mean(axis=0))
        np.testing.assert_allclose(model.scale.numpy(),x.std(axis=0).clip(1e-4))
        normalized=np.clip((x-model.mean.numpy())/model.scale.numpy(),-8,8)
        np.testing.assert_array_equal(model.reference_features.numpy(),normalized[:,model.reference_columns.numpy()])
        np.testing.assert_array_equal(model.reference_targets.numpy(),app.data['targets'][indices,3])

    @unittest.skipUnless((OUTPUT/'accumulator_experiment/metrics.json').exists(),'Optional accumulator experiment has not been run.')
    def test_experiment_reload_and_metrics_match_saved_test_outputs(self):
        directory=OUTPUT/'accumulator_experiment'
        report=json.loads((directory/'metrics.json').read_text())
        with np.load(directory/'test_predictions.npz') as saved:
            indices=saved['cycle_indices']
            np.testing.assert_array_equal(saved['targets'],app.data['targets'][indices])
            for kind in report['models']:
                restored,_=load_model(directory/f'{kind}.pt')
                predicted,scores=predict(restored,app.data['features'][indices])
                np.testing.assert_array_equal(predicted,saved[f'{kind}_predictions'])
                self.assertEqual(evaluation_metrics(saved['targets'],predicted),report['models'][kind]['test'])
                for c,values in zip(COMPONENTS,scores):
                    np.testing.assert_array_equal(values,saved[f"{kind}_{c['id']}_scores"])
        self.assertEqual(report['decision'],'retain_existing_deployment')
        self.assertEqual(app.report['selected_model'],'linear')

    @unittest.skipUnless((OUTPUT/'accumulator_experiment/metrics.json').exists(),'Optional accumulator experiment has not been run.')
    def test_experiment_candidate_selection_uses_validation_only(self):
        report=json.loads((OUTPUT/'accumulator_experiment/metrics.json').read_text())
        candidate=report['models']['linear_knn']
        best=max(candidate['validation_candidates'],key=lambda t:t['mean_macro_f1'])
        self.assertEqual(best['neighbors'],candidate['neighbors'])
        self.assertEqual(best['feature_set'],candidate['feature_set'])
        self.assertEqual(best['mean_macro_f1'],candidate['validation']['mean_macro_f1'])
        self.assertEqual(report['selected_model'],max(report['models'],key=lambda k:report['models'][k]['validation']['mean_macro_f1']))


if __name__=='__main__': unittest.main()
