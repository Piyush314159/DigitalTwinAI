"""Check real-data provenance, split isolation, feature extraction and metrics."""
import itertools
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

import app
from hydraulics import (SENSORS, COMPONENTS, FEATURE_NAMES, OUTPUT, find_raw_data,
                        summarize_cycle, evaluation_metrics, grouped_split, file_hash)
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

    def test_removed_model_kind_cannot_be_loaded(self):
        checkpoint=torch.load(OUTPUT/'model.pt',map_location='cpu',weights_only=True)
        checkpoint['kind']='mlp'
        with tempfile.TemporaryDirectory() as folder:
            path=Path(folder)/'old_model.pt'
            torch.save(checkpoint,path)
            with self.assertRaisesRegex(ValueError,'Only the retained linear classifier'):
                load_model(path)

    def test_saved_report_recomputes_from_exact_test_outputs(self):
        with np.load(OUTPUT/'test_predictions.npz') as saved:
            idx=saved['cycle_indices']
            np.testing.assert_array_equal(app.data['targets'][idx],saved['targets'])
            self.assertEqual(set(saved.files),{'targets','cycle_indices','linear_predictions',
                                               *[f'linear_{c["id"]}_scores' for c in COMPONENTS]})
            self.assertEqual(evaluation_metrics(saved['targets'],saved['linear_predictions']),
                             app.report['models']['linear']['test'])
        self.assertEqual(file_hash(OUTPUT/'model.pt'),app.report['checkpoint_sha256'])

    def test_source_manifest_matches_downloaded_measurements(self):
        manifest=json.loads((OUTPUT/'source_manifest.json').read_text())
        directory=find_raw_data()
        for name in ('profile.txt','VS1.txt','PS1.txt'):
            self.assertEqual(file_hash(directory/name),manifest[name]['sha256'])

    def test_deployment_selection_uses_validation_metric(self):
        self.assertEqual(app.report['selected_model'],'linear')
        self.assertEqual(set(app.report['models']),{'linear'})


if __name__=='__main__': unittest.main()
