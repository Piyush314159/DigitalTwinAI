"""Measured-sequence integrity and conditional GRU experiment regression checks."""
import itertools
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

import app
from hydraulics import OUTPUT, COMPONENTS, evaluation_metrics, file_hash
from model import HydraulicGRU, load_model, predict
from scripts.compare_gru import measured_sequences, experiment_split, freeze_protocol

DIRECTORY=OUTPUT/'gru_experiment'


class GRUInputTests(unittest.TestCase):
    def test_sequence_time_and_channel_order_match_real_measurements(self):
        sequences=measured_sequences(app.data)
        self.assertEqual(sequences.shape,(2205,60,14))
        np.testing.assert_array_equal(sequences[7,:,0],app.data['traces'][7,0])
        np.testing.assert_array_equal(sequences[7,:,13],app.data['traces'][7,13])

    def test_hidden_memory_resets_for_each_cycle(self):
        x=measured_sequences(app.data)[:3]
        for layers in (1,2):
            model=HydraulicGRU(np.zeros(14),np.ones(14),layers)
            a,sa=predict(model,x[1:2]);b,sb=predict(model,x)
            np.testing.assert_array_equal(a[0],b[1])
            for p,q in zip(sa,sb):np.testing.assert_allclose(p[0],q[1],atol=1e-6)

    def test_gru_rejects_wrong_axis_order_and_incomplete_sequences(self):
        model=HydraulicGRU(np.zeros(14),np.ones(14))
        with self.assertRaisesRegex(ValueError,'60 time points'):
            predict(model,app.data['traces'][:1])
        x=measured_sequences(app.data)[:1].copy();x[0,7,2]=np.nan
        with self.assertRaisesRegex(ValueError,'Incomplete'):
            predict(model,x)

    def test_new_split_excludes_old_test_groups_and_never_splits_a_condition(self):
        split,excluded=experiment_split(app.data['profile'])
        groups=[{tuple(row) for row in app.data['profile'][indices,:4]}
                for indices in [*split.values(),excluded]]
        for a,b in itertools.combinations(groups,2):self.assertFalse(a&b)
        np.testing.assert_array_equal(np.sort(np.concatenate([*split.values(),excluded])),np.arange(2205))


@unittest.skipUnless((DIRECTORY/'metrics.json').exists(),'Run the optional GRU experiment first.')
class GRUExperimentTests(unittest.TestCase):
    def setUp(self):
        self.report=json.loads((DIRECTORY/'metrics.json').read_text())
        self.plan=json.loads((DIRECTORY/'protocol.json').read_text())

    def test_protocol_selection_and_layer_condition_are_consistent(self):
        self.assertEqual(freeze_protocol(app.data)[1],self.plan)
        r=self.report;models=r['models'];decision=json.loads((DIRECTORY/'selection.json').read_text())
        gain=models['gru_1']['validation']['mean_macro_f1']-models['linear']['validation']['mean_macro_f1']
        self.assertEqual(decision,r['layer_decision'])
        self.assertEqual(decision['two_layers_trained'],gain>self.plan['settings']['minimum_improvement'])
        self.assertEqual('gru_2' in models,decision['two_layers_trained'])
        self.assertEqual(r['selected_model'],max(models,key=lambda k:models[k]['validation']['mean_macro_f1']))
        self.assertEqual(file_hash(DIRECTORY/'protocol.json'),r['protocol_sha256'])
        for kind,sha in r['checkpoint_sha256'].items():self.assertEqual(file_hash(DIRECTORY/f'{kind}.pt'),sha)

    def test_sequence_normalizer_uses_only_training_cycles(self):
        model,meta=load_model(DIRECTORY/'gru_1.pt')
        train=np.array(self.plan['partitions']['train']);x=measured_sequences(app.data)[train]
        np.testing.assert_array_equal(meta['training_cycle_indices'],train)
        np.testing.assert_allclose(model.mean.numpy(),x.mean(axis=(0,1)))
        np.testing.assert_allclose(model.scale.numpy(),x.std(axis=(0,1)).clip(1e-4))

    def test_reload_and_exact_test_metrics_are_reproducible(self):
        with np.load(DIRECTORY/'test_predictions.npz') as saved:
            indices=saved['cycle_indices']
            np.testing.assert_array_equal(indices,self.plan['partitions']['test'])
            np.testing.assert_array_equal(saved['targets'],app.data['targets'][indices])
            for kind in self.report['models']:
                model,meta=load_model(DIRECTORY/f'{kind}.pt')
                x=measured_sequences(app.data)[indices] if kind.startswith('gru_') else app.data['features'][indices]
                predictions,scores=predict(model,x)
                np.testing.assert_array_equal(predictions,saved[f'{kind}_predictions'])
                self.assertEqual(evaluation_metrics(saved['targets'],predictions),self.report['models'][kind]['test'])
                for c,values in zip(COMPONENTS,scores):
                    np.testing.assert_array_equal(values,saved[f"{kind}_{c['id']}_scores"])

    def test_sequence_checkpoint_rejects_wrong_channel_order(self):
        meta=torch.load(DIRECTORY/'gru_1.pt',weights_only=True)
        meta['sensor_ids']=list(reversed(meta['sensor_ids']))
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'invalid.pt';torch.save(meta,path)
            with self.assertRaisesRegex(ValueError,'sequence schema mismatch'):load_model(path)


if __name__=='__main__':unittest.main()
