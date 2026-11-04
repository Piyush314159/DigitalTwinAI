"""Measured multirate input, grouping and saved TCN evidence checks."""
import itertools
import json
import unittest

import numpy as np
import torch

import app
from hydraulics import (COMPONENTS, OUTPUT, evaluation_metrics, file_hash,
                        find_raw_data)
from scripts.compare_tcn import (HIGH_SENSORS, LOW_SENSORS, SIGNAL_CACHE,
                                 frozen_protocol, measured_multirate, new_tcn)

DIRECTORY = OUTPUT / 'tcn_experiment'


class TCNInputTests(unittest.TestCase):
    def test_multirate_cache_uses_original_sensor_rows_in_order(self):
        high, low = measured_multirate(app.data)
        self.assertEqual(high.shape, (2205, 9, 600))
        self.assertEqual(low.shape, (2205, 5, 60))
        self.assertEqual([sensor['id'] for sensor in HIGH_SENSORS],
                         [*[f'PS{i}' for i in range(1, 7)], 'EPS1', 'FS1', 'FS2'])
        self.assertEqual([sensor['id'] for sensor in LOW_SENSORS],
                         [*[f'TS{i}' for i in range(1, 5)], 'VS1'])
        directory = find_raw_data()
        for name, observed, rate in [('PS1', high[7, 0], 100),
                                      ('FS1', high[7, 7], 10),
                                      ('TS1', low[7, 0], 1)]:
            with (directory / f'{name}.txt').open() as stream:
                raw = np.fromstring(next(itertools.islice(stream, 7, 8)), sep='\t')
            expected = raw.reshape(600, 10).mean(axis=1) if rate == 100 else raw
            np.testing.assert_allclose(observed, expected, rtol=1e-6, atol=1e-6)

    def test_grouped_inner_and_outer_partitions_do_not_share_conditions(self):
        plan = frozen_protocol(app.data)
        profile = app.data['profile']
        seen = set()
        for fold in plan['folds']:
            train = np.array(fold['train'])
            validation = np.array(fold['validation'])
            inner_train = np.array(fold['inner_train'])
            inner_validation = np.array(fold['inner_validation'])
            self.assertFalse(set(validation) & seen)
            seen.update(validation.tolist())
            for a, b in ((train, validation), (inner_train, inner_validation)):
                left = {tuple(row) for row in profile[a, :4]}
                right = {tuple(row) for row in profile[b, :4]}
                self.assertFalse(left & right)
            self.assertEqual(set(inner_train) | set(inner_validation), set(train))
        self.assertEqual(seen, set(range(len(profile))))

    def test_normalizers_use_training_cycles_and_reject_incomplete_inputs(self):
        high, low = measured_multirate(app.data)
        train = np.array(frozen_protocol(app.data)['folds'][0]['inner_train'])
        model = new_tcn(app.data, high, low, train)
        np.testing.assert_allclose(model.feature_mean.numpy(), app.data['features'][train].mean(axis=0))
        np.testing.assert_allclose(model.high_mean.numpy(), high[train].mean(axis=(0, 2))[:, None])
        np.testing.assert_allclose(model.low_mean.numpy(), low[train].mean(axis=(0, 2))[:, None])
        features = torch.from_numpy(app.data['features'][:1])
        hi = torch.from_numpy(high[:1])
        lo = torch.from_numpy(low[:1])
        self.assertEqual([head.shape[1] for head in model(features, hi, lo)],
                         [len(component['codes']) for component in COMPONENTS])
        with self.assertRaisesRegex(ValueError, 'input schema'):
            model(features, lo, hi)
        damaged = hi.clone()
        damaged[0, 0, 0] = float('nan')
        with self.assertRaisesRegex(ValueError, 'incomplete'):
            model(features, damaged, lo)


@unittest.skipUnless((DIRECTORY / 'metrics.json').exists(), 'Run the TCN comparison first.')
class TCNEvidenceTests(unittest.TestCase):
    def test_exact_predictions_reproduce_report_without_changing_deployment(self):
        report = json.loads((DIRECTORY / 'metrics.json').read_text())
        self.assertEqual(file_hash(DIRECTORY / 'protocol.json'), report['protocol_sha256'])
        self.assertEqual(file_hash(DIRECTORY / 'out_of_fold_predictions.npz'),
                         report['predictions_sha256'])
        self.assertEqual(file_hash(SIGNAL_CACHE), report['signal_cache_sha256'])
        self.assertEqual(report['dataset_fingerprint'], str(app.data['fingerprint']))
        with np.load(DIRECTORY / 'out_of_fold_predictions.npz') as saved:
            indices = saved['cycle_indices']
            np.testing.assert_array_equal(indices, np.arange(2205))
            np.testing.assert_array_equal(saved['targets'], app.data['targets'])
            for kind in ('linear', 'tcn'):
                self.assertEqual(evaluation_metrics(saved['targets'],
                                                    saved[f'{kind}_predictions']),
                                 report['models'][kind]['out_of_fold'])
                for fold, scores in zip(frozen_protocol(app.data)['folds'], report['folds']):
                    validation = np.array(fold['validation'])
                    self.assertEqual(evaluation_metrics(saved['targets'][validation],
                                                        saved[f'{kind}_predictions'][validation]),
                                     scores[kind]['validation'])
        self.assertEqual(file_hash(OUTPUT / 'model.pt'), app.report['checkpoint_sha256'])
        self.assertEqual(app.checkpoint['kind'], 'tcn')


if __name__ == '__main__':
    unittest.main()
