"""Check grouped tree-comparison evidence without requiring optional sklearn."""
import itertools
import json
import unittest

import numpy as np

import app
from hydraulics import OUTPUT, evaluation_metrics, file_hash, grouped_split, upgrade_split
from scripts.compare_gru import experiment_split

DIRECTORY = OUTPUT / 'tree_experiment'


@unittest.skipUnless((DIRECTORY / 'metrics.json').exists(), 'Run the optional tree comparison first.')
class TreeExperimentTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.protocol = json.loads((DIRECTORY / 'protocol.json').read_text())
        cls.report = json.loads((DIRECTORY / 'metrics.json').read_text())

    def test_previous_test_groups_are_excluded_and_outer_folds_are_disjoint(self):
        profile = app.data['profile']
        _, groups = np.unique(profile[:, :4], axis=0, return_inverse=True)
        old = grouped_split(profile)['test']
        accumulator = upgrade_split(profile)['test']
        gru = experiment_split(profile)[0]['test']
        expected_exclusion = np.union1d(np.union1d(old, accumulator), gru)
        np.testing.assert_array_equal(self.protocol['excluded_inspected_test_indices'], expected_exclusion)
        development = np.array(self.protocol['development_indices'])
        np.testing.assert_array_equal(np.sort(np.r_[development, expected_exclusion]), np.arange(len(profile)))
        validation_sets = []
        for fold in self.protocol['folds']:
            train = np.array(fold['train'])
            validation = np.array(fold['validation'])
            inner_train = np.array(fold['inner_train'])
            inner_validation = np.array(fold['inner_validation'])
            self.assertFalse(set(groups[train]) & set(groups[validation]))
            self.assertFalse(set(groups[inner_train]) & set(groups[inner_validation]))
            self.assertEqual(set(inner_train) | set(inner_validation), set(train))
            self.assertEqual(set(train) | set(validation), set(development))
            validation_sets.append(set(validation))
        for a, b in itertools.combinations(validation_sets, 2):
            self.assertFalse(a & b)
        self.assertEqual(set.union(*validation_sets), set(development))

    def test_saved_predictions_reproduce_reported_scores(self):
        self.assertEqual(file_hash(DIRECTORY / 'protocol.json'), self.report['protocol_sha256'])
        self.assertEqual(file_hash(DIRECTORY / 'out_of_fold_predictions.npz'),
                         self.report['predictions_sha256'])
        with np.load(DIRECTORY / 'out_of_fold_predictions.npz') as saved:
            indices = saved['cycle_indices']
            np.testing.assert_array_equal(indices, self.protocol['development_indices'])
            np.testing.assert_array_equal(saved['targets'], app.data['targets'][indices])
            for kind, values in self.report['models'].items():
                predictions = saved[f'{kind}_predictions']
                self.assertEqual(evaluation_metrics(saved['targets'], predictions), values['out_of_fold'])
                for fold, scores in zip(self.protocol['folds'], self.report['folds']):
                    positions = np.searchsorted(indices, fold['validation'])
                    self.assertEqual(evaluation_metrics(saved['targets'][positions],
                                                        predictions[positions])['mean_macro_f1'],
                                     scores[kind]['mean_macro_f1'])
        self.assertEqual(self.report['tree_better_folds'], sum(
            fold['boosted_trees']['mean_macro_f1'] > fold['linear']['mean_macro_f1']
            for fold in self.report['folds']))


if __name__ == '__main__':
    unittest.main()
