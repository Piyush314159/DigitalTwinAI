"""Operational contracts and fixed-model robustness integrity."""

import copy
import csv
import hashlib
import io
import json
from pathlib import Path
import unittest
from unittest.mock import patch

import numpy as np
import torch

import app
from anomaly import DEFAULT_CHECKPOINT, make_dataset
from scripts.evaluate_robustness import apply_sensor_challenge, summarize


class MonitoringTests(unittest.TestCase):
    def setUp(self):
        self.client = app.app.test_client()
        self.position_patch = patch.object(app, 'current_step', 0)
        self.position_patch.start()
        self.addCleanup(self.position_patch.stop)

    def test_masked_and_invalid_readings_are_unavailable(self):
        sample = copy.deepcopy(app.shift_data[0])
        sample['masked_features'][0, 0] = float('nan')
        sample['sensor_mask'][7, 4] = 0
        sample['masked_features'][7, 4] = 999999
        values = app.observed_values(sample)
        self.assertIsNone(values[0][0])
        self.assertIsNone(values[7][4])
        self.assertTrue(np.isfinite([v for row in values for v in row if v is not None]).all())

    def test_no_model_means_unknown_status_not_normal(self):
        with patch.object(app, 'anomaly_detector', None):
            state = self.client.get('/api/state').json
            self.assertEqual(state['modelStatus'], 'not_trained')
            self.assertIsNone(state['scores'])
            self.assertIsNone(state['flagged'])
            self.assertIsNone(state['summary']['flaggedStations'])
            self.assertTrue(all(point['flaggedCount'] is None for point in state['history']))
            self.assertEqual(self.client.get('/api/config').json['capabilities'][0]['status'], 'not_trained')

    def test_seek_reset_and_repeated_reads_are_consistent(self):
        initial = self.client.get('/api/state').json
        forward = self.client.post('/api/seek', json={'step': 20}).json
        self.assertEqual(forward['step'], 20)
        self.assertEqual(len(forward['history']), 21)
        self.assertEqual(forward, self.client.get('/api/state').json)
        self.assertEqual(self.client.post('/api/reset', json={}).json, initial)
        self.assertEqual(self.client.post('/api/step', json={}).json['step'], 1)

    def test_invalid_seek_does_not_change_position(self):
        for value in (-1, 30, True, '2', 1.5, None):
            response = self.client.post('/api/seek', json={'step': value})
            self.assertEqual(response.status_code, 400)
            self.assertEqual(app.current_step, 0)
        self.assertEqual(self.client.post('/api/seek', json=[]).status_code, 400)

    def test_flags_alert_counts_and_history_share_the_same_scores(self):
        if app.anomaly_detector is None:
            self.skipTest('Train the saved detector to run its integration check.')
        state = self.client.post('/api/seek', json={'step': 20}).json
        expected = [score >= state['threshold'] for score in state['scores']]
        self.assertEqual(expected, state['flagged'])
        self.assertEqual(sum(expected), state['summary']['flaggedStations'])
        self.assertEqual(sum(expected), state['history'][-1]['flaggedCount'])
        self.assertEqual(state['scores'], self.client.get('/api/anomaly').json['scores'])
        evaluation = self.client.get('/api/validation').json
        self.assertEqual(evaluation['metrics']['threshold'], state['threshold'])

    def test_simulator_answers_do_not_affect_the_operational_response(self):
        initial = self.client.post('/api/seek', json={'step': 3}).json
        changed = copy.deepcopy(app.shift_data)
        for step in changed:
            step['ground_truth_anomaly'][:] = 1
            step['features'][:] = 999999
            step['active_scenarios'] = ['injected answers must stay out of monitoring']
        with patch.object(app, 'shift_data', changed):
            self.assertEqual(initial, self.client.get('/api/state').json)
        serialized = json.dumps(initial)
        for forbidden in ('ground_truth_anomaly', 'active_scenarios', 'risk_class', 'confidence'):
            self.assertNotIn(forbidden, serialized)

    def test_snapshot_exports_observations_and_unambiguous_units(self):
        response = self.client.get('/api/snapshot.csv')
        rows = list(csv.DictReader(io.StringIO(response.data.decode())))
        self.assertEqual(len(rows), 40)
        self.assertEqual(rows[7]['vibration_mm/s'], '')
        self.assertIn('throughput_target_fraction', rows[0])
        self.assertNotIn('throughput_uph', rows[0])
        self.assertEqual(rows[0]['simulated_time'], '08:00')
        self.assertIn('attachment', response.headers['Content-Disposition'])

    def test_original_prediction_capabilities_are_explicitly_untrained(self):
        config = self.client.get('/api/config').json
        capabilities = {c['name']: c['status'] for c in config['capabilities']}
        self.assertEqual(capabilities['Queue forecasting'], 'not_trained')
        self.assertEqual(capabilities['Confidence calibration'], 'not_validated')
        response = self.client.post('/api/whatif', json={})
        self.assertEqual(response.status_code, 409)


class RobustnessTests(unittest.TestCase):
    def test_sensor_challenges_change_only_inputs(self):
        original = make_dataset([901, 902], steps=12)
        changed = apply_sensor_challenge(original, outage_rate=0.20, noise_fraction=0.2,
                                         scale=torch.ones(12))
        self.assertTrue(torch.equal(original.targets, changed.targets))
        self.assertEqual(original.manifests, changed.manifests)
        self.assertTrue(torch.all(changed.masks <= original.masks))
        self.assertTrue(torch.equal(changed.readings[changed.masks == 0],
                                    torch.zeros_like(changed.readings[changed.masks == 0])))
        self.assertFalse(torch.equal(original.readings, changed.readings))

    def test_healthy_only_f1_and_recall_are_explicitly_undefined(self):
        metrics = summarize(np.zeros(5), np.array([0.1, 0.2, 0.3, 0.9, 0.99]), 0.94)
        self.assertIsNone(metrics['f1'])
        self.assertIsNone(metrics['recall'])
        self.assertIsNone(metrics['average_precision'])
        self.assertEqual(metrics['false_positive'], 1)
        self.assertEqual(metrics['false_alarm_rate'], 0.2)

    def test_default_generator_still_reproduces_the_original_training_data(self):
        path = DEFAULT_CHECKPOINT.with_name('dataset_manifest.json')
        if not path.exists():
            self.skipTest('Original training artifacts are unavailable.')
        entries = json.loads(path.read_text())['train'][:3]
        repeated = make_dataset([s['seed'] for s in entries])
        self.assertEqual([s['data_sha256'] for s in repeated.manifests],
                         [s['data_sha256'] for s in entries])

    def test_saved_robustness_results_reproduce_from_outputs_and_frozen_threshold(self):
        directory = DEFAULT_CHECKPOINT.parent / 'robustness'
        if not (directory / 'metrics.json').exists():
            self.skipTest('Run the declared robustness evaluation first.')
        report = json.loads((directory / 'metrics.json').read_text())
        self.assertEqual(report['checkpoint_sha256'], hashlib.sha256(DEFAULT_CHECKPOINT.read_bytes()).hexdigest())
        data = np.load(directory / 'predictions.npz')
        for case in report['cases']:
            key = case['id']
            recomputed = summarize(data[f'{key}_targets'], data[f'{key}_scores'], report['threshold'])
            self.assertEqual(recomputed, case['detector'])
        for key in ('sensor_outages', 'measurement_noise'):
            np.testing.assert_array_equal(data[f'{key}_targets'], data['reference_targets'])


if __name__ == '__main__':
    unittest.main()
