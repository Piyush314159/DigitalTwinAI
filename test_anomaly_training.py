"""Checks for data leakage, missing sensors, batching, and saved inference."""

from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import copy

import numpy as np
import torch

from anomaly import (
    ShiftDataset, StationAnomalyDetector, classification_metrics, fit_normalizer,
    load_detector, make_dataset,
)
from assembly_line import build_assembly_graph
from simulator import ProductionSimulator


class AnomalyTrainingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)
        cls.stations, cls.edges, _ = build_assembly_graph()

    def make_model(self):
        torch.manual_seed(7)
        return StationAnomalyDetector(torch.zeros(12), torch.ones(12),
                                      hidden_channels=16, dropout=0.0).eval()

    def test_explicit_normal_shift_has_no_injected_faults(self):
        sim = ProductionSimulator(self.stations, scenarios=[], seed=7)
        for step in sim.generate_shift():
            self.assertEqual(step['active_scenarios'], [])
            self.assertEqual(float(step['ground_truth_anomaly'].sum()), 0.0)

    def test_shift_generation_is_reproducible_and_seeds_do_not_overlap(self):
        a = make_dataset([101, 102], steps=12)
        repeated = make_dataset([101, 102], steps=12)
        separate = make_dataset([201, 202], steps=12)
        self.assertTrue(torch.equal(a.readings, repeated.readings))
        self.assertEqual(a.manifests, repeated.manifests)
        self.assertFalse({s['data_sha256'] for s in a.manifests} &
                         {s['data_sha256'] for s in separate.manifests})
        self.assertEqual(tuple(a.targets.shape), (2, 12, 40))

    def test_normalizer_ignores_unobserved_values(self):
        x = torch.tensor([[[[2., 1000000.], [4., -1000000.]]]])
        mask = torch.tensor([[[[1., 0.], [1., 0.]]]])
        dataset = ShiftDataset(x, mask, torch.zeros(1, 1, 2), [])
        mean, scale = fit_normalizer(dataset)
        torch.testing.assert_close(mean, torch.tensor([3., 0.]))
        self.assertEqual(float(scale[0]), 1.0)
        self.assertTrue(torch.isfinite(scale).all())

    def test_unavailable_readings_cannot_change_predictions(self):
        model = self.make_model()
        x = torch.randn(1, 3, 40, 12)
        mask = torch.ones_like(x)
        mask[:, :, 7, 4] = 0
        changed = x.clone()
        changed[:, :, 7, 4] = float('nan')
        with torch.no_grad():
            a = model(x, self.edges, mask)
            b = model(changed, self.edges, mask)
        torch.testing.assert_close(a, b)
        self.assertTrue(torch.isfinite(b).all())

    def test_batched_shifts_do_not_exchange_messages(self):
        model = self.make_model()
        x = torch.randn(2, 3, 40, 12)
        masks = torch.ones_like(x)
        with torch.no_grad():
            together = model(x, self.edges, masks)
            separate = torch.cat([model(x[i:i+1], self.edges, masks[i:i+1])
                                  for i in range(2)])
        torch.testing.assert_close(together, separate, atol=1e-6, rtol=1e-5)

    def test_online_sequence_matches_offline_and_checkpoint_reload(self):
        model = self.make_model()
        x = torch.randn(1, 4, 40, 12)
        masks = torch.ones_like(x)
        with torch.no_grad():
            expected = model(x, self.edges, masks)
            hidden, actual = None, []
            for t in range(4):
                score, hidden = model.forward_step(x[0, t], self.edges, masks[0, t], hidden)
                actual.append(score)
        torch.testing.assert_close(expected[0], torch.stack(actual))
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'detector.pt'
            torch.save({
                'format_version': 1, 'task': 'current_station_anomaly',
                'model_config': {'hidden_channels': 16, 'dropout': 0.0},
                'state_dict': model.state_dict(), 'threshold': 0.5,
            }, path)
            restored, checkpoint = load_detector(path)
            with torch.no_grad():
                actual = restored(x, self.edges, masks)
        self.assertEqual(checkpoint['threshold'], 0.5)
        torch.testing.assert_close(expected, actual, rtol=0, atol=0)

    def test_average_precision_handles_tied_scores(self):
        metrics = classification_metrics([0, 1], [0.5, 0.5], 0.5)
        self.assertEqual(metrics['average_precision'], 0.5)
        self.assertEqual(metrics['precision'], 0.5)
        self.assertEqual(metrics['recall'], 1.0)
        self.assertAlmostEqual(metrics['f1'], 2 / 3)

    def test_dashboard_detector_ignores_simulator_answers_and_hidden_readings(self):
        import app
        prefix = copy.deepcopy(app.shift_data[:4])
        changed = copy.deepcopy(prefix)
        for step in changed:
            step['ground_truth_anomaly'][:] = 1
            step['active_scenarios'] = ['do not feed this answer to the model']
            step['features'][:] = 999999
            step['masked_features'][step['sensor_mask'] == 0] = float('nan')
        with patch.object(app, 'anomaly_detector', self.make_model()), \
             patch.object(app, 'anomaly_checkpoint', {'threshold': 0.5}):
            self.assertEqual(app.run_trained_anomaly(prefix), app.run_trained_anomaly(changed))

    def test_dashboard_repeated_reads_are_idempotent_and_absence_is_explicit(self):
        import app
        with patch.object(app, 'anomaly_detector', self.make_model()), \
             patch.object(app, 'anomaly_checkpoint', {'threshold': 0.5}), \
             patch.object(app, 'current_step', 3):
            client = app.app.test_client()
            a = client.get('/api/anomaly').json
            b = client.get('/api/anomaly').json
            self.assertEqual(a, b)
            self.assertEqual(a['steps_observed'], 4)
            self.assertEqual(len(a['scores']), 40)
        with patch.object(app, 'anomaly_detector', None):
            self.assertEqual(client.get('/api/anomaly').json['status'], 'not_trained')
            self.assertEqual(client.get('/api/anomaly/validation').json['status'], 'not_trained')


if __name__ == '__main__':
    unittest.main()
