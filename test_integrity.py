import unittest
import torch

from model import load_model, prediction_arrays


class IntegrityTests(unittest.TestCase):
    def test_missing_sensor_values_produce_finite_outputs(self):
        model, _, _ = load_model()
        x = torch.ones(3, 12)
        x[1, 2] = float('nan')
        mask = torch.ones_like(x)
        mask[1, 2] = 0
        with torch.no_grad():
            decoded = prediction_arrays(model(x, torch.tensor([[0, 1], [1, 2]]), mask=mask))
        for value in decoded.values():
            self.assertTrue(torch.isfinite(torch.as_tensor(value)).all())
        self.assertFalse(model.is_trained)

    def test_operational_status_uses_only_trained_detection(self):
        import app
        import numpy as np
        import copy
        first = copy.deepcopy(app.shift_data[0])
        changed = copy.deepcopy(first)
        changed['ground_truth_anomaly'] = np.ones_like(first['ground_truth_anomaly'])
        a = app.run_trained_anomaly([first])
        b = app.run_trained_anomaly([changed])
        self.assertEqual(a, b)
        client = app.app.test_client()
        state = client.get('/api/state').json
        self.assertNotIn('risk_class', state['predictions'])
        self.assertNotIn('confidence', state['predictions'])
        self.assertNotIn('queue', state['predictions'])
        self.assertEqual(client.get('/api/validation').json['status'], state['modelStatus'])
        self.assertEqual(client.post('/api/whatif', json={}).json['status'], 'not_trained')
