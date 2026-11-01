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

    def test_no_training_claims_or_ground_truth_dependencies(self):
        import app
        import numpy as np
        import copy
        first = copy.deepcopy(app.shift_data[0])
        changed = copy.deepcopy(first)
        changed['ground_truth_anomaly'] = np.ones_like(first['ground_truth_anomaly'])
        app.h_states = None
        a = app.run_inference(first)
        app.h_states = None
        b = app.run_inference(changed)
        for key in ('queue', 'anomaly_score', 'risk_class', 'risk_probs', 'confidence'):
            self.assertEqual(a[key], b[key])
        client = app.app.test_client()
        self.assertEqual(client.get('/api/validation').json['status'], 'not_trained')
        self.assertEqual(client.post('/api/whatif', json={}).json['status'], 'not_trained')
