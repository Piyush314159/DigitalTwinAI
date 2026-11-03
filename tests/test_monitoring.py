"""End-to-end real-data API checks, including label exclusion and abstention."""
import csv
import io
import unittest
from unittest.mock import patch

import numpy as np
import app


class MonitoringTests(unittest.TestCase):
    def setUp(self):
        self.client=app.app.test_client()
        a=patch.object(app,'current_partition','test');b=patch.object(app,'current_position',0)
        a.start();b.start();self.addCleanup(a.stop);self.addCleanup(b.stop)

    def test_state_uses_real_cycles_and_all_four_components(self):
        state=self.client.get('/api/state').json
        self.assertEqual(state['source'],'real_test_rig')
        self.assertEqual(state['totalRecords'],328)
        self.assertEqual(state['cycle'],int(app.partitions['test'][0])+1)
        self.assertEqual(len(state['observations']),14)
        self.assertEqual(len(state['conditions']),4)
        self.assertEqual(state,self.client.get('/api/state').json)

    def test_profile_labels_do_not_affect_predictions_or_sensor_readings(self):
        expected=self.client.get('/api/state').json
        changed={**app.data,'targets':np.zeros_like(app.data['targets']),
                 'profile':np.zeros_like(app.data['profile'])}
        with patch.object(app,'data',changed):
            self.assertEqual(expected,self.client.get('/api/state').json)
        self.assertNotIn('profile',expected)
        self.assertNotIn('reference',expected)

    def test_incomplete_inputs_are_not_treated_as_optimal_conditions(self):
        x=app.data['features'][0].copy();x[0]=np.nan
        result=app.estimate_conditions(x)
        self.assertEqual(result['status'],'insufficient_data')
        self.assertEqual(result['conditions'],[])

    def test_missing_model_does_not_create_random_predictions(self):
        with patch.object(app,'model',None):
            state=self.client.get('/api/state').json
        self.assertEqual(state['modelStatus'],'not_trained')
        self.assertEqual(state['conditions'],[])
        self.assertIsNone(state['summary']['flaggedComponents'])

    def test_review_count_comes_from_predicted_classes(self):
        state=self.client.get('/api/state').json
        self.assertEqual(state['summary']['flaggedComponents'],sum(c['class_index']!=0 for c in state['conditions']))

    def test_seek_partition_end_and_reset(self):
        last=self.client.post('/api/seek',json={'position':327}).json
        self.assertEqual(last['position'],327)
        self.assertEqual(self.client.post('/api/step').json['status'],'end_of_records')
        self.assertEqual(self.client.post('/api/reset').json['position'],0)
        full=self.client.post('/api/partition',json={'partition':'all'}).json
        self.assertEqual(full['totalRecords'],2205)
        self.assertEqual(full['cycle'],1)
        self.assertEqual(self.client.post('/api/step').json['cycle'],2)

    def test_invalid_positions_and_partitions_do_not_mutate_state(self):
        for value in (-1,328,True,None,1.2,'1'):
            self.assertEqual(self.client.post('/api/seek',json={'position':value}).status_code,400)
            self.assertEqual(app.current_position,0)
        for value in ('unknown',[],None):
            self.assertEqual(self.client.post('/api/partition',json={'partition':value}).status_code,400)

    def test_reference_labels_are_separate_explicit_annotations(self):
        reference=self.client.get('/api/reference').json
        self.assertTrue(reference['annotationOnly'])
        cycle=reference['cycle']-1
        self.assertEqual([v['code'] for v in reference['conditions']],app.data['profile'][cycle,:4].tolist())

    def test_csv_contains_measured_signals_and_separate_predictions(self):
        response=self.client.get('/api/snapshot.csv')
        rows=list(csv.DictReader(io.StringIO(response.data.decode())))
        self.assertEqual(len(rows),18)
        self.assertEqual(sum(r['record_type']=='sensor' for r in rows),14)
        self.assertEqual(sum(r['record_type']=='prediction' for r in rows),4)
        self.assertEqual(rows[0]['unit'],'bar')
        self.assertNotIn('recorded_condition',rows[0])

    def test_evaluation_is_real_data_and_matches_model(self):
        evaluation=self.client.get('/api/validation').json
        self.assertEqual(evaluation['report']['source'],'real_test_rig')
        self.assertEqual(evaluation['report']['selected_model'],app.checkpoint['kind'])
        self.assertEqual(evaluation['robustness']['checkpoint_sha256'],app.report['checkpoint_sha256'])
        self.assertEqual(self.client.get('/api/evaluation.json').status_code,200)


if __name__=='__main__': unittest.main()
