"""End-to-end real-data API checks, including label exclusion and abstention."""
import csv
import io
import json
import unittest
from unittest.mock import patch

import numpy as np
import app


class MonitoringTests(unittest.TestCase):
    def setUp(self):
        self.client=app.app.test_client()

    def test_state_uses_real_cycles_and_all_four_components(self):
        state=self.client.get('/api/state').json
        self.assertEqual(state['source'],'real_test_rig')
        self.assertEqual(state['totalRecords'],len(app.partitions['test']))
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
        end=len(app.partitions['test'])-1
        last=self.client.post('/api/seek',json={'position':end}).json
        self.assertEqual(last['position'],end)
        self.assertEqual(self.client.post('/api/step',json={'partition':'test','position':end}).json['status'],'end_of_records')
        self.assertEqual(self.client.post('/api/reset',json={'partition':'test','position':end}).json['position'],0)
        full=self.client.post('/api/partition',json={'partition':'all'}).json
        self.assertEqual(full['totalRecords'],2205)
        self.assertEqual(full['cycle'],1)
        self.assertEqual(self.client.post('/api/step',json={'partition':'all','position':0}).json['cycle'],2)

    def test_invalid_positions_and_partitions_do_not_mutate_state(self):
        for value in (-1,len(app.partitions['test']),True,None,1.2,'1'):
            self.assertEqual(self.client.post('/api/seek',json={'position':value}).status_code,400)
            self.assertEqual(self.client.get('/api/state').json['position'],0)
        for value in ('unknown',[],None):
            self.assertEqual(self.client.post('/api/partition',json={'partition':value}).status_code,400)

    def test_two_clients_and_tabs_have_independent_replay_and_exports(self):
        other=app.app.test_client()
        selected={'partition':'all','position':7}
        first=self.client.post('/api/seek',json=selected).json
        self.assertEqual(first['cycle'],8)
        for client in (self.client,other):
            self.assertEqual(client.get('/api/state').json['position'],0)
            self.assertEqual(client.get('/api/config').json['defaultPartition'],'test')
        self.assertEqual(other.post('/api/step',json={'partition':'test','position':2}).json['position'],3)
        self.assertEqual(self.client.get('/api/state',query_string=selected).json['cycle'],8)
        self.assertEqual(self.client.get('/api/reference',query_string=selected).json['cycle'],8)
        rows=list(csv.DictReader(io.StringIO(self.client.get('/api/snapshot.csv',query_string=selected).text)))
        self.assertTrue(all(row['cycle']=='8' and row['partition']=='all' for row in rows))
        self.assertEqual(other.get('/api/state').json['position'],0)

    def test_nonfinite_readings_produce_strict_json_and_empty_csv_cells(self):
        cycle=int(app.partitions['test'][0])
        for invalid in (np.nan,np.inf,-np.inf):
            with self.subTest(invalid=invalid):
                changed={**app.data,'features':app.data['features'].copy(),'traces':app.data['traces'].copy()}
                changed['features'][cycle,0]=invalid
                changed['traces'][cycle,0,10]=invalid
                with patch.object(app,'data',changed):
                    response=self.client.get('/api/state')
                    def reject_constant(value):
                        raise ValueError(f'Non-standard JSON number: {value}')
                    state=json.loads(response.text,parse_constant=reject_constant)
                    self.assertEqual(state['modelStatus'],'insufficient_data')
                    self.assertEqual(state['conditions'],[])
                    self.assertIsNone(state['summary']['flaggedComponents'])
                    self.assertIsNone(state['observations'][0]['mean'])
                    self.assertIsNone(state['observations'][0]['trace'][10])
                    self.assertEqual(state['dataQuality'],{'complete':False,'invalidChannels':['PS1']})
                    rows=list(csv.DictReader(io.StringIO(self.client.get('/api/snapshot.csv').text)))
                    self.assertEqual(len(rows),14)
                    self.assertEqual(rows[0]['mean'],'')

    def test_missing_trace_alone_also_abstains(self):
        changed={**app.data,'traces':app.data['traces'].copy()}
        changed['traces'][app.partitions['test'][0],1,0]=np.nan
        with patch.object(app,'data',changed):
            state=self.client.get('/api/state').json
        self.assertEqual(state['modelStatus'],'insufficient_data')
        self.assertEqual(state['dataQuality']['invalidChannels'],['PS2'])

    def test_invalid_get_selections_are_rejected_for_readings_labels_and_exports(self):
        for path in ('/api/state','/api/reference','/api/snapshot.csv'):
            for selection in ({'partition':'unknown'},{'position':'NaN'},{'position':'1.5'},
                              {'position':'-1'},{'position':len(app.partitions['test'])}):
                self.assertEqual(self.client.get(path,query_string=selection).status_code,400)

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
        self.assertEqual(set(evaluation['report']['models']),{'linear'})
        self.assertEqual(evaluation['robustness']['checkpoint_sha256'],app.report['checkpoint_sha256'])
        self.assertEqual(set(evaluation),{'status','report','robustness'})
        self.assertEqual(self.client.get('/api/evaluation.json').status_code,200)


if __name__=='__main__': unittest.main()
