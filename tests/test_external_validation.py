"""External data schema, source overlap and frozen checkpoint checks."""
import itertools
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

import app
from hydraulics import OUTPUT, SENSORS, file_hash, find_raw_data, summarize_cycle
from scripts.external_validation import MANIFEST, _read_candidates, load_candidates, read_external


def write_fixture(directory, change_pressure):
    """A small parser fixture; its labels are deliberately artificial test values."""
    (directory / 'metadata.json').write_text(json.dumps({
        'source_id': 'test-fixture', 'machine_id': 'fixture-bench',
        'collection_period': 'test-only', 'independence_basis': 'different_machine',
        'label_provenance': 'test fixture', 'sensor_alignment': 'test fixture'}))
    (directory / 'profile.txt').write_text(
        '100\t100\t0\t130\n20\t90\t1\t115\n3\t80\t2\t100\n100\t73\t0\t90\n')
    original = find_raw_data()
    for sensor in SENSORS:
        with (original / f"{sensor['id']}.txt").open() as source, \
                (directory / f"{sensor['id']}.txt").open('w') as target:
            for line in itertools.islice(source, 4):
                if change_pressure and sensor['id'] == 'PS1':
                    values = np.fromstring(line, sep=' ', dtype=np.float64)
                    values[0] += 10
                    line = '\t'.join(map(str, values)) + '\n'
                target.write(line)


class ExternalDataTests(unittest.TestCase):
    def test_rejects_cycle_rows_reused_from_training_source(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            write_fixture(directory, change_pressure=False)
            with self.assertRaisesRegex(ValueError, 'exactly match UCI training'):
                read_external(directory, app.data['features'])

    def test_new_rows_follow_native_feature_and_tcn_sequence_rules(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            write_fixture(directory, change_pressure=True)
            values, metadata, source_hash = read_external(directory, app.data['features'])
            self.assertEqual(values['features'].shape, (4, 140))
            self.assertEqual(values['high'].shape, (4, 9, 600))
            self.assertEqual(values['low'].shape, (4, 5, 60))
            self.assertEqual(metadata['machine_id'], 'fixture-bench')
            self.assertEqual(len(source_hash), 64)
            with (directory / 'PS1.txt').open() as source:
                raw = np.fromstring(next(source), sep=' ')
            np.testing.assert_allclose(values['features'][0, :10], summarize_cycle(raw))
            np.testing.assert_allclose(values['high'][0, 0], raw.reshape(600, 10).mean(axis=1),
                                       rtol=1e-6, atol=1e-6)
            with (directory / 'TS1.txt').open() as source:
                temperature = np.fromstring(next(source), sep=' ')
            np.testing.assert_allclose(values['low'][0, 0], temperature)

    def test_rejects_missing_condition_class(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            write_fixture(directory, change_pressure=True)
            (directory / 'profile.txt').write_text('100\t100\t0\t130\n')
            with self.assertRaisesRegex(ValueError, 'omit classes'):
                read_external(directory, app.data['features'])


@unittest.skipUnless(MANIFEST.exists(), 'Prepare external candidates first.')
class FrozenCandidateTests(unittest.TestCase):
    def test_candidate_hashes_source_and_normalization_are_frozen(self):
        manifest = _read_candidates()
        self.assertEqual(manifest['training_dataset_fingerprint'], str(app.data['fingerprint']))
        self.assertEqual(manifest['training_cycles'], 2205)
        self.assertEqual(manifest['candidates']['tcn']['epochs'], 33)
        self.assertEqual(file_hash(OUTPUT / 'model.pt'), app.report['checkpoint_sha256'])
        linear, tcn = load_candidates(manifest)
        self.assertEqual(linear.network.weight.shape[1], 140)
        self.assertEqual(tcn.feature_mean.shape[0], 140)


if __name__ == '__main__':
    unittest.main()
