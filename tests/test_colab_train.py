"""Check sample-weighted aggregation independently of GPU training."""
import json
import csv
import tempfile
import unittest
from pathlib import Path

import numpy as np
from colab_train import RunMonitor, average_updates, save_round_metrics


class AggregationTests(unittest.TestCase):
    def test_weighted_three_client_average_preserves_inputs(self):
        updates = [([np.array([value], dtype=np.float32)], count)
                   for value, count in [(1, 2), (4, 3), (10, 5)]]
        result = average_updates(updates)
        np.testing.assert_allclose(result[0], [6.4])
        self.assertEqual(updates[0][0][0][0], 1)
        self.assertEqual(result[0].dtype, np.float32)

    def test_empty_weight_rejected(self):
        with self.assertRaises(ValueError):
            average_updates([([np.array([1.0])], 0)])

    def test_validation_exports_preserve_weights_and_all_rounds(self):
        with tempfile.TemporaryDirectory() as folder:
            output = Path(folder)
            values = [{'client': 'client1', 'validation_images': 1,
                       'metrics': {key: .2 for key in ('precision', 'recall', 'mAP50', 'mAP50_95')}},
                      {'client': 'client2', 'validation_images': 3,
                       'metrics': {key: .6 for key in ('precision', 'recall', 'mAP50', 'mAP50_95')}}]
            scores = save_round_metrics(output, 36, values)
            self.assertAlmostEqual(scores['mAP50'], .5)
            detail = json.loads((output / 'metrics_clients_round_036.json').read_text())
            self.assertEqual(detail['clients'], values)
            save_round_metrics(output, 37, values)
            with (output / 'global_validation_metrics.csv').open(newline='') as stream:
                rows = list(csv.DictReader(stream))
            self.assertEqual([row['round'] for row in rows], ['36', '37'])
            self.assertAlmostEqual(float(rows[0]['recall']), .5)
            self.assertFalse(list(output.glob('*.tmp')))

    def test_monitor_persists_progress(self):
        with tempfile.TemporaryDirectory() as folder:
            monitor = RunMonitor(Path(folder))
            monitor.update(round=2, client='client3', phase='training')
            status = json.loads((Path(folder) / 'status.json').read_text())
            self.assertEqual(status['round'], 2)
            self.assertEqual(status['client'], 'client3')
            monitor.close()
            completed = json.loads((Path(folder) / 'status.json').read_text())
            self.assertEqual(completed['phase'], 'complete')
