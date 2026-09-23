"""Check sample-weighted aggregation independently of GPU training."""
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
from colab_train import RunMonitor, average_updates


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
