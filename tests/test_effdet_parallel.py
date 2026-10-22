"""Verify trial assignment boundaries without starting training or using a GPU."""
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from parallel_effdet import screen


class AssignedScreeningTests(unittest.TestCase):
    def exercise(self, first, last):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root/'configs/tuning').mkdir(parents=True)
            (root/'configs/tuning/grid_v1.json').write_text(json.dumps(dict(screen_rounds=5, local_epochs=1)))
            output = root/'output'
            (output/'efficientdet_d0').mkdir(parents=True)
            trials = [(f'trial_{i:02d}', {'index': i}) for i in range(1, 17)]
            with patch.object(screen.torch.cuda, 'is_available', return_value=True), \
                 patch.object(screen.torch.cuda, 'get_device_name', return_value='Tesla T4'), \
                 patch.object(screen, 'implementation_hash', return_value=screen.BASE_HASH), \
                 patch.object(screen, 'sha256', return_value=screen.INITIAL_SHA), \
                 patch.object(screen, 'profile', return_value={'verified': True}), \
                 patch.object(screen, 'hardware', return_value={'gpu': 'Tesla T4'}), \
                 patch.object(screen, 'load_protocol', return_value={'dataset_checksums_sha256': 'dataset'}), \
                 patch.object(screen.search, 'configurations', return_value=iter(trials)), \
                 patch.object(screen.search, 'ensure_trial') as ensure, \
                 patch.object(screen.search, 'rank_trials', return_value=[]), \
                 patch.object(screen.search, 'write_ranking'), \
                 patch('subprocess.run') as subprocess_run:
                screen.main(['--root', str(root), '--output', str(output), '--first', str(first), '--last', str(last)])
                self.assertEqual([call.args[2] for call in ensure.call_args_list],
                    [f'trial_{i:02d}' for i in range(first, last+1)])
                self.assertTrue(all(call.args[4] == 5 for call in ensure.call_args_list))
                self.assertEqual(subprocess_run.call_count, 1)  # preflight only; no full-search controller
                status = json.loads((output/'efficientdet_d0'/f'parallel_{first:02d}_{last:02d}_status.json').read_text())
                self.assertEqual(status['phase'], 'assigned_screening_complete')
                self.assertFalse((output/'plan.json').exists())
                self.assertFalse((output/'efficientdet_d0'/'screening_selection.json').exists())

    def test_4gmail_owns_only_twelve_to_sixteen(self):
        self.exercise(12, 16)

    def test_71gmail_stops_before_twelve(self):
        self.exercise(1, 11)

    def test_reject_overlapping_range(self):
        with self.assertRaises(ValueError):
            screen.main(['--root', '.', '--output', '.', '--first', '8', '--last', '16'])


if __name__ == '__main__':
    unittest.main()
