import argparse
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import torch
from comparison import runner as r
from continuation import efficientdet as e
from continuation import search as s
from benchmarks import efficientdet_fp32 as b


class Adapter:
    def __init__(self):
        self.net = torch.nn.Linear(2, 1)
        self.device = torch.device('cpu')
        self.model_id = 'efficientdet_d0'

    def raw(self, images, targets):
        loss = ((self.net(images[0])-targets[0]['target'])**2).mean()
        return loss, {'class_loss': loss.detach()}

    def loss(self, images, targets):
        loss, components = self.raw(images, targets)
        return loss, {k: float(v) for k, v in components.items()}


class ContinuationTests(unittest.TestCase):
    def test_full_local_loop_preserves_update_and_statistics(self):
        adapter = Adapter()
        initial = {k: v.clone() for k, v in adapter.net.state_dict().items()}
        batch = ([torch.tensor([[1., 2.]])], [dict(target=torch.tensor([[.5]]), image_id=torch.tensor(1))])
        args = argparse.Namespace(lr=.001, rounds=30, optimizer='adamw', weight_decay=.0005,
                                  batch=1, workers=0, local_epochs=1, seed=42, max_train_batches=None)
        with patch.object(r, 'loader', return_value=[batch]*3):
            old = r.local_train(adapter, None, args, 4, 1, lambda **kw: None)
            expected = {k: v.clone() for k, v in adapter.net.state_dict().items()}
            adapter.net.load_state_dict(initial)
            with patch.object(b, 'tensor_loss', side_effect=lambda a, x, y: a.raw(x, y)):
                new = e.local_train(adapter, None, args, 4, 1, lambda **kw: None)
        self.assertTrue(b.compare_states(expected, adapter.net.state_dict())['bitwise_equal'])
        for key in ('unique_training_images', 'image_presentations', 'optimizer_steps', 'mean_loss', 'learning_rate'):
            self.assertEqual(old[key], new[key])

    def test_subprocess_only_redirects_training_runner(self):
        with patch.object(s, 'REAL_RUN') as run:
            s.dispatch(['python', '-m', 'tuning.runner', '--root', 'data'], check=True)
            run.assert_called_with(['python', '-m', 'continuation.runner', '--root', 'data'], check=True)
            s.dispatch(['python', '-m', 'tuning.preflight'], check=True)
            run.assert_called_with(['python', '-m', 'tuning.preflight'], check=True)

    def test_benchmark_evidence_rejects_missing_or_failed_pairs(self):
        evidence = json.loads(Path('reports/comparison/efficientdet_fp32_benchmark_20261019/gpu_benchmark.json').read_text())
        e.validate_evidence(evidence)
        evidence['cases'][0]['runs'][0]['equivalence']['bitwise_equal'] = False
        with self.assertRaises(ValueError):
            e.validate_evidence(evidence)

    def test_extension_identity_and_evidence_are_strict_without_rewriting_old_files(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            model = root/'model'
            evidence = json.loads(Path('reports/comparison/efficientdet_fp32_benchmark_20261019/gpu_benchmark.json').read_text())
            (root/'continuation').mkdir()
            (root/'continuation/benchmark_evidence.json').write_text(json.dumps(evidence))
            for number in range(1, 6):
                trial = model/'trials'/f'trial_{number:02d}'
                trial.mkdir(parents=True)
                (trial/'weights.pt').write_bytes(b'weights')
                (trial/'latest.json').write_text(json.dumps(dict(round=5, checkpoint='weights.pt', checkpoint_sha256=e.sha256(trial/'weights.pt'))))
                (trial/'settings.json').write_text(json.dumps(dict(config=dict(implementation_hash=e.BASE_HASH))))
            before = b.protected_snapshot(model)
            with patch.object(e, 'implementation_hash', return_value=e.BASE_HASH), patch.object(e, 'hardware', return_value=evidence['hardware']), patch.object(e, 'identity', return_value={'extension': 'one'}):
                first = e.profile(model, root, 'cpu')
                self.assertEqual(first, e.profile(model, root, 'cpu'))
                with patch.object(e, 'identity', return_value={'extension': 'two'}):
                    with self.assertRaises(ValueError):
                        e.profile(model, root, 'cpu')
            after = b.protected_snapshot(model)
            self.assertEqual(before, {k: v for k, v in after.items() if k != 'execution_profile.json'})


if __name__ == '__main__':
    unittest.main()
