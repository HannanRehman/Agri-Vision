import tempfile
from pathlib import Path
import unittest
from unittest.mock import patch

import torch

from benchmarks import efficientdet_fp32 as b
from tuning.models import create_adapter


class TinyAdapter:
    def __init__(self, bad_loss=False, bad_gradient=False):
        self.device = torch.device('cpu')
        self.net = torch.nn.Linear(2, 1)
        self.bad_loss = bad_loss
        if bad_gradient:
            self.net.weight.register_hook(lambda grad: grad * float('nan'))

    def load(self, state):
        self.net.load_state_dict(state)

    def state(self):
        return {k: v.detach().clone() for k, v in self.net.state_dict().items()}

    def raw(self, images, targets):
        loss = ((self.net(images[0])-targets[0])**2).mean()
        if self.bad_loss:
            loss = loss * float('nan')
        return loss, {'class_loss': loss.detach()}

    def loss(self, images, targets):
        loss, components = self.raw(images, targets)
        return loss, {k: float(v) for k, v in components.items()}


class BenchmarkTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(4)

    def test_steps_preserve_updates_and_losses(self):
        adapter = TinyAdapter()
        initial = adapter.state()
        batches = [([torch.tensor([[1., 2.], [3., 4.]])], [torch.tensor([[.5], [.9]])])]
        with patch.object(b, 'tensor_loss', side_effect=lambda a, x, y: a.raw(x, y)):
            original = b.run_segment(adapter, initial, batches, False, 4, 42, .0001, .0005)
            optimized = b.run_segment(adapter, initial, batches, True, 4, 42, .0001, .0005)
        self.assertTrue(b.compare_states(original[1], optimized[1])['bitwise_equal'])
        self.assertTrue(torch.equal(original[2], optimized[2]))
        self.assertAlmostEqual(original[0]['mean_loss'], optimized[0]['mean_loss'])

    def test_nonfinite_loss_never_updates_weights(self):
        for optimized in (False, True):
            adapter = TinyAdapter(bad_loss=True)
            initial = adapter.state()
            optimizer = torch.optim.AdamW(adapter.net.parameters())
            with patch.object(b, 'tensor_loss', side_effect=lambda a, x, y: a.raw(x, y)):
                with self.assertRaises(ValueError):
                    b.step(adapter, optimizer, [torch.ones(1, 2)], [torch.ones(1, 1)], optimized)
            self.assertTrue(b.compare_states(initial, adapter.state())['bitwise_equal'])

    def test_nonfinite_gradient_never_updates_weights(self):
        for optimized in (False, True):
            adapter = TinyAdapter(bad_gradient=True)
            initial = adapter.state()
            optimizer = torch.optim.AdamW(adapter.net.parameters())
            with patch.object(b, 'tensor_loss', side_effect=lambda a, x, y: a.raw(x, y)):
                with self.assertRaises((ValueError, RuntimeError)):
                    b.step(adapter, optimizer, [torch.ones(1, 2)], [torch.ones(1, 1)], optimized)
            self.assertTrue(b.compare_states(initial, adapter.state())['bitwise_equal'])

    def test_output_cannot_be_inside_search_or_cover_it(self):
        with tempfile.TemporaryDirectory() as temp:
            parent = Path(temp)
            model = parent/'grid_search/v1/efficientdet_d0'
            for output in (model, model/'benchmark', parent/'grid_search/new', parent):
                with self.assertRaises(ValueError):
                    b.check_output_path(output, model, parent/'code')
            b.check_output_path(parent/'benchmark_results', model, parent/'code')

    def test_snapshot_detects_mutation(self):
        with tempfile.TemporaryDirectory() as temp:
            folder = Path(temp)
            (folder/'latest.json').write_text('original')
            before = b.protected_snapshot(folder)
            (folder/'latest.json').write_text('changed')
            self.assertNotEqual(before, b.protected_snapshot(folder))

    def test_real_efficientdet_loss_and_update_equivalence(self):
        # Small CPU engineering check; the T4 benchmark uses 512 and 640.
        with tempfile.TemporaryDirectory() as temp:
            torch.manual_seed(42)
            adapter = create_adapter('efficientdet_d0', 'cpu', Path(temp), 128, pretrained=False)
            initial = adapter.state()
            batches = [([torch.rand(3, 100, 120)],
                        [dict(boxes=torch.tensor([[10., 10., 60., 70.]]), labels=torch.tensor([1]))])]
            original = b.run_segment(adapter, initial, batches, False, 1, 42, .0001, .0005)
            optimized = b.run_segment(adapter, initial, batches, True, 1, 42, .0001, .0005)
            self.assertTrue(b.compare_states(original[1], optimized[1])['passed'])
            self.assertTrue(torch.equal(original[2], optimized[2]))


if __name__ == '__main__':
    unittest.main()
