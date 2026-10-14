"""Verify chained continuation and recovery using synthetic tensors, without training."""
import ast
from contextlib import redirect_stdout
import csv
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import torch
from comparison import runner as r
from comparison.data import sha256
from scripts import continue_comparison as c
from test_30_round_workflow import Adapter, Dataset, NAMES


class EfficientDetContinuationTests(unittest.TestCase):
    def test_notebook_uses_completed_branch_and_pinned_round15(self):
        root = Path(__file__).resolve().parents[1]
        notebook = json.loads((root/'AgriVision_EfficientDet_Continue_30Rounds.ipynb').read_text())
        for cell in notebook['cells']:
            if cell['cell_type'] == 'code':
                ast.parse(''.join(cell['source']))
                self.assertEqual(cell['outputs'], [])
                self.assertIsNone(cell['execution_count'])
        context = {}
        exec(''.join(notebook['cells'][1]['source']), context)
        self.assertEqual(context['MODELS'], ['efficientdet_d0'])
        self.assertEqual(context['TARGET_ROUNDS'], 30)
        self.assertEqual(context['EXPECTED_SOURCE_ROUND'], 15)
        self.assertTrue(str(context['SOURCES']['efficientdet_d0']).endswith('_to15'))
        self.assertTrue(str(context['RUNS']['efficientdet_d0']).endswith('_to30_from15'))
        self.assertEqual(json.loads((root/'AgriVision_Comparison.ipynb').read_text())['cells'], notebook['cells'])

    def test_round15_weights_and_best_survive_interruption_then_complete30(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            source, output = root/'to15', root/'to30_from15'
            (source/'checkpoints').mkdir(parents=True)
            (source/'results/metrics').mkdir(parents=True)
            (root/'configs/comparison').mkdir(parents=True)
            spec = root/'configs/comparison/efficientdet_d0.json'
            spec.write_text('{}')
            protocol = dict(dataset_checksums_sha256='synthetic', clients=['client1','client2','client3'])
            config = dict(model='efficientdet_d0', mode='main', rounds=15, local_epochs=1,
                          batch=4, seed=42, lr=.0001, optimizer='adamw', weight_decay=.0001,
                          max_train_batches=None, validation_limit=None,
                          continuation=dict(original_rounds=10),
                          dataset_checksums_sha256='synthetic', implementation_sha256=r.implementation_hash(),
                          model_spec_sha256=sha256(spec))
            config_hash = c.canonical_hash(config)
            for number, name in [(0,'initial'), (11,'global_round_011'), (15,'global_round_015')]:
                state = dict(weight=torch.tensor([[float(number)]]), bias=torch.tensor([0.]))
                r.atomic_torch(source/f'checkpoints/{name}.pt',
                               dict(round=number, names=NAMES, config_hash=config_hash, model_state=state))
            settings = dict(config=config, config_hash=config_hash, diagnostic_only=False,
                            hardware=dict(gpu='CPU', torch=torch.__version__, packages={}),
                            initial_checkpoint_sha256=sha256(source/'checkpoints/initial.pt'))
            r.atomic_json(source/'settings.json', settings)
            r.atomic_json(source/'latest.json', dict(round=15, checkpoint='checkpoints/global_round_015.pt',
                          checkpoint_sha256=sha256(source/'checkpoints/global_round_015.pt'),
                          best_map=.6, best_checkpoint='checkpoints/global_round_011.pt'))
            metrics = dict(map_50_95=.5, map50=.7, precision=.6, recall=.8, f1=.68, images=1)
            for number in range(1,16):
                r.atomic_json(source/f'results/metrics/round_{number:03d}.json',
                              dict(round=number, round_seconds=1., pooled_validation=metrics))
            source_hashes = {str(p):sha256(p) for p in source.rglob('*') if p.is_file()}
            observed_rounds = []
            def synthetic_step(adapter, dataset, args, number, index, progress):
                import math
                effective_lr = args.lr*(.1+.9*(1+math.cos(math.pi*(number-1)/(args.rounds-1)))/2)
                self.assertAlmostEqual(effective_lr, .00001)
                self.assertGreaterEqual(number, 16)
                if number == 16:
                    self.assertEqual(adapter.state()['weight'].item(), 15.)
                observed_rounds.append(number)
                adapter.load({k:v+.01 for k,v in adapter.state().items()})
                return dict(unique_training_images=1, learning_rate=effective_lr)
            def interrupted_step(adapter, dataset, args, number, index, progress):
                if number == 17:
                    raise RuntimeError('synthetic interruption')
                return synthetic_step(adapter,dataset,args,number,index,progress)
            command = ['--root',str(root),'--source',str(source),'--output',str(output),'--rounds','30','--workers','0']
            with patch.object(c,'load_protocol',return_value=protocol), patch.object(c,'CropDataset',Dataset), \
                 patch.object(torch.cuda,'is_available',return_value=True), \
                 patch.object(torch.cuda,'get_device_name',return_value='CPU'), \
                 patch.object(r,'create_adapter',side_effect=lambda *args,**kwargs:Adapter()), \
                 patch.object(r,'evaluate',return_value=metrics), redirect_stdout(io.StringIO()):
                with patch.object(r,'local_train',side_effect=interrupted_step):
                    with self.assertRaisesRegex(RuntimeError,'synthetic interruption'):
                        c.main(command)
                self.assertEqual(json.loads((output/'latest.json').read_text())['round'],16)
                with patch.object(r,'local_train',side_effect=synthetic_step):
                    c.main(command)
                with patch.object(r,'local_train',side_effect=AssertionError('completed run repeated')):
                    c.main(command)
            self.assertEqual(observed_rounds, [number for number in range(16,31) for _ in range(3)])
            self.assertEqual(json.loads((output/'latest.json').read_text())['best_checkpoint'], 'checkpoints/global_round_011.pt')
            self.assertEqual(json.loads((output/'status.json').read_text())['completed_rounds'],30)
            with (output/'results/metrics/global_validation_metrics.csv').open(newline='') as saved:
                self.assertEqual([int(row['round']) for row in csv.DictReader(saved)], list(range(1,31)))
            self.assertEqual(source_hashes, {str(p):sha256(p) for p in source.rglob('*') if p.is_file()})


if __name__ == '__main__':
    unittest.main()
