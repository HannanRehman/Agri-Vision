"""Grid, schedule, checkpoint recovery and real input-size checks; no data training."""
import argparse
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
from tuning import runner as t
from tuning.models import create_adapter
from tuning.search import configurations, rank_trials
from tuning import search

ROOT = Path(__file__).resolve().parents[1]
NAMES = {i:f'class_{i}' for i in range(24)}


class Dataset:
    names = NAMES
    def __init__(self, *args, **kwargs):
        pass


class Adapter:
    def __init__(self):
        self.net = torch.nn.Linear(1, 1)
    def load(self, state):
        self.net.load_state_dict(state)
    def state(self):
        return {key:value.detach().clone() for key,value in self.net.state_dict().items()}


class GridTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(4)

    def test_full_four_dimensional_grid_and_per_model_ranges(self):
        grid = json.loads((ROOT/'configs/tuning/grid_v1.json').read_text())
        all_trials = []
        for model in grid['models']:
            trials = list(configurations(grid, model, 'data', 'code'))
            self.assertEqual(len(trials), 16)
            self.assertEqual(len({t.digest(config) for _,config in trials}),16)
            for _, config in trials:
                self.assertEqual(config['schedule_rounds'],30)
                self.assertEqual(config['local_epochs'],1)
            all_trials.extend(trials)
        self.assertEqual(len(all_trials),64)

    def test_short_trial_retains_full_schedule_prefix(self):
        config = dict(lr=.001, schedule_rounds=30)
        self.assertAlmostEqual(t.learning_rate(config,1),.001)
        self.assertAlmostEqual(t.learning_rate(config,30),.0001)
        self.assertGreater(t.learning_rate(config,5), .0001)
        self.assertLess(t.learning_rate(config,5), .001)
        self.assertAlmostEqual(t.learning_rate(config,10), .0001)
        self.assertAlmostEqual(t.learning_rate(config,20), .0001)
        self.assertEqual(t.learning_rate(dict(config,model='yolo_v8n'),30),.001)
        # Training to round 5 must not prematurely reach the terminal rate.
        self.assertNotAlmostEqual(t.learning_rate(config,5),.0001)

    def test_native_yolo_explicit_parameters_preserve_baseline_warmup(self):
        config = dict(model='yolo_v8n',lr=.0001785, schedule_rounds=30, weight_decay=.001, batch=4, resolution=512)
        options = t.native_options(config,3,2)
        self.assertEqual(options['optimizer'],'AdamW')
        self.assertEqual((options['batch'],options['imgsz']),(4,512))
        self.assertEqual(options['weight_decay'],.001)
        self.assertAlmostEqual(options['lr0'],t.learning_rate(config,3))
        self.assertEqual(options['warmup_bias_lr'],0.)
        self.assertNotIn('warmup_epochs',options)
        self.assertNotIn('nbs',options)

    def test_ssd_actual_transform_and_shared_weights_across_sizes(self):
        for model in ('ssdlite','ssd_mobilenetv3'):
            weights = None
            for size in (320,512):
                r.seed_all(42)
                adapter = create_adapter(model,'cpu',ROOT,size,False)
                if weights is None:
                    weights = adapter.state()
                else:
                    adapter.load(weights)
                    self.assertEqual(r.state_hash(adapter.state()),r.state_hash(weights))
                adapter.net.eval()
                transformed,_ = adapter.net.transform([torch.rand(3,110,170)],None)
                self.assertEqual(tuple(transformed.tensors.shape[-2:]),(size,size))

    def test_efficientdet_image_and_anchor_sizes_and_checkpoint_reload(self):
        weights = None
        for size in (512,640):
            r.seed_all(42)
            adapter = create_adapter('efficientdet_d0','cpu',ROOT,size,False)
            if weights is None:
                weights = adapter.state()
            else:
                adapter.load(weights)
            self.assertEqual(tuple(adapter.net.config.image_size),(size,size))
            self.assertEqual(tuple(adapter.prediction_bench.anchors.image_size),(size,size))
            self.assertEqual(tuple(adapter.batch([torch.rand(3,120,180)]).shape[-2:]),(size,size))
            self.assertEqual(r.state_hash(adapter.state()),r.state_hash(weights))

    def test_rank_uses_requested_budget_not_later_confirmation(self):
        with tempfile.TemporaryDirectory() as temporary:
            folder=Path(temporary)
            trials=[('trial_01',{}),('trial_02',{})]
            for name, scores in [('trial_01',[.4,.3]),('trial_02',[.3,.9])]:
                run=folder/'trials'/name
                (run/'results/metrics').mkdir(parents=True)
                r.atomic_json(run/'latest.json',dict(round=2))
                for n, score in enumerate(scores,1):
                    r.atomic_json(run/f'results/metrics/round_{n:03d}.json',dict(round=n,pooled_validation=dict(map_50_95=score)))
            self.assertEqual(rank_trials(folder,trials,1)[0]['trial_id'],'trial_01')
            self.assertEqual(rank_trials(folder,trials,2)[0]['trial_id'],'trial_02')

    def test_trial_interruption_extension_final_restart_and_completed_skip(self):
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary); model=root/'model';model.mkdir()
            initial_path=model/'initial.pt'
            initial=dict(model_state=dict(weight=torch.tensor([[1.]]),bias=torch.tensor([0.])),names=NAMES,round=0)
            r.atomic_torch(initial_path,initial)
            config=dict(model='ssdlite',lr=.001,weight_decay=.0005,batch=4,resolution=320,
                        optimizer='sgd',schedule_rounds=30,seed=42,dataset_hash='data',implementation_hash='code')
            protocol=dict(dataset_checksums_sha256='data',clients=['one','two','three'])
            calls=[]
            def step(adapter,root,config,state,protocol,number,workers,status):
                calls.append(number)
                return {k:v+number/100 for k,v in state.items()},[]
            def evaluate(adapter,*args,**kwargs):
                score=float(adapter.state()['weight'].item())/10
                return dict(map_50_95=score,map50=score,precision=.5,recall=.5,f1=.5,images=1)
            interrupted=root/'interrupted';direct=root/'direct';final=root/'fresh_final'
            with patch.object(t,'implementation_hash',return_value='code'), \
                 patch.object(t,'hardware',return_value={'gpu':'CPU'}), \
                 patch.object(t,'load_protocol',return_value=protocol), \
                 patch.object(t,'CropDataset',Dataset), \
                 patch.object(t,'initialization',return_value=(initial_path,initial)), \
                 patch.object(t,'create_adapter',side_effect=lambda *args:Adapter()), \
                 patch.object(r,'evaluate',side_effect=evaluate), redirect_stdout(io.StringIO()):
                with patch.object(t,'standard_step',side_effect=step):
                    t.run_trial(root,interrupted,model,config,2,torch.device('cpu'),0)
                self.assertEqual(calls,[1,2])
                with patch.object(t,'standard_step',side_effect=RuntimeError('synthetic stop')):
                    with self.assertRaisesRegex(RuntimeError,'synthetic stop'):
                        t.run_trial(root,interrupted,model,config,4,torch.device('cpu'),0)
                self.assertEqual(json.loads((interrupted/'latest.json').read_text())['round'],2)
                calls.clear()
                with patch.object(t,'standard_step',side_effect=step):
                    t.run_trial(root,interrupted,model,config,4,torch.device('cpu'),0)
                self.assertEqual(calls,[3,4])
                with patch.object(t,'standard_step',side_effect=AssertionError('repeated training')):
                    t.run_trial(root,interrupted,model,config,4,torch.device('cpu'),0)
                with patch.object(t,'standard_step',side_effect=step):
                    t.run_trial(root,direct,model,config,4,torch.device('cpu'),0)
                    calls.clear()
                    t.run_trial(root,final,model,config,4,torch.device('cpu'),0,True)
                self.assertEqual(calls,[1,2,3,4])
                with self.assertRaisesRegex(ValueError,'configuration'):
                    t.run_trial(root,interrupted,model,dict(config,lr=.002),4,torch.device('cpu'),0)
            for folder in (interrupted,direct,final):
                latest=json.loads((folder/'latest.json').read_text())
                state=torch.load(folder/latest['checkpoint'],weights_only=True)['model_state']
                self.assertAlmostEqual(state['weight'].item(),1.1,places=6)
                with (folder/'results/metrics/global_validation_metrics.csv').open() as stream:
                    self.assertEqual([int(row['round']) for row in csv.DictReader(stream)],[1,2,3,4])
            self.assertEqual(len(list((interrupted/'checkpoints').glob('*.pt'))),1)
            self.assertIsNone(json.loads((interrupted/'latest.json').read_text())['best_checkpoint'])
            self.assertIsNotNone(json.loads((final/'latest.json').read_text())['best_checkpoint'])

    def test_automatic_all_trials_two_finalists_and_fresh_final_resume(self):
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary); output=root/'grid'
            grid=json.loads((ROOT/'configs/tuning/grid_v1.json').read_text())
            grid_path=root/'grid.json';r.atomic_json(grid_path,grid)
            calls=[]; hardware={'gpu':'Tesla T4'}
            def fake_run(command,check):
                if 'tuning.preflight' in command:
                    return
                def argument(flag): return command[command.index(flag)+1]
                run=Path(argument('--output'));model_folder=Path(argument('--model-folder'))
                config=json.loads(Path(argument('--config')).read_text());stop=int(argument('--stop-round'))
                prior=json.loads((run/'latest.json').read_text())['round'] if (run/'latest.json').exists() else 0
                calls.append((run.name,prior,stop))
                (run/'checkpoints').mkdir(parents=True,exist_ok=True)
                initial=model_folder/'initial.pt'
                if not initial.exists():initial.write_bytes(b'common untrained weights')
                checkpoint=run/'checkpoints/latest.pt';checkpoint.write_bytes(str(stop).encode())
                settings=dict(config=config,hardware=hardware,loader_workers=0,
                              initial_checkpoint=str(initial),initial_checkpoint_sha256=search.sha256(initial))
                r.atomic_json(run/'settings.json',settings)
                trial_index=int(run.name.split('_')[-1]) if run.name.startswith('trial_') else 16
                for n in range(prior+1,stop+1):
                    score=trial_index/100+n/1000
                    metrics=dict(map_50_95=score,map50=score,precision=score,recall=score,f1=score)
                    r.atomic_json(run/f'results/metrics/round_{n:03d}.json',dict(round=n,pooled_validation=metrics))
                r.atomic_json(run/'latest.json',dict(round=stop,checkpoint='checkpoints/latest.pt',checkpoint_sha256=search.sha256(checkpoint)))
            command=['--root',str(root),'--output',str(output),'--grid',str(grid_path),'--models','ssdlite','--workers','0']
            with patch.object(torch.cuda,'is_available',return_value=True), \
                 patch.object(torch.cuda,'get_device_name',return_value='Tesla T4'), \
                 patch.object(search,'hardware',return_value=hardware), \
                 patch.object(search,'implementation_hash',return_value='fixed-code'), \
                 patch.object(search,'load_protocol',return_value=dict(dataset_checksums_sha256='data')), \
                 patch.object(search.subprocess,'run',side_effect=fake_run), redirect_stdout(io.StringIO()):
                search.main(command)
                self.assertEqual(len(calls),19)
                self.assertEqual(sum(prior==0 and stop==5 for _,prior,stop in calls),16)
                self.assertEqual(sum(prior==5 and stop==10 for _,prior,stop in calls),2)
                self.assertEqual(calls[-1],('final_training',0,30))
                calls.clear()
                search.main(command)
                self.assertEqual(calls,[])
            selection=json.loads((output/'ssdlite/selected_configuration.json').read_text())
            self.assertEqual(selection['trial_id'],'trial_16')
            self.assertEqual(json.loads((output/'search_status.json').read_text())['phase'],'complete')


if __name__ == '__main__':
    unittest.main()
