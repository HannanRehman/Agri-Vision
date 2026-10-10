"""Synthetic state/recovery checks; no detector training or optimizer steps."""
from contextlib import redirect_stdout
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from types import SimpleNamespace

import numpy as np
import torch
from scripts import continue_comparison as c
from scripts import yolo_native_train as y
from comparison import runner as r
from comparison.data import sha256


NAMES = {i:f'class{i}' for i in range(24)}


class Dataset:
    names=NAMES
    rows=[dict(client='client1',image_id=1,key='a',image='a.jpg')]
    def __init__(self,*args,**kwargs):pass
    def __len__(self):return 1


class Adapter:
    def __init__(self):
        self.net=torch.nn.Linear(1,1)
    def load(self,state):self.net.load_state_dict(state)
    def state(self):return {k:v.detach().clone() for k,v in self.net.state_dict().items()}


class WorkflowTests(unittest.TestCase):
    def test_extension_preserves_old_schedule_without_lr_jump(self):
        expected=[.001*(.1+.9*(1+np.cos(np.pi*(n-1)/9))/2) for n in range(1,11)]
        for n,value in enumerate(expected,1):self.assertAlmostEqual(c.extension_rate(.001,10,n),value)
        for n in range(11,31):self.assertAlmostEqual(c.extension_rate(.001,10,n),.0001)
        self.assertAlmostEqual(c.extension_rate(.0001,10,11),.00001)
        with self.assertRaises(ValueError):c.extension_rate(.001,10,0)

    def test_historical_numpy_aggregation_and_integer_buffers(self):
        initial=dict(w=torch.tensor([1.,2.]),counter=torch.tensor(0))
        a=dict(w=torch.tensor([3.,4.]),counter=torch.tensor(7))
        b=dict(w=torch.tensor([9.,12.]),counter=torch.tensor(12))
        actual=y.historical_average([(a,1),(b,3)],initial)
        expected=a['w'].numpy()*.25+b['w'].numpy()*.75
        np.testing.assert_array_equal(actual['w'].numpy(),expected)
        self.assertEqual(actual['counter'].item(),0)
        b['w'][0]=float('nan')
        with self.assertRaises(ValueError):y.historical_average([(a,1),(b,3)],initial)

    def test_native_settings_restore_historical_training(self):
        self.assertEqual(y.NATIVE_ARGS['optimizer'],'auto')
        self.assertEqual(y.NATIVE_ARGS['epochs'],1)
        self.assertEqual(y.NATIVE_ARGS['batch'],8)
        self.assertEqual(y.NATIVE_ARGS['imgsz'],640)
        self.assertEqual(y.NATIVE_ARGS['warmup_epochs'],3.)
        self.assertIsNone(y.NATIVE_ARGS['freeze'])
        self.assertTrue(y.NATIVE_ARGS['amp'])

    def test_absolute_native_yaml_lists(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp)
            path=y.runtime_yaml(root,Dataset(),root/'lists')
            import yaml
            value=yaml.safe_load(path.read_text())
            self.assertEqual(value['names'],NAMES)
            self.assertEqual(Path(value['train']).read_text().strip(),str((root/'a.jpg').resolve()))

    def test_source_protection_recovery_and_completed_run_skip(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);source=root/'original';source.mkdir()
            (source/'checkpoints').mkdir();(source/'results/metrics').mkdir(parents=True)
            (root/'configs/comparison').mkdir(parents=True)
            spec=root/'configs/comparison/ssdlite.json';spec.write_text('{}')
            protocol=dict(dataset_checksums_sha256='synthetic',clients=['client1','client2','client3'])
            config=dict(model='ssdlite',mode='main',rounds=3,local_epochs=1,batch=1,seed=42,
                        lr=.001,optimizer='sgd',weight_decay=.0005,max_train_batches=None,validation_limit=None,
                        dataset_checksums_sha256='synthetic',implementation_sha256=r.implementation_hash(),model_spec_sha256=sha256(spec))
            config_hash=c.canonical_hash(config)
            state=dict(weight=torch.tensor([[1.]]),bias=torch.tensor([0.]))
            for number,name in [(0,'initial'),(2,'global_round_002')]:
                r.atomic_torch(source/f'checkpoints/{name}.pt',dict(round=number,names=NAMES,config_hash=config_hash,model_state=state))
            settings=dict(config=config,config_hash=config_hash,diagnostic_only=False,
                          hardware=dict(gpu='CPU',torch=torch.__version__,packages={}),
                          initial_checkpoint_sha256=sha256(source/'checkpoints/initial.pt'))
            r.atomic_json(source/'settings.json',settings)
            r.atomic_json(source/'latest.json',dict(round=2,checkpoint='checkpoints/global_round_002.pt',
                          checkpoint_sha256=sha256(source/'checkpoints/global_round_002.pt'),best_map=.2,best_checkpoint='checkpoints/global_round_002.pt'))
            originals={str(p):sha256(p) for p in source.rglob('*') if p.is_file()}
            interrupted=root/'interrupted';direct=root/'direct'
            metrics=dict(map_50_95=.3,map50=.4,precision=.5,recall=.6,f1=.54,images=1)
            calls=[]
            def synthetic_step(adapter,dataset,args,number,index,status):
                # Deterministic synthetic tensor update, not a training step.
                factor=.1+.9*(1+np.cos(np.pi*(number-1)/max(args.rounds-1,1)))/2
                self.assertAlmostEqual(args.lr*factor,c.extension_rate(.001,3,number))
                new={k:v+number/100+index/1000 for k,v in adapter.state().items()}
                adapter.load(new)
                return dict(unique_training_images=1,learning_rate=c.extension_rate(.001,3,number))
            def fail_second(*args,**kwargs):
                calls.append(args[4])
                if len(calls)==2:raise RuntimeError('synthetic interruption')
                return synthetic_step(*args,**kwargs)
            common=['--root',str(root),'--source',str(source),'--rounds','4','--workers','0']
            with patch.object(c,'load_protocol',return_value=protocol),patch.object(c,'CropDataset',Dataset), \
                 patch.object(torch.cuda,'is_available',return_value=True),patch.object(torch.cuda,'get_device_name',return_value='CPU'), \
                 patch.object(r,'create_adapter',side_effect=lambda *a,**kw:Adapter()),patch.object(r,'evaluate',return_value=metrics), \
                 redirect_stdout(io.StringIO()):
                with patch.object(r,'local_train',side_effect=fail_second):
                    with self.assertRaisesRegex(RuntimeError,'synthetic interruption'):c.main(common+['--output',str(interrupted)])
                self.assertEqual(json.loads((interrupted/'latest.json').read_text())['round'],2)
                with patch.object(r,'local_train',side_effect=synthetic_step):
                    c.main(common+['--output',str(interrupted)])
                    c.main(common+['--output',str(direct)])
                with patch.object(r,'local_train',side_effect=AssertionError('completed model retrained')):
                    c.main(common+['--output',str(interrupted)])
                with self.assertRaisesRegex(ValueError,'Continuation settings'):
                    c.main(['--root',str(root),'--source',str(source),'--rounds','5','--output',str(interrupted)])
                changed=json.loads((source/'latest.json').read_text());changed['checkpoint_sha256']='bad'
                r.atomic_json(source/'latest.json',changed)
                with self.assertRaisesRegex(ValueError,'checkpoint changed'):c.validate_source(root,source,torch.device('cpu'))
                r.atomic_json(source/'latest.json',dict(round=2,checkpoint='checkpoints/global_round_002.pt',
                              checkpoint_sha256=sha256(source/'checkpoints/global_round_002.pt'),best_map=.2,best_checkpoint='checkpoints/global_round_002.pt'))
            recovered=torch.load(interrupted/'checkpoints/global_round_004.pt',weights_only=True)
            expected=torch.load(direct/'checkpoints/global_round_004.pt',weights_only=True)
            self.assertEqual(r.state_hash(recovered['model_state']),r.state_hash(expected['model_state']))
            self.assertEqual(originals,{str(p):sha256(p) for p in source.rglob('*') if p.is_file()})
            kept={p.name for p in (interrupted/'checkpoints').glob('*.pt')}
            self.assertEqual(kept,{'initial.pt','global_round_003.pt','global_round_004.pt'})

    def test_unsafe_checkpoint_path_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            with self.assertRaises(ValueError):c.safe_checkpoint(Path(temp),'../outside.pt')

    def test_native_partial_round_resume_keeps_same_client_initialization(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);(root/'yolov8n.pt').write_bytes(b'synthetic pretrained placeholder')
            protocol=dict(dataset_checksums_sha256='synthetic',clients=['client1','client2','client3'])
            class Model:
                def __init__(self):self.model=torch.nn.Linear(1,1)
                def val(self,**kwargs):return SimpleNamespace(box=SimpleNamespace(mp=.5,mr=.4,map50=.3,map=.2))
            def synthetic_fit(model,data,work,seed,extra=None):
                local=work/'local';local.mkdir(parents=True)
                (local/'args.yaml').write_text('optimizer: auto\n')
                (local/'results.csv').write_text('epoch,train/cls_loss\n1,1.0\n')
                model.trainer=SimpleNamespace(optimizer=SimpleNamespace(param_groups=[dict(lr=.0001)]))
                model.native_transfer_verified=True
                # No optimizer: artificial state change checks round orchestration only.
                return {k:v.detach().clone()+.1 for k,v in model.model.state_dict().items()},local
            calls=[]
            def fail_second(*args,**kwargs):
                calls.append(1)
                if len(calls)==2:raise RuntimeError('synthetic native interruption')
                return synthetic_fit(*args,**kwargs)
            env=dict(gpu='CPU',torch=torch.__version__,packages={})
            pilot=root/'pilot.json';pilot.write_text(json.dumps(dict(phase='passed',pipeline_hash=y.pipeline_hash(),hardware=env,
                                                                   dataset_sha256='synthetic',seed=42)))
            common=['--root',str(root),'--rounds','2','--pilot-record',str(pilot)]
            interrupted=root/'interrupted';direct=root/'direct'
            metrics=dict(map_50_95=.3,map50=.4,precision=.5,recall=.6,f1=.54,images=1)
            with patch.object(y,'load_protocol',return_value=protocol),patch.object(y,'CropDataset',Dataset), \
                 patch.object(y,'hardware',return_value=env),patch.object(y,'create_yolo',side_effect=lambda *a:Model()), \
                 patch.object(y,'NativePredictor'),patch.object(r,'evaluate',return_value=metrics), \
                 patch.object(torch.cuda,'is_available',return_value=True),patch.object(torch.cuda,'manual_seed_all'), \
                 patch.object(torch.cuda,'empty_cache'),redirect_stdout(io.StringIO()):
                with patch.object(y,'native_train',side_effect=fail_second):
                    with self.assertRaisesRegex(RuntimeError,'synthetic native interruption'):y.main(common+['--output',str(interrupted)])
                self.assertEqual(json.loads((interrupted/'latest.json').read_text())['round'],0)
                with patch.object(y,'native_train',side_effect=synthetic_fit):
                    y.main(common+['--output',str(interrupted)])
                    y.main(common+['--output',str(direct)])
                with patch.object(y,'native_train',side_effect=AssertionError('completed native run retrained')):
                    y.main(common+['--output',str(interrupted)])
                with self.assertRaisesRegex(ValueError,'resume config'):
                    y.main(['--root',str(root),'--rounds','3','--output',str(interrupted)])
            recovered=torch.load(interrupted/'checkpoints/global_round_002.pt',weights_only=True)
            expected=torch.load(direct/'checkpoints/global_round_002.pt',weights_only=True)
            self.assertEqual(r.state_hash(recovered['model_state']),r.state_hash(expected['model_state']))
            record=json.loads((interrupted/'results/metrics/round_001.json').read_text())
            self.assertEqual(len({item['round_start_sha256'] for item in record['clients']}),1)
            self.assertIn('native_validation_image_weighted_summary',record)


if __name__=='__main__':unittest.main()
