import json
from contextlib import redirect_stdout
import io
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
from PIL import Image
import torch

from comparison.data import CropDataset
from comparison.evaluation import DetectionEvaluator
from comparison.runner import fedavg,state_hash
from comparison import runner
from comparison import final_evaluate


def target(boxes,labels,image_id=1):
    return dict(boxes=torch.tensor(boxes,dtype=torch.float32).reshape(-1,4),
                labels=torch.tensor(labels,dtype=torch.int64),image_id=torch.tensor(image_id),
                original_size=torch.tensor([100,100]))


def prediction(boxes,labels,scores):
    return dict(boxes=torch.tensor(boxes,dtype=torch.float32).reshape(-1,4),
                labels=torch.tensor(labels,dtype=torch.int64),scores=torch.tensor(scores,dtype=torch.float32))


class EvaluationTests(unittest.TestCase):
    def test_perfect_detection_and_missing_class(self):
        evaluator = DetectionEvaluator({0:'healthy',1:'disease'})
        evaluator.update([target([[10,10,30,30]],[0])],[prediction([[10,10,30,30]],[0],[.9])])
        result = evaluator.compute()
        self.assertAlmostEqual(result['map_50_95'],1.)
        self.assertEqual(result['precision'],1.)
        self.assertEqual(result['recall'],1.)
        self.assertIsNone(result['per_class'][1]['map50'])

    def test_wrong_class_is_false_positive_and_missed_truth(self):
        evaluator = DetectionEvaluator({0:'healthy',1:'disease'})
        evaluator.update([target([[10,10,30,30]],[0])],[prediction([[10,10,30,30]],[1],[.9])])
        result = evaluator.compute()
        self.assertEqual((result['true_positives'],result['false_positives'],result['false_negatives']),(0,1,1))

    def test_ap_uses_low_score_predictions_but_pr_uses_cutoff(self):
        evaluator = DetectionEvaluator({0:'crop'})
        evaluator.update([target([[10,10,30,30]],[0])],[prediction([[10,10,30,30]],[0],[.1])])
        result = evaluator.compute()
        self.assertAlmostEqual(result['map50'],1.)
        self.assertEqual(result['recall'],0.)

    def test_empty_predictions_are_supported(self):
        evaluator = DetectionEvaluator({0:'crop'})
        evaluator.update([target([[10,10,30,30]],[0])],[prediction([],[],[])])
        result = evaluator.compute()
        self.assertEqual(result['map50'],0.)
        self.assertEqual(result['false_negatives'],1)

    def test_duplicate_predictions_match_gt_once(self):
        evaluator = DetectionEvaluator({0:'crop'})
        evaluator.update([target([[10,10,30,30]],[0])],
                         [prediction([[10,10,30,30],[10,10,30,30]],[0,0],[.9,.8])])
        result = evaluator.compute()
        self.assertEqual(result['precision'],.5)
        self.assertEqual(result['recall'],1.)
        with self.assertRaises(ValueError):
            evaluator.update([target([[10,10,30,30]],[0])],[prediction([],[],[])])


class AggregationTests(unittest.TestCase):
    def test_weighting_and_integer_buffers(self):
        a = dict(weight=torch.tensor([1.,3.]),counter=torch.tensor(0))
        b = dict(weight=torch.tensor([5.,7.]),counter=torch.tensor(0))
        output = fedavg([(a,1),(b,3)])
        self.assertTrue(torch.equal(output['weight'],torch.tensor([4.,6.])))
        self.assertEqual(output['counter'].dtype,torch.int64)
        self.assertTrue(torch.equal(a['weight'],torch.tensor([1.,3.])))
        b['counter'] = torch.tensor(1)
        with self.assertRaises(ValueError):
            fedavg([(a,1),(b,3)])

    def test_nonfinite_update_and_zero_samples_rejected(self):
        with self.assertRaises(ValueError):
            fedavg([(dict(x=torch.tensor(float('nan'))),1)])
        with self.assertRaises(ValueError):
            fedavg([(dict(x=torch.tensor(1.)),0)])

    def test_hash_includes_scalar_integer_and_float_values(self):
        a = dict(x=torch.tensor(1.),n=torch.tensor(0))
        b = dict(x=torch.tensor(2.),n=torch.tensor(0))
        self.assertNotEqual(state_hash(a),state_hash(b))


class DatasetTests(unittest.TestCase):
    def test_ids_boxes_and_horizontal_flip(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary); directory = root/'manifests/non_chili_v1'; directory.mkdir(parents=True)
            (directory/'class_map.json').write_text(json.dumps({'names':{i:f'class{i}' for i in range(24)}}))
            Image.fromarray(np.zeros((100,200,3),dtype=np.uint8)).save(root/'sample.jpg')
            (root/'sample.txt').write_text('23 0.25 0.5 0.2 0.4\n')
            (directory/'retained.csv').write_text('key,split,client,image,label\nsample,train,client1,sample.jpg,sample.txt\n')
            dataset = CropDataset(root,'train','client1')
            image,item = dataset[0]
            self.assertEqual(tuple(image.shape),(3,100,200))
            self.assertEqual(item['labels'].tolist(),[23])
            self.assertTrue(torch.allclose(item['boxes'],torch.tensor([[30.,30.,70.,70.]])))
            dataset.augment = True
            with patch('torch.rand',return_value=torch.tensor(0.)):
                _,item = dataset[0]
            self.assertTrue(torch.allclose(item['boxes'],torch.tensor([[130.,30.,170.,70.]])))
            with self.assertRaises(ValueError):
                CropDataset(root,'valid',augment=True)


class TinyAdapter:
    def __init__(self):
        self.net = torch.nn.Linear(1,1)
        self.device = torch.device('cpu')
        self.metadata = {'test_adapter':True}
    def state(self):
        return {k:v.detach().clone() for k,v in self.net.state_dict().items()}
    def load(self,state):
        self.net.load_state_dict(state)
    def loss(self,images,targets):
        value = torch.stack([image.mean().reshape(1) for image in images])
        loss = (self.net(value)-1).square().mean()
        return loss,{'test_loss':float(loss.detach())}
    def predict(self,images):
        return [prediction([],[],[]) for _ in images]


class RecoveryTests(unittest.TestCase):
    def test_partial_round_restarts_and_matches_uninterrupted_training(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary); manifests = root/'manifests/non_chili_v1'; manifests.mkdir(parents=True)
            config = root/'configs/comparison'; config.mkdir(parents=True)
            (config/'ssdlite.json').write_text('{"input_size_candidate":320}')
            (manifests/'class_map.json').write_text(json.dumps({'names':{i:f'class{i}' for i in range(24)}}))
            text = 'key,split,client,image,label\n'
            for index,(split,client) in enumerate([('train','client1'),('train','client2'),('train','client3'),('valid','client1')]):
                Image.fromarray(np.full((32,32,3),40+index*30,dtype=np.uint8)).save(root/f'{index}.jpg')
                (root/f'{index}.txt').write_text('0 0.5 0.5 0.5 0.5\n')
                text += f'{index},{split},{client},{index}.jpg,{index}.txt\n'
            (manifests/'retained.csv').write_text(text)
            protocol = {'dataset_checksums_sha256':'synthetic_test','clients':['client1','client2','client3']}
            actual_train = runner.local_train
            calls = []
            def interrupt_second_client(*args,**kwargs):
                calls.append(args[4])
                if len(calls)==2:
                    raise RuntimeError('simulated interrupted client')
                return actual_train(*args,**kwargs)
            common = ['--root',str(root),'--model','ssdlite','--mode','pilot','--device','cpu',
                      '--batch','1','--threads','1','--workers','0','--max-train-batches','1','--validation-limit','1']
            interrupted = root/'interrupted'; direct = root/'direct'
            with patch.object(runner,'load_protocol',return_value=protocol), \
                 patch.object(runner,'create_adapter',side_effect=lambda *args,**kwargs:TinyAdapter()), \
                 redirect_stdout(io.StringIO()):
                with patch.object(runner,'local_train',side_effect=interrupt_second_client):
                    with self.assertRaisesRegex(RuntimeError,'simulated interrupted'):
                        runner.main(common+['--output',str(interrupted)])
                self.assertFalse((interrupted/'latest.json').exists())
                runner.main(common+['--output',str(interrupted),'--resume'])
                runner.main(common+['--output',str(direct)])
                with patch.object(runner,'local_train',side_effect=AssertionError('completed round repeated')):
                    runner.main(common+['--output',str(interrupted),'--resume'])
                with self.assertRaisesRegex(ValueError,'Resume settings'):
                    runner.main(common+['--output',str(interrupted),'--resume','--rounds','2'])
                with patch.object(final_evaluate,'load_protocol',return_value=protocol), \
                     patch.object(final_evaluate,'create_adapter',side_effect=lambda *args,**kwargs:TinyAdapter()), \
                     patch('sys.argv',['final_evaluate','--root',str(root),'--run',str(interrupted),
                                       '--split','valid','--device','cpu','--batch','1','--workers','0','--profile-images','1']):
                    final_evaluate.main()
                evaluation = json.loads((interrupted/'results/evaluations/best_valid.json').read_text())
                self.assertEqual(evaluation['runtime']['measured_images'],1)
                self.assertEqual(evaluation['metrics']['images'],1)
                self.assertTrue(evaluation['diagnostic_only'])
            recovered = torch.load(interrupted/'checkpoints/global_round_001.pt',weights_only=True)
            uninterrupted = torch.load(direct/'checkpoints/global_round_001.pt',weights_only=True)
            self.assertEqual(state_hash(recovered['model_state']),state_hash(uninterrupted['model_state']))
            record = json.loads((interrupted/'results/metrics/round_001.json').read_text())
            self.assertEqual(len({client['round_start_sha256'] for client in record['clients']}),1)
            self.assertEqual(record['pooled_validation']['images'],1)
            self.assertIn('client1',record['pooled_validation']['per_client'])


if __name__ == '__main__':
    unittest.main()
