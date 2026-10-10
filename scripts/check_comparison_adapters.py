"""Local engineering checks: gradients, tiny overfit, label conversion and save/reload."""
import argparse
from datetime import datetime,timezone
import json
from pathlib import Path
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import torch

from comparison.data import CropDataset,load_protocol
from comparison.models import create_adapter
from comparison.runner import seed_all,state_hash,atomic_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--models',nargs='+',default=['ssdlite','ssd_mobilenetv3','yolo_v8n','efficientdet_d0'])
    parser.add_argument('--pretrained',action='store_true')
    parser.add_argument('--device',default='cpu')
    parser.add_argument('--steps',type=int,default=1)
    parser.add_argument('--report',type=Path,default=ROOT/'reports/comparison/adapter_checks.json')
    args = parser.parse_args()
    if args.steps < 1:
        parser.error('steps must be positive')
    torch.set_num_threads(4)
    load_protocol(ROOT)
    dataset = CropDataset(ROOT,'train',limit=2,seed=42)
    batch = [dataset[i] for i in range(2)]
    images,targets = zip(*batch)
    report = dict(created_at_utc=datetime.now(timezone.utc).isoformat(),device=args.device,
                  pretrained=args.pretrained,engineering_check_only=True,models={})
    for name in args.models:
        print(f'CHECK {name}',flush=True); seed_all(42)
        adapter = create_adapter(name,args.device,ROOT,args.pretrained)
        initial = adapter.state()
        optimizer = torch.optim.SGD(adapter.net.parameters(),lr=.001,momentum=.9)
        losses = []
        for step in range(args.steps):
            optimizer.zero_grad(set_to_none=True)
            loss,_ = adapter.loss(images,targets)
            if not torch.isfinite(loss):
                raise AssertionError('Non-finite loss')
            loss.backward()
            gradients = [p.grad for p in adapter.net.parameters() if p.grad is not None]
            if not gradients or not all(torch.isfinite(g).all() for g in gradients):
                raise AssertionError('Missing or non-finite gradients')
            torch.nn.utils.clip_grad_norm_(adapter.net.parameters(),10.,error_if_nonfinite=True)
            optimizer.step(); losses.append(float(loss.detach()))
        for key,value in initial.items():
            if any(key.endswith(suffix) for suffix in ['running_mean','running_var','num_batches_tracked']):
                if not torch.equal(value,adapter.state()[key]):
                    raise AssertionError(f'Frozen normalization buffer changed: {key}')
        before_state = adapter.state()
        before_predictions = adapter.predict(images)
        if state_hash(before_state) != state_hash(adapter.state()):
            raise AssertionError('Inference changed model state')
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/'checkpoint.pt'
            torch.save(before_state,path)
            restored = create_adapter(name,args.device,ROOT,False)
            restored.load(torch.load(path,map_location='cpu',weights_only=True))
            after_predictions = restored.predict(images)
        for before,after in zip(before_predictions,after_predictions):
            for key in before:
                torch.testing.assert_close(before[key],after[key],rtol=1e-5,atol=1e-5)
            if len(before['labels']) and (before['labels'].min()<0 or before['labels'].max()>23):
                raise AssertionError('Wrong output class convention')
        if args.steps>=5 and losses[-1]>=losses[0]:
            raise AssertionError('Tiny training-only overfit check did not decrease loss')
        report['models'][name] = dict(status='passed',steps=args.steps,losses=losses,
                                     predictions_per_image=[len(p['scores']) for p in before_predictions],
                                     finite_gradients=True,frozen_bn_buffers=True,
                                     save_reload_prediction_consistency=True,
                                     state_sha256=state_hash(before_state),initialization=adapter.metadata)
        atomic_json(args.report,report)
        print(f'PASS {name}: loss {losses[0]:.4f} -> {losses[-1]:.4f}',flush=True)


if __name__=='__main__':
    main()
