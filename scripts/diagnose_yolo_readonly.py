"""Audit YOLO checkpoints and validation exports without optimizer steps or training."""
import ast
import copy
import csv
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
import torch
from torchvision.ops import box_iou
from comparison.data import CropDataset, sha256
from comparison.models import create_adapter
from comparison.runner import implementation_hash, state_hash
from comparison.evaluation import DetectionEvaluator

RUN = ROOT / 'experiments/yolo_v8n_non_chili_seed42_20261009_224152_449800'
OUT = ROOT / 'reports/comparison/yolo_diagnosis_20261010'
OUT.mkdir(parents=True, exist_ok=True)
torch.set_num_threads(4)

def save(name, value):
    (OUT / name).write_text(json.dumps(value, indent=2, allow_nan=False), encoding='utf-8')

def checkpoint(number):
    return RUN / ('checkpoints/initial.pt' if number == 0 else f'checkpoints/global_round_{number:03d}.pt')

def describe(values):
    a = np.asarray(values, dtype=float)
    return dict(count=len(a), maximum=float(a.max()) if len(a) else None,
                quantiles={str(q):float(np.quantile(a,q)) for q in [.5,.9,.99]} if len(a) else {},
                above_025=int((a>=.25).sum()), above_01=int((a>=.1).sum()))

def targets_without_pixels(dataset):
    targets = {}
    for row in dataset.rows:
        w,h = int(row['width']),int(row['height'])
        boxes,labels = [],[]
        for line in (ROOT / row['label']).read_text().splitlines():
            if not line.strip(): continue
            label,x,y,bw,bh = map(float,line.split())
            labels.append(int(label)); boxes.append([(x-bw/2)*w,(y-bh/2)*h,(x+bw/2)*w,(y+bh/2)*h])
        b = torch.tensor(boxes,dtype=torch.float32).reshape(-1,4)
        b[:,[0,2]]=b[:,[0,2]].clamp(0,w); b[:,[1,3]]=b[:,[1,3]].clamp(0,h)
        targets[row['image_id']] = dict(image_id=torch.tensor(row['image_id']),boxes=b,
                                        labels=torch.tensor(labels,dtype=torch.long),original_size=torch.tensor([h,w]))
    return targets

settings=json.loads((RUN/'settings.json').read_text())
initial_saved=torch.load(checkpoint(0),map_location='cpu',weights_only=True)
initial=initial_saved['model_state']
latest=json.loads((RUN/'latest.json').read_text())
report=dict(training_performed=False,optimizer_steps_performed=0,device='cpu',
            versions=dict(torch=torch.__version__,ultralytics=__import__('ultralytics').__version__),
            original_implementation_hash=implementation_hash(),
            recorded_implementation_hash=settings['config']['implementation_sha256'],
            initial_checkpoint_hash_matches=sha256(checkpoint(0))==settings['initial_checkpoint_sha256'],
            final_checkpoint_hash_matches=sha256(checkpoint(10))==latest['checkpoint_sha256'],
            checkpoint_integrity=[], curves=[])
for number in range(11):
    saved=torch.load(checkpoint(number),map_location='cpu',weights_only=True)
    state=saved['model_state']
    expected=(settings['initial_state_sha256'] if number==0 else
              json.loads((RUN/f'logs/client_client1_round_{number+1:03d}.json').read_text())['round_start_sha256'] if number<10 else None)
    digest=state_hash(state)
    report['checkpoint_integrity'].append(dict(round=number,finite=all(torch.isfinite(x).all().item() for x in state.values()),
        metadata_matches=saved['round']==number and saved['config_hash']==settings['config_hash'] and saved['names']==initial_saved['names'],
        state_hash=digest,next_client_start_hash_matches=(digest==expected) if expected else None,
        changed_bn_buffers=[k for k,v in state.items() if any(k.endswith(s) for s in ['running_mean','running_var','num_batches_tracked']) and not torch.equal(v,initial[k])]))
    if number:
        metrics=json.loads((RUN/f'results/metrics/round_{number:03d}.json').read_text())
        report['curves'].append(dict(round=number,**{k:metrics['pooled_validation'][k] for k in ['map50','map_50_95','true_positives','false_positives','false_negatives']},
            client_losses=[c['mean_loss'] for c in metrics['clients']],
            unique_training_images=sum(c['unique_training_images'] for c in metrics['clients']),
            identical_client_start=len(set(c['round_start_sha256'] for c in metrics['clients']))==1))
save('audit.json',report)
print('Checkpoint audit complete',flush=True)

from ultralytics import YOLO
source=YOLO(str(ROOT/'yolov8n.pt')).model.state_dict()
prefix='model.22.cv3.'
classification_keys=[k for k in initial if k.startswith(prefix)]
transfer=[]
for key in classification_keys:
    compatible=key in source and source[key].shape==initial[key].shape
    transfer.append(dict(key=key,coco_shape=list(source[key].shape) if key in source else None,
                         crop_shape=list(initial[key].shape),compatible=compatible,
                         copied_identically=bool(torch.equal(initial[key],source[key])) if compatible else False))
report['classification_transfer']=transfer
report['classification_transfer_summary']=dict(total_state_tensors=len(transfer),
    compatible=sum(x['compatible'] for x in transfer),incompatible=sum(not x['compatible'] for x in transfer),
    coco_hidden_width=int(source[prefix+'0.0.conv.weight'].shape[0]),crop_hidden_width=int(initial[prefix+'0.0.conv.weight'].shape[0]),
    fresh_bn_layers=sum(k.startswith(prefix) and k.endswith('running_var') for k in initial),
    fresh_bn_means_zero=all(torch.count_nonzero(v)==0 for k,v in initial.items() if k.startswith(prefix) and k.endswith('running_mean')),
    fresh_bn_variances_one=all(torch.equal(v,torch.ones_like(v)) for k,v in initial.items() if k.startswith(prefix) and k.endswith('running_var')))

dataset=CropDataset(ROOT,'valid')
truth=targets_without_pixels(dataset)
report['saved_prediction_exports']=[]
for path in sorted((RUN/'results/predictions').glob('*.jsonl')):
    rows=[json.loads(line) for line in path.open(encoding='utf-8')]
    report['saved_prediction_exports'].append(dict(file=path.name,images=len(rows),
        valid_complete_ids=set(r['image_id'] for r in rows)==set(truth),
        confidence=describe([score for r in rows for score in r['scores']])))
best_rows=[json.loads(line) for line in (RUN/'results/predictions/validation_round_008.jsonl').open(encoding='utf-8')]
evaluator=DetectionEvaluator(dataset.names)
for row in best_rows:
    p=dict(boxes=torch.tensor(row['boxes'],dtype=torch.float32).reshape(-1,4),
           labels=torch.tensor(row['labels'],dtype=torch.long),scores=torch.tensor(row['scores'],dtype=torch.float32))
    evaluator.update([truth[row['image_id']]],[p])
report['replayed_best_validation']=evaluator.compute()
expected=json.loads((RUN/'results/metrics/round_008.json').read_text())['pooled_validation']
report['replay_matches_original']=all(abs(report['replayed_best_validation'][k]-expected[k])<1e-9 for k in ['map50','map_50_95','precision','recall'])
all_scores,all_matches=[],[]
for image_id,target in evaluator.targets.items():
    pred=evaluator.predictions[image_id]
    for cls in pred['labels'].unique().tolist():
        mask=pred['labels']==cls
        scores=pred['scores'][mask]; boxes=pred['boxes'][mask]
        overlaps=box_iou(boxes,target['boxes'][target['labels']==cls])
        matched=set()
        for score,overlap in zip(scores.tolist(),overlaps):
            eligible=[(float(v),j) for j,v in enumerate(overlap) if j not in matched and v>=.5]
            if eligible: matched.add(max(eligible)[1])
            all_scores.append(score);all_matches.append(bool(eligible))
a=np.asarray(all_scores);m=np.asarray(all_matches,dtype=bool); gt=sum(len(t['labels']) for t in truth.values())
report['threshold_sweep']=[]
for cutoff in [.25,.1,.05,.01,.005,.001]:
    selected=a>=cutoff; tp=int(m[selected].sum()); count=int(selected.sum())
    precision=tp/count if count else 0.; recall=tp/gt
    report['threshold_sweep'].append(dict(confidence=cutoff,precision=precision,recall=recall,
        true_positives=tp,false_positives=count-tp,false_negatives=gt-tp))
save('audit.json',report)
print('Saved predictions replayed; AP matches:',report['replay_matches_original'],flush=True)

# Stratified fixed validation sample: first image for each foreground class.
chosen=[];seen=set()
for i,row in enumerate(dataset.rows):
    labels=set(truth[row['image_id']]['labels'].tolist())
    if labels-seen: chosen.append(i);seen|=labels
    if len(seen)==24: break
chosen=chosen[:24]
samples=[dataset[i] for i in chosen];images=[x[0] for x in samples];targets=[x[1] for x in samples]
report['inference_sample']=[dict(image_id=int(t['image_id']),key=t['key'],labels=t['labels'].tolist()) for t in targets]
adapter=create_adapter('yolo_v8n','cpu',ROOT,pretrained=False)
report['checkpoint_sample_inference']=[]
for number in [0,1,8,10]:
    adapter.load(torch.load(checkpoint(number),map_location='cpu',weights_only=True)['model_state'])
    before=state_hash(adapter.state());outputs=adapter.predict(images)
    report['checkpoint_sample_inference'].append(dict(round=number,confidence=describe([float(x) for p in outputs for x in p['scores']]),
        inference_changed_state=state_hash(adapter.state())!=before))
    save('audit.json',report)
    print('Read-only sample inference round',number,flush=True)

# Use native Predictor on identical preprocessed tensors, with a separate copied model.
adapter.load(torch.load(checkpoint(8),map_location='cpu',weights_only=True)['model_state'])
adapter.net.names=dataset.names
canonical=adapter.predict(images)
native=YOLO('yolov8n.yaml')
native.model=copy.deepcopy(adapter.net);native.task='detect'
results=native.predict(source=adapter.batch(images),imgsz=640,conf=.001,iou=.55,max_det=100,device='cpu',verbose=False,save=False)
comparisons=[]
for image,target,ours,result in zip(images,targets,canonical,results):
    b=result.boxes.xyxy.cpu().clone();h,w=image.shape[-2:]
    b*=b.new_tensor([w/640,h/640,w/640,h/640]);b[:,[0,2]]=b[:,[0,2]].clamp(0,w);b[:,[1,3]]=b[:,[1,3]].clamp(0,h)
    same_count=len(b)==len(ours['boxes'])
    comparisons.append(dict(image_id=int(target['image_id']),same_count=same_count,
       same_labels=bool(same_count and torch.equal(result.boxes.cls.cpu().long(),ours['labels'])),
       maximum_score_difference=float((result.boxes.conf.cpu()-ours['scores']).abs().max()) if same_count and len(b) else None,
       maximum_box_difference_pixels=float((b-ours['boxes']).abs().max()) if same_count and len(b) else None))
report['native_predictor_comparison']=comparisons
report['original_implementation_hash_after']=implementation_hash()
report['checkpoints_unchanged']=sha256(checkpoint(0))==settings['initial_checkpoint_sha256'] and sha256(checkpoint(10))==latest['checkpoint_sha256']
save('audit.json',report)
print(json.dumps({'output':str(OUT/'audit.json'),'training_performed':False,'replay_matches_original':report['replay_matches_original'],
                 'transfer':report['classification_transfer_summary'],'native_comparison':comparisons}))
