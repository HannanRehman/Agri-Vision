"""Evaluate the validation-selected checkpoint and profile batch-1 inference."""
import argparse
import csv
from datetime import datetime,timezone
import json
from pathlib import Path
import time

import numpy as np
import torch

from .data import CropDataset,load_protocol,sha256
from .models import create_adapter
from .runner import atomic_json,evaluate,implementation_hash,seed_all,synchronize


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,default=Path.cwd())
    parser.add_argument('--run',type=Path,required=True)
    parser.add_argument('--split',choices=['valid','test'],default='test')
    parser.add_argument('--device',default='cuda')
    parser.add_argument('--batch',type=int,default=8)
    parser.add_argument('--workers',type=int,default=2)
    parser.add_argument('--profile-images',type=int,default=30)
    args = parser.parse_args()
    if min(args.batch,args.profile_images)<1 or args.workers<0:
        parser.error('Invalid evaluation settings')
    torch.set_num_threads(4); seed_all(42)
    root,run = args.root.resolve(),args.run.resolve()
    settings = json.loads((run/'settings.json').read_text())
    latest = json.loads((run/'latest.json').read_text())
    protocol = load_protocol(root)
    if protocol['dataset_checksums_sha256'] != settings['config']['dataset_checksums_sha256']:
        raise ValueError('Dataset differs from the training dataset')
    if settings['config']['implementation_sha256'] != implementation_hash():
        raise ValueError('Training/evaluation code differs from the recorded run')
    if settings['diagnostic_only'] and args.split=='test':
        raise ValueError('Pilot runs cannot be used for final held-out test results')
    if latest['round'] != settings['config']['rounds']:
        raise ValueError('Finish the declared round budget before final evaluation')
    checkpoint = run/latest['best_checkpoint']
    saved = torch.load(checkpoint,map_location='cpu',weights_only=True)
    if saved['config_hash'] != settings['config_hash']:
        raise ValueError('Checkpoint config mismatch')
    adapter = create_adapter(settings['config']['model'],args.device,root,pretrained=False)
    adapter.load(saved['model_state'])
    dataset = CropDataset(root,args.split)
    destination = run/'results/evaluations'/f'best_{args.split}.json'
    if destination.exists():
        raise FileExistsError('Final evaluation already exists; preserve it rather than overwrite')
    destination.parent.mkdir(parents=True,exist_ok=True)
    prediction_file = run/'results/predictions'/f'best_{args.split}.jsonl'
    metrics = evaluate(adapter,dataset,args,prediction_file)
    # Use validation images for runtime measurement even when scoring the test split.
    profiling = CropDataset(root,'valid',limit=args.profile_images,seed=42)
    tensors = [profiling[i][0] for i in range(len(profiling))]
    for _ in range(5):
        adapter.predict([tensors[0]])
    if adapter.device.type=='cuda':
        torch.cuda.reset_peak_memory_stats(adapter.device)
    milliseconds = []
    for image in tensors:
        synchronize(adapter.device); started = time.perf_counter()
        adapter.predict([image]); synchronize(adapter.device)
        milliseconds.append((time.perf_counter()-started)*1000)
    hardware = torch.cuda.get_device_name(adapter.device) if adapter.device.type=='cuda' else 'CPU'
    output = dict(created_at_utc=datetime.now(timezone.utc).isoformat(),split=args.split,
                  selected_round=saved['round'],selected_by='highest pooled validation mAP50-95 within declared budget',
                  checkpoint=checkpoint.name,checkpoint_sha256=sha256(checkpoint),metrics=metrics,
                  diagnostic_only=settings['diagnostic_only'],hardware=hardware,
                  runtime=dict(batch_size=1,warmup_iterations=5,measured_images=len(tensors),
                               mean_ms=float(np.mean(milliseconds)),median_ms=float(np.median(milliseconds)),
                               p95_ms=float(np.percentile(milliseconds,95)),
                               scope='CPU-resident decoded tensors to predictions, including device transfer, adapter preprocessing and NMS; disk loading excluded',
                               peak_cuda_allocated_bytes=torch.cuda.max_memory_allocated(adapter.device) if adapter.device.type=='cuda' else None),
                  model=dict(parameters=settings['parameter_count'],input_size=settings['input_size'],
                             checkpoint_bytes=checkpoint.stat().st_size,
                             tensor_state_bytes=sum(t.numel()*t.element_size() for t in saved['model_state'].values())))
    atomic_json(destination,output)
    per_class = run/'results/metrics'/f'best_{args.split}_per_class.csv'
    with per_class.open('w',newline='',encoding='utf-8') as stream:
        writer = csv.DictWriter(stream,fieldnames=list(metrics['per_class'][0])); writer.writeheader(); writer.writerows(metrics['per_class'])
    print(json.dumps({k:v for k,v in output.items() if k!='metrics'},indent=2),flush=True)
    print(f'Saved full pooled metrics and per-class CSV to {destination}',flush=True)


if __name__=='__main__':
    main()
