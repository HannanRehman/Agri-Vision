"""Benchmarked FP32 execution; original data, schedule, optimizer and FedAvg retained."""
import argparse
import json
import math
from pathlib import Path

import torch

from benchmarks.efficientdet_fp32 import step
from comparison import runner as r
from comparison.data import sha256
from tuning.runner import digest, hardware, implementation_hash

BASE_HASH = '8d2e2a0363d7e5d6d0410a8bafda25f56e6ed6ae7505440fe3ec79a230d263f1'


def identity():
    root = Path(__file__).resolve().parents[1]
    files = [*Path(__file__).parent.glob('*.py'), root/'benchmarks/efficientdet_fp32.py',
             root/'benchmarks/__init__.py']
    return {p.relative_to(root).as_posix(): sha256(p) for p in sorted(files)}


def validate_evidence(report):
    if (report.get('phase') != 'complete' or report.get('precision') != 'FP32'
            or not report.get('training_search_unchanged_verified')):
        raise ValueError('Completed FP32 benchmark evidence required')
    if {(x['resolution'], x['batch']) for x in report['cases']} != {(512, 2), (512, 4), (640, 2), (640, 4)}:
        raise ValueError('Incomplete benchmark coverage')
    for case in report['cases']:
        if len(case['runs']) != 3 or case['median_speedup'] <= 1:
            raise ValueError('Benchmark speed evidence differs')
        if not all(x['equivalence']['passed'] and x['equivalence']['bitwise_equal']
                   and x['equivalence']['losses_match'] for x in case['runs']):
            raise ValueError('Benchmark equivalence failed')


def profile(folder, root, device):
    if implementation_hash() != BASE_HASH:
        raise ValueError('Original search implementation changed')
    evidence = root/'continuation/benchmark_evidence.json'
    report = json.loads(evidence.read_text())
    validate_evidence(report)
    current_hardware = hardware(device)
    if current_hardware != report['hardware']:
        raise ValueError('Benchmark and continuation environments differ')
    # Existing hashes describe the unmodified original source. This extension has
    # its own strict fingerprint; neither old configs nor checkpoint hashes change.
    value = dict(version='efficientdet_fp32_overhead_v1', base_implementation_hash=BASE_HASH,
                 extension_files=identity(), benchmark_sha256=sha256(evidence), hardware=current_hardware,
                 changes=['consolidated gradient guard via unchanged norm clipping',
                          'GPU loss components until logging', 'GPU float64 loss accumulation'],
                 precision='FP32', preserved_screening_trials=list(range(1, 6)),
                 initial_continuation_trial=6,
                 note='Earlier screening rounds retain original execution. All new rounds, including '
                      'finalist extensions, record this execution profile. No claim of universal bitwise equivalence.')
    path = folder/'execution_profile.json'
    if path.exists():
        if json.loads(path.read_text()) != value:
            raise ValueError('Execution extension, evidence or environment changed')
    else:
        # Establish the transition only when its declared five completed trials exist.
        for number in range(1, 6):
            trial = folder/'trials'/f'trial_{number:02d}'
            latest = json.loads((trial/'latest.json').read_text())
            settings = json.loads((trial/'settings.json').read_text())
            if latest['round'] < 5 or settings['config']['implementation_hash'] != BASE_HASH:
                raise ValueError('Five completed original screening trials required')
            checkpoint = (trial/latest['checkpoint']).resolve()
            if not checkpoint.is_relative_to(trial.resolve()) or sha256(checkpoint) != latest['checkpoint_sha256']:
                raise ValueError('Saved trial checkpoint differs')
        r.atomic_json(path, value)
    return value


def local_train(adapter, dataset, args, round_number, client_index, status):
    if adapter.model_id != 'efficientdet_d0' or args.optimizer != 'adamw':
        raise ValueError('This extension is only for EfficientDet AdamW')
    r.seed_all(args.seed+round_number*100+client_index)
    lr = args.lr*(.1+.9*(1+math.cos(math.pi*(round_number-1)/max(args.rounds-1, 1)))/2)
    optimizer = torch.optim.AdamW([p for p in adapter.net.parameters() if p.requires_grad],
                                 lr=lr, weight_decay=args.weight_decay)
    seen, steps, used_ids = 0, 0, set()
    total = torch.zeros((), dtype=torch.float64, device=adapter.device)
    r.synchronize(adapter.device)
    import time
    started = time.perf_counter()
    for epoch in range(args.local_epochs):
        data = r.loader(dataset, args.batch, args.workers, True,
                        args.seed+round_number*100+client_index*10+epoch)
        for index, (images, targets) in enumerate(data):
            if args.max_train_batches is not None and index >= args.max_train_batches:
                break
            loss, components = step(adapter, optimizer, images, targets, optimized=True)
            total += loss.double()*len(images)
            seen += len(images)
            steps += 1
            used_ids.update(int(t['image_id']) for t in targets)
            if index % 25 == 0:
                status(epoch=epoch+1, batch=index+1, loss=float(loss),
                       components={k: float(v) for k, v in components.items()})
                print(f'  epoch {epoch+1} batch {index+1}: loss={float(loss):.4f}', flush=True)
    r.synchronize(adapter.device)
    if not steps:
        raise ValueError('No optimizer steps performed')
    return dict(unique_training_images=len(used_ids), image_presentations=seen, optimizer_steps=steps,
                mean_loss=float(total)/seen, learning_rate=lr, seconds=time.perf_counter()-started)
