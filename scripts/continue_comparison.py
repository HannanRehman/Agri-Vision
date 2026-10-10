"""Audited continuation of existing non-YOLO runs; original folders remain untouched."""
import argparse
from copy import deepcopy
from datetime import datetime, timezone
import gc
import hashlib
from importlib.metadata import version
import json
from pathlib import Path
import shutil
import time

import torch
from comparison import runner as r
from comparison.data import CropDataset, load_protocol, sha256


def canonical_hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def extension_rate(base_lr, original_rounds, round_number):
    """Original cosine through its last round, then constant at its terminal LR."""
    import math
    if round_number < 1 or original_rounds < 2:
        raise ValueError('Invalid schedule round')
    index = min(round_number, original_rounds)
    return base_lr * (.1 + .9 * (1 + math.cos(math.pi*(index-1)/(original_rounds-1))) / 2)


def validate_source(root, source, device):
    settings = json.loads((source/'settings.json').read_text())
    config = settings['config']
    protocol = load_protocol(root)
    if settings['config_hash'] != canonical_hash(config):
        raise ValueError('Source settings hash differs')
    if config['model'] not in ('ssdlite', 'ssd_mobilenetv3', 'efficientdet_d0'):
        raise ValueError('Only the three existing alternative detectors can continue here')
    if settings['diagnostic_only'] or config['mode'] != 'main':
        raise ValueError('Cannot continue a diagnostic pilot as a main run')
    if config['dataset_checksums_sha256'] != protocol['dataset_checksums_sha256']:
        raise ValueError('Source dataset differs')
    if config['implementation_sha256'] != r.implementation_hash():
        raise ValueError('Original comparison implementation changed')
    if config['model_spec_sha256'] != sha256(root/f"configs/comparison/{config['model']}.json"):
        raise ValueError('Model specification changed')
    hardware = settings['hardware']
    current_gpu = torch.cuda.get_device_name(device) if device.type == 'cuda' else 'CPU'
    if hardware['gpu'] != current_gpu or hardware['torch'] != torch.__version__:
        raise ValueError('Resume GPU/torch differs from source; use the recorded environment')
    for name, expected in hardware['packages'].items():
        if version(name) != expected:
            raise ValueError(f'Resume package differs: {name}')
    if sha256(source/'checkpoints/initial.pt') != settings['initial_checkpoint_sha256']:
        raise ValueError('Source initial checkpoint changed')
    latest = json.loads((source/'latest.json').read_text())
    checkpoint = safe_checkpoint(source, latest['checkpoint'])
    if sha256(checkpoint) != latest['checkpoint_sha256']:
        raise ValueError('Source latest checkpoint changed')
    saved = torch.load(checkpoint, map_location='cpu', weights_only=True)
    names = CropDataset(root, 'valid').names
    if saved['config_hash'] != settings['config_hash'] or saved['names'] != names or saved['round'] != latest['round']:
        raise ValueError('Source checkpoint metadata differs')
    if not 0 < latest['round'] <= config['rounds']:
        raise ValueError('Invalid committed source round')
    if any(v.is_floating_point() and not torch.isfinite(v).all() for v in saved['model_state'].values()):
        raise ValueError('Non-finite source state')
    return settings, latest, protocol


def safe_checkpoint(folder, relative):
    path = (folder/relative).resolve()
    if not path.is_relative_to(folder.resolve()) or path.suffix != '.pt':
        raise ValueError('Invalid checkpoint path')
    return path


def prepare_branch(root, source, output, target, device):
    original, source_latest, protocol = validate_source(root, source, device)
    if target <= original['config']['rounds']:
        raise ValueError('Continuation target must exceed the original budget')
    if output == source or output.is_relative_to(source):
        raise ValueError('Use a separate continuation folder')
    config = deepcopy(original['config'])
    config.update(rounds=target, learning_rate_schedule='original cosine through original budget; constant terminal LR thereafter',
                  continuation=dict(original_rounds=original['config']['rounds'],
                                    source_config_hash=original['config_hash'],
                                    source_commit=source_latest,
                                    source_settings_sha256=sha256(source/'settings.json'),
                                    continuation_script_sha256=sha256(Path(__file__))))
    config_hash = canonical_hash(config)
    if output.exists():
        settings = json.loads((output/'settings.json').read_text())
        if settings['config_hash'] != config_hash or settings['config'] != config:
            raise ValueError('Continuation settings or source lineage changed')
        return settings, protocol
    output.mkdir(parents=True)
    for directory in ['checkpoints', 'logs', 'results/metrics', 'results/predictions', 'results/evaluations']:
        (output/directory).mkdir(parents=True)
    # Re-encode metadata under the continuation config while preserving exact tensors.
    paths = {'checkpoints/initial.pt', source_latest['checkpoint'], source_latest['best_checkpoint']}
    for relative in sorted(paths):
        source_path = safe_checkpoint(source, relative)
        state = torch.load(source_path, map_location='cpu', weights_only=True)
        if state['config_hash'] != original['config_hash'] or state['round'] > source_latest['round']:
            raise ValueError('Source best/initial checkpoint metadata differs')
        state['config_hash'] = config_hash
        state['lineage'] = dict(source_checkpoint_sha256=sha256(source_path), source_config_hash=original['config_hash'])
        r.atomic_torch(output/relative, state)
    for path in (source/'results/metrics').glob('round_*.json'):
        if int(path.stem.split('_')[-1]) <= source_latest['round']:
            shutil.copy2(path, output/'results/metrics'/path.name)
    settings = deepcopy(original)
    settings.update(config=config, config_hash=config_hash, source_run=str(source),
                    created_at_utc=datetime.now(timezone.utc).isoformat(),
                    initial_checkpoint_sha256=sha256(output/'checkpoints/initial.pt'))
    r.atomic_json(output/'settings.json', settings)
    latest = dict(source_latest, checkpoint_sha256=sha256(output/source_latest['checkpoint']))
    r.atomic_json(output/'latest.json', latest)
    r.rewrite_metrics(output)
    return settings, protocol


def prune_branch(output, latest):
    # Only weights produced in this new branch; originals and metrics are preserved.
    keep = {'initial.pt', Path(latest['checkpoint']).name, Path(latest['best_checkpoint']).name}
    for path in (output/'checkpoints').glob('global_round_*.pt'):
        if path.name not in keep:
            path.unlink()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=Path.cwd())
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--rounds', type=int, default=30)
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--workers', type=int, default=2)
    args = parser.parse_args(argv)
    device = torch.device(args.device)
    if device.type != 'cuda' or not torch.cuda.is_available():
        raise RuntimeError('Continuation main training requires a GPU')
    if args.workers < 0:
        raise ValueError('Invalid worker count')
    torch.set_num_threads(4)
    root, source, output = args.root.resolve(), args.source.resolve(), args.output.resolve()
    settings, protocol = prepare_branch(root, source, output, args.rounds, device)
    config = settings['config']; r.seed_all(config['seed'])
    train_args = argparse.Namespace(**{k:config[k] for k in ['model','mode','rounds','local_epochs','batch','seed','lr','optimizer','weight_decay','max_train_batches','validation_limit']}, workers=args.workers)
    dataset = CropDataset(root, 'valid')
    latest = json.loads((output/'latest.json').read_text())
    path = safe_checkpoint(output, latest['checkpoint'])
    if sha256(path) != latest['checkpoint_sha256']:
        raise ValueError('Continuation checkpoint changed')
    saved = torch.load(path, map_location='cpu', weights_only=True)
    if saved['config_hash'] != settings['config_hash'] or saved['round'] != latest['round'] or saved['names'] != dataset.names:
        raise ValueError('Continuation checkpoint metadata differs')
    adapter = r.create_adapter(config['model'], device, root, pretrained=False)
    adapter.net.names = dataset.names
    global_state = saved['model_state']
    for metric in (output/'results/metrics').glob('round_*.json'):
        if int(metric.stem.split('_')[-1]) > latest['round']:
            metric.unlink()
    r.rewrite_metrics(output)
    def status(**values):
        r.atomic_json(output/'status.json', dict(updated_at_utc=datetime.now(timezone.utc).isoformat(), **values))
    original_rounds = config['continuation']['original_rounds']
    try:
        for number in range(latest['round']+1, args.rounds+1):
            started = time.perf_counter(); start_hash = r.state_hash(global_state)
            updates, clients = [], []
            local_args = deepcopy(train_args)
            for index, client in enumerate(protocol['clients']):
                adapter.load(global_state)
                if r.state_hash(adapter.state()) != start_hash:
                    raise ValueError('Client initialization differs')
                train = CropDataset(root, 'train', client, augment=True)
                def progress(**values):
                    status(phase='training', round=number, client=client, **values)
                # Keep actual-round randomness while independently clamping the LR schedule.
                # local_train normally couples these; use an equivalent scaled base LR.
                local_args.rounds = args.rounds
                import math
                factor = .1+.9*(1+math.cos(math.pi*(number-1)/max(args.rounds-1,1)))/2
                local_args.lr = extension_rate(config['lr'], original_rounds, number)/factor
                result = r.local_train(adapter, train, local_args, number, index, progress)
                result.update(client=client, round_start_sha256=start_hash)
                updates.append((adapter.state(), result['unique_training_images'])); clients.append(result)
                r.atomic_json(output/f'logs/client_{client}_round_{number:03d}.json', result)
            global_state = r.fedavg(updates); del updates
            adapter.load(global_state); status(phase='pooled_validation', round=number)
            # Raw all-round exports consumed substantial Drive space; keep metrics instead.
            metrics = r.evaluate(adapter, dataset, train_args)
            round_path = output/f'checkpoints/global_round_{number:03d}.pt'
            r.atomic_torch(round_path, dict(model_state=global_state, round=number, config_hash=settings['config_hash'], names=dataset.names))
            best_map, best_path = latest['best_map'], latest['best_checkpoint']
            if metrics['map_50_95'] > best_map:
                best_map, best_path = metrics['map_50_95'], round_path.relative_to(output).as_posix()
            r.atomic_json(output/f'results/metrics/round_{number:03d}.json', dict(round=number, round_seconds=time.perf_counter()-started,
                          pooled_validation=metrics, clients=clients, diagnostic_only=False,
                          validation_scope='full frozen pooled validation', continuation_schedule=config['learning_rate_schedule']))
            latest = dict(round=number, checkpoint=round_path.relative_to(output).as_posix(), checkpoint_sha256=sha256(round_path), best_map=best_map, best_checkpoint=best_path)
            r.atomic_json(output/'latest.json', latest); r.rewrite_metrics(output); prune_branch(output, latest)
            print(f"ROUND {number}/{args.rounds} committed: AP95={metrics['map_50_95']:.4f}", flush=True)
            gc.collect()
        status(phase='complete', completed_rounds=latest['round'], best_checkpoint=latest['best_checkpoint'])
    except BaseException as error:
        status(phase='interrupted_or_failed', error=f'{type(error).__name__}: {error}', recovery='Rerun identical command; repeat incomplete round')
        raise


if __name__ == '__main__':
    main()
