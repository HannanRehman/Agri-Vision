"""Resumable grid trial preserving architecture-specific baseline training schedules."""
import argparse
from copy import deepcopy
from datetime import datetime, timezone
import gc
import hashlib
from importlib.metadata import version
import json
import math
from pathlib import Path
import shutil
import tempfile
import time

import torch
from comparison import runner as r
from comparison.data import CropDataset, load_protocol, sha256
from scripts import yolo_native_train as y
from tuning.models import create_adapter, NativePredictor


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def implementation_hash():
    files = [*Path(__file__).parent.glob('*.py'), Path(y.__file__),
             Path(y.__file__).with_name('continue_comparison.py')]
    return digest(dict(common=r.implementation_hash(), files={p.name: sha256(p) for p in sorted(files)}))


def hardware(device):
    return dict(gpu=torch.cuda.get_device_name(device) if device.type == 'cuda' else 'CPU',
                torch=str(torch.__version__), packages={name: version(name) for name in
                ('torchvision', 'ultralytics', 'pycocotools', 'effdet', 'timm', 'numpy')})


def learning_rate(config, number):
    if config.get('model') == 'yolo_v8n':
        return config['lr']  # Native baseline restarts the same local schedule per fit.
    horizon = config.get('lr_horizon_rounds', 10)
    index = min(number, horizon)
    return config['lr']*(.1+.9*(1+math.cos(math.pi*(index-1)/(horizon-1)))/2)


def initialization(root, model_folder, config, device):
    """One common freshly transferred initialization for all configurations of a model."""
    path = model_folder/'initial.pt'
    names = CropDataset(root, 'valid').names
    fingerprint = digest(dict(model=config['model'], seed=config['seed'], names=names,
                              implementation=config['implementation_hash'], dataset=config['dataset_hash']))
    model_folder.mkdir(parents=True, exist_ok=True)
    if path.exists():
        saved = torch.load(path, map_location='cpu', weights_only=True)
        if saved['fingerprint'] != fingerprint or saved['names'] != names:
            raise ValueError('Common initialization/code/data changed')
    else:
        r.seed_all(config['seed'])
        if config['model'] == 'yolo_v8n':
            model = y.create_yolo(root, names)
            state = {k:v.detach().cpu().clone() for k,v in model.model.state_dict().items()}
            del model
        else:
            adapter = create_adapter(config['model'], device, root, config['resolution'], True)
            state = adapter.state()
            del adapter
        saved = dict(model_state=state, names=names, round=0, fingerprint=fingerprint)
        r.atomic_torch(path, saved)
    if not all(torch.isfinite(t).all() for t in saved['model_state'].values() if t.is_floating_point()):
        raise ValueError('Non-finite common initialization')
    return path, saved


def native_options(config, number, workers):
    # Native local warmup/augmentation/normalization and effective-batch policy remain
    # as in the native baseline. Explicit AdamW enables the searched lr0 values.
    return dict(optimizer='AdamW', lr0=config['lr'],
                weight_decay=config['weight_decay'], batch=config['batch'],
                imgsz=config['resolution'], workers=workers,
                momentum=.9, warmup_bias_lr=0.0)


def native_step(root, output, config, state, names, protocol, number, device, workers, status):
    clients, updates = [], []
    start_hash = r.state_hash(y.floating_state(state))
    with tempfile.TemporaryDirectory(prefix='agrivision_grid_yolo_') as temporary:
        work = Path(temporary)
        for index, client in enumerate(protocol['clients']):
            status(phase='training', round=number, client=client)
            r.seed_all(config['seed'])
            model = y.create_yolo(root, names)
            current = model.model.state_dict()
            model.model.load_state_dict({k:state[k] if v.is_floating_point() else v for k,v in current.items()}, strict=True)
            train = CropDataset(root, 'train', client)
            valid = CropDataset(root, 'valid', client)
            data = y.runtime_yaml(root, train, work/client/'data', valid)
            trained, local = y.native_train(model, data, work/client/'fit', config['seed'],
                                             native_options(config, number, workers))
            if type(model.trainer.optimizer).__name__ != 'AdamW':
                raise ValueError('YOLO optimizer differs from requested AdamW')
            if model.trainer.args.imgsz != config['resolution'] or model.trainer.args.batch != config['batch']:
                raise ValueError('YOLO ignored the searched size/batch')
            if not math.isclose(model.trainer.args.lr0, learning_rate(config, number)):
                raise ValueError('YOLO ignored the searched learning rate')
            record = dict(client=client, unique_training_images=len(train), round_start_sha256=start_hash,
                          requested_lr0=learning_rate(config, number), optimizer='AdamW',
                          final_learning_rates=[float(g['lr']) for g in model.trainer.optimizer.param_groups],
                          optimizer_weight_decays=[float(g['weight_decay']) for g in model.trainer.optimizer.param_groups],
                          native_transfer_verified=model.native_transfer_verified,
                          gradient_accumulation=int(model.trainer.accumulate), nominal_batch_size=y.NATIVE_ARGS['nbs'])
            log = output/'logs'/f'{client}_round_{number:03d}'
            log.mkdir(exist_ok=True, parents=True)
            for name in ('args.yaml', 'results.csv'):
                shutil.copy2(local/name, log/name)
            clients.append(record)
            updates.append((trained, len(train)))
            del model
            gc.collect()
            if device.type == 'cuda':
                torch.cuda.empty_cache()
    return y.historical_average(updates, state), clients


def standard_step(adapter, root, config, state, protocol, number, workers, status):
    # Keep actual-round randomness while reproducing the baseline's original
    # 10-round cosine followed by a constant terminal rate through round 30.
    factor = .1+.9*(1+math.cos(math.pi*(number-1)/(config['schedule_rounds']-1)))/2
    args = argparse.Namespace(model=config['model'], optimizer=config['optimizer'], lr=learning_rate(config, number)/factor,
                              weight_decay=config['weight_decay'], batch=config['batch'],
                              rounds=config['schedule_rounds'], local_epochs=1, seed=config['seed'],
                              workers=workers, max_train_batches=None)
    updates, clients = [], []
    start_hash = r.state_hash(state)
    for index, client in enumerate(protocol['clients']):
        adapter.load(state)
        if r.state_hash(adapter.state()) != start_hash:
            raise ValueError('Client did not receive the intended global state')
        dataset = CropDataset(root, 'train', client, augment=True)
        def progress(**values):
            status(phase='training', round=number, client=client, **values)
        result = r.local_train(adapter, dataset, args, number, index, progress)
        result.update(client=client, round_start_sha256=start_hash)
        updates.append((adapter.state(), result['unique_training_images']))
        clients.append(result)
    return r.fedavg(updates), clients


def prune_checkpoints(output, latest, retain_best):
    keep = {Path(latest['checkpoint']).name}
    if retain_best and latest.get('best_checkpoint'):
        keep.add(Path(latest['best_checkpoint']).name)
    folder = output/'checkpoints'
    assert folder.resolve().is_relative_to(output.resolve())
    for path in folder.glob('global_round_*.pt'):
        if path.name not in keep:
            path.unlink()


def run_trial(root, output, model_folder, config, stop_round, device, workers=2, retain_best=False):
    if not 1 <= stop_round <= config['schedule_rounds']:
        raise ValueError('Invalid trial budget')
    if config['implementation_hash'] != implementation_hash():
        raise ValueError('Tuning implementation changed')
    protocol = load_protocol(root)
    if config['dataset_hash'] != protocol['dataset_checksums_sha256'] or len(protocol['clients']) != 3:
        raise ValueError('Dataset/client protocol differs')
    initial_path, initial = initialization(root, model_folder, config, device)
    config_hash, current_hardware = digest(config), hardware(device)
    for directory in ('checkpoints', 'logs', 'results/metrics'):
        (output/directory).mkdir(parents=True, exist_ok=True)
    settings_path = output/'settings.json'
    settings = dict(config=config, config_hash=config_hash, hardware=current_hardware,
                    initial_checkpoint_sha256=sha256(initial_path), initial_state_sha256=r.state_hash(initial['model_state']),
                    initial_checkpoint=str(initial_path), retain_best_checkpoint=retain_best,
                    loader_workers=workers, diagnostic_only=False, split='full frozen pooled validation')
    if settings_path.exists():
        if json.loads(settings_path.read_text()) != settings:
            raise ValueError('Trial configuration, initialization or environment changed')
    else:
        r.atomic_json(settings_path, settings)
    latest_path = output/'latest.json'
    latest = json.loads(latest_path.read_text()) if latest_path.exists() else dict(
        round=0, checkpoint=None, best_map=-1., best_round=0, best_checkpoint=None)
    if latest['round'] > stop_round:
        raise ValueError('Requested stop is before a committed round')
    state = initial['model_state']
    if latest['round']:
        path = (output/latest['checkpoint']).resolve()
        if not path.is_relative_to(output.resolve()) or sha256(path) != latest['checkpoint_sha256']:
            raise ValueError('Trial checkpoint changed')
        saved = torch.load(path, map_location='cpu', weights_only=True)
        if saved['round'] != latest['round'] or saved['config_hash'] != config_hash or saved['names'] != initial['names']:
            raise ValueError('Trial checkpoint metadata differs')
        state = saved['model_state']
    for metric in (output/'results/metrics').glob('round_*.json'):
        if int(metric.stem.split('_')[-1]) > latest['round']:
            metric.unlink()
    r.rewrite_metrics(output)
    def status(**values):
        r.atomic_json(output/'status.json', dict(updated_at_utc=datetime.now(timezone.utc).isoformat(), **values))
    if latest['round'] == stop_round:
        status(phase='budget_complete', completed_rounds=latest['round'], stop_round=stop_round)
        return latest
    adapter = None
    if config['model'] != 'yolo_v8n':
        adapter = create_adapter(config['model'], device, root, config['resolution'], False)
        adapter.load(state)
    valid = CropDataset(root, 'valid')
    try:
        for number in range(latest['round']+1, stop_round+1):
            started = time.perf_counter()
            if config['model'] == 'yolo_v8n':
                state, clients = native_step(root, output, config, state, initial['names'], protocol,
                                             number, device, workers, status)
                model = y.create_yolo(root, initial['names'])
                model.model.load_state_dict(state, strict=True)
                predictor = NativePredictor(model, device, config['resolution'])
            else:
                state, clients = standard_step(adapter, root, config, state, protocol, number, workers, status)
                adapter.load(state)
                predictor = adapter
            status(phase='pooled_validation', round=number)
            metrics = r.evaluate(predictor, valid, argparse.Namespace(batch=min(4, config['batch']), workers=workers))
            if config['model'] == 'yolo_v8n':
                del predictor, model
            checkpoint = output/f'checkpoints/global_round_{number:03d}.pt'
            r.atomic_torch(checkpoint, dict(model_state=state, round=number, config_hash=config_hash, names=initial['names']))
            best_map, best_round = latest['best_map'], latest['best_round']
            best_checkpoint = latest.get('best_checkpoint')
            if metrics['map_50_95'] > best_map:
                best_map, best_round = metrics['map_50_95'], number
                best_checkpoint = checkpoint.relative_to(output).as_posix() if retain_best else None
            r.atomic_json(output/f'results/metrics/round_{number:03d}.json',
                          dict(round=number, round_seconds=time.perf_counter()-started,
                               pooled_validation=metrics, clients=clients, diagnostic_only=False,
                               validation_scope='full frozen pooled validation', searched_config=config))
            latest = dict(round=number, checkpoint=checkpoint.relative_to(output).as_posix(),
                          checkpoint_sha256=sha256(checkpoint), best_map=best_map,
                          best_round=best_round, best_checkpoint=best_checkpoint)
            r.atomic_json(latest_path, latest)
            r.rewrite_metrics(output)
            prune_checkpoints(output, latest, retain_best)
            status(phase='round_committed', completed_rounds=number, stop_round=stop_round)
            print(f"COMMITTED {config['model']} round {number}/{stop_round}: mAP50-95={metrics['map_50_95']:.5f}", flush=True)
            gc.collect()
            if device.type == 'cuda':
                torch.cuda.empty_cache()
        status(phase='budget_complete', completed_rounds=latest['round'], stop_round=stop_round)
        return latest
    except BaseException as error:
        status(phase='interrupted_or_failed', error=f'{type(error).__name__}: {error}',
               completed_rounds=latest['round'], recovery='Rerun identical notebook; incomplete round repeats')
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--model-folder', type=Path, required=True)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--stop-round', type=int, required=True)
    parser.add_argument('--workers', type=int, default=2)
    parser.add_argument('--retain-best', action='store_true')
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError('A GPU runtime is required')
    torch.set_num_threads(4)
    run_trial(args.root.resolve(), args.output.resolve(), args.model_folder.resolve(),
              json.loads(args.config.read_text()), args.stop_round, torch.device('cuda:0'),
              args.workers, args.retain_best)


if __name__ == '__main__':
    main()
