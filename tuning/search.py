"""Exhaustive four-parameter screening, finalist confirmation and fresh final training."""
import argparse
import csv
import itertools
import json
from pathlib import Path
import subprocess
import sys

import torch
from comparison import runner as r
from comparison.data import load_protocol, sha256
from tuning.runner import digest, hardware, implementation_hash

DIMENSIONS = ('lr', 'weight_decay', 'batch', 'resolution')


def configurations(grid, model, dataset_hash, code_hash):
    spec = grid['models'][model]
    values = [spec[key] for key in DIMENSIONS]
    if any(len(v) != 2 or len(set(v)) != 2 for v in values):
        raise ValueError('Each of the four dimensions must have two distinct values')
    for index, combination in enumerate(itertools.product(*values), 1):
        config = dict(model=model, **dict(zip(DIMENSIONS, combination)), optimizer=spec['optimizer'],
                      schedule_rounds=grid['schedule_rounds'], seed=grid['seed'], local_epochs=1,
                      dataset_hash=dataset_hash, implementation_hash=code_hash,
                      lr_horizon_rounds=10 if model != 'yolo_v8n' else None,
                      schedule='original 10-round cosine then terminal rate through 30' if model != 'yolo_v8n'
                               else 'constant searched lr0; original native warmup resets each local fit',
                      normalization='native adaptive BN' if model == 'yolo_v8n' else 'frozen running BN; train affine',
                      optimizer_state='reset per client per round',
                      aggregation='historical NumPy floating FedAvg' if model == 'yolo_v8n' else 'image-weighted FedAvg',
                      native_warmup='original per-fit 3 epochs; warmup bias LR 0 as resolved by baseline AdamW' if model == 'yolo_v8n' else None)
        yield f'trial_{index:02d}', config


def metric_at_budget(folder, budget):
    latest_file = folder/'latest.json'
    if not latest_file.exists():
        return None
    latest = json.loads(latest_file.read_text())
    records = [json.loads(path.read_text()) for path in sorted((folder/'results/metrics').glob('round_*.json'))
               if int(path.stem.split('_')[-1]) <= min(budget, latest['round'])]
    if latest['round'] < budget or len(records) != budget:
        return None
    return max(records, key=lambda record: record['pooled_validation']['map_50_95'])


def rank_trials(model_folder, trials, budget):
    ranked = []
    for trial_id, config in trials:
        record = metric_at_budget(model_folder/'trials'/trial_id, budget)
        if record is not None:
            ranked.append(dict(trial_id=trial_id, config=config, best_round=record['round'],
                               metrics=record['pooled_validation']))
    return sorted(ranked, key=lambda row: (-row['metrics']['map_50_95'], row['trial_id']))


def write_ranking(path, ranked, budget):
    fields = ['rank', 'trial_id', 'budget', *DIMENSIONS, 'best_round',
              'precision', 'recall', 'f1', 'map50', 'map_50_95']
    temp = path.with_suffix('.tmp')
    with temp.open('w', encoding='utf-8', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for rank, row in enumerate(ranked, 1):
            writer.writerow(dict(rank=rank, trial_id=row['trial_id'], budget=budget,
                                 **{key:row['config'][key] for key in DIMENSIONS}, best_round=row['best_round'],
                                 **{key:row['metrics'][key] for key in fields[-5:]}))
    temp.replace(path)


def ensure_trial(root, model_folder, trial_id, config, stop, workers, final=False):
    output = model_folder/'final_training' if final else model_folder/'trials'/trial_id
    latest_file = output/'latest.json'
    if latest_file.exists() and json.loads(latest_file.read_text())['round'] >= stop:
        # Still validate the config and initialization identity before skipping.
        settings = json.loads((output/'settings.json').read_text())
        if (settings['config'] != config or settings['hardware'] != hardware(torch.device('cuda:0'))
                or settings['loader_workers'] != workers):
            raise ValueError('Completed trial config/environment differs')
        if sha256(settings['initial_checkpoint']) != settings['initial_checkpoint_sha256']:
            raise ValueError('Common initialization changed')
        latest = json.loads(latest_file.read_text())
        checkpoint = (output/latest['checkpoint']).resolve()
        if not checkpoint.is_relative_to(output.resolve()) or sha256(checkpoint) != latest['checkpoint_sha256']:
            raise ValueError('Completed checkpoint changed')
        return
    configs = model_folder/'configs'
    configs.mkdir(exist_ok=True)
    config_file = configs/('selected_final.json' if final else f'{trial_id}.json')
    if config_file.exists() and json.loads(config_file.read_text()) != config:
        raise ValueError('Saved grid configuration changed')
    r.atomic_json(config_file, config)
    command = [sys.executable, '-m', 'tuning.runner', '--root', str(root), '--output', str(output),
               '--model-folder', str(model_folder), '--config', str(config_file),
               '--stop-round', str(stop), '--workers', str(workers)]
    if final:
        command.append('--retain-best')
    subprocess.run(command, check=True)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--grid', type=Path)
    parser.add_argument('--models', nargs='+', required=True)
    parser.add_argument('--workers', type=int, default=2)
    args = parser.parse_args(argv)
    if not torch.cuda.is_available():
        raise RuntimeError('Select a GPU runtime before running the search')
    device = torch.device('cuda:0')
    if 'T4' not in torch.cuda.get_device_name(device):
        raise RuntimeError('Use T4 for comparison with the recorded experiments')
    root, output = args.root.resolve(), args.output.resolve()
    grid_path = args.grid or root/'configs/tuning/grid_v1.json'
    grid = json.loads(grid_path.read_text())
    protocol = load_protocol(root)
    if grid['local_epochs'] != 1 or not 1 <= grid['screen_rounds'] < grid['finalist_rounds'] < grid['schedule_rounds']:
        raise ValueError('Invalid search budgets')
    if len(set(args.models)) != len(args.models) or any(model not in grid['models'] for model in args.models):
        raise ValueError('Invalid models')
    output.mkdir(parents=True, exist_ok=True)
    identity = dict(grid=grid, grid_sha256=sha256(grid_path), models=args.models,
                    code_hash=implementation_hash(), dataset_hash=protocol['dataset_checksums_sha256'],
                    hardware=hardware(device), selection='validation mAP50-95; deterministic trial ID tie break')
    plan = output/'plan.json'
    if plan.exists() and json.loads(plan.read_text()) != identity:
        raise ValueError('Grid/code/data/environment changed. Use the original setup to resume.')
    r.atomic_json(plan, identity)
    def status(**values):
        r.atomic_json(output/'search_status.json', values)
    try:
        for model in args.models:
            model_folder = output/model
            model_folder.mkdir(exist_ok=True)
            status(phase='preflight', model=model)
            subprocess.run([sys.executable, '-m', 'tuning.preflight', '--root', str(root),
                            '--model-folder', str(model_folder), '--model', model, '--grid', str(grid_path)], check=True)
            trials = list(configurations(grid, model, protocol['dataset_checksums_sha256'], identity['code_hash']))
            for index, (trial_id, config) in enumerate(trials, 1):
                status(phase='screening', model=model, trial=trial_id, trial_number=index, total_trials=len(trials), budget=grid['screen_rounds'])
                print(f"GRID {model}: {trial_id} ({index}/{len(trials)}) { {k:config[k] for k in DIMENSIONS} }", flush=True)
                ensure_trial(root, model_folder, trial_id, config, grid['screen_rounds'], args.workers)
                write_ranking(model_folder/'screening_results.csv', rank_trials(model_folder, trials, grid['screen_rounds']), grid['screen_rounds'])
            screening = rank_trials(model_folder, trials, grid['screen_rounds'])
            if len(screening) != len(trials):
                raise RuntimeError('Not all grid trials completed their screening budget')
            finalist_ids = [row['trial_id'] for row in screening[:grid['finalists']]]
            r.atomic_json(model_folder/'screening_selection.json', dict(budget=grid['screen_rounds'], finalists=finalist_ids))
            selected_trials = [(trial_id, config) for trial_id, config in trials if trial_id in finalist_ids]
            for trial_id, config in selected_trials:
                status(phase='finalist_confirmation', model=model, trial=trial_id, budget=grid['finalist_rounds'])
                ensure_trial(root, model_folder, trial_id, config, grid['finalist_rounds'], args.workers)
            confirmed = rank_trials(model_folder, selected_trials, grid['finalist_rounds'])
            if len(confirmed) != len(selected_trials):
                raise RuntimeError('Finalist confirmation incomplete')
            write_ranking(model_folder/'finalist_results.csv', confirmed, grid['finalist_rounds'])
            winner = confirmed[0]
            r.atomic_json(model_folder/'selected_configuration.json', dict(
                trial_id=winner['trial_id'], config=winner['config'], selection_budget=grid['finalist_rounds'],
                selection_metrics=winner['metrics'], selection_round=winner['best_round'],
                note='Selected among screened finalists; not a full-budget exhaustive optimum'))
            status(phase='final_training', model=model, selected_trial=winner['trial_id'], budget=grid['schedule_rounds'])
            ensure_trial(root, model_folder, winner['trial_id'], winner['config'], grid['schedule_rounds'], args.workers, final=True)
            record = metric_at_budget(model_folder/'final_training', grid['schedule_rounds'])
            r.atomic_json(model_folder/'final_best_metrics.json', dict(round=record['round'], pooled_validation=record['pooled_validation']))
            print(f"FINISHED {model}: full 30-round selected-configuration run saved", flush=True)
        status(phase='complete', models=args.models, final_rounds=grid['schedule_rounds'])
    except BaseException as error:
        status(phase='interrupted_or_failed', error=f'{type(error).__name__}: {error}',
               recovery='Run all again with identical notebook and uploads; saved trials/rounds are skipped')
        raise


if __name__ == '__main__':
    main()
