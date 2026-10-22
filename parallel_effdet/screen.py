"""Run assigned screening trials only; combine all results before selection."""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
from unittest.mock import patch

import torch
from comparison import runner as r
from comparison.data import load_protocol, sha256
from continuation.efficientdet import BASE_HASH, profile
from continuation.search import dispatch
from tuning import search
from tuning.runner import digest, hardware, implementation_hash

INITIAL_SHA = 'dd0a92a8261e15f8776deba607c6ef0b9bab4389f3705a656c4a8050b45c6fce'


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--first', type=int, required=True)
    parser.add_argument('--last', type=int, required=True)
    parser.add_argument('--workers', type=int, default=2)
    args = parser.parse_args(argv)
    if (args.first, args.last) not in ((1, 11), (12, 16)) or args.workers != 2:
        raise ValueError('Only assigned ranges 1-11 and 12-16 with two workers are supported')
    if not torch.cuda.is_available() or 'T4' not in torch.cuda.get_device_name(0):
        raise ValueError('A T4 runtime is required')
    root, output = args.root.resolve(), args.output.resolve()
    model = output/'efficientdet_d0'
    if implementation_hash() != BASE_HASH or sha256(model/'initial.pt') != INITIAL_SHA:
        raise ValueError('Original implementation or shared initialization differs')
    execution = profile(model, root, torch.device('cuda:0'))
    grid_path = root/'configs/tuning/grid_v1.json'
    grid = json.loads(grid_path.read_text())
    protocol = load_protocol(root)
    if grid['screen_rounds'] != 5 or grid['local_epochs'] != 1:
        raise ValueError('Original five-round screening protocol required')
    all_trials = list(search.configurations(grid, 'efficientdet_d0',
        protocol['dataset_checksums_sha256'], BASE_HASH))
    trials = all_trials[args.first-1:args.last]
    driver = Path(__file__)
    identity = dict(version='efficientdet_parallel_screening_v1', first=args.first, last=args.last,
        workers=args.workers, trials=[name for name, _ in trials],
        configs={name:config for name, config in trials},
        grid_sha256=sha256(grid_path), initial_sha256=INITIAL_SHA,
        execution_profile_sha256=digest(execution), hardware=hardware(torch.device('cuda:0')),
        driver_sha256=sha256(driver))
    assignment = model/f'parallel_{args.first:02d}_{args.last:02d}_assignment.json'
    if assignment.exists() and json.loads(assignment.read_text()) != identity:
        raise ValueError('Assigned screening setup changed')
    r.atomic_json(assignment, identity)
    status_file = model/f'parallel_{args.first:02d}_{args.last:02d}_status.json'

    def status(**values):
        r.atomic_json(status_file, dict(updated_at_utc=datetime.now(timezone.utc).isoformat(), **values))

    try:
        # Preserve preflight behavior and the original runtime/data identity checks.
        import subprocess
        import sys
        subprocess.run([sys.executable, '-u', '-m', 'tuning.preflight', '--root', str(root),
            '--model-folder', str(model), '--model', 'efficientdet_d0', '--grid', str(grid_path)], check=True)
        for trial_id, config in trials:
            status(phase='screening', trial=trial_id, budget=5, assigned_first=args.first, assigned_last=args.last)
            print(f'ASSIGNED EfficientDet: {trial_id}; five-round screening only', flush=True)
            with patch.object(search.subprocess, 'run', dispatch):
                search.ensure_trial(root, model, trial_id, config, 5, args.workers)
            search.write_ranking(model/f'parallel_{args.first:02d}_{args.last:02d}_results.csv',
                search.rank_trials(model, trials, 5), 5)
        status(phase='assigned_screening_complete', assigned_first=args.first, assigned_last=args.last,
            next_step='Combine all 16 screening trials on 71gmail before finalist selection and final training')
        print('Assigned screening complete. This job stops here; combine results before selection.', flush=True)
    except BaseException as error:
        status(phase='interrupted_or_failed', error=f'{type(error).__name__}: {error}',
            recovery='Run all again with the same notebook; committed rounds are preserved')
        raise


if __name__ == '__main__':
    main()
