"""Run the unchanged trial controller with a separately verified FP32 step extension."""
import argparse
from pathlib import Path
from unittest.mock import patch

import torch

from comparison import runner as r
from tuning import runner
from tuning.runner import digest
from continuation.efficientdet import local_train, profile


def main():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument('--model-folder', type=Path, required=True)
    parser.add_argument('--root', type=Path, required=True)
    args, _ = parser.parse_known_args()
    execution = profile(args.model_folder, args.root, torch.device('cuda:0'))
    atomic = r.atomic_json

    def record(path, value):
        # Only annotate NEW metric records; never rewrite historical configs/settings.
        if Path(path).name.startswith('round_') and Path(path).suffix == '.json' and 'pooled_validation' in value:
            value = dict(value, execution_profile=dict(version=execution['version'], sha256=digest(execution)))
        return atomic(path, value)

    with patch.object(r, 'local_train', local_train), patch.object(r, 'atomic_json', record):
        runner.main()


if __name__ == '__main__':
    main()
