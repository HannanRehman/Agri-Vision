"""Continue original screening/confirmation/final controller with explicit execution provenance."""
import argparse
from pathlib import Path
import subprocess
from unittest.mock import patch

import torch
from tuning import search
from continuation.efficientdet import profile


def dispatch(command, *args, **kwargs):
    command = list(command)
    if '-m' in command and command[command.index('-m')+1] == 'tuning.runner':
        command[command.index('-m')+1] = 'continuation.runner'
    return REAL_RUN(command, *args, **kwargs)


REAL_RUN = subprocess.run


def main():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--models', nargs='+', required=True)
    args, _ = parser.parse_known_args()
    if args.models != ['efficientdet_d0']:
        raise ValueError('This continuation is limited to EfficientDet')
    profile(args.output/'efficientdet_d0', args.root, torch.device('cuda:0'))
    print('Verified FP32 extension. Original trials/configs/checkpoints remain intact.', flush=True)
    with patch.object(search.subprocess, 'run', dispatch):
        search.main()


if __name__ == '__main__':
    main()
