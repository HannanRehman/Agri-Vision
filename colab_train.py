"""Single-GPU, sequential three-client FedAvg simulation for Colab."""

import argparse
import gc
import json
import shutil
from pathlib import Path

import numpy as np
import torch

from federated_client import AgriVisionClient
from model import class_names, create_model, get_parameters, set_parameters
from server import weighted_metrics
from split_dataset import check_partition


def average_updates(updates):
    total = sum(count for _, count in updates)
    if total <= 0:
        raise ValueError('No training samples')
    return [sum(arrays[i] * (count / total) for arrays, count in updates)
            for i in range(len(updates[0][0]))]


def release_gpu():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--rounds', type=int, default=1)
    parser.add_argument('--local-epochs', type=int, default=1)
    parser.add_argument('--batch', type=int, default=4)
    parser.add_argument('--imgsz', type=int, default=320)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--initial-checkpoint', type=Path)
    args = parser.parse_args()
    if min(args.rounds, args.local_epochs, args.batch, args.imgsz) < 1:
        parser.error('Training settings must be positive')
    if not torch.cuda.is_available():
        raise RuntimeError('Select a GPU runtime in Colab before training')
    args.output.mkdir(parents=True, exist_ok=False)
    for line in check_partition():
        print(line, flush=True)
    model = create_model()
    start_round = 0
    if args.initial_checkpoint:
        saved = torch.load(args.initial_checkpoint, map_location='cpu', weights_only=True)
        if saved['class_names'] != class_names():
            raise ValueError('Checkpoint class mapping differs from dataset')
        model.model.load_state_dict(saved['model_state_dict'], strict=True)
        start_round = int(saved['round'])
    parameters = get_parameters(model)
    metadata = {**vars(args), 'torch': torch.__version__, 'gpu': torch.cuda.get_device_name(0)}
    (args.output / 'settings.json').write_text(json.dumps(metadata, default=str, indent=2))
    for round_number in range(start_round + 1, start_round + args.rounds + 1):
        config = {'server_round': round_number, 'local_epochs': args.local_epochs,
                  'batch': args.batch, 'imgsz': args.imgsz}
        updates = []
        for client_id in ('client1', 'client2', 'client3'):
            print(f'ROUND {round_number}: training {client_id}', flush=True)
            client = AgriVisionClient(client_id, '0')
            # Every client receives the SAME global parameters for this round.
            arrays, count, _ = client.fit(parameters, config)
            if not all(np.isfinite(array).all() for array in arrays):
                raise ValueError(f'Non-finite update from {client_id}')
            updates.append((arrays, count))
            # Preserve completed local epochs for inspection, not partial-round resume.
            shutil.copytree(client.run_dir, args.output / client_id, dirs_exist_ok=True)
            del client
            release_gpu()
        parameters = average_updates(updates)
        del updates
        set_parameters(model, parameters)
        checkpoint = args.output / f'global_round_{round_number:03d}.pt'
        temporary = checkpoint.with_suffix('.tmp')
        torch.save({'model_state_dict': model.model.state_dict(),
                    'class_names': class_names(), 'round': round_number}, temporary)
        temporary.replace(checkpoint)
        print(f'Saved global checkpoint to Drive: {checkpoint}', flush=True)
        results = []
        for client_id in ('client1', 'client2', 'client3'):
            client = AgriVisionClient(client_id, '0')
            _, count, metrics = client.evaluate(parameters, config)
            results.append((count, metrics))
            del client
            release_gpu()
        metrics = weighted_metrics(results)
        (args.output / f'metrics_round_{round_number:03d}.json').write_text(json.dumps(metrics, indent=2))
        print(f'ROUND {round_number} validation: {metrics}', flush=True)
    # Verify that the saved global checkpoint can be loaded by the project model.
    saved = torch.load(checkpoint, map_location='cpu', weights_only=True)
    model.model.load_state_dict(saved['model_state_dict'], strict=True)
    print(f'COMPLETE: {checkpoint}', flush=True)


if __name__ == '__main__':
    main()
