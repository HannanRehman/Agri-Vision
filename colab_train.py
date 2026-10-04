"""Single-GPU, sequential three-client FedAvg simulation for Colab."""

import argparse
import csv
import gc
from importlib.metadata import version
import json
import shutil
import threading
from datetime import datetime, timezone
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


def save_round_metrics(output, round_number, results):
    """Keep aggregation inputs and a CSV of every completed validation round."""
    output = Path(output)
    metrics = weighted_metrics([(item['validation_images'], item['metrics']) for item in results])
    detail = {'round': round_number, 'split': 'val',
              'aggregation': 'validation-image-weighted client metrics', 'clients': results}
    for filename, value in [(f'metrics_clients_round_{round_number:03d}.json', detail),
                            (f'metrics_round_{round_number:03d}.json', metrics)]:
        target = output / filename
        temporary = target.with_suffix('.tmp')
        temporary.write_text(json.dumps(value, indent=2))
        temporary.replace(target)
    target = output / 'global_validation_metrics.csv'
    temporary = target.with_suffix('.tmp')
    with temporary.open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=['round', 'precision', 'recall', 'mAP50', 'mAP50_95'])
        writer.writeheader()
        for path in sorted(output.glob('metrics_round_*.json')):
            writer.writerow({'round': int(path.stem.rsplit('_', 1)[1]), **json.loads(path.read_text())})
    temporary.replace(target)
    return metrics


class RunMonitor:
    """Persist visible progress and emit a heartbeat while long YOLO calls run."""

    def __init__(self, output):
        self.output = output
        self.state = {'phase': 'starting'}
        self.lock = threading.RLock()
        self.stop_event = threading.Event()
        self.thread = threading.Thread(target=self._run, daemon=True)
        self._write()
        self.thread.start()

    def update(self, **values):
        with self.lock:
            self.state.update(values)
        self._write()

    def _snapshot(self):
        with self.lock:
            snapshot = dict(self.state)
        snapshot['updated_at_utc'] = datetime.now(timezone.utc).isoformat()
        if torch.cuda.is_available():
            gib = 1024 ** 3
            snapshot['cuda_allocated_gb'] = round(torch.cuda.memory_allocated() / gib, 3)
            snapshot['cuda_reserved_gb'] = round(torch.cuda.memory_reserved() / gib, 3)
        return snapshot

    def _write(self):
        with self.lock:
            snapshot = self._snapshot()
            temporary = self.output / 'status.tmp'
            temporary.write_text(json.dumps(snapshot, indent=2))
            temporary.replace(self.output / 'status.json')

    def _run(self):
        while not self.stop_event.wait(60):
            self._write()
            snapshot = self._snapshot()
            print(
                'HEARTBEAT: '
                f"round={snapshot.get('round', '-')} "
                f"client={snapshot.get('client', '-')} "
                f"phase={snapshot['phase']} "
                f"cuda_reserved={snapshot.get('cuda_reserved_gb', 0)}GB",
                flush=True,
            )

    def close(self):
        self.stop_event.set()
        self.thread.join(timeout=2)
        self.update(phase='complete')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--rounds', type=int, default=1)
    parser.add_argument('--local-epochs', type=int, default=1)
    parser.add_argument('--batch', type=int, default=16)
    parser.add_argument('--imgsz', type=int, default=320)
    parser.add_argument('--workers', type=int, default=2)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--initial-checkpoint', type=Path)
    args = parser.parse_args()
    if min(args.rounds, args.local_epochs, args.batch, args.imgsz) < 1:
        parser.error('Training settings must be positive')
    if args.workers < 0:
        parser.error('workers cannot be negative')
    if not torch.cuda.is_available():
        raise RuntimeError('Select a GPU runtime in Colab before training')
    args.output.mkdir(parents=True, exist_ok=False)
    monitor = RunMonitor(args.output)
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
    metadata = {**vars(args), 'torch': torch.__version__, 'gpu': torch.cuda.get_device_name(0),
                'start_round': start_round, 'end_round': start_round + args.rounds,
                'class_names': class_names(),
                'packages': {name: version(name) for name in ('ultralytics', 'flwr', 'numpy', 'PyYAML')}}
    (args.output / 'settings.json').write_text(json.dumps(metadata, default=str, indent=2))
    for round_number in range(start_round + 1, start_round + args.rounds + 1):
        config = {'server_round': round_number, 'local_epochs': args.local_epochs,
                  'batch': args.batch, 'imgsz': args.imgsz, 'workers': args.workers}
        updates = []
        for client_id in ('client1', 'client2', 'client3'):
            monitor.update(round=round_number, client=client_id, phase='training')
            print(f'ROUND {round_number}: training {client_id}', flush=True)
            client = AgriVisionClient(client_id, '0')
            # Every client receives the SAME global parameters for this round.
            arrays, count, _ = client.fit(parameters, config)
            if not all(np.isfinite(array).all() for array in arrays):
                raise ValueError(f'Non-finite update from {client_id}')
            updates.append((arrays, count))
            # Preserve completed local epochs for inspection, not partial-round resume.
            monitor.update(round=round_number, client=client_id, phase='copying local results')
            local_run = client.run_dir / f'round_{round_number:03d}'
            if not local_run.is_dir():
                raise FileNotFoundError(f'Expected local training output: {local_run}')
            shutil.copytree(local_run, args.output / client_id / local_run.name)
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
        monitor.update(round=round_number, client='-', phase='checkpoint saved')
        print(f'Saved global checkpoint to Drive: {checkpoint}', flush=True)
        results = []
        for client_id in ('client1', 'client2', 'client3'):
            monitor.update(round=round_number, client=client_id, phase='validating global model')
            client = AgriVisionClient(client_id, '0')
            _, count, metrics = client.evaluate(parameters, config)
            results.append({'client': client_id, 'validation_images': count, 'metrics': metrics})
            del client
            release_gpu()
        metrics = save_round_metrics(args.output, round_number, results)
        monitor.update(round=round_number, client='-', phase='round complete', metrics=metrics)
        print(f'ROUND {round_number} validation: {metrics}', flush=True)
    # Verify that the saved global checkpoint can be loaded by the project model.
    saved = torch.load(checkpoint, map_location='cpu', weights_only=True)
    model.model.load_state_dict(saved['model_state_dict'], strict=True)
    monitor.close()
    print(f'COMPLETE: {checkpoint}', flush=True)


if __name__ == '__main__':
    main()
