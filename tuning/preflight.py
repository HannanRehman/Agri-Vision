"""One real GPU batch per batch/resolution pair, isolated from search metrics."""
import argparse
import gc
import itertools
import json
from pathlib import Path
import tempfile

import torch
from comparison import runner as r
from comparison.data import CropDataset, load_protocol
from scripts import yolo_native_train as y
from tuning.models import create_adapter, NativePredictor
from tuning.runner import initialization, hardware, implementation_hash, native_options
from tuning.search import configurations


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--model-folder', type=Path, required=True)
    parser.add_argument('--model', required=True)
    parser.add_argument('--grid', type=Path, required=True)
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError('GPU required for memory preflight')
    torch.set_num_threads(4)
    root, folder, device = args.root.resolve(), args.model_folder.resolve(), torch.device('cuda:0')
    grid = json.loads(args.grid.read_text())
    spec = grid['models'][args.model]
    protocol = load_protocol(root)
    code_hash = implementation_hash()
    trials = list(configurations(grid, args.model, protocol['dataset_checksums_sha256'], code_hash))
    identity = dict(model=args.model, spec=spec, code_hash=code_hash, seed=grid['seed'],
                    dataset_hash=protocol['dataset_checksums_sha256'], hardware=hardware(device))
    report_path = folder/'preflight.json'
    if report_path.exists():
        saved = json.loads(report_path.read_text())
        if saved.get('phase') == 'passed' and saved['identity'] == identity:
            print('GPU batch/resolution preflight already passed:', args.model, flush=True)
            return
    _, initial = initialization(root, folder, trials[0][1], device)
    results = []
    try:
        for batch, size in itertools.product(spec['batch'], spec['resolution']):
            config = dict(trials[0][1], batch=batch, resolution=size)
            r.seed_all(grid['seed'])
            dataset = CropDataset(root, 'train', protocol['clients'][0], limit=batch, seed=grid['seed'])
            print(f'PREFLIGHT {args.model}: batch={batch}, actual resolution={size}', flush=True)
            torch.cuda.reset_peak_memory_stats()
            if args.model == 'yolo_v8n':
                with tempfile.TemporaryDirectory(prefix='agrivision_grid_probe_') as temporary:
                    work = Path(temporary)
                    model = y.create_yolo(root, dataset.names)
                    model.model.load_state_dict(initial['model_state'], strict=True)
                    data = y.runtime_yaml(root, dataset, work/'data')
                    _, local = y.native_train(model, data, work/'fit', grid['seed'], native_options(config, 1, 0))
                    if type(model.trainer.optimizer).__name__ != 'AdamW' or model.trainer.args.imgsz != size or model.trainer.args.batch != batch:
                        raise ValueError('Native trainer did not apply grid parameters')
                    predictor = NativePredictor(model, device, size)
                    predicted = predictor.predict([dataset[0][0]])
                    del model, predictor
            else:
                adapter = create_adapter(args.model, device, root, size, False)
                adapter.load(initial['model_state'])
                train_args = argparse.Namespace(optimizer=config['optimizer'], lr=config['lr'], weight_decay=config['weight_decay'],
                                                batch=batch, rounds=30, local_epochs=1, seed=grid['seed'], workers=0, max_train_batches=1)
                info = r.local_train(adapter, dataset, train_args, 1, 0, lambda **values: None)
                if not info['optimizer_steps'] or not torch.isfinite(torch.tensor(info['mean_loss'])):
                    raise ValueError('GPU probe failed to produce a finite optimizer step')
                image = dataset[0][0]
                if args.model.startswith('ssd'):
                    adapter.net.eval()
                    transformed, _ = adapter.net.transform([image.to(device)], None)
                    if tuple(transformed.tensors.shape[-2:]) != (size, size):
                        raise ValueError('SSD internal transform ignored resolution')
                    del transformed
                elif tuple(adapter.net.config.image_size) != (size, size) or tuple(adapter.batch([image]).shape[-2:]) != (size, size):
                    raise ValueError('EfficientDet anchors/input size differ')
                predicted = adapter.predict([image])
                del adapter
            if len(predicted) != 1 or not torch.isfinite(predicted[0]['boxes']).all():
                raise ValueError('Invalid prediction from GPU probe')
            results.append(dict(batch=batch, resolution=size, phase='passed',
                                peak_allocated_gpu_bytes=torch.cuda.max_memory_allocated()))
            del predicted
            gc.collect()
            torch.cuda.empty_cache()
        r.atomic_json(report_path, dict(phase='passed', identity=identity, probes=results,
                                       diagnostic_only=True, search_weights_from_probe=False))
    except BaseException as error:
        r.atomic_json(report_path, dict(phase='failed', identity=identity, probes=results,
                                       error=f'{type(error).__name__}: {error}', diagnostic_only=True))
        raise


if __name__ == '__main__':
    main()
