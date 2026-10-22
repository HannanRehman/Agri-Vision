"""Compare FP32 step overhead on fixed training-only batches without saving weights."""
import argparse
import gc
import json
from pathlib import Path
import statistics
import tempfile
import time

import torch

from comparison import runner as r
from comparison.data import CropDataset, sha256, load_protocol
from comparison.models import freeze_batchnorm
from tuning.models import create_adapter
from tuning.runner import hardware


def tensor_loss(adapter, images, targets):
    """Same EfficientDet loss computation; leave diagnostic scalars on the device."""
    adapter.training_bench.train()
    freeze_batchnorm(adapter.net)
    count = max(max(len(t['boxes']) for t in targets), 1)
    boxes = torch.full((len(targets), count, 4), -1., device=adapter.device)
    labels = torch.full((len(targets), count), -1., device=adapter.device)
    for index, (image, target) in enumerate(zip(images, targets)):
        height, width = image.shape[-2:]
        scaled = target['boxes'].to(adapter.device) * boxes.new_tensor(
            [adapter.size / width, adapter.size / height] * 2)
        boxes[index, :len(scaled)] = scaled[:, [1, 0, 3, 2]]
        labels[index, :len(scaled)] = target['labels'].to(adapter.device) + 1
    result = adapter.training_bench(adapter.batch(images), dict(bbox=boxes, cls=labels))
    return result['loss'], {k: result[k].detach() for k in ('class_loss', 'box_loss')}


def step(adapter, optimizer, images, targets, optimized):
    parameters = [p for p in adapter.net.parameters() if p.requires_grad]
    optimizer.zero_grad(set_to_none=True)
    loss, components = tensor_loss(adapter, images, targets) if optimized else adapter.loss(images, targets)
    # Preserve the original loss guard, clipping threshold and optimizer order.
    if loss.ndim != 0 or not torch.isfinite(loss):
        raise ValueError('Non-finite or non-scalar training loss')
    loss.backward()
    if not optimized and any(p.grad is not None and not torch.isfinite(p.grad).all() for p in parameters):
        raise ValueError('Non-finite gradients')
    # A non-finite gradient makes the total norm non-finite; reject before stepping.
    torch.nn.utils.clip_grad_norm_(parameters, 10., error_if_nonfinite=True)
    optimizer.step()
    return loss.detach(), components


def compare_states(reference, candidate, atol=1e-7, rtol=1e-6):
    if reference.keys() != candidate.keys():
        return dict(passed=False, reason='state keys differ')
    maximum = 0.
    exact = True
    passed = True
    for key, value in reference.items():
        other = candidate[key]
        exact = exact and torch.equal(value, other)
        if value.is_floating_point():
            maximum = max(maximum, float((value - other).abs().max()) if value.numel() else 0.)
            passed = passed and bool(torch.allclose(value, other, atol=atol, rtol=rtol))
        else:
            passed = passed and torch.equal(value, other)
    return dict(passed=passed, bitwise_equal=exact, max_absolute_difference=maximum, atol=atol, rtol=rtol)


def protected_snapshot(folder):
    return {p.relative_to(folder).as_posix(): sha256(p)
            for p in sorted(folder.rglob('*')) if p.is_file()}


def check_output_path(output, model_folder, root):
    output, model_folder, root = output.resolve(), model_folder.resolve(), root.resolve()
    if (output == root or output.is_relative_to(model_folder.parent.parent)
            or model_folder.is_relative_to(output) or root.is_relative_to(output)):
        raise ValueError('Benchmark output must be separate from the search and code/data root')


def run_segment(adapter, initial, batches, optimized, steps, seed, lr, weight_decay):
    adapter.load(initial)
    r.seed_all(seed)
    optimizer = torch.optim.AdamW([p for p in adapter.net.parameters() if p.requires_grad],
                                 lr=lr, weight_decay=weight_decay)
    losses = []
    weighted_loss = torch.zeros((), device=adapter.device, dtype=torch.float64) if optimized else 0.
    # Log to local disk only. Drive latency is intentionally not simulated in this test.
    with tempfile.TemporaryDirectory(prefix='effdet_bench_') as temporary:
        status = Path(temporary) / 'status.json'
        r.synchronize(adapter.device)
        if adapter.device.type == 'cuda':
            torch.cuda.reset_peak_memory_stats(adapter.device)
        start = time.perf_counter()
        images_seen = 0
        for index in range(steps):
            images, targets = batches[index % len(batches)]
            loss, components = step(adapter, optimizer, images, targets, optimized)
            losses.append(loss)
            weighted_loss += loss.double() * len(images) if optimized else float(loss) * len(images)
            images_seen += len(images)
            if index % 25 == 0:
                r.atomic_json(status, dict(batch=index+1, loss=float(loss),
                                          components={k: float(v) for k, v in components.items()}))
        r.synchronize(adapter.device)
        elapsed = time.perf_counter() - start
    loss_values = torch.stack(losses).cpu()
    result = dict(seconds=elapsed, images=images_seen, images_per_second=images_seen/elapsed,
                  mean_loss=float(weighted_loss)/images_seen,
                  peak_allocated_bytes=torch.cuda.max_memory_allocated(adapter.device)
                  if adapter.device.type == 'cuda' else None)
    state = adapter.state()
    del optimizer
    return result, state, loss_values


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--model-folder', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--steps', type=int, default=30)
    parser.add_argument('--warmup', type=int, default=3)
    parser.add_argument('--repeats', type=int, default=3)
    parser.add_argument('--workers', type=int, default=2)
    args = parser.parse_args(argv)
    if min(args.steps, args.warmup, args.repeats) < 1:
        raise ValueError('Positive step/warmup/repeat counts required')
    check_output_path(args.output, args.model_folder, args.root)
    if not torch.cuda.is_available() or 'T4' not in torch.cuda.get_device_name(0):
        raise RuntimeError('Use a T4 GPU for the benchmark')
    torch.set_num_threads(4)
    protocol = load_protocol(args.root)
    device = torch.device('cuda:0')
    before = protected_snapshot(args.model_folder)
    report = dict(diagnostic_only=True, precision='FP32', architecture='efficientdet_d0',
                  hardware=hardware(device), dataset_hash=protocol['dataset_checksums_sha256'],
                  steps=args.steps, warmup=args.warmup, repeats=args.repeats,
                  scope='Cached CPU training batches; includes transfer/forward/backward/optimizer/local logging. '
                        'Excludes dataset decoding, Drive latency, validation and FedAvg. No full-round speedup claim.',
                  cases=[], training_search_modified=False, automatic_migration=False)
    args.output.mkdir(parents=True, exist_ok=True)
    saved = torch.load(args.model_folder/'initial.pt', map_location='cpu', weights_only=True)
    settings = json.loads((args.model_folder/'trials/trial_05/settings.json').read_text())
    if report['hardware'] != settings['hardware']:
        raise ValueError('GPU/package environment differs from the saved search')
    initial = saved['model_state']
    try:
        for size in (512, 640):
            for batch in (2, 4):
                # Same training sample, augmentation RNG and tensors for both variants.
                r.seed_all(42)
                dataset = CropDataset(args.root, 'train', protocol['clients'][0], augment=True,
                                      limit=batch*4, seed=42)
                batches = list(r.loader(dataset, batch, args.workers, shuffle=False))
                adapter = create_adapter('efficientdet_d0', device, args.root, size, pretrained=False)
                runs = []
                for repeat in range(args.repeats):
                    pair = {}
                    order = (False, True) if repeat % 2 == 0 else (True, False)
                    for optimized in order:
                        label = 'optimized' if optimized else 'original'
                        print(f'BENCH {size}px batch={batch} repeat={repeat+1} {label}', flush=True)
                        run_segment(adapter, initial, batches, optimized, args.warmup, 42,
                                    settings['config']['lr'], settings['config']['weight_decay'])
                        pair[label] = run_segment(adapter, initial, batches, optimized, args.steps, 42,
                                                  settings['config']['lr'], settings['config']['weight_decay'])
                    original, candidate = pair['original'], pair['optimized']
                    equivalence = compare_states(original[1], candidate[1])
                    equivalence['losses_match'] = bool(torch.allclose(original[2], candidate[2], atol=1e-7, rtol=1e-6))
                    equivalence['passed'] &= equivalence['losses_match']
                    if not equivalence['passed']:
                        raise RuntimeError(f'Parameter/loss equivalence failed: {equivalence}')
                    runs.append(dict(original=original[0], optimized=candidate[0], equivalence=equivalence,
                                     speedup=original[0]['seconds']/candidate[0]['seconds']))
                    del pair
                speedup = statistics.median(x['speedup'] for x in runs)
                report['cases'].append(dict(resolution=size, batch=batch, runs=runs, median_speedup=speedup,
                                            step_time_reduction_percent=100*(1-1/speedup)))
                r.atomic_json(args.output/'benchmark.json', report)
                del adapter, batches, dataset
                gc.collect()
                torch.cuda.empty_cache()
        report['phase'] = 'complete'
        report['decision'] = 'Review measured speed and equivalence; existing search is not migrated automatically.'
    except BaseException as error:
        report.update(phase='failed_or_interrupted', error=f'{type(error).__name__}: {error}')
        raise
    finally:
        unchanged = before == protected_snapshot(args.model_folder)
        report['training_search_unchanged_verified'] = unchanged
        report['training_search_modified'] = not unchanged
        r.atomic_json(args.output/'benchmark.json', report)
        if not unchanged:
            raise RuntimeError('Search files changed during benchmark; investigate concurrent training')
    print(json.dumps(report, indent=2), flush=True)


if __name__ == '__main__':
    main()
