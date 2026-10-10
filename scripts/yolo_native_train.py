"""24-class YOLO FedAvg with historical native Ultralytics local training."""
import argparse
from copy import deepcopy
import csv
from datetime import datetime, timezone
import gc
from importlib.metadata import version
import json
from pathlib import Path
import shutil
import tempfile
import time

import numpy as np
import torch
import yaml
from comparison.data import CropDataset, load_protocol, sha256
from comparison import runner as r
from scripts.continue_comparison import canonical_hash, safe_checkpoint


NATIVE_ARGS = dict(epochs=1, batch=8, imgsz=640, workers=2, optimizer='auto',
                   lr0=.01, lrf=.01, momentum=.937, weight_decay=.0005,
                   warmup_epochs=3., warmup_momentum=.8, warmup_bias_lr=.1,
                   nbs=64, hsv_h=.015, hsv_s=.7, hsv_v=.4, degrees=0.,
                   translate=.1, scale=.5, shear=0., perspective=0., flipud=0.,
                   fliplr=.5, mosaic=1., mixup=0., copy_paste=0., close_mosaic=10,
                   freeze=None, amp=True, val=False, plots=False, save=True)


def create_yolo(root, names):
    from ultralytics import YOLO
    from ultralytics.nn.tasks import DetectionModel
    model = YOLO(str(root/'yolov8n.pt') if (root/'yolov8n.pt').exists() else 'yolov8n.pt')
    pretrained = model.model
    if len(pretrained.names) != 80:
        raise ValueError('Initialization must be the COCO 80-class yolov8n.pt')
    detector = DetectionModel(deepcopy(pretrained.yaml), nc=24, verbose=False)
    detector.load(pretrained)
    detector.names = names
    model.model = detector
    return model


def floating_state(state):
    return {k:v.detach().cpu().clone() for k,v in state.items() if v.is_floating_point()}


def historical_average(updates, global_state):
    """Exactly the old NumPy weighted floating aggregation; integer buffers unchanged."""
    total = sum(count for _,count in updates)
    if total <= 0 or any(count <= 0 for _,count in updates):
        raise ValueError('Positive training counts required')
    output = {k:v.clone() for k,v in global_state.items()}
    for key, reference in global_state.items():
        if not reference.is_floating_point():
            continue
        if any(state[key].shape != reference.shape or state[key].dtype != reference.dtype or not torch.isfinite(state[key]).all()
               for state,_ in updates):
            raise ValueError(f'Invalid client tensor: {key}')
        arrays = [state[key].detach().cpu().numpy() for state,_ in updates]
        averaged = sum(array*(count/total) for array,(_,count) in zip(arrays,updates))
        output[key] = torch.as_tensor(averaged, dtype=reference.dtype).clone()
    return output


def runtime_yaml(root, dataset, destination, validation=None):
    """Absolute manifest lists avoid Ultralytics resolving portable YAML from its settings root."""
    destination.mkdir(parents=True, exist_ok=True)
    for split, rows in [('train', dataset.rows), ('val', (validation or dataset).rows)]:
        (destination/f'{split}.txt').write_text(''.join(str((root/row['image']).resolve())+'\n' for row in rows), encoding='utf-8')
    path = destination/'data.yaml'
    path.write_text(yaml.safe_dump(dict(path=str(root.resolve()), train=str(destination/'train.txt'),
                                      val=str(destination/'val.txt'), names=dataset.names, nc=24)), encoding='utf-8')
    return path


class NativePredictor:
    def __init__(self, model, device):
        from ultralytics import YOLO
        # Native inference fuses Conv/BN in place. Isolate it from resumable training state.
        self.model = YOLO('yolov8n.yaml', task='detect', verbose=False)
        self.model.model = deepcopy(model.model).float()
        self.model.overrides = deepcopy(model.overrides)
        self.device = torch.device(device)
    @torch.inference_mode()
    def predict(self, images):
        # Native predictor accepts BGR arrays and returns original-image coordinates.
        arrays = [(image.mul(255).round().byte().permute(1,2,0).numpy()[:,:,::-1]).copy() for image in images]
        results = self.model.predict(source=arrays, imgsz=640, device=str(self.device), conf=.001,
                                     iou=.55, max_det=100, verbose=False, save=False)
        return [dict(boxes=result.boxes.xyxy.cpu(), labels=result.boxes.cls.long().cpu(), scores=result.boxes.conf.cpu()) for result in results]


def hardware(device):
    return dict(gpu=torch.cuda.get_device_name(device), torch=torch.__version__,
                packages={name:version(name) for name in ['ultralytics','torchvision','pycocotools','numpy','PyYAML']})


def pipeline_hash():
    return canonical_hash(dict(native=sha256(Path(__file__)), common=r.implementation_hash(),
                               helper=sha256(Path(__file__).with_name('continue_comparison.py'))))


def native_csv(output):
    path=output/'results/metrics/native_validation_metrics.csv'
    temporary=path.with_suffix('.tmp')
    with temporary.open('w',newline='',encoding='utf-8') as stream:
        writer=csv.DictWriter(stream,fieldnames=['round','precision','recall','map50','map_50_95'])
        writer.writeheader()
        for item in sorted((output/'results/metrics').glob('round_*.json')):
            record=json.loads(item.read_text())
            writer.writerow(dict(round=record['round'],**record['native_validation_image_weighted_summary']))
    temporary.replace(path)


def native_train(model, data, work, seed, extra=None):
    options = dict(NATIVE_ARGS, data=str(data), device='0', project=str(work), name='local', exist_ok=False,
                   seed=seed, deterministic=True)
    options.update(extra or {})
    model.predictor = None
    expected=floating_state(model.model.state_dict())
    def verify_received_state(trainer):
        actual=floating_state(trainer.model.state_dict())
        if expected.keys()!=actual.keys() or any(not torch.equal(expected[k],actual[k]) for k in expected):
            raise ValueError('Native trainer did not receive the intended global floating state')
        model.native_transfer_verified=True
    model.add_callback('on_pretrain_routine_end',verify_received_state)
    model.train(**options)
    if not getattr(model,'native_transfer_verified',False):
        raise ValueError('Native initialization verification callback did not run')
    # Native train restores the best/last EMA weights, as in the historical fit call.
    local = Path(model.trainer.save_dir)
    trained = {k:v.detach().cpu().clone().float() if v.is_floating_point() else v.detach().cpu().clone()
               for k,v in model.model.state_dict().items()}
    return trained, local


def learning_pilot(root, output, device, seed):
    r.seed_all(seed); protocol = load_protocol(root)
    train = CropDataset(root, 'train', limit=48, seed=seed)
    model = create_yolo(root, train.names)
    initial = model.model.state_dict()
    before_bn = {k:v.clone() for k,v in initial.items() if k.endswith('running_mean') and '.cv3.' in k}
    output.mkdir(parents=True, exist_ok=False)
    with tempfile.TemporaryDirectory(prefix='agrivision_yolo_pilot_') as temporary:
        work = Path(temporary); data = runtime_yaml(root, train, work/'data')
        _, local = native_train(model, data, work/'fit', seed,
                               dict(epochs=12, workers=0, mosaic=0., hsv_h=0., hsv_s=0., hsv_v=0., translate=0., scale=0., fliplr=0.))
        shutil.copy2(local/'results.csv', output/'results.csv')
        shutil.copy2(local/'args.yaml', output/'args.yaml')
        rows = list(csv.DictReader((local/'results.csv').open()))
        rows = [{k.strip():v for k,v in row.items()} for row in rows]
        first, last = float(rows[0]['train/cls_loss']), float(rows[-1]['train/cls_loss'])
        model.predictor = None
        evaluator_args = argparse.Namespace(batch=8, workers=0)
        metrics = r.evaluate(NativePredictor(model, device), train, evaluator_args)
        after = model.model.state_dict()
        changed_bn = sum(not torch.equal(value.float(),after[key].cpu().float()) for key,value in before_bn.items())
        sample = [train[0][0]]
        expected = NativePredictor(model, device).predict(sample)[0]
        state = {k:v.detach().cpu().clone() for k,v in after.items()}
        test_path = work/'reload.pt'; r.atomic_torch(test_path,dict(model_state=state))
        restored = create_yolo(root, train.names)
        restored.model.load_state_dict(torch.load(test_path,weights_only=True)['model_state'],strict=True)
        restored.model.to(device); restored.predictor=None
        actual = NativePredictor(restored,device).predict(sample)[0]
        reload_ok = all(expected[k].shape == actual[k].shape and torch.allclose(expected[k],actual[k],atol=1e-3,rtol=1e-4) for k in expected)
    passed = bool(last < first*.95 and metrics['map50'] >= .05 and changed_bn > 0 and reload_ok)
    report = dict(phase='passed' if passed else 'failed', diagnostic_only=True,
                  train_images=len(train), train_map50=metrics['map50'], first_cls_loss=first,last_cls_loss=last,
                  classifier_bn_layers_changed=changed_bn, save_reload_passed=reload_ok,
                  native_trainer_transfer_verified=model.native_transfer_verified,
                  dataset_sha256=protocol['dataset_checksums_sha256'], pipeline_hash=pipeline_hash(),
                  hardware=hardware(device), seed=seed, epochs=12,
                  gate='classification loss decreases >=5%, training AP50 >=.05, BN adapts, save/reload matches',
                  metrics_are_not_final_results=True)
    r.atomic_json(output/'pilot.json', report)
    if not passed:
        raise RuntimeError('YOLO learning pilot failed. Main training is blocked; inspect pilot.json and results.csv')
    print(json.dumps(report,indent=2),flush=True)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=Path.cwd())
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--pilot-record', type=Path)
    parser.add_argument('--pilot', action='store_true')
    parser.add_argument('--rounds', type=int, default=30)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--device', default='cuda:0')
    args = parser.parse_args(argv)
    root, output = args.root.resolve(), args.output.resolve()
    device = torch.device(args.device)
    if device.type!='cuda' or not torch.cuda.is_available():
        raise RuntimeError('Native YOLO training requires a Colab GPU')
    if device.index not in (None,0) or args.rounds < 1:
        raise ValueError('Use GPU 0 and a positive round budget')
    torch.set_num_threads(4)
    if args.pilot:
        learning_pilot(root, output, device, args.seed); return
    protocol = load_protocol(root); dataset = CropDataset(root,'valid')
    current_hardware = hardware(device)
    config = dict(model='yolo_v8n', mode='main', rounds=args.rounds, seed=args.seed,
                  native_args=NATIVE_ARGS, dataset_checksums_sha256=protocol['dataset_checksums_sha256'],
                  pipeline_hash=pipeline_hash(), aggregation='historical NumPy training-image-weighted floating tensors; integer buffers unchanged',
                  normalization='native adaptive BatchNorm', optimizer_state='native optimizer and warmup reset per client fit, as historical',
                  evaluator='shared pooled COCO AP; micro P/R conf .25 IoU .5; native client summaries separately labelled')
    config_hash = canonical_hash(config)
    if output.exists():
        settings = json.loads((output/'settings.json').read_text())
        if settings['config_hash'] != config_hash or settings['config'] != config or settings['hardware'] != current_hardware:
            raise ValueError('Native YOLO resume config/code/dataset/environment differs')
    else:
        if not args.pilot_record:
            raise ValueError('A passed learning pilot is required')
        pilot = json.loads(args.pilot_record.read_text())
        if pilot['phase']!='passed' or pilot['pipeline_hash']!=pipeline_hash() or pilot['hardware']!=current_hardware or pilot['dataset_sha256']!=protocol['dataset_checksums_sha256'] or pilot['seed']!=args.seed:
            raise ValueError('Pilot did not pass for this pipeline/environment/dataset/seed')
        output.mkdir(parents=True)
        for directory in ['checkpoints','logs','results/metrics','results/predictions','results/evaluations']:
            (output/directory).mkdir(parents=True)
        r.seed_all(args.seed); initial = create_yolo(root, dataset.names)
        global_state = {k:v.detach().cpu().clone() for k,v in initial.model.state_dict().items()}
        r.atomic_torch(output/'checkpoints/initial.pt',dict(model_state=global_state, round=0, config_hash=config_hash,names=dataset.names))
        settings = dict(config=config,config_hash=config_hash,hardware=current_hardware,diagnostic_only=False,
                        model=dict(architecture='YOLOv8n native Ultralytics',input_size=640),input_size=640,
                        parameter_count=sum(p.numel() for p in initial.model.parameters()),
                        initial_checkpoint_sha256=sha256(output/'checkpoints/initial.pt'),
                        pretrained_sha256=sha256(root/'yolov8n.pt'), pilot=pilot)
        r.atomic_json(output/'settings.json',settings)
        r.atomic_json(output/'latest.json',dict(round=0,checkpoint='checkpoints/initial.pt',checkpoint_sha256=sha256(output/'checkpoints/initial.pt'),best_map=-1.,best_checkpoint=None))
        del initial; gc.collect()
    latest = json.loads((output/'latest.json').read_text())
    path = safe_checkpoint(output,latest['checkpoint'])
    if sha256(path)!=latest['checkpoint_sha256'] or sha256(output/'checkpoints/initial.pt')!=settings['initial_checkpoint_sha256']:
        raise ValueError('Native checkpoint changed')
    saved = torch.load(path,map_location='cpu',weights_only=True)
    if saved['config_hash']!=config_hash or saved['round']!=latest['round'] or saved['names']!=dataset.names:
        raise ValueError('Native checkpoint metadata differs')
    global_state = saved['model_state']
    for metric in (output/'results/metrics').glob('round_*.json'):
        if int(metric.stem.split('_')[-1])>latest['round']:metric.unlink()
    r.rewrite_metrics(output)
    native_csv(output)
    def status(**values):
        r.atomic_json(output/'status.json',dict(updated_at_utc=datetime.now(timezone.utc).isoformat(),**values))
    try:
        for number in range(latest['round']+1,args.rounds+1):
            started=time.perf_counter(); start_hash=r.state_hash(floating_state(global_state)); updates=[];clients=[]
            with tempfile.TemporaryDirectory(prefix='agrivision_yolo_round_') as temporary:
                work=Path(temporary)
                for client in protocol['clients']:
                    status(phase='training',round=number,client=client)
                    print(f'ROUND {number}/{args.rounds}: {client}, native Ultralytics training',flush=True)
                    # Historical clients created the same canonical model then received floating weights.
                    model=create_yolo(root,dataset.names)
                    state=model.model.state_dict()
                    model.model.load_state_dict({k:global_state[k] if v.is_floating_point() else v for k,v in state.items()},strict=True)
                    if r.state_hash(floating_state(model.model.state_dict()))!=start_hash:raise ValueError('Client initialization differs')
                    train=CropDataset(root,'train',client); valid=CropDataset(root,'valid',client)
                    data=runtime_yaml(root,train,work/client/'data',valid)
                    trained, local=native_train(model,data,work/client/'fit',args.seed)
                    updates.append((trained,len(train)))
                    log=output/f'logs/{client}_round_{number:03d}';log.mkdir(exist_ok=True)
                    for filename in ['args.yaml','results.csv']:
                        shutil.copy2(local/filename,log/filename)
                    resolved=yaml.safe_load((local/'args.yaml').read_text())
                    record=dict(client=client,unique_training_images=len(train),round_start_sha256=start_hash,
                                resolved_args=resolved,actual_optimizer=type(model.trainer.optimizer).__name__,
                                native_trainer_transfer_verified=model.native_transfer_verified,
                                final_learning_rates=[float(g['lr']) for g in model.trainer.optimizer.param_groups])
                    clients.append(record)
                    del model;gc.collect();torch.cuda.empty_cache()
                global_state=historical_average(updates,global_state);del updates
                model=create_yolo(root,dataset.names);model.model.load_state_dict(global_state,strict=True)
                native_clients=[]
                for client in protocol['clients']:
                    status(phase='native_validation',round=number,client=client)
                    train=CropDataset(root,'train',client);valid=CropDataset(root,'valid',client)
                    data=runtime_yaml(root,train,work/client/'eval_data',valid)
                    result=model.val(data=str(data),split='val',imgsz=640,batch=8,device='0',workers=2,plots=False,
                                     project=str(work/'validation'),name=client,verbose=False)
                    native_clients.append(dict(client=client,images=len(valid),precision=float(result.box.mp),recall=float(result.box.mr),map50=float(result.box.map50),map_50_95=float(result.box.map)))
                status(phase='pooled_validation',round=number)
                model.predictor=None
                metrics=r.evaluate(NativePredictor(model,device),dataset,argparse.Namespace(batch=8,workers=2))
                native_summary={k:sum(item[k]*item['images'] for item in native_clients)/len(dataset) for k in ['precision','recall','map50','map_50_95']}
                round_path=output/f'checkpoints/global_round_{number:03d}.pt'
                r.atomic_torch(round_path,dict(model_state=global_state,round=number,config_hash=config_hash,names=dataset.names))
                best_map,best_path=latest['best_map'],latest['best_checkpoint']
                if metrics['map_50_95']>best_map:best_map,best_path=metrics['map_50_95'],round_path.relative_to(output).as_posix()
                r.atomic_json(output/f'results/metrics/round_{number:03d}.json',dict(round=number,round_seconds=time.perf_counter()-started,
                              pooled_validation=metrics,clients=clients,native_clients=native_clients,
                              native_validation_image_weighted_summary=native_summary,diagnostic_only=False))
                latest=dict(round=number,checkpoint=round_path.relative_to(output).as_posix(),checkpoint_sha256=sha256(round_path),best_map=best_map,best_checkpoint=best_path)
                r.atomic_json(output/'latest.json',latest);r.rewrite_metrics(output);native_csv(output)
                from scripts.continue_comparison import prune_branch
                prune_branch(output,latest)
                print(f"ROUND {number} committed: pooled AP95={metrics['map_50_95']:.4f}; native precision={native_summary['precision']:.4f}",flush=True)
                del model;gc.collect();torch.cuda.empty_cache()
        status(phase='complete',completed_rounds=latest['round'],best_checkpoint=latest['best_checkpoint'])
    except BaseException as error:
        status(phase='interrupted_or_failed',error=f'{type(error).__name__}: {error}',recovery='Rerun identical command; incomplete round repeats')
        raise


if __name__=='__main__':
    main()
