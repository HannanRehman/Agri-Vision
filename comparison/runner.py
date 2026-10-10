"""Sequential FedAvg with pooled validation, atomic round commits and explicit resume."""
import argparse
import csv
from datetime import datetime,timezone
import gc
import hashlib
from importlib.metadata import version
import json
import math
from pathlib import Path
import random
import time

import numpy as np
import torch
from torch.utils.data import DataLoader

from .data import CropDataset,collate,load_protocol,sha256
from .evaluation import DetectionEvaluator
from .models import create_adapter


DEFAULTS = {'ssdlite':('sgd',.001,.0005,320,8),
            'ssd_mobilenetv3':('sgd',.001,.0005,320,8),
            'yolo_v8n':('sgd',.005,.0005,640,8),
            'efficientdet_d0':('adamw',.0001,.0001,512,4)}


def seed_all(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def state_hash(state):
    digest = hashlib.sha256()
    for key,value in sorted(state.items()):
        digest.update(key.encode()); digest.update(str(value.dtype).encode()); digest.update(str(tuple(value.shape)).encode())
        digest.update(value.detach().cpu().contiguous().reshape(-1).view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def implementation_hash():
    digest = hashlib.sha256()
    for path in sorted(Path(__file__).parent.glob('*.py')):
        digest.update(path.name.encode()); digest.update(path.read_bytes())
    return digest.hexdigest()


def fedavg(updates):
    if not updates or any(count <= 0 for _,count in updates):
        raise ValueError('Positive image counts required')
    keys = updates[0][0].keys()
    total = sum(count for _,count in updates)
    if any(state.keys() != keys for state,_ in updates):
        raise ValueError('Client state keys differ')
    output = {}
    for key in keys:
        reference = updates[0][0][key]
        if any(state[key].shape != reference.shape or state[key].dtype != reference.dtype for state,_ in updates):
            raise ValueError(f'State shape/type mismatch: {key}')
        if reference.is_floating_point():
            accumulator = torch.zeros_like(reference,dtype=torch.float64)
            for state,count in updates:
                if not torch.isfinite(state[key]).all():
                    raise ValueError(f'Non-finite client update: {key}')
                accumulator.add_(state[key].double(),alpha=count/total)
            output[key] = accumulator.to(reference.dtype)
        else:
            if not all(torch.equal(reference,state[key]) for state,_ in updates):
                raise ValueError(f'Non-floating buffers differ: {key}; cannot average counters')
            output[key] = reference.clone()
    return output


def atomic_json(path,value):
    path = Path(path); path.parent.mkdir(parents=True,exist_ok=True)
    temporary = path.with_suffix(path.suffix+'.tmp')
    temporary.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n',encoding='utf-8')
    temporary.replace(path)


def atomic_torch(path,value):
    temporary = path.with_suffix('.tmp')
    torch.save(value,temporary); temporary.replace(path)


def loader(dataset,batch,workers,shuffle=False,seed=42):
    generator = torch.Generator().manual_seed(seed)
    return DataLoader(dataset,batch_size=batch,shuffle=shuffle,num_workers=workers,
                      collate_fn=collate,generator=generator,pin_memory=torch.cuda.is_available())


def synchronize(device):
    if device.type == 'cuda':
        torch.cuda.synchronize(device)


def local_train(adapter,dataset,args,round_number,client_index,status):
    seed_all(args.seed+round_number*100+client_index)
    learning_rate = args.lr*(.1+.9*(1+math.cos(math.pi*(round_number-1)/max(args.rounds-1,1)))/2)
    parameters = [p for p in adapter.net.parameters() if p.requires_grad]
    optimizer = (torch.optim.SGD(parameters,lr=learning_rate,momentum=.9,weight_decay=args.weight_decay)
                 if args.optimizer == 'sgd' else torch.optim.AdamW(parameters,lr=learning_rate,weight_decay=args.weight_decay))
    seen,steps,loss_total,used_ids = 0,0,0.,set()
    synchronize(adapter.device); started = time.perf_counter()
    for epoch in range(args.local_epochs):
        data = loader(dataset,args.batch,args.workers,True,args.seed+round_number*100+client_index*10+epoch)
        for batch_index,(images,targets) in enumerate(data):
            if args.max_train_batches is not None and batch_index >= args.max_train_batches:
                break
            optimizer.zero_grad(set_to_none=True)
            loss,components = adapter.loss(images,targets)
            if loss.ndim != 0 or not torch.isfinite(loss):
                raise ValueError('Non-finite or non-scalar training loss')
            loss.backward()
            if any(p.grad is not None and not torch.isfinite(p.grad).all() for p in parameters):
                raise ValueError('Non-finite gradients')
            torch.nn.utils.clip_grad_norm_(parameters,10.,error_if_nonfinite=True)
            optimizer.step()
            seen += len(images); steps += 1; loss_total += float(loss.detach())*len(images)
            used_ids.update(int(t['image_id']) for t in targets)
            if batch_index % 25 == 0:
                status(epoch=epoch+1,batch=batch_index+1,loss=float(loss.detach()),components=components)
                print(f'  epoch {epoch+1} batch {batch_index+1}: loss={float(loss.detach()):.4f}',flush=True)
    synchronize(adapter.device)
    if not steps:
        raise ValueError('No optimizer steps performed')
    return dict(unique_training_images=len(used_ids),image_presentations=seen,optimizer_steps=steps,
                mean_loss=loss_total/seen,learning_rate=learning_rate,seconds=time.perf_counter()-started)


def evaluate(adapter,dataset,args,prediction_file=None):
    evaluator = DetectionEvaluator(dataset.names,max_detections=100)
    synchronize(adapter.device); started = time.perf_counter()
    stream = Path(prediction_file).open('w',encoding='utf-8') if prediction_file else None
    try:
        for batch_index,(images,targets) in enumerate(loader(dataset,args.batch,args.workers)):
            predictions = adapter.predict(images)
            evaluator.update(targets,predictions)
            if stream:
                for target,prediction in zip(targets,predictions):
                    stream.write(json.dumps(dict(image_id=int(target['image_id']),key=target['key'],
                                                 **{k:v.cpu().tolist() for k,v in prediction.items()}))+'\n')
            if batch_index % 100 == 0:
                print(f'  validation batch {batch_index+1}',flush=True)
    finally:
        if stream:
            stream.close()
    synchronize(adapter.device)
    elapsed = time.perf_counter()-started
    result = evaluator.compute()
    client_metrics = {}
    for client in sorted({row['client'] for row in dataset.rows}):
        subset = DetectionEvaluator(dataset.names,max_detections=100)
        for row in dataset.rows:
            if row['client']==client:
                image_id = row['image_id']
                subset.update([{**evaluator.targets[image_id],'image_id':torch.tensor(image_id)}],
                              [evaluator.predictions[image_id]])
        client_metrics[client] = subset.compute()
    result.update(inference_and_loader_seconds=elapsed,inference_and_loader_images_per_second=len(dataset)/elapsed,
                  per_client=client_metrics,
                  throughput_scope='Includes loader/adapter/postprocessing; excludes COCO metric calculation; not batch-1 latency')
    return result


def rewrite_metrics(output):
    records = [json.loads(p.read_text()) for p in sorted((output/'results/metrics').glob('round_*.json'))]
    destination = output/'results/metrics/global_validation_metrics.csv'
    temporary = destination.with_suffix('.tmp')
    with temporary.open('w',newline='',encoding='utf-8') as stream:
        fields = ['round','map_50_95','map50','precision','recall','f1','images','round_seconds']
        writer = csv.DictWriter(stream,fieldnames=fields); writer.writeheader()
        for record in records:
            writer.writerow({key:record[key] if key in record else record['pooled_validation'][key] for key in fields})
    temporary.replace(destination)


def parser():
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument('--root',type=Path,default=Path.cwd())
    result.add_argument('--model',choices=DEFAULTS,default='ssdlite')
    result.add_argument('--mode',choices=['pilot','main'],default='pilot')
    result.add_argument('--device',default='cuda')
    result.add_argument('--output',type=Path,required=True)
    result.add_argument('--rounds',type=int)
    result.add_argument('--local-epochs',type=int,default=1)
    result.add_argument('--batch',type=int)
    result.add_argument('--workers',type=int,default=2)
    result.add_argument('--threads',type=int,default=4)
    result.add_argument('--seed',type=int,default=42)
    result.add_argument('--lr',type=float)
    result.add_argument('--optimizer',choices=['sgd','adamw'])
    result.add_argument('--weight-decay',type=float)
    result.add_argument('--max-train-batches',type=int)
    result.add_argument('--validation-limit',type=int)
    result.add_argument('--no-pretrained',action='store_true')
    result.add_argument('--resume',action='store_true',help='Recover the most recent committed round in --output')
    return result


def main(argv=None):
    args = parser().parse_args(argv)
    optimizer,lr,decay,size,batch = DEFAULTS[args.model]
    args.rounds = args.rounds if args.rounds is not None else (1 if args.mode=='pilot' else 10)
    args.batch = args.batch if args.batch is not None else batch
    args.lr = args.lr if args.lr is not None else lr
    args.weight_decay = args.weight_decay if args.weight_decay is not None else decay
    args.optimizer = args.optimizer or optimizer
    if args.mode=='pilot':
        args.max_train_batches = args.max_train_batches if args.max_train_batches is not None else 2
        args.validation_limit = args.validation_limit if args.validation_limit is not None else 24
    elif args.max_train_batches is not None or args.validation_limit is not None or args.no_pretrained:
        raise ValueError('Main runs require full partitions, full validation and pretrained initialization')
    if min(args.rounds,args.local_epochs,args.batch,args.threads,args.lr)<=0 or args.workers<0 or args.weight_decay<0:
        raise ValueError('Invalid training settings')
    if any(value is not None and value < 1 for value in [args.max_train_batches,args.validation_limit]):
        raise ValueError('Subset limits must be positive')
    device = torch.device(args.device)
    if device.type=='cuda' and not torch.cuda.is_available():
        raise RuntimeError('Select a Colab GPU runtime; local installation is CPU-only')
    if args.mode=='main' and device.type!='cuda':
        raise RuntimeError('Full main runs require GPU; CPU is allowed for limited pilots only')
    torch.set_num_threads(args.threads)
    args.root,args.output = args.root.resolve(),args.output.resolve()
    protocol = load_protocol(args.root)
    model_spec = json.loads((args.root/f'configs/comparison/{args.model}.json').read_text())
    if model_spec['input_size_candidate'] != size:
        raise ValueError('Model specification input size differs from implemented adapter')
    seed_all(args.seed)
    dataset = CropDataset(args.root,'valid',limit=args.validation_limit,seed=args.seed)
    run_config = {key:str(value) if isinstance(value,Path) else value for key,value in vars(args).items() if key not in ['resume','output','root','threads','workers','device']}
    run_config.update(dataset_checksums_sha256=protocol['dataset_checksums_sha256'],
                      implementation_sha256=implementation_hash(),
                      model_spec_sha256=sha256(args.root/f'configs/comparison/{args.model}.json'),
                      normalization_policy='freeze running BN stats on all models; train affine parameters',
                      optimizer_state='reset per client per round',learning_rate_schedule='global-round cosine to 0.1 initial LR',
                      augmentation='shared training-only horizontal flip p=0.5; adapter resize/normalization',
                      evaluator='pooled pycocotools COCO bbox; P/R micro at conf .25 IoU .5',
                      gradient_clip_norm=10.,amp=False,deterministic_algorithms='seeded and cuDNN deterministic; not guaranteed bit-identical across hardware')
    config_hash = hashlib.sha256(json.dumps(run_config,sort_keys=True).encode()).hexdigest()
    if args.resume:
        existing = json.loads((args.output/'settings.json').read_text())
        if existing['config_hash'] != config_hash:
            raise ValueError('Resume settings/dataset differ. Round budget and schedule must stay unchanged.')
        if existing['hardware']['gpu'] != (torch.cuda.get_device_name(device) if device.type=='cuda' else 'CPU'):
            raise ValueError('Resume GPU type differs from recorded hardware')
        if existing['hardware']['torch'] != torch.__version__ or any(existing['hardware']['packages'][name] != version(name) for name in existing['hardware']['packages']):
            raise ValueError('Resume package versions differ from the recorded environment')
    else:
        args.output.mkdir(parents=True,exist_ok=False)
    for directory in ['checkpoints','logs','results/metrics','results/predictions','results/evaluations']:
        (args.output/directory).mkdir(parents=True,exist_ok=True)
    def status(**values):
        atomic_json(args.output/'status.json',dict(updated_at_utc=datetime.now(timezone.utc).isoformat(),**values))
    try:
        status(phase='initializing',model=args.model,mode=args.mode)
        adapter = create_adapter(args.model,device,args.root,pretrained=not args.no_pretrained)
        adapter.net.names = dataset.names
        if device.type=='cuda':
            torch.cuda.reset_peak_memory_stats(device)
        settings = dict(config=run_config,config_hash=config_hash,model=adapter.metadata,
                        hardware=dict(gpu=torch.cuda.get_device_name(device) if device.type=='cuda' else 'CPU',
                                      python=__import__('sys').version,torch=torch.__version__,cuda=torch.version.cuda,
                                      packages={name:version(name) for name in ['torchvision','ultralytics','pycocotools','effdet','timm']}),
                        parameter_count=sum(p.numel() for p in adapter.net.parameters()),
                        trainable_parameter_count=sum(p.numel() for p in adapter.net.parameters() if p.requires_grad),
                        input_size=size,validation_images=len(dataset),diagnostic_only=args.mode=='pilot',
                        created_at_utc=datetime.now(timezone.utc).isoformat())
        initial_path = args.output/'checkpoints/initial.pt'
        if not args.resume:
            initial_state = adapter.state()
            atomic_torch(initial_path,dict(model_state=initial_state,round=0,config_hash=config_hash,names=dataset.names))
            settings['initial_checkpoint_sha256'] = sha256(initial_path)
            settings['initial_state_sha256'] = state_hash(initial_state)
            atomic_json(args.output/'settings.json',settings)
        else:
            settings = existing
            if sha256(initial_path) != settings['initial_checkpoint_sha256']:
                raise ValueError('Initialization checkpoint changed')
        latest_path = args.output/'latest.json'
        latest = json.loads(latest_path.read_text()) if args.resume and latest_path.exists() else dict(round=0,checkpoint='checkpoints/initial.pt',best_map=-1.,best_checkpoint=None)
        checkpoint = args.output/latest['checkpoint']
        if latest.get('checkpoint_sha256') and sha256(checkpoint) != latest['checkpoint_sha256']:
            raise ValueError('Committed checkpoint changed')
        saved = torch.load(checkpoint,map_location='cpu',weights_only=True)
        if saved['config_hash'] != config_hash or saved['names'] != dataset.names:
            raise ValueError('Checkpoint metadata differs')
        global_state = saved['model_state']
        start_round = int(latest['round'])
        if saved['round'] != start_round:
            raise ValueError('Commit/checkpoint round mismatch')
        best_map,best_checkpoint = latest['best_map'],latest['best_checkpoint']
        # Remove an uncommitted result from an interrupted round before rebuilding CSVs.
        for metric_path in (args.output/'results/metrics').glob('round_*.json'):
            if int(metric_path.stem.split('_')[-1]) > start_round:
                metric_path.unlink()
        rewrite_metrics(args.output)
        for round_number in range(start_round+1,args.rounds+1):
            round_start = time.perf_counter(); start_hash = state_hash(global_state)
            updates,clients = [],[]
            for client_index,client in enumerate(protocol['clients']):
                adapter.load(global_state)
                if state_hash(adapter.state()) != start_hash:
                    raise ValueError('Clients did not receive identical global initialization')
                # Pilot dataset is bounded before loading, so batches do not scan the full source.
                limit = args.max_train_batches*args.batch*args.local_epochs if args.mode=='pilot' else None
                train = CropDataset(args.root,'train',client,augment=True,limit=limit,seed=args.seed+client_index)
                print(f'ROUND {round_number}: {client}, {len(train)} training images, {args.local_epochs} local epoch(s)',flush=True)
                def client_status(**values):
                    status(phase='training',round=round_number,client=client,**values)
                result = local_train(adapter,train,args,round_number,client_index,client_status)
                result.update(client=client,round_start_sha256=start_hash)
                updates.append((adapter.state(),result['unique_training_images'])); clients.append(result)
                atomic_json(args.output/f'logs/client_{client}_round_{round_number:03d}.json',result)
            global_state = fedavg(updates); del updates
            adapter.load(global_state)
            status(phase='pooled_validation',round=round_number)
            predictions = args.output/f'results/predictions/validation_round_{round_number:03d}.jsonl'
            metrics = evaluate(adapter,dataset,args,predictions)
            round_path = args.output/f'checkpoints/global_round_{round_number:03d}.pt'
            atomic_torch(round_path,dict(model_state=global_state,round=round_number,config_hash=config_hash,names=dataset.names))
            if metrics['map_50_95'] > best_map:
                best_map,best_checkpoint = metrics['map_50_95'],round_path.relative_to(args.output).as_posix()
            record = dict(round=round_number,round_seconds=time.perf_counter()-round_start,
                          pooled_validation=metrics,clients=clients,diagnostic_only=args.mode=='pilot',
                          validation_scope='fixed subset engineering check' if args.mode=='pilot' else 'full frozen pooled validation')
            atomic_json(args.output/f'results/metrics/round_{round_number:03d}.json',record)
            commit = dict(round=round_number,checkpoint=round_path.relative_to(args.output).as_posix(),
                          checkpoint_sha256=sha256(round_path),best_map=best_map,best_checkpoint=best_checkpoint)
            # Commit last: interruption beforehand repeats the whole round from the last commit.
            atomic_json(latest_path,commit); rewrite_metrics(args.output)
            print(f'ROUND {round_number} complete: pooled mAP50-95={metrics["map_50_95"]:.4f}',flush=True)
            gc.collect()
        # Reload checkpoint and verify its tensor state; prediction consistency is covered by pilot tests.
        final = torch.load(args.output/(best_checkpoint or 'checkpoints/initial.pt'),map_location='cpu',weights_only=True)
        adapter.load(final['model_state'])
        if state_hash(adapter.state()) != state_hash(final['model_state']):
            raise ValueError('Save/reload state mismatch')
        status(phase='complete',completed_rounds=args.rounds,mode=args.mode,best_checkpoint=best_checkpoint,
               best_validation_map_50_95=best_map,diagnostic_only=args.mode=='pilot',
               peak_cuda_allocated_bytes=torch.cuda.max_memory_allocated(device) if device.type=='cuda' else None)
        print(f'COMPLETE: {args.output}',flush=True)
    except BaseException as error:
        status(phase='interrupted_or_failed',error=f'{type(error).__name__}: {error}',
               recovery='Use identical settings with --resume; incomplete global round repeats')
        raise


if __name__ == '__main__':
    main()
