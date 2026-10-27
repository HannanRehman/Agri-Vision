"""Download a bounded batch from connector references, then audit the merged run."""
from concurrent.futures import ThreadPoolExecutor
import csv
import hashlib
import json
from pathlib import Path, PurePosixPath
import sys
import time
import urllib.request
import zipfile

ROOT = Path(__file__).resolve().parents[1]
CACHE = ROOT/'saved_models/efficientdet_merged_re19gmail_20261024'
REPORT = ROOT/'reports/comparison/efficientdet_merge_re19gmail_20261024'
INITIAL = 'dd0a92a8261e15f8776deba607c6ef0b9bab4389f3705a656c4a8050b45c6fce'


def sha(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream,'sha256').hexdigest()


def download(item):
    relative=PurePosixPath(item['relative'])
    if relative.is_absolute() or '..' in relative.parts or any(':' in part for part in relative.parts):
        raise ValueError('Invalid relative download path')
    CACHE.mkdir(parents=True,exist_ok=True)
    base=CACHE.resolve()
    path=base.joinpath(*relative.parts)
    path.parent.mkdir(parents=True,exist_ok=True)
    if not path.parent.resolve().is_relative_to(base):
        raise ValueError('Download parent escaped the cache')
    for attempt in range(3):
        try:
            with urllib.request.urlopen(item['url'],timeout=15) as source, path.open('wb') as target:
                while block := source.read(1024*1024):
                    target.write(block)
            break
        except Exception as error:
            if attempt==2:
                raise RuntimeError('Download failed for '+item['relative']+' ('+type(error).__name__+')') from None
            time.sleep(.5*(attempt+1))
    if path.stat().st_size != item['bytes']:
        raise ValueError('Size mismatch: '+item['relative'])
    return dict(relative=item['relative'],bytes=path.stat().st_size,sha256=sha(path))


def validate(cache=CACHE):
    import torch
    def read(relative):
        return json.loads((cache/relative).read_text(encoding='utf-8'))
    if sha(cache/'initial.pt') != INITIAL:
        raise ValueError('Shared initialization differs')
    if sha(cache/'ROOT/agrivision_grid_search_v1_code.zip') != '5453a56a97c2aca78224cec346c259e5f8a0a5eda2a93eb17b3677f06689c049':
        raise ValueError('Original code bundle differs')
    if sha(cache/'ROOT/efficientdet_fp32_continuation_v1.zip') != '35e010acf0d7bfebd3ff585d7cc33bdabd9bc009371d9e01b2fcc4f27ca5820e':
        raise ValueError('Optimized FP32 extension differs')
    plan=read('PLAN/plan.json')
    if plan['models'] != ['efficientdet_d0']:
        raise ValueError('Merged plan must select EfficientDet only')
    trials={}; ranked=[]
    reference=read('trials/trial_01/settings.json')
    if reference['hardware']!=plan['hardware']:
        raise ValueError('Plan hardware differs from the copied trials')
    parts=read('DATASET/parts.json')
    if len(parts['parts'])!=14 or parts['joined_sha256']!='7a2920e4a4e0405ee157e9b40ac6306b54d6933b20a306b44a4b7c6871f20868':
        raise ValueError('Dataset parts manifest differs')
    execution=read('execution_profile.json')
    with zipfile.ZipFile(cache/'ROOT/efficientdet_fp32_continuation_v1.zip') as extension:
        for filename,expected in execution['extension_files'].items():
            if hashlib.sha256(extension.read(filename)).hexdigest()!=expected:
                raise ValueError('Execution profile source differs: '+filename)
        if hashlib.sha256(extension.read('continuation/benchmark_evidence.json')).hexdigest()!=execution['benchmark_sha256']:
            raise ValueError('Benchmark evidence differs from execution profile')
    execution_digest=hashlib.sha256(json.dumps(execution,sort_keys=True).encode()).hexdigest()
    for number in range(1,17):
        name=f'trial_{number:02d}'; path=cache/'trials'/name
        latest=read(f'trials/{name}/latest.json')
        settings=read(f'trials/{name}/settings.json')
        config=read(f'configs/{name}.json')
        if settings['config']!=config or settings['hardware']!=reference['hardware']:
            raise ValueError('Config/environment differs: '+name)
        config_hash=hashlib.sha256(json.dumps(config,sort_keys=True).encode()).hexdigest()
        if config_hash!=settings['config_hash']:
            raise ValueError('Saved configuration fingerprint differs: '+name)
        if config['dataset_hash']!=plan['dataset_hash'] or config['implementation_hash']!=plan['code_hash']:
            raise ValueError('Plan identity differs: '+name)
        if settings['initial_checkpoint_sha256']!=INITIAL or settings['initial_checkpoint']!=reference['initial_checkpoint']:
            raise ValueError('Initialization identity differs: '+name)
        if settings['loader_workers']!=2 or settings['retain_best_checkpoint'] or settings['diagnostic_only']:
            raise ValueError('Unexpected screening settings: '+name)
        checkpoint=(path/latest['checkpoint']).resolve()
        if not checkpoint.is_relative_to(path.resolve()) or sha(checkpoint)!=latest['checkpoint_sha256']:
            raise ValueError('Checkpoint hash differs: '+name)
        saved=torch.load(checkpoint,map_location='cpu',weights_only=True)
        names={int(k):v for k,v in saved['names'].items()}
        expected_names={int(k):v for k,v in json.loads((ROOT/'manifests/non_chili_v1/class_map.json').read_text())['names'].items()}
        if saved['round']!=latest['round'] or saved['config_hash']!=settings['config_hash'] or names!=expected_names:
            raise ValueError('Checkpoint contents differ: '+name)
        del saved
        if latest['round']< (5 if number<16 else 3):
            raise ValueError('Copied progress is incomplete: '+name)
        metric_records=[]; metric_hashes={}
        for current in range(1,latest['round']+1):
            relative=f'results/metrics/round_{current:03d}.json'
            record=json.loads((path/relative).read_text())
            if record['round']!=current or record['pooled_validation']['images']!=4157:
                raise ValueError('Metric record differs: '+name)
            if number>=6 and record.get('execution_profile',{}).get('sha256')!=execution_digest:
                raise ValueError('Optimized round provenance differs: '+name)
            metric_records.append(record)
            metric_hashes[relative]=sha(path/relative)
        trials[name]=dict(source_account='71gmail' if number<=11 else '4gmail',
            round=latest['round'],checkpoint_sha256=latest['checkpoint_sha256'],
            settings_sha256=sha(path/'settings.json'),config_sha256=sha(cache/'configs'/f'{name}.json'),
            metric_sha256=metric_hashes)
        if latest['round']>=5:
            best=max(metric_records[:5],key=lambda x:x['pooled_validation']['map_50_95'])
            ranked.append(dict(trial_id=name,budget=5,**{k:config[k] for k in ('lr','weight_decay','batch','resolution')},
                best_round=best['round'],**{k:best['pooled_validation'][k] for k in ('precision','recall','f1','map50','map_50_95')}))
    ranked.sort(key=lambda x:(-x['map_50_95'],x['trial_id']))
    REPORT.mkdir(parents=True,exist_ok=True)
    with (REPORT/'screening_results.csv').open('w',newline='') as stream:
        fields=['rank','trial_id','budget','lr','weight_decay','batch','resolution','best_round','precision','recall','f1','map50','map_50_95']
        writer=csv.DictWriter(stream,fieldnames=fields);writer.writeheader()
        for i,row in enumerate(ranked,1):writer.writerow(dict(rank=i,**row))
    marker=dict(version='combined_effdet_re19gmail_v1',account='hannanre19@gmail.com',
        initial_checkpoint_sha256=INITIAL,plan_sha256=sha(cache/'PLAN/plan.json'),
        execution_profile_sha256=sha(cache/'execution_profile.json'),trials=trials,
        completed_screening_trials=len(ranked),gpu_training_started=False)
    (REPORT/'combined_effdet_re19gmail.json').write_text(json.dumps(marker,indent=2)+'\n')
    summary=dict(trials_verified=len(trials),completed_screening_trials=len(ranked),
        trial_16_saved_round=trials['trial_16']['round'],all_checkpoint_bytes_verified=True,
        all_settings_and_configs_verified=True,all_committed_round_metrics_verified=True,
        verified_model_file_count=sum(1 for p in cache.rglob('*') if p.is_file()),training_started=False)
    (REPORT/'validation.json').write_text(json.dumps(summary,indent=2)+'\n')
    print(json.dumps(summary,indent=2))


if __name__=='__main__':
    if '--download' in sys.argv:
        items=json.load(sys.stdin)
        with ThreadPoolExecutor(max_workers=4) as pool:
            results=list(pool.map(download,items))
        print(json.dumps(results))
    else:
        validate()
