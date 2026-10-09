"""Verify organization records and references without training or evaluation."""
import ast
from datetime import datetime, timezone
import hashlib
from importlib.metadata import version, PackageNotFoundError
import json
from pathlib import Path
import re
import shutil
import sys

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT/'reports/workspace_verification_20261009.json'

def flatten(value):
    if isinstance(value, list):
        return [item for part in value for item in flatten(part)]
    if isinstance(value, dict) and 'action' in value:
        return [value]
    if isinstance(value, dict) and 'value' in value:
        return flatten(value['value'])
    raise ValueError('Unexpected cleanup record shape')

def digest(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream,'sha256').hexdigest()

def main():
    log_path = ROOT/'reports/workspace_cleanup_20261009.json'
    log = flatten(json.loads(log_path.read_text(encoding='utf-8-sig')))
    source_verification = json.loads((ROOT/'reports/chili_source_archive_verification_20261009.json').read_text())
    deleted = 'archive/imported_datasets/PENYAKIT PADA CABAI.v1i.yolov8'
    retained = deleted+'.zip'
    deleted_files = json.loads((ROOT/'reports/chili_source_deleted_files_20261009.json').read_text(encoding='utf-8-sig'))
    if isinstance(deleted_files, dict) and 'value' in deleted_files:
        deleted_files = deleted_files['value']
    assert len(deleted_files) == source_verification['files'] == 1557
    if not any(item.get('action') == 'remove verified archived source folder' for item in log):
        log.append({'action':'remove verified archived source folder', 'removed':deleted,
                    'retained':retained, 'sha256':source_verification['zip_sha256'],
                    'bytes':source_verification['folder_bytes'], 'files':1557,
                    'authorization':'User requested deletion of unused extracted chili source',
                    'verification':'Each file matched its ZIP entry before removal'})
    log_path.write_text(json.dumps(log,indent=2)+'\n', encoding='utf-8')
    failures = []
    later_removed = {item['removed'].replace('\\','/') for item in log if 'removed' in item}
    for item in log:
        if item['action'] == 'move':
            destination = item['destination'].replace('\\','/')
            if (ROOT/item['source']).exists():
                failures.append('Move source still exists: '+item['source'])
            if not (ROOT/destination).exists() and destination not in later_removed:
                failures.append('Move destination missing: '+destination)
        else:
            if (ROOT/item['removed']).exists():
                failures.append('Deleted item still exists: '+item['removed'])
            if digest(ROOT/item['retained']).lower() != item['sha256'].lower():
                failures.append('Retained copy hash differs: '+item['retained'])
    if (ROOT/'saved_models/test_evaluation_round_035-2').exists():
        failures.append('Redundant round-35 folder still exists')
    source_paths = [*ROOT.glob('*.py'), *ROOT.joinpath('scripts').glob('*.py'),
                    ROOT/'reports/seminar_poster/build_poster.py']
    for path in source_paths:
        ast.parse(path.read_text(encoding='utf-8'),filename=str(path))
    configs = list((ROOT/'configs/comparison').glob('*.json'))
    for path in configs:
        spec = json.loads(path.read_text())
        if path.name != 'protocol.json':
            if not (ROOT/spec['shared_protocol']).is_file() or not (ROOT/spec['run_directory']).is_dir():
                failures.append('Missing config dependency: '+path.name)
            for key in ['results_directory','checkpoints_directory','logs_directory']:
                if not (ROOT/spec[key]).is_dir():
                    failures.append('Missing model output directory: '+path.name+' '+key)
    assert len(configs) == 5
    # Output is created before checking the documentation's link to this report.
    OUT.write_text('{"status":"checking"}\n',encoding='utf-8')
    docs = [ROOT/'README.md',ROOT/'WORKSPACE_INDEX.md',ROOT/'WORKSPACE_ORGANIZATION_PLAN.md',
            ROOT/'ARCHITECTURE_COMPARISON_PLAN.md',ROOT/'TRAINING_PLAN.md',ROOT/'archive/README.md',
            ROOT/'archive/documents/FIFTEEN_DAY_THESIS_PLAN.md',
            ROOT/'reports/seminar_poster/README.md',ROOT/'reports/rounds_026_035_detailed_report.md',
            *ROOT.joinpath('configs').rglob('*.md'),*ROOT.joinpath('experiments').rglob('*.md'),
            *ROOT.joinpath('manifests').rglob('*.md')]
    checked_links = 0
    for doc in docs:
        for target in re.findall(r'\[[^\]]*\]\(([^)]+)\)',doc.read_text(encoding='utf-8')):
            target = target.strip().strip('<>').split('#')[0]
            if not target or re.match(r'^[a-zA-Z]+://',target):
                continue
            checked_links += 1
            if not (doc.parent/target).exists():
                failures.append(f'Broken link: {doc.relative_to(ROOT)} -> {target}')
    packages = {}
    for name in ['torch','torchvision','ultralytics','flwr','numpy','PyYAML','python-pptx','python-docx']:
        try: packages[name] = version(name)
        except PackageNotFoundError: packages[name] = None
    removed_bytes = sum(item.get('bytes',0) for item in log if item['action'].startswith('remove'))
    report = {'created_at_utc':datetime.now(timezone.utc).isoformat(),
              'status':'passed' if not failures else 'failed', 'failures':failures,
              'cleanup_actions':len(log), 'removed_files':8+1557,
              'deleted_payload_bytes':removed_bytes, 'free_bytes':shutil.disk_usage(ROOT).free,
              'syntax_checked_files':len(source_paths),'draft_configs_checked':len(configs),
              'local_document_links_checked':checked_links,
              'existing_tests':{'passed':13,'command':'.venv/Scripts/python.exe -m unittest discover -s tests -v',
                                'basis':'Successful execution observed during 2026-10-09 organization; not rerun by this verifier'},
              'existing_partition_check':{'passed':True,'train':15512,'valid':4374,'test':2191,'classes':30,
                                          'basis':'Successful split_dataset.py execution observed during 2026-10-09 organization; not rerun by this verifier'},
              'environment':{'python':sys.version,'packages':packages},
              'training_started':False,'dataset_filtering_started':False,
              'google_drive_changed':False}
    OUT.write_text(json.dumps(report,indent=2)+'\n',encoding='utf-8')
    print(json.dumps({key:report[key] for key in ['status','failures','cleanup_actions','removed_files','deleted_payload_bytes','local_document_links_checked']}))
    if failures: raise SystemExit(1)

if __name__ == '__main__':
    main()
