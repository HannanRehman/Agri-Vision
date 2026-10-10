"""Summarize saved validation runs and verify every locally saved Drive file."""
import csv
import hashlib
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ARCHIVE = ROOT / 'archive/comparison_download_20261010'
REPORT = ROOT / 'reports/comparison/main_run_review_20261010'
REPORT.mkdir(parents=True, exist_ok=True)

def sha(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()

inventory = json.loads((ROOT / 'reports/comparison/main_runs_drive_inventory_20261010.json').read_text())
verified, missing = [], []
for item in inventory:
    if item['file_or_folder'] != 'file':
        continue
    path = ROOT / 'experiments' / item['path']
    if not path.is_file() or path.stat().st_size != int(item['size']):
        missing.append(dict(id=item['id'], path=item['path'], bytes=int(item['size'])))
        continue
    verified.append(dict(id=item['id'], path=item['path'], bytes=path.stat().st_size, sha256=sha(path), local_path=str(path), parent=item['parent'], url=item['url']))
(ARCHIVE / 'verified_local.json').write_text(json.dumps(verified, indent=2))
(ARCHIVE / 'missing_local.json').write_text(json.dumps(missing, indent=2))

rows = []
details = []
for run in sorted((ROOT / 'experiments').glob('*_non_chili_seed42_20261009_224152_449800')):
    settings = json.loads((run / 'settings.json').read_text())
    latest = json.loads((run / 'latest.json').read_text())
    number = int(Path(latest['best_checkpoint']).stem.split('_')[-1])
    metrics = json.loads((run / f'results/metrics/round_{number:03d}.json').read_text())['pooled_validation']
    row = dict(model=settings['config']['model'], completed_rounds=latest['round'], best_round=number,
               **{key:metrics[key] for key in ['precision','recall','f1','map50','map_50_95']})
    rows.append(row)
    checks = {}
    for label, relative, expected in [('initial', 'checkpoints/initial.pt', settings['initial_checkpoint_sha256']),
                                      ('latest', latest['checkpoint'], latest['checkpoint_sha256'])]:
        path = run / relative
        checks[label] = bool(path.exists() and sha(path) == expected)
    details.append(dict(**row, checkpoint_hash_checks=checks, settings=settings))
with (REPORT / 'validation_summary.csv').open('w', newline='') as stream:
    writer = csv.DictWriter(stream, fieldnames=rows[0].keys())
    writer.writeheader()
    writer.writerows(rows)
snapshot = dict(captured_at_utc=datetime.now(timezone.utc).isoformat(), verified_files=len(verified),
                verified_bytes=sum(x['bytes'] for x in verified), missing_files=len(missing),
                free_f_bytes=shutil.disk_usage(ROOT).free, runs=details)
(REPORT / 'verification.json').write_text(json.dumps(snapshot, indent=2))
table = ['| Model | Saved rounds | Best round | Precision | Recall | F1 | mAP50 | mAP50–95 |',
         '|---|---:|---:|---:|---:|---:|---:|---:|']
for row in rows:
    table.append('| '+row['model']+' | '+str(row['completed_rounds'])+' | '+str(row['best_round'])+' | '+
                 ' | '.join(f"{100*row[key]:.2f}%" for key in ['precision','recall','f1','map50','map_50_95'])+' |')
text = '# Agri Vision comparison: saved validation results\n\n'
text += f"Snapshot: {snapshot['captured_at_utc']}.\n\n"+'\n'.join(table)+'\n\n'
text += ('These scores come from the same 4,157-image pooled validation split with 24 non-chili classes. '
         'They are not held-out test scores. Best checkpoints are selected by validation mAP50–95. '
         'Precision/recall use confidence 0.25 and class-aware IoU 0.5; AP sweeps scores above 0.001.\n\n'
         'One local epoch means each client visits its training split once in each global round. '
         'The completed models ran 10 global rounds, each with three clients and one local epoch.\n\n'
         'EfficientDet committed round 2 and stopped during round 3 pooled validation. '
         'The cause of the Colab disconnect is unconfirmed. Resume with the identical code, settings, '
         'package versions and Tesla T4; round 3 will repeat from the round-2 checkpoint. '
         'Keep the total budget at 10 rounds. The EfficientDet-only resume notebook is prepared.\n\n'
         'YOLO is abnormal: all ten rounds have zero true positives and zero false positives at confidence 0.25. '
         'This means no predictions survived that cutoff. Its nonzero AP comes from lower-confidence predictions. '
         'Do not use this run to claim YOLO is inherently inferior. The custom YOLO training/initialization and '
         'prediction path need investigation. A lower reporting cutoff alone will not repair its very low AP.\n\n'
         'SSDLite improved from 10.98% to 39.06% mAP50–95. SSD–MobileNetV3 improved from 1.18% to 30.30%. '
         'EfficientDet improved from 29.93% to 46.68% in two committed rounds. Its lead is provisional until '
         'the common round budget is completed and selected checkpoints are evaluated on the held-out test set.\n\n'
         'The experiment shares data, clients, seed, round budget, augmentation and evaluator. Input sizes, '
         'optimizers, batch sizes and pretraining/head initialization differ by model. These are practical '
         'training configurations, not a controlled architecture-only ablation.\n\n')
text += f"Local backup: {len(verified)} files verified by byte size and SHA-256; {len(missing)} files still missing or incomplete. "
text += 'Private transfer URLs and the cleanup audit stay in the ignored local archive directory.\n'
(REPORT / 'README.md').write_text(text, encoding='utf-8')
print(json.dumps({key:snapshot[key] for key in ['verified_files','verified_bytes','missing_files','free_f_bytes']}))
print(json.dumps([{k:v for k,v in x.items() if k!='settings'} for x in details]))
