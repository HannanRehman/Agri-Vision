"""Build and verify the upload package for the round-35 continuation."""
import ast
import hashlib
import json
from pathlib import Path
import zipfile

ROOT = Path(__file__).resolve().parents[1]


def sha256(path):
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def main():
    dataset = ROOT / 'original_dataset_colab.zip'
    checkpoint = ROOT / 'saved_models/global_round_035.pt'
    import sys
    sys.path.insert(0, str(ROOT))
    import torch
    from model import class_names, create_model
    saved = torch.load(checkpoint, map_location='cpu', weights_only=True)
    assert saved['round'] == 35 and saved['class_names'] == class_names()
    model = create_model()
    model.model.load_state_dict(saved['model_state_dict'], strict=True)
    print('Verified round-35 checkpoint architecture and classes.', flush=True)
    dataset_hash = sha256(dataset)
    assert dataset_hash == 'a21b497dbced6a8797e3f10fa4af6baafad734533417b37fff6f149deabc0827'
    expected_counts = {'train': 15512, 'valid': 4374, 'test': 2191}
    with zipfile.ZipFile(dataset) as archive:
        counts = {split: sum(len(Path(n).parts) == 4 and Path(n).parts[:3] == ('original_dataset', split, 'images')
                             and Path(n).suffix.lower() in {'.jpg', '.jpeg', '.png', '.bmp', '.webp'}
                             for n in archive.namelist()) for split in expected_counts}
    assert counts == expected_counts
    files = ['model.py', 'federated_client.py', 'server.py', 'split_dataset.py', 'colab_train.py',
             'requirements.txt', 'predict.py', 'evaluate.py', 'client1.py', 'client2.py', 'client3.py',
             'yolov8n.pt', 'original_dataset/data.yaml', 'dataset/client1/data.yaml',
             'dataset/client2/data.yaml', 'dataset/client3/data.yaml']
    bundle = ROOT / 'agrivision_colab_code.zip'
    with zipfile.ZipFile(bundle, 'w', compression=zipfile.ZIP_DEFLATED) as archive:
        for name in files:
            data = (ROOT / name).read_bytes()
            entry = zipfile.ZipInfo(name, date_time=(2026, 1, 1, 0, 0, 0))
            entry.compress_type = zipfile.ZIP_DEFLATED
            archive.writestr(entry, data)
    with zipfile.ZipFile(bundle) as archive:
        assert archive.testzip() is None
        assert all(archive.read(name) == (ROOT / name).read_bytes() for name in files)
    hashes = {'dataset': dataset_hash, 'code': sha256(bundle), 'checkpoint': sha256(checkpoint)}
    notebook_path = ROOT / 'AgriVision_Colab.ipynb'
    notebook = json.loads(notebook_path.read_text(encoding='utf-8'))
    def set_cell(index, source):
        notebook['cells'][index]['source'] = source.splitlines(keepends=True)
    set_cell(0, '''# Agri Vision — continue rounds 36–40

Upload `original_dataset_colab.zip`, the newly rebuilt `agrivision_colab_code.zip`, and `saved_models/global_round_035.pt` to **MyDrive/Agri Vision/**. The checkpoint must be directly in that Drive folder as `global_round_035.pt`. The existing dataset ZIP can be reused if its checksum matches. Replace the old code ZIP with the new bundle.

Select **Runtime → Change runtime type → GPU**, then **Runtime → Run all** and approve Drive access. This notebook runs **five additional rounds (36–40)** from round 35, with the same 30-class model, dataset, three-client FedAvg, one local epoch, batch 8, image size 640, and two workers. Training hyperparameters are retained for a comparable continuation; optimizer/warmup behaviour has not been changed.

All earlier result folders are preserved. Each new run has a unique folder. Completed global rounds can be resumed; an interrupted partial round must be repeated. Per-round global and individual-client validation metrics, sample counts, a combined CSV, local training outputs, and package versions are saved to Drive. The held-out test split is not evaluated by this notebook.
''')
    set_cell(1, f'''from pathlib import Path
ROUNDS = 5
LOCAL_EPOCHS = 1
BATCH = 8
IMGSZ = 640
WORKERS = 2
EXPECTED_START_ROUND = 35
EXPECTED_SPLIT_COUNTS = {{'train': 15512, 'valid': 4374, 'test': 2191}}
EXPECTED_DATASET_SHA256 = '{hashes['dataset']}'
EXPECTED_CODE_SHA256 = '{hashes['code']}'
EXPECTED_CHECKPOINT_SHA256 = '{hashes['checkpoint']}'
DATASET_ZIP = '/content/drive/MyDrive/Agri Vision/original_dataset_colab.zip'
CODE_ZIP = '/content/drive/MyDrive/Agri Vision/agrivision_colab_code.zip'
OUTPUT_ROOT = '/content/drive/MyDrive/Agri Vision/colab_results'
# Required starting point for this rounds 36–40 continuation.
INITIAL_CHECKPOINT = '/content/drive/MyDrive/Agri Vision/global_round_035.pt'
''')
    set_cell(2, ''.join(notebook['cells'][2]['source']).replace('round-25 checkpoint', 'round-35 checkpoint'))
    set_cell(7, '''## Train rounds 36–40 and save to Drive

Each round trains all three clients from the same global weights, aggregates their updates, then validates the global model on all three validation shards. Watch for `ROUND 40 validation` and `COMPLETE`. Checkpoints are saved before validation; use `status.json` and the metric files to confirm a fully completed round. The CSV and per-client JSON files are saved after validation finishes.
''')
    set_cell(9, '''## Review the completed five rounds

The next cell checks that all rounds 36–40 have checkpoints and saved validation scores, displays the combined CSV, and identifies the strongest new checkpoint by each mAP metric. The final checkpoint is not automatically the best. Compare these results with your rounds 26–35 report using validation scores. Preserve every run folder; test evaluation remains a separate step.
''')
    review = '''import csv
expected_rounds = list(range(EXPECTED_START_ROUND + 1, EXPECTED_START_ROUND + ROUNDS + 1))
status = json.loads((RUN / 'status.json').read_text())
if status.get('phase') != 'complete' or status.get('round') != expected_rounds[-1]:
    raise RuntimeError(f'Run did not complete: {status}')
for round_number in expected_rounds:
    for filename in (f'global_round_{round_number:03d}.pt', f'metrics_round_{round_number:03d}.json', f'metrics_clients_round_{round_number:03d}.json'):
        if not (RUN / filename).is_file():
            raise FileNotFoundError(RUN / filename)
with (RUN / 'global_validation_metrics.csv').open(newline='') as stream:
    rows = list(csv.DictReader(stream))
if [int(row['round']) for row in rows] != expected_rounds:
    raise ValueError('Validation CSV does not cover every expected round')
print('Verified completed rounds:', expected_rounds)
print('round | precision | recall | mAP50 | mAP50-95')
for row in rows:
    print(row['round'], '|', ' | '.join(f"{float(row[key])*100:.2f}%" for key in ('precision', 'recall', 'mAP50', 'mAP50_95')))
for key in ('mAP50', 'mAP50_95'):
    best = max(rows, key=lambda row: float(row[key]))
    print(f"Best NEW round by {key}: {best['round']} ({float(best[key])*100:.2f}%)")
print('Download the entire run folder plus the sibling .log and _input_sha256.json files.')
print('Run folder:', RUN)
print('Log:', log_path)
'''
    if len(notebook['cells']) == 10:
        notebook['cells'].append({'cell_type': 'code', 'execution_count': None, 'metadata': {}, 'outputs': [], 'source': []})
    set_cell(10, review)
    for cell in notebook['cells']:
        if cell['cell_type'] == 'code':
            cell['outputs'] = []
            cell['execution_count'] = None
            ast.parse(''.join(cell['source']))
    notebook_path.write_text(json.dumps(notebook, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')
    manifest = {'start_round': 35, 'end_round': 40, 'additional_rounds': 5, 'hashes': hashes,
                'files': {'dataset': 'original_dataset_colab.zip', 'code': 'agrivision_colab_code.zip',
                          'checkpoint': 'saved_models/global_round_035.pt', 'notebook': 'AgriVision_Colab.ipynb'}}
    (ROOT / 'reports/rounds_036_040_upload_manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    print(json.dumps(manifest, indent=2))


if __name__ == '__main__':
    main()
