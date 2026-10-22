"""Package the benchmark-backed EfficientDet execution extension and its notebook."""
import ast
import hashlib
import json
from pathlib import Path
import zipfile

ROOT = Path(__file__).resolve().parents[1]
REPORT = ROOT/'reports/comparison/efficientdet_fp32_continuation_20261019'
REPORT.mkdir(parents=True, exist_ok=True)
evidence = ROOT/'reports/comparison/efficientdet_fp32_benchmark_20261019/gpu_benchmark.json'
files = {f'continuation/{p.name}': p.read_bytes() for p in (ROOT/'continuation').glob('*.py')}
files.update({'benchmarks/__init__.py': (ROOT/'benchmarks/__init__.py').read_bytes(),
              'benchmarks/efficientdet_fp32.py': (ROOT/'benchmarks/efficientdet_fp32.py').read_bytes(),
              'continuation/benchmark_evidence.json': evidence.read_bytes()})
bundle = REPORT/'efficientdet_fp32_continuation_v1.zip'
with zipfile.ZipFile(bundle, 'w', zipfile.ZIP_DEFLATED) as archive:
    for name, data in sorted(files.items()):
        info = zipfile.ZipInfo(name, date_time=(2026, 10, 19, 0, 0, 0))
        info.compress_type = zipfile.ZIP_DEFLATED
        archive.writestr(info, data)
checksum = hashlib.sha256(bundle.read_bytes()).hexdigest()
path = ROOT/'AgriVision_GridSearch_EfficientDet_71gmail.ipynb'
notebook = json.loads(path.read_text(encoding='utf-8'))
notebook['cells'][0]['source'] = ['# Agri Vision: EfficientDet optimized FP32 continuation - 71gmail\n', '\n',
    'Use rehmanfx71@gmail.com, T4 GPU, then Run all. Keep 4gmail stopped. Five screening trials are complete; continue trial 6 from its last committed round (round 1 if none saved). This is the production training notebook, not the benchmark.\n', '\n',
    'Only the three benchmarked FP32 overhead reductions are enabled. Original data, settings, precision, validation, checkpoint hashes and FedAvg remain unchanged. A separate execution profile fingerprints the extension and benchmark evidence; every new round records its version. All remaining screening, finalist confirmation and final training stages continue automatically. Earlier rounds retain original execution provenance.\n']
for cell in notebook['cells']:
    if cell['cell_type'] != 'code':
        continue
    text = ''.join(cell['source'])
    if cell.get('id') == 'verify-migration-71':
        text = text.replace('range(1, 5)', 'range(1, 6)')
    if 'command = [sys.executable' in text and ('"tuning.search"' in text or '"continuation.search"' in text):
        text = text.replace('"tuning.search"', '"continuation.search"')
        prefix = f'''extension = DRIVE_ROOT / "efficientdet_fp32_continuation_v1.zip"
assert sha(extension) == "{checksum}", "Wrong optimized continuation bundle"
with zipfile.ZipFile(extension) as archive:
    for name in archive.namelist():
        assert (WORKSPACE / name).resolve().is_relative_to(WORKSPACE.resolve())
        assert name.startswith(("continuation/", "benchmarks/")), "Extension must not replace original training modules"
    archive.extractall(WORKSPACE)
print("Verified optimized FP32 extension; continuing existing EfficientDet search.", flush=True)
'''
        # Regeneration replaces an existing prefix rather than duplicating it.
        text = text[text.index('command = [sys.executable'):]
        text = prefix + text
    cell['source'] = text.splitlines(keepends=True)
    cell['outputs'] = []
    cell['execution_count'] = None
    ast.parse(text)
path.write_text(json.dumps(notebook, indent=2, ensure_ascii=False)+'\n', encoding='utf-8')
record = dict(bundle=bundle.name, sha256=checksum, files={name: hashlib.sha256(data).hexdigest() for name, data in files.items()},
              benchmark_evidence_sha256=hashlib.sha256(evidence.read_bytes()).hexdigest(),
              original_bundle_sha256=hashlib.sha256((ROOT/'agrivision_grid_search_v1_code.zip').read_bytes()).hexdigest(),
              gpu_training_started=False)
assert record['original_bundle_sha256'] == '5453a56a97c2aca78224cec346c259e5f8a0a5eda2a93eb17b3677f06689c049'
(REPORT/'package.json').write_text(json.dumps(record, indent=2)+'\n', encoding='utf-8')
print(json.dumps(record, indent=2))
