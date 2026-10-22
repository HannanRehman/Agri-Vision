"""Package the separate benchmark without changing the existing search bundle."""
import ast
import hashlib
import json
from pathlib import Path
import zipfile

ROOT = Path(__file__).resolve().parents[1]
REPORT = ROOT/'reports/comparison/efficientdet_fp32_benchmark_20261019'
REPORT.mkdir(parents=True, exist_ok=True)
BUNDLE = REPORT/'efficientdet_fp32_benchmark_v1.zip'
with zipfile.ZipFile(BUNDLE, 'w', zipfile.ZIP_DEFLATED) as archive:
    for name in ('benchmarks/__init__.py', 'benchmarks/efficientdet_fp32.py', 'tests/test_effdet_benchmark.py'):
        info = zipfile.ZipInfo(name, date_time=(2026, 10, 19, 0, 0, 0))
        info.compress_type = zipfile.ZIP_DEFLATED
        archive.writestr(info, (ROOT/name).read_bytes())
digest = hashlib.sha256(BUNDLE.read_bytes()).hexdigest()
original = json.loads((ROOT/'AgriVision_GridSearch_EfficientDet_71gmail.ipynb').read_text(encoding='utf-8'))
cells = original['cells'][:4]
cells[0]['source'] = ['# Agri Vision: EfficientDet FP32 speed benchmark\n', '\n',
    'Open as rehmanfx71@gmail.com with T4 GPU. Keep the training search stopped. Run all. This performs short engineering measurements on training-only images, checks equivalent losses and parameter updates, and writes a separate benchmark report. It does not resume trials, save trained weights, or select hyperparameters.\n', '\n',
    'Both loops remain full precision. Batch sizes 2/4 and resolutions 512/640 are tested three times with alternating order. Cached identical CPU batches isolate training-step overhead; data decoding, validation, FedAvg and Drive latency are excluded from the speed estimate.\n']
source = ''.join(cells[1]['source']).replace('/content/agrivision_grid_v1', '/content/agrivision_effdet_benchmark')
source = source.replace('OUTPUT = DRIVE_ROOT / "grid_search" / "grid_v1_20261017"',
                        'MODEL_FOLDER = DRIVE_ROOT / "grid_search/grid_v1_20261017/efficientdet_d0"\nOUTPUT = DRIVE_ROOT / "benchmark_results/efficientdet_fp32_v1"')
cells[1]['source'] = source.splitlines(keepends=True)
source = ''.join(cells[3]['source'])
source = '\n'.join(line for line in source.splitlines() if '"unittest"' not in line)+'\n'
cells[3]['source'] = source.splitlines(keepends=True)
extra = f'''benchmark_zip = DRIVE_ROOT / "efficientdet_fp32_benchmark_v1.zip"
assert sha(benchmark_zip) == "{digest}", "Wrong benchmark bundle"
with zipfile.ZipFile(benchmark_zip) as bundle:
    for name in bundle.namelist():
        assert (WORKSPACE / name).resolve().is_relative_to(WORKSPACE.resolve())
    bundle.extractall(WORKSPACE)
command = [sys.executable, "-u", "-m", "benchmarks.efficientdet_fp32",
           "--root", str(WORKSPACE), "--model-folder", str(MODEL_FOLDER),
           "--output", str(OUTPUT), "--steps", "30", "--warmup", "3", "--repeats", "3"]
with subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                      text=True, bufsize=1, env=dict(os.environ, PYTHONUNBUFFERED="1")) as process:
    for line in process.stdout:
        print(line, end="", flush=True)
    code = process.wait()
if code:
    raise subprocess.CalledProcessError(code, command)
print("Benchmark complete. Report:", OUTPUT / "benchmark.json")
print("No search migration or full training was started.")
'''
cells.append(dict(cell_type='code', metadata={}, source=extra.splitlines(keepends=True),
                  outputs=[], execution_count=None, id='benchmark-only'))
cells.append(dict(cell_type='markdown', metadata={}, source=[
    'Review median speedups, memory and equivalence for every batch/resolution pair. Do not infer full-round savings directly from cached-batch measurements. No benefit, inconsistent timings or failed equivalence means do not migrate. A worthwhile result still requires an explicit reviewed continuation with versioned code, rather than editing stored hashes or trial identities.\n'], id='interpretation'))
for cell in cells:
    if cell['cell_type'] == 'code':
        cell['outputs'] = []
        cell['execution_count'] = None
        ast.parse(''.join(cell['source']))
original['cells'] = cells
name = 'AgriVision_EfficientDet_FP32_Benchmark_71gmail.ipynb'
original['metadata']['colab']['name'] = name
(ROOT/name).write_text(json.dumps(original, indent=2, ensure_ascii=False)+'\n', encoding='utf-8')
report = dict(bundle=BUNDLE.name, sha256=digest, notebook=name, gpu_benchmark_run=False,
              existing_search_bundle_unchanged=hashlib.sha256((ROOT/'agrivision_grid_search_v1_code.zip').read_bytes()).hexdigest()
              == '5453a56a97c2aca78224cec346c259e5f8a0a5eda2a93eb17b3677f06689c049')
assert report['existing_search_bundle_unchanged']
(REPORT/'package.json').write_text(json.dumps(report, indent=2)+'\n', encoding='utf-8')
print(json.dumps(report, indent=2))
