"""Build isolated grid-search bundle and three auto-resume Colab notebooks; never train."""
import ast
import hashlib
import json
from pathlib import Path
import zipfile

ROOT = Path(__file__).resolve().parents[1]


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def cell(source, code=False):
    item = dict(cell_type='code' if code else 'markdown', metadata={}, source=source.splitlines(keepends=True))
    if code:
        ast.parse(source)
        item.update(outputs=[], execution_count=None)
    return item


def main():
    files = [*sorted((ROOT/'comparison').glob('*.py')), *sorted((ROOT/'tuning').glob('*.py')),
             *sorted((ROOT/'configs/comparison').glob('*.json')), ROOT/'configs/tuning/grid_v1.json',
             ROOT/'requirements-grid.txt', ROOT/'scripts/yolo_native_train.py',
             ROOT/'scripts/continue_comparison.py', ROOT/'tests/test_grid_search.py']
    archive = ROOT/'agrivision_grid_search_v1_code.zip'
    with zipfile.ZipFile(archive, 'w', zipfile.ZIP_DEFLATED) as bundle:
        for path in files:
            bundle.write(path, path.relative_to(ROOT).as_posix())
    with zipfile.ZipFile(archive) as bundle:
        assert bundle.testzip() is None
    dataset = json.loads((ROOT/'reports/non_chili_v1/upload_package.json').read_text())
    jobs = [('19gmail', ['yolo_v8n'], 'AgriVision_GridSearch_YOLO.ipynb'),
            ('4gmail', ['efficientdet_d0'], 'AgriVision_GridSearch_EfficientDet.ipynb'),
            ('40gmail', ['ssdlite'], 'AgriVision_GridSearch_SSD_Pair.ipynb')]
    notebooks = []
    extraction = '''from google.colab import drive
drive.mount("/content/drive")
import hashlib, json, shutil, zipfile

def sha(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()

WORKSPACE.mkdir(parents=True, exist_ok=True)
dataset_zip = WORKSPACE.parent / "non_chili_v1_colab_dataset.zip"
source_zip = DRIVE_ROOT / dataset_zip.name
if source_zip.is_file():
    if not dataset_zip.is_file() or sha(dataset_zip) != EXPECTED_DATASET_SHA256:
        shutil.copyfile(source_zip, dataset_zip)
else:
    if not dataset_zip.is_file() or sha(dataset_zip) != EXPECTED_DATASET_SHA256:
        parts_dir = DRIVE_ROOT / "non_chili_v1_upload_parts"
        manifest = json.loads((parts_dir / "parts.json").read_text())
        assert manifest["joined_sha256"] == EXPECTED_DATASET_SHA256, "Wrong dataset parts manifest"
        specs = manifest["parts"]
        assert specs and [p["name"] for p in specs] == [dataset_zip.name + f".part{i:02d}" for i in range(1, len(specs)+1)]
        partial = dataset_zip.with_suffix(".partial")
        with partial.open("wb") as output:
            for index, spec in enumerate(specs, 1):
                path = parts_dir / spec["name"]
                assert path.is_file() and path.stat().st_size == spec["bytes"], f"Missing/damaged part {index}"
                with path.open("rb") as stream:
                    shutil.copyfileobj(stream, output, 16*1024*1024)
                print(f"Dataset part {index}/{len(specs)} copied", flush=True)
        assert sha(partial) == EXPECTED_DATASET_SHA256, "Dataset parts damaged"
        partial.replace(dataset_zip)
assert sha(dataset_zip) == EXPECTED_DATASET_SHA256
dataset_marker = WORKSPACE / ".dataset_verified"
if not dataset_marker.exists() or dataset_marker.read_text() != EXPECTED_DATASET_SHA256:
    with zipfile.ZipFile(dataset_zip) as bundle:
        for name in bundle.namelist():
            assert (WORKSPACE/name).resolve().is_relative_to(WORKSPACE.resolve())
        bundle.extractall(WORKSPACE)
    dataset_marker.write_text(EXPECTED_DATASET_SHA256)
code_zip = WORKSPACE.parent / "agrivision_grid_search_v1_code.zip"
shutil.copyfile(DRIVE_ROOT/code_zip.name, code_zip)
assert sha(code_zip) == EXPECTED_CODE_SHA256, "Wrong grid-search code bundle"
with zipfile.ZipFile(code_zip) as bundle:
    for name in bundle.namelist():
        assert (WORKSPACE/name).resolve().is_relative_to(WORKSPACE.resolve())
    bundle.extractall(WORKSPACE)
print("Verified local dataset and isolated grid-search code.")
'''
    environment = '''import os, subprocess, sys
os.chdir(WORKSPACE)
sys.path.insert(0, str(WORKSPACE))
import torch, torchvision
assert torch.cuda.is_available(), "Select Runtime > Change runtime type > T4 GPU"
assert "T4" in torch.cuda.get_device_name(0), "Use T4 for the comparison"
torch_before, vision_before = torch.__version__, torchvision.__version__
print("GPU:", torch.cuda.get_device_name(0), "torch:", torch_before, "torchvision:", vision_before)
subprocess.run([sys.executable, "-m", "pip", "install", "--no-deps", "-r", "requirements-grid.txt"], check=True)
# Preserve the existing CUDA packages while installing supporting libraries.
subprocess.run([sys.executable, "-m", "pip", "install", "huggingface_hub", "safetensors", "packaging", "tqdm", "omegaconf==2.4.0",
                "matplotlib", "opencv-python", "requests", "scipy", "psutil", "polars>=0.20", "ultralytics-thop>=2.0.18"], check=True)
assert torch.__version__ == torch_before and torchvision.__version__ == vision_before
subprocess.run([sys.executable, "-m", "unittest", "discover", "-s", "tests", "-p", "test_grid_search.py", "-v"], check=True)
'''
    for account, models, name in jobs:
        intro = f'''# Agri Vision: hyperparameter grid search — {account}

This notebook runs **{', '.join(models)}** only. Open using the assigned account, select **T4 GPU**, then **Run all**. The first execution checks actual batch/resolution compatibility on a few training images, then automatically runs the search and final training. It stops after the assigned model completes; SSD-MobileNetV3 is assigned separately to 78gmail.

Each model has **16 combinations**: two learning rates × two weight decays × two batch sizes × two resolutions. Every combination starts from the same cached COCO-transfer initialization for its architecture. Each trains five full federated rounds; the top two continue to ten for confirmation. The selected configuration then starts a **fresh 30-round run** from the original cached initialization. Selection uses pooled validation mAP50–95. Test images are not used.

The completed baseline runs and their code ZIP remain separate. Reuse the existing dataset parts. Keep the uploaded grid code/settings unchanged to resume. A partial round repeats; committed rounds/trials are skipped. Colab GPU availability is variable.
'''
        settings = f'''from pathlib import Path
DRIVE_ROOT = Path("/content/drive/MyDrive/Agri Vision")
WORKSPACE = Path("/content/agrivision_grid_v1")
OUTPUT = DRIVE_ROOT / "grid_search" / "grid_v1_20261017"
MODELS = {models!r}
WORKERS = 2
EXPECTED_DATASET_SHA256 = "{dataset['sha256']}"
EXPECTED_CODE_SHA256 = "{sha(archive)}"
'''
        cells = [cell(intro), cell(settings, True), cell(extraction, True), cell(environment, True),
                 cell('''## Automatic search, confirmation and final training
Each model's `preflight.json` records batch/resolution compatibility. `screening_results.csv` ranks all sixteen five-round trials, and `finalist_results.csv` ranks the two ten-round finalists. `selected_configuration.json` identifies the winner. The runner then trains that configuration from scratch through round 30 under `final_training/`, retaining validation-best and latest weights.
'''), cell('''subprocess.run([sys.executable, "-m", "tuning.search", "--root", str(WORKSPACE),
                "--output", str(OUTPUT), "--models", *MODELS, "--workers", str(WORKERS)], check=True)
print("Completed models:", MODELS)
print("Results:", OUTPUT)
''', True), cell('''## Resume and result locations
If the session stops, reopen this same notebook on the same account, select T4 and Run all. No changes to settings or model order are needed. The current phase is recorded in `grid_search/grid_v1_20261017/search_status.json`; each trial has its own status and committed `latest.json`. Only the assigned model runs; there is no second model queued on 40gmail.

Trial validation CSVs and per-class round JSONs are retained. Screening stores latest-only weights to conserve Drive space; final training retains best and latest. GPU preflight weights are discarded. No held-out testing is performed. The five-/ten-round selection is budgeted screening, not proof of the best possible thirty-round configuration.
''')]
        for index, item in enumerate(cells):
            item['id'] = f'grid-v1-{index:02d}'
        notebook = dict(nbformat=4, nbformat_minor=5, cells=cells,
                        metadata=dict(kernelspec=dict(display_name='Python 3', language='python', name='python3'),
                                      language_info=dict(name='python'), colab=dict(name=name), accelerator='GPU'))
        destination = ROOT/name
        destination.write_text(json.dumps(notebook, indent=2, ensure_ascii=False)+'\n', encoding='utf-8')
        notebooks.append(dict(account=account, models=models, file=name, bytes=destination.stat().st_size,
                              sha256=sha(destination)))
    report = dict(code_bundle=archive.name, code_bytes=archive.stat().st_size, code_sha256=sha(archive),
                  dataset_sha256=dataset['sha256'], notebooks=notebooks, gpu_training_started=False)
    destination = ROOT/'reports/comparison/grid_search_20261017'
    destination.mkdir(parents=True, exist_ok=True)
    (destination/'package.json').write_text(json.dumps(report, indent=2)+'\n', encoding='utf-8')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
