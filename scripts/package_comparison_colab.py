"""Build a distinct code bundle and GPU pilot/main notebook for the 24-class comparison."""
import hashlib
import json
from pathlib import Path
import zipfile

ROOT = Path(__file__).resolve().parents[1]


def digest(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream,'sha256').hexdigest()


def main():
    dataset = json.loads((ROOT/'reports/non_chili_v1/upload_package.json').read_text())
    files = [*sorted((ROOT/'comparison').glob('*.py')),*sorted((ROOT/'configs/comparison').glob('*.json')),
             ROOT/'requirements-comparison.txt',ROOT/'scripts/check_comparison_adapters.py',
             ROOT/'tests/test_comparison.py']
    destination = ROOT/'agrivision_comparison_code.zip'
    with zipfile.ZipFile(destination,'w',compression=zipfile.ZIP_DEFLATED) as archive:
        for path in files:
            archive.write(path,path.relative_to(ROOT).as_posix())
    with zipfile.ZipFile(destination) as archive:
        if archive.testzip() is not None:
            raise ValueError('Code bundle CRC check failed')
    code_hash = digest(destination)
    def markdown(text):
        return dict(cell_type='markdown',metadata={},source=text.splitlines(keepends=True))
    def code(text):
        return dict(cell_type='code',execution_count=None,metadata={},outputs=[],source=text.splitlines(keepends=True))
    cells = [markdown('''# Agri Vision: 24-class detector comparison

Upload **non_chili_v1_colab_dataset.zip** and **agrivision_comparison_code.zip** to **MyDrive/Agri Vision/**.
Alternatively, upload **non_chili_v1_colab_dataset.zip.part01** and **non_chili_v1_colab_dataset.zip.part02** in place of the dataset ZIP. The notebook joins the two parts on Colab's local disk and verifies the original ZIP hash.
For connector uploads, keep the smaller parts and **parts.json** in **MyDrive/Agri Vision/non_chili_v1_upload_parts/**. The notebook reads the manifest and joins every listed part automatically.
Choose a GPU runtime (use the same GPU type for all main runs). Default execution runs an **SSDLite GPU engineering pilot**, not all main experiments.
Main runs require the pilot completion record from this notebook. Each model starts fresh from pretrained weights. Existing 30-class results are preserved.
'''),code(f'''from pathlib import Path
MODE = "pilot"  # Set "main" only after reviewing the completed GPU pilot.
MODELS = ["yolo_v8n", "ssdlite", "ssd_mobilenetv3", "efficientdet_d0"]
ROUNDS = 10
LOCAL_EPOCHS = 1
SEED = 42
WORKERS = 2
PROFILE_IMAGES = 30
# Defaults: batch 8 for YOLO/SSD/SSDLite; batch 4 for EfficientDet-D0.
EXPECTED_DATASET_SHA256 = "{dataset['sha256']}"
EXPECTED_CODE_SHA256 = "{code_hash}"
DRIVE_ROOT = Path("/content/drive/MyDrive/Agri Vision")
WORKSPACE = Path("/content/agrivision_comparison")
RESUME_RUNS = {{}}  # Optional model_id -> existing Drive run folder; preserve settings/budget.
FINAL_TEST = False  # Enable after configurations/checkpoints are selected; never tune using test scores.
'''),code('''from google.colab import drive
drive.mount("/content/drive")
import hashlib, json, shutil, zipfile
def sha(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream,"sha256").hexdigest()
WORKSPACE.mkdir(parents=True, exist_ok=True)
def upload_path(filename):
    path = DRIVE_ROOT / filename
    if path.is_file() or filename != "non_chili_v1_colab_dataset.zip":
        return path
    parts_dir = DRIVE_ROOT / "non_chili_v1_upload_parts"
    manifest_path = parts_dir / "parts.json"
    if manifest_path.is_file():
        manifest = json.loads(manifest_path.read_text())
        assert manifest["joined_sha256"] == EXPECTED_DATASET_SHA256, "Wrong dataset parts manifest"
        specs = manifest["parts"]
        assert specs and [item["name"] for item in specs] == [filename + f".part{index:02d}" for index in range(1, len(specs) + 1)], "Missing or incorrectly ordered parts manifest"
        parts = [parts_dir / item["name"] for item in specs]
        assert all(part.is_file() and part.stat().st_size == item["bytes"] for part, item in zip(parts, specs)), "Upload all listed dataset parts with their complete sizes first"
    else:
        parts = [DRIVE_ROOT / (filename + suffix) for suffix in (".part01", ".part02")]
        assert all(part.is_file() for part in parts), "Upload the dataset ZIP or the dataset parts folder first"
    joined = WORKSPACE.parent / filename
    if joined.is_file() and sha(joined) == EXPECTED_DATASET_SHA256:
        return joined
    temporary = joined.with_suffix(joined.suffix + ".partial")
    with temporary.open("wb") as output:
        for part in parts:
            with part.open("rb") as source:
                shutil.copyfileobj(source, output, length=16 * 1024 * 1024)
    assert sha(temporary) == EXPECTED_DATASET_SHA256, "Dataset parts are damaged or do not belong together"
    temporary.replace(joined)
    print(f"Joined {len(parts)} dataset parts on Colab's local disk.")
    return joined
for filename, expected in [("non_chili_v1_colab_dataset.zip", EXPECTED_DATASET_SHA256),
                           ("agrivision_comparison_code.zip", EXPECTED_CODE_SHA256)]:
    path = upload_path(filename)
    assert path.is_file(), f"Upload {filename} first"
    assert sha(path) == expected, f"Wrong or damaged upload: {filename}"
    with zipfile.ZipFile(path) as archive:
        for name in archive.namelist():
            assert (WORKSPACE / name).resolve().is_relative_to(WORKSPACE.resolve()), "Unsafe archive path"
        archive.extractall(WORKSPACE)
print("Verified and extracted both uploads.")
'''),code('''import os, subprocess, sys
os.chdir(WORKSPACE)
# Protect Colab's CUDA-enabled torch/torchvision pair from dependency replacement.
import torch, torchvision
torch_before, vision_before = torch.__version__, torchvision.__version__
assert torch.cuda.is_available(), "Select Runtime -> Change runtime type -> GPU"
gpu = torch.cuda.get_device_name(0)
print("GPU:", gpu, "torch:", torch_before, "torchvision:", vision_before)
subprocess.run([sys.executable, "-m", "pip", "install", "--no-deps", "-r", "requirements-comparison.txt"], check=True)
# Optional model dependencies, pinned where tested. --no-deps above preserves core CUDA packages.
subprocess.run([sys.executable, "-m", "pip", "install", "huggingface_hub", "safetensors", "packaging", "tqdm", "omegaconf==2.4.0",
                "polars>=0.20", "ultralytics-thop>=2.0.18", "matplotlib", "opencv-python", "requests", "scipy", "psutil"], check=True)
assert torch.__version__ == torch_before and torchvision.__version__ == vision_before
subprocess.run([sys.executable, "-m", "unittest", "discover", "-s", "tests", "-p", "test_comparison.py", "-v"], check=True)
'''),markdown('''## GPU pilot and setup checks
The pilot uses two batches per client and 24 fixed validation images. These metrics are diagnostics, not thesis results.
The tiny overfit check uses training images only. It must decrease loss and preserve save/reload predictions.
Dependency/adapter checks cover all four models, including EfficientDet before long runs.
'''),code('''from datetime import datetime, timezone
import json
OUTPUT_ROOT = DRIVE_ROOT / "comparison_results"
OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
GPU_PILOT_RECORD = OUTPUT_ROOT / "gpu_pilot_passed.json"
GPU_PILOT_HASH = None
if MODE == "pilot":
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S_%f")
    pilot_dir = OUTPUT_ROOT / f"ssdlite_gpu_pilot_{stamp}"
    subprocess.run([sys.executable, "scripts/check_comparison_adapters.py", "--models", "ssdlite",
                    "--pretrained", "--steps", "8", "--device", "cuda", "--report", str(OUTPUT_ROOT / f"ssdlite_overfit_{stamp}.json")], check=True)
    subprocess.run([sys.executable, "scripts/check_comparison_adapters.py", "--models", "ssd_mobilenetv3", "yolo_v8n", "efficientdet_d0",
                    "--pretrained", "--device", "cuda", "--report", str(OUTPUT_ROOT / f"all_adapter_checks_{stamp}.json")], check=True)
    subprocess.run([sys.executable, "-m", "comparison.runner", "--root", str(WORKSPACE), "--model", "ssdlite",
                    "--mode", "pilot", "--device", "cuda", "--batch", "8", "--workers", str(WORKERS),
                    "--local-epochs", str(LOCAL_EPOCHS), "--seed", str(SEED), "--max-train-batches", "2", "--validation-limit", "24",
                    "--output", str(pilot_dir)], check=True)
    # Resume completed pilot: should verify state and finish without repeating training.
    subprocess.run([sys.executable, "-m", "comparison.runner", "--root", str(WORKSPACE), "--model", "ssdlite",
                    "--mode", "pilot", "--device", "cuda", "--batch", "8", "--workers", str(WORKERS),
                    "--local-epochs", str(LOCAL_EPOCHS), "--seed", str(SEED), "--max-train-batches", "2", "--validation-limit", "24",
                    "--output", str(pilot_dir), "--resume"], check=True)
    from comparison.runner import implementation_hash
    GPU_PILOT_RECORD.write_text(json.dumps({"gpu": gpu, "code_hash": implementation_hash(), "pilot_dir": str(pilot_dir),
                                          "phase": "passed", "local_epochs": LOCAL_EPOCHS}, indent=2))
    print("GPU pilot passed; inspect Drive outputs before setting MODE=main.")
elif MODE != "main":
    raise ValueError("MODE must be pilot or main")
'''),markdown('''## Main training
When `MODE="main"`, each listed detector trains serially for the same declared round and local-epoch budget.
Record resolved settings after pilots. Default learning rates are candidates, not an equal tuning study.
Budget and code cannot change on resume because that would silently change the schedule. A partial round restarts from the last committed global round.
Main training uses every client training image and the full pooled validation split. Test evaluation is separate.
'''),code('''RUNS = {}
if MODE == "main":
    from comparison.runner import implementation_hash
    assert GPU_PILOT_RECORD.is_file(), "Run the GPU pilot first"
    record = json.loads(GPU_PILOT_RECORD.read_text())
    assert record["phase"] == "passed" and record["gpu"] == gpu
    assert record["code_hash"] == implementation_hash(), "Code changed; rerun the GPU pilot"
    assert record["local_epochs"] == LOCAL_EPOCHS, "Epoch setting changed; rerun the pilot"
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S_%f")
    for model in MODELS:
        output = Path(RESUME_RUNS[model]) if model in RESUME_RUNS else OUTPUT_ROOT / f"{model}_non_chili_seed{SEED}_{stamp}"
        command = [sys.executable, "-m", "comparison.runner", "--root", str(WORKSPACE), "--model", model,
                   "--mode", "main", "--device", "cuda", "--rounds", str(ROUNDS), "--local-epochs", str(LOCAL_EPOCHS),
                   "--seed", str(SEED), "--workers", str(WORKERS), "--output", str(output)]
        if model in RESUME_RUNS:
            command.append("--resume")
        subprocess.run(command, check=True)
        RUNS[model] = output
        (OUTPUT_ROOT / f"main_runs_{stamp}.json").write_text(json.dumps({k:str(v) for k,v in RUNS.items()}, indent=2))
    print("Completed main run folders:", RUNS)
'''),code('''if MODE == "main" and FINAL_TEST:
    for model, output in RUNS.items():
        subprocess.run([sys.executable, "-m", "comparison.final_evaluate", "--root", str(WORKSPACE), "--run", str(output),
                        "--split", "test", "--device", "cuda", "--profile-images", str(PROFILE_IMAGES)], check=True)
    subprocess.run([sys.executable, "-m", "comparison.compare_results", "--runs", *map(str, RUNS.values()),
                    "--output", str(OUTPUT_ROOT / "architecture_comparison_test.csv")], check=True)
else:
    print("Held-out test evaluation was not requested. Validation metrics and checkpoints are saved to Drive.")
''')]
    notebook = dict(nbformat=4,nbformat_minor=5,cells=cells,
                    metadata=dict(kernelspec=dict(display_name='Python 3',language='python',name='python3'),
                                  language_info=dict(name='python'),colab=dict(name='AgriVision_Comparison.ipynb'),accelerator='GPU'))
    for index,cell in enumerate(cells):
        cell['id'] = f'comparison-{index:02d}'
        if cell['cell_type']=='code':
            compile(''.join(cell['source']),f'notebook cell {index}','exec')
    (ROOT/'AgriVision_Comparison.ipynb').write_text(json.dumps(notebook,indent=2)+'\n',encoding='utf-8')
    report = dict(code_bundle=destination.name,code_bytes=destination.stat().st_size,code_sha256=code_hash,
                  dataset_bundle=dataset['filename'],dataset_sha256=dataset['sha256'],
                  notebook='AgriVision_Comparison.ipynb',default_mode='pilot',main_gpu_runs_completed=False)
    (ROOT/'reports/comparison/colab_upload_package.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2))


if __name__=='__main__':
    main()
