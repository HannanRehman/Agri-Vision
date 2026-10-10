"""Package separate native YOLO and three-detector continuation notebooks; never train."""
import ast
import hashlib
import json
from pathlib import Path
import shutil
import zipfile

ROOT = Path(__file__).resolve().parents[1]


def sha(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream,'sha256').hexdigest()


def cell(text, code=False):
    result = dict(cell_type='code' if code else 'markdown',metadata={},source=text.splitlines(keepends=True))
    if code:result.update(execution_count=None,outputs=[])
    return result


def notebook(name, cells):
    for index,item in enumerate(cells):
        item['id'] = f'run30-{index:02d}'
        if item['cell_type']=='code':ast.parse(''.join(item['source']))
    result=dict(nbformat=4,nbformat_minor=5,cells=cells,
                metadata=dict(kernelspec=dict(display_name='Python 3',language='python',name='python3'),
                              language_info=dict(name='python'),colab=dict(name=name),accelerator='GPU'))
    (ROOT/name).write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8')


def main():
    original = ROOT/'AgriVision_Comparison_10Round.ipynb'
    if not original.exists():shutil.copy2(ROOT/'AgriVision_Comparison.ipynb',original)
    old=json.loads(original.read_text(encoding='utf-8'))
    extract=''.join(old['cells'][2]['source']).replace('agrivision_comparison_code.zip','agrivision_30_round_code.zip')
    environment=''.join(old['cells'][3]['source'])
    environment=environment.replace('os.chdir(WORKSPACE)', 'os.chdir(WORKSPACE)\nsys.path.insert(0, str(WORKSPACE))')
    environment=environment.replace('subprocess.run([sys.executable, "-m", "unittest", "discover", "-s", "tests", "-p", "test_comparison.py", "-v"], check=True)',
                                    'for pattern in ["test_comparison.py", "test_30_round_workflow.py"]:\n    subprocess.run([sys.executable, "-m", "unittest", "discover", "-s", "tests", "-p", pattern, "-v"], check=True)')
    files=[*sorted((ROOT/'comparison').glob('*.py')),*sorted((ROOT/'configs/comparison').glob('*.json')),
           ROOT/'requirements-comparison.txt',ROOT/'tests/test_comparison.py',ROOT/'tests/test_30_round_workflow.py',
           ROOT/'scripts/yolo_native_train.py',ROOT/'scripts/continue_comparison.py']
    bundle=ROOT/'agrivision_30_round_code.zip'
    with zipfile.ZipFile(bundle,'w',compression=zipfile.ZIP_DEFLATED) as archive:
        for path in files:archive.write(path,path.relative_to(ROOT).as_posix())
    with zipfile.ZipFile(bundle) as archive:assert archive.testzip() is None
    dataset=json.loads((ROOT/'reports/non_chili_v1/upload_package.json').read_text())
    common=f'''from pathlib import Path
DRIVE_ROOT = Path("/content/drive/MyDrive/Agri Vision")
WORKSPACE = Path("/content/agrivision_30_round")
EXPECTED_DATASET_SHA256 = "{dataset['sha256']}"
EXPECTED_CODE_SHA256 = "{sha(bundle)}"
TARGET_ROUNDS = 30
SEED = 42
WORKERS = 2
'''
    yolo=[cell('''# Agri Vision: native YOLOv8n, rounds 1–30

Use your **separate YOLO account**. Upload `agrivision_30_round_code.zip` and the existing `non_chili_v1_colab_dataset.zip` to `MyDrive/Agri Vision/`. The same verified 14-part dataset folder is also supported. The 24-class data remains unchanged.

Choose a **T4 GPU**. **Run all** first performs a small training-only learning check, then starts the fresh 30-round main run if the check passes. Rerunning resumes the last completed global round automatically; an interrupted partial round repeats. Pilot weights are never used for the main run.

Native Ultralytics 8.4.48 training restores the historical method: three clients, one local epoch, batch 8, 640 images, native optimizer/warmup/augmentation and adapting BatchNorm. Seed 42 and the new 24-class dataset are declared differences from the old experiment. Metrics and initial/best/latest weights are saved; bulky client weights and every-round raw prediction exports are not retained.
'''),cell(common,True),cell(extract,True),cell(environment,True),cell('''## Learning check and recoverable main run
The pilot uses 48 training images for 12 small epochs with augmentation disabled, and evaluates only those training images. Its scores are diagnostics. Main training starts from fresh pretrained initialization with the historical augmentation settings. A failed pilot stops this notebook.
'''),cell('''import json
from datetime import datetime, timezone
from comparison.data import load_protocol
from scripts.yolo_native_train import pipeline_hash, hardware
load_protocol(WORKSPACE)
assert "T4" in gpu, "Select a T4 for the agreed historical GPU environment"
OUTPUT_ROOT = DRIVE_ROOT / "comparison_results"
OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
RUN = OUTPUT_ROOT / "yolo_v8n_native_non_chili_seed42_30round"
PILOT_RECORD = OUTPUT_ROOT / "yolo_native_learning_pilot_seed42.json"
# Resume does not repeat the pilot or initialize a new run.
if not RUN.exists():
    passed = False
    if PILOT_RECORD.exists():
        record = json.loads(PILOT_RECORD.read_text())
        passed = (record.get("phase") == "passed" and record.get("pipeline_hash") == pipeline_hash()
                  and record.get("hardware") == hardware(torch.device("cuda:0"))
                  and record.get("dataset_sha256") == load_protocol(WORKSPACE)["dataset_checksums_sha256"]
                  and record.get("seed") == SEED)
    if not passed:
        stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S_%f")
        pilot = OUTPUT_ROOT / f"yolo_native_learning_pilot_{stamp}"
        subprocess.run([sys.executable, "-m", "scripts.yolo_native_train", "--root", str(WORKSPACE),
                        "--pilot", "--seed", str(SEED), "--output", str(pilot)], check=True)
        shutil.copy2(pilot / "pilot.json", PILOT_RECORD)
subprocess.run([sys.executable, "-m", "scripts.yolo_native_train", "--root", str(WORKSPACE),
                "--rounds", str(TARGET_ROUNDS), "--seed", str(SEED), "--pilot-record", str(PILOT_RECORD),
                "--output", str(RUN)], check=True)
print("YOLO run:", RUN)
print(json.loads((RUN / "latest.json").read_text()))
''',True),cell('''## Progress and results
While training, the output prints the round, client, native losses and validation. In Drive, `status.json` gives the phase; `latest.json` is the committed global round. The main CSV is `results/metrics/global_validation_metrics.csv`. Each round's JSON includes pooled comparison metrics and separately labelled native YOLO client metrics. Best selection uses pooled validation AP50–95, not the final round automatically. Held-out test evaluation is a separate action after configuration selection.

If the runtime stops, reopen this same notebook on the same account with the same uploaded bundle, select T4, and Run all. Do not delete the run folder or change the budget/packages. The weights, configuration, hardware and hashes are checked before resume.
''')]
    notebook('AgriVision_YOLO_Native_30Rounds.ipynb',yolo)
    alternative=[cell('''# Agri Vision: continue the three alternative detectors toward round 30

Use the **current account and existing Agri Vision Drive**. YOLO is excluded. Choose T4 and Run all to train **EfficientDet → SSDLite → SSD–MobileNetV3 sequentially**. YOLO may run at the same time on its other account.

Upload `agrivision_30_round_code.zip` to `MyDrive/Agri Vision/`; the existing dataset parts remain usable. This notebook creates separate linked `_to30` result folders and preserves original runs. It checks the original code, model specifications, dataset, GPU, package versions and checkpoint hashes before continuing.

EfficientDet resumes from its latest committed round (last verified: 2, so next 3). SSDLite and SSD resume from round 10, next 11. Original cosine learning rates are preserved through round 10; rounds 11–30 use the original terminal learning rate (10% of original initial LR), with the same per-client optimizer reset. This is a documented continuation, not a fresh thirty-round schedule.
'''),cell(common+'''STAMP = "20261009_224152_449800"
MODELS = ["efficientdet_d0", "ssdlite", "ssd_mobilenetv3"]
SOURCE_ROOT = DRIVE_ROOT / "comparison_results"
SOURCES = {model: SOURCE_ROOT / f"{model}_non_chili_seed42_{STAMP}" for model in MODELS}
RUNS = {model: SOURCE_ROOT / f"{model}_non_chili_seed42_{STAMP}_to30" for model in MODELS}
''',True),cell(extract,True),cell(environment,True),cell('''## Validate all sources before starting any model
This catches missing checkpoints or mismatched environments before spending training time. GPU access may still end during a run; completed rounds remain resumable. The other two models start after the preceding model finishes. Rerunning skips already completed continuations.
'''),cell('''from scripts.continue_comparison import validate_source
for model in MODELS:
    settings, latest, protocol = validate_source(WORKSPACE, SOURCES[model], torch.device("cuda:0"))
    assert settings["config"]["seed"] == SEED and settings["config"]["local_epochs"] == 1
    assert settings["config"]["model"] == model
    print(model, "source committed round:", latest["round"], "next:", latest["round"] + 1)
print("All source preflight checks passed.")
''',True),cell('''for model in MODELS:
    print(f"Starting/resuming {model} toward round {TARGET_ROUNDS}", flush=True)
    subprocess.run([sys.executable, "-m", "scripts.continue_comparison", "--root", str(WORKSPACE),
                    "--source", str(SOURCES[model]), "--output", str(RUNS[model]),
                    "--rounds", str(TARGET_ROUNDS), "--workers", str(WORKERS)], check=True)
    print(model, json.loads((RUNS[model] / "latest.json").read_text()))
''',True),cell('''## Results, resume and storage
Each `_to30` folder records original lineage, schedule, pooled/client metrics and initial/best/latest checkpoints. Source metrics through its committed round are copied for a complete curve. Source folders are not modified. Intermediate weights in the new continuation folder are pruned only after a completed round is committed; validation metrics remain. All-round raw prediction exports are omitted to conserve Drive space.

An interruption repeats the incomplete round. Reopen this notebook on the current account, select T4 and Run all; the first incomplete model resumes and completed models finish without retraining. Do not change the source folders, upload bundle or package versions during a continuation.

These are continued ten-round experiments. The thesis must report their schedules separately from the fresh native-YOLO thirty-round experiment. Final held-out evaluation is deferred until configuration/checkpoint selection.
''')]
    notebook('AgriVision_Comparison.ipynb',alternative)
    notebook('AgriVision_Alternatives_Continue_30Rounds.ipynb',alternative)
    report=dict(code_bundle=bundle.name,code_bytes=bundle.stat().st_size,code_sha256=sha(bundle),
                dataset_bundle=dataset['filename'],dataset_sha256=dataset['sha256'],
                notebooks=['AgriVision_YOLO_Native_30Rounds.ipynb','AgriVision_Comparison.ipynb','AgriVision_Alternatives_Continue_30Rounds.ipynb'],
                original_comparison_code_sha256=sha(ROOT/'agrivision_comparison_code.zip'),
                gpu_training_started=False)
    (ROOT/'reports/comparison/30_round_upload_package.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2))


if __name__=='__main__':main()
