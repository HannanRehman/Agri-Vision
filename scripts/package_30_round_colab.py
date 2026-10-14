"""Package separate native YOLO and three-detector continuation notebooks; never train."""
import ast
import hashlib
import json
from pathlib import Path
import shutil
import zipfile
from copy import deepcopy

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
        if item['cell_type']=='code':
            ast.parse(''.join(item['source']))
            item['outputs'] = []
            item['execution_count'] = None
    result=dict(nbformat=4,nbformat_minor=5,cells=cells,
                metadata=dict(kernelspec=dict(display_name='Python 3',language='python',name='python3'),
                              language_info=dict(name='python'),colab=dict(name=name),accelerator='GPU'))
    (ROOT/name).write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8')


def efficientdet_only_notebook(alternative):
    """Retain the historical 15-round notebook and prepare the current extension."""
    cells = deepcopy(alternative)
    cells[0] = cell('''# Agri Vision: EfficientDet-D0 only, through round 15

Use **4gmail**, select a **T4 GPU**, and **Run all**. Only EfficientDet runs; SSDLite, SSD-MobileNetV3 and YOLO are excluded. The original EfficientDet checkpoint last verified at round 2 resumes at round 3, and training stops after round 15 is committed.

The existing code ZIP and 14 dataset parts are reused. The original run remains untouched; continued results are saved in a separate `_to15` folder. Original learning rates are preserved through round 10; rounds 11-15 use the original terminal learning rate. Rerunning resumes the last completed round in `_to15`; an incomplete round repeats. No new pilot is run.
''')
    source = ''.join(cells[1]['source'])
    source = source.replace('TARGET_ROUNDS = 30', 'TARGET_ROUNDS = 15')
    source = source.replace('MODELS = ["efficientdet_d0", "ssdlite", "ssd_mobilenetv3"]', 'MODELS = ["efficientdet_d0"]')
    source = source.replace('_to30"', '_to15"')
    cells[1] = cell(source, True)
    cells[4] = cell('''## Validate the EfficientDet source before training
The original dataset, code, model settings, package versions, GPU and checkpoint hashes must match. No other detector source is required for this notebook.
''')
    cells[-1] = cell('''## Results and automatic resume
Results are written to `comparison_results/efficientdet_d0_non_chili_seed42_20261009_224152_449800_to15/`. Check `latest.json` for the committed round and `status.json` for progress. The pooled CSV and per-round JSONs retain all completed metrics; initial, best and latest checkpoints are preserved.

If interrupted, reopen this same notebook on 4gmail, select T4 and Run all. Completed rounds are skipped and an unfinished round repeats. Once round 15 is saved, the notebook finishes without starting another model. Keep the uploaded bundle, original source and continuation settings unchanged. Extending this run later requires a reviewed continuation setup; do not simply edit this notebook's target.

The held-out test is reserved for final evaluation after checkpoint selection.
''')
    notebook('AgriVision_EfficientDet_Continue_15Rounds.ipynb', deepcopy(cells))
    efficientdet_round16_to30_notebook(cells)


def efficientdet_round16_to30_notebook(template):
    """Continue the verified completed 15-round branch, using the unchanged runner."""
    cells = deepcopy(template)
    cells[0] = cell('''# Agri Vision: EfficientDet-D0, rounds 16–30

Use **4gmail**, select a **T4 GPU**, and **Run all**. This notebook continues only EfficientDet-D0 from the saved round-15 checkpoint through round 30. The round-15 source and its validation-best round-11 checkpoint were verified on 15 October 2026. No new pilot is needed.

Reuse the existing code ZIP and dataset parts. Results go into a separate `_to30_from15` folder. Original training settings are retained: 24 classes, three clients, one local epoch, batch 4, 512-pixel inputs, AdamW and seed 42. Rounds 16–30 retain the existing terminal learning rate of 0.00001; optimizer state resets per client as in the completed run. This is a continuation of the existing schedule.
''')
    settings = ''.join(cells[1]['source'])
    settings = settings.replace('TARGET_ROUNDS = 15', 'TARGET_ROUNDS = 30')
    settings = settings.replace('f"{model}_non_chili_seed42_{STAMP}"',
                                'f"{model}_non_chili_seed42_{STAMP}_to15"')
    settings = settings.replace('RUNS = {model: SOURCE_ROOT / f"{model}_non_chili_seed42_{STAMP}_to15"',
                                'RUNS = {model: SOURCE_ROOT / f"{model}_non_chili_seed42_{STAMP}_to30_from15"')
    settings += '''EXPECTED_SOURCE_ROUND = 15
EXPECTED_SOURCE_CONFIG_HASH = "5bac396d04beaeab8c865b095b691522fcceba3f4e147d839142a1199be70089"
EXPECTED_SOURCE_CHECKPOINT_SHA256 = "3815ef3cc290b35d529ecd6b2d626a61f1a3ec66ae56e0fd7f2991681c5e391d"
'''
    cells[1] = cell(settings, True)
    cells[4] = cell('''## Verify the completed round-15 source and resume position
Preflight verifies the source weights, configuration, data, original implementation, T4 GPU and recorded package versions. A missing or mismatched round-15 source stops execution before training. After interruption, the new branch resumes from its own latest committed round.
''')
    cells[5] = cell('''from scripts.continue_comparison import validate_source
for model in MODELS:
    settings, latest, protocol = validate_source(WORKSPACE, SOURCES[model], torch.device("cuda:0"))
    config = settings["config"]
    assert config["model"] == model and config["seed"] == SEED and config["local_epochs"] == 1
    assert config["rounds"] == EXPECTED_SOURCE_ROUND and latest["round"] == EXPECTED_SOURCE_ROUND, "Expected the completed round-15 source"
    assert settings["config_hash"] == EXPECTED_SOURCE_CONFIG_HASH, "Source configuration differs"
    assert latest["checkpoint_sha256"] == EXPECTED_SOURCE_CHECKPOINT_SHA256, "Wrong round-15 checkpoint"
    assert config["batch"] == 4 and config["optimizer"] == "adamw" and settings["input_size"] == 512
    assert config["lr"] == 0.0001 and config["continuation"]["original_rounds"] == 10
    assert len(protocol["clients"]) == 3 and settings["validation_images"] == 4157
    source_status = json.loads((SOURCES[model] / "status.json").read_text())
    assert source_status["phase"] == "complete" and source_status["completed_rounds"] == 15
    committed = latest["round"]
    if RUNS[model].exists():
        branch = json.loads((RUNS[model] / "latest.json").read_text())
        committed = branch["round"]
        assert EXPECTED_SOURCE_ROUND <= committed <= TARGET_ROUNDS, "Invalid continuation commit"
    if committed == TARGET_ROUNDS:
        print(model, "already completed round 30; no rounds will repeat.")
    else:
        print(model, "resume from committed round:", committed, "next training round:", committed + 1, "target:", TARGET_ROUNDS)
print("Round-15 source and environment preflight checks passed.")
''', True)
    cells[-1] = cell('''## Results and automatic resume
The new run is saved under `comparison_results/efficientdet_d0_non_chili_seed42_20261009_224152_449800_to30_from15/`. On its first run, the runner copies metrics for rounds 1–15 and the initial, latest and validation-best checkpoints, then begins round 16. The `_to15` source is retained unchanged.

The runner saves pooled and per-class validation metrics, client logs, and best/latest checkpoints after each round. Check `status.json` for the active phase and `latest.json` for the last fully committed round. The complete history is `results/metrics/global_validation_metrics.csv`; per-class results are in each `round_NNN.json` under `pooled_validation.per_class`. The round-11 best remains selected unless a later round improves pooled mAP50–95. Intermediate new-branch weights are pruned after commits to conserve Drive space; metrics remain.

If Colab stops, reopen this same notebook on 4gmail, select T4 and Run all. Completed rounds are skipped; an unfinished round repeats from the last committed global checkpoint. Training stops after round 30 and starts no other model. Keep both source and continuation folders, code bundle and package versions unchanged. Held-out test evaluation remains separate.
''')
    notebook('AgriVision_EfficientDet_Continue_30Rounds.ipynb', cells)
    notebook('AgriVision_Comparison.ipynb', deepcopy(cells))


def ssd_pair_notebook(alternative):
    """Separate-account workflow for the two saved SSD experiments."""
    cells = deepcopy(alternative)
    cells[0] = cell('''# Agri Vision: SSDLite and SSD-MobileNetV3, rounds 11-30

Use the separate account assigned to these two detectors, choose **T4 GPU**, and **Run all**. Only **SSDLite followed by SSD-MobileNetV3** runs, sequentially on this GPU. YOLO and EfficientDet run on their own accounts.

The copied original source folders, code ZIP and dataset parts must be under `MyDrive/Agri Vision/`. Both source runs last committed round 10; the next round is 11. The original sources remain untouched, with separate `_to30` result folders for each detector. Rounds 11-30 use each source's original terminal learning rate. No new pilot is required for these unchanged, validated continuations.
''')
    cells[1] = cell(''.join(cells[1]['source']).replace(
        'MODELS = ["efficientdet_d0", "ssdlite", "ssd_mobilenetv3"]',
        'MODELS = ["ssdlite", "ssd_mobilenetv3"]'), True)
    cells[4] = cell('''## Validate both copied SSD sources before training
This checks the original code, dataset, T4 GPU, package versions, settings and checkpoint hashes. A different Gmail is acceptable; the source bytes and recorded environment must match. EfficientDet files are not required.
''')
    notebook('AgriVision_SSD_Pair_Continue_30Rounds.ipynb', cells)


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
    notebook('AgriVision_Alternatives_Continue_30Rounds.ipynb',alternative)
    efficientdet_only_notebook(alternative)
    ssd_pair_notebook(alternative)
    report=dict(code_bundle=bundle.name,code_bytes=bundle.stat().st_size,code_sha256=sha(bundle),
                dataset_bundle=dataset['filename'],dataset_sha256=dataset['sha256'],
                notebooks=['AgriVision_YOLO_Native_30Rounds.ipynb','AgriVision_Comparison.ipynb','AgriVision_Alternatives_Continue_30Rounds.ipynb'],
                original_comparison_code_sha256=sha(ROOT/'agrivision_comparison_code.zip'),
                gpu_training_started=False)
    (ROOT/'reports/comparison/30_round_upload_package.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2))


if __name__=='__main__':main()
