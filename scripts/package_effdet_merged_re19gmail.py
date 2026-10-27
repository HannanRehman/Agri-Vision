"""Build the verified combined-search continuation notebook; never launch training."""
import ast
import hashlib
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
REPORT=ROOT/'reports/comparison/efficientdet_merge_re19gmail_20261024'
NAME='AgriVision_EfficientDet_Combined_Resume_re19gmail.ipynb'


def main():
    marker_path=REPORT/'combined_effdet_re19gmail.json'
    marker=json.loads(marker_path.read_text())
    marker_sha=hashlib.sha256(marker_path.read_bytes()).hexdigest()
    notebook=json.loads((ROOT/'AgriVision_GridSearch_EfficientDet_71gmail.ipynb').read_text(encoding='utf-8'))
    notebook['metadata']['colab']['name']=NAME
    notebook['cells'][0]['source']=[
        '# Agri Vision: combined EfficientDet continuation - re19gmail\n','\n',
        'Use hannanre19@gmail.com (different from the YOLO account). Keep the old 4gmail and 71gmail EfficientDet jobs stopped. Select T4 GPU and Run all. This notebook contains the verified combination of trials 1-11 from 71gmail and trials 12-16 from 4gmail.\n','\n',
        f'{marker["completed_screening_trials"]} screening trials are complete. Trial 16 has {marker["trials"]["trial_16"]["round"]} committed rounds in this snapshot. Completed work is skipped; an unfinished round repeats from its last committed global checkpoint. Once all 16 five-round trials are complete, the global top two continue to round 10. The selected configuration then starts its fresh 30-round final run automatically. No manual model switching is required.\n','\n',
        'The original dataset, code, common initialization, hardware checks and optimized FP32 execution profile are retained. This preparation does not start GPU training.\n',
    ]
    for c in notebook['cells']:
        c['metadata']={}
        if c['cell_type']=='code':
            source=''.join(c['source'])
            source=source.replace('rehmanfx71@gmail.com','hannanre19@gmail.com')
            source=source.replace('71gmail_resume.json','combined_effdet_re19gmail.json')
            c['source']=source.splitlines(keepends=True)
            c['outputs']=[];c['execution_count']=None
    verification=f'''# Validate the combined snapshot and exact resume environment before training.
marker_path = DRIVE_ROOT / "combined_effdet_re19gmail.json"
assert sha(marker_path) == "{marker_sha}", "Wrong combined-search marker"
snapshot = json.loads(marker_path.read_text())
model_folder = OUTPUT / "efficientdet_d0"
initial = model_folder / "initial.pt"
assert sha(initial) == snapshot["initial_checkpoint_sha256"], "Common initialization differs"
assert sha(OUTPUT/"plan.json") == snapshot["plan_sha256"], "Saved search plan differs"
assert sha(model_folder/"execution_profile.json") == snapshot["execution_profile_sha256"], "Execution profile differs"
from importlib.metadata import version
reference = json.loads((model_folder/"trials/trial_01/settings.json").read_text())
actual = {{"gpu": torch.cuda.get_device_name(0), "torch": str(torch.__version__),
    "packages": {{name: version(name) for name in reference["hardware"]["packages"]}}}}
assert actual == reference["hardware"], f"Resume environment differs: {{actual}}"
assert set(snapshot["trials"]) == {{f"trial_{{n:02d}}" for n in range(1,17)}}
for name, source in sorted(snapshot["trials"].items()):
    trial = model_folder/"trials"/name
    settings = json.loads((trial/"settings.json").read_text())
    assert sha(trial/"settings.json") == source["settings_sha256"], f"Settings changed: {{name}}"
    assert sha(model_folder/"configs"/f"{{name}}.json") == source["config_sha256"], f"Configuration changed: {{name}}"
    assert settings["initial_checkpoint"] == str(initial)
    assert settings["initial_checkpoint_sha256"] == snapshot["initial_checkpoint_sha256"]
    assert settings["hardware"] == actual
    latest = json.loads((trial/"latest.json").read_text())
    assert latest["round"] >= source["round"], f"Committed progress lost: {{name}}"
    checkpoint = (trial/latest["checkpoint"]).resolve()
    assert checkpoint.is_relative_to(trial.resolve())
    assert sha(checkpoint) == latest["checkpoint_sha256"], f"Damaged checkpoint: {{name}}"
    if latest["round"] == source["round"]:
        assert latest["checkpoint_sha256"] == source["checkpoint_sha256"]
    for relative, expected in source["metric_sha256"].items():
        assert sha(trial/relative) == expected, f"Historical metric changed: {{name}}/{{relative}}"
    assert all((trial/"results/metrics"/f"round_{{n:03d}}.json").is_file() for n in range(1,latest["round"]+1))
    print(name, "verified saved round", latest["round"], flush=True)
print("All 16 combined trials verified. Remaining screening, global confirmation and final training will resume.", flush=True)
'''
    for c in notebook['cells']:
        if c.get('id')=='verify-migration-71':
            c['id']='verify-combined-re19gmail'
            c['source']=verification.splitlines(keepends=True)
    notebook['cells'][-1]['source']=[
        '## Resume and results\n',
        'Always reopen this notebook as hannanre19@gmail.com, select T4 and Run all. All checkpoints/settings and paths remain under My Drive/Agri Vision/grid_search/grid_v1_20261017/efficientdet_d0/. Keep this folder named Agri Vision.\n','\n',
        'The copied snapshot marker is fixed. Its checks accept later committed rounds while preserving original source settings and metrics. Completed screening trials are skipped after identity checks. screening_results.csv ranks all completed screening trials, finalist_results.csv ranks the global finalists, and final_training/ stores the selected fresh 30-round run with best/latest weights. The notebook exits automatically after the final 30 rounds.\n','\n',
        'Do not run old EfficientDet notebooks concurrently. The archived partial trial-12 copy on 71gmail is deliberately excluded; the completed 4gmail trial 12 is the official copy. This experiment uses custom sequential three-client FedAvg, not Flower.\n',
    ]
    for c in notebook['cells']:
        if c['cell_type']=='code':ast.parse(''.join(c['source']))
    output=ROOT/NAME
    output.write_text(json.dumps(notebook,indent=2,ensure_ascii=False)+'\n',encoding='utf-8')
    readme=REPORT/'START_HERE_EfficientDet_re19gmail.md'
    readme.write_text(
        '# EfficientDet combined continuation on re19gmail\n\n'
        'Account: hannanre19@gmail.com. This is distinct from hannanrehman19@gmail.com, which ran YOLO.\n\n'
        f'Open {NAME}, choose T4 GPU, then Run all. Keep Agri Vision in My Drive under that exact name. Authorize Drive as the new account. No settings edits are required.\n\n'
        f'The copy contains trials 1-11 from 71gmail and 12-16 from 4gmail: {marker["completed_screening_trials"]} completed five-round trials and trial 16 saved through round {marker["trials"]["trial_16"]["round"]}. An incomplete round repeats; already committed rounds/trials are preserved.\n\n'
        'The notebook automatically finishes screening, globally selects the best two trials, extends them to round 10, then starts a fresh 30-round run of the winning configuration. The same shared initial weights and verified optimized FP32 code are used. Strict original code/data/environment checks remain enabled.\n\n'
        'Required uploads are this notebook, combined_effdet_re19gmail.json, agrivision_grid_search_v1_code.zip, efficientdet_fp32_continuation_v1.zip, all dataset parts and parts.json in non_chili_v1_upload_parts/, and the combined grid_search/ tree. All copied trial checkpoint bytes are SHA256-verified before handoff; the notebook rechecks them before training.\n\n'
        'Keep 4gmail and 71gmail EfficientDet sessions stopped. Their source results remain intact. Run all again on the new account after interruptions. GPU availability and speed depend on the session; no completion-time guarantee is implied. Nothing was trained during preparation.\n',encoding='utf-8')
    package=dict(notebook=NAME,notebook_sha256=hashlib.sha256(output.read_bytes()).hexdigest(),
        marker_sha256=marker_sha,training_started=False)
    (REPORT/'package.json').write_text(json.dumps(package,indent=2)+'\n')
    print(json.dumps(package,indent=2))


if __name__=='__main__':main()
