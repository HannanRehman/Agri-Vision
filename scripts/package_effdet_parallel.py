"""Package separate, resumable EfficientDet screening ranges without training."""
import ast
import hashlib
import json
from pathlib import Path
import zipfile

ROOT = Path(__file__).resolve().parents[1]
REPORT = ROOT/'reports/comparison/efficientdet_parallel_20261022'


def main():
    REPORT.mkdir(parents=True, exist_ok=True)
    bundle = REPORT/'efficientdet_parallel_screening_v1.zip'
    with zipfile.ZipFile(bundle, 'w', zipfile.ZIP_DEFLATED) as archive:
        for path in sorted((ROOT/'parallel_effdet').glob('*.py')):
            ast.parse(path.read_text())
            info = zipfile.ZipInfo(path.relative_to(ROOT).as_posix(), date_time=(2026, 10, 22, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            archive.writestr(info, path.read_bytes())
    checksum = hashlib.sha256(bundle.read_bytes()).hexdigest()
    outputs = []
    for account, first, last, name in [
        ('4gmail', 12, 16, 'AgriVision_GridSearch_EfficientDet.ipynb'),
        ('71gmail', 1, 11, 'AgriVision_EfficientDet_Trials01_11_71gmail.ipynb')]:
        notebook = json.loads((ROOT/'AgriVision_GridSearch_SSD_Pair.ipynb').read_text(encoding='utf-8'))
        notebook['metadata']['colab']['name'] = name
        notebook['cells'][0]['source'] = [
            f'# EfficientDet assigned screening trials {first}-{last}: {account}\n',
            '\n',
            'Select T4 and Run all. This notebook only runs its assigned five-round screening trials, using the original common initialization and benchmarked FP32 execution. Committed trials/rounds are skipped. An unfinished round repeats.\n',
            '\n',
            '4gmail handles trials 12-16; 71gmail handles trials 1-11. Each saves independently. This notebook stops after its assigned range. Combine all 16 screening results on 71gmail before choosing finalists and starting final training. No local finalist selection is performed.\n',
        ]
        for cell in notebook['cells']:
            cell['metadata'] = {}
            if cell['cell_type'] == 'code':
                source = ''.join(cell['source']).replace("MODELS = ['ssdlite', 'ssd_mobilenetv3']", "MODELS = ['efficientdet_d0']")
                cell['source'] = source.splitlines(keepends=True)
                cell['outputs'] = []
                cell['execution_count'] = None
        notebook['cells'][4]['source'] = ['## Assigned screening only\n',
            'Trial folders and checkpoints retain their original paths. Separate parallel status, assignment and ranking files record this job. The existing search plan and full screening ranking are preserved.\n']
        source = f'''extension = DRIVE_ROOT / "efficientdet_fp32_continuation_v1.zip"
assert sha(extension) == "35e010acf0d7bfebd3ff585d7cc33bdabd9bc009371d9e01b2fcc4f27ca5820e"
parallel = DRIVE_ROOT / "efficientdet_parallel_screening_v1.zip"
assert sha(parallel) == "{checksum}"
for path, prefixes in [(extension, ("continuation/", "benchmarks/")), (parallel, ("parallel_effdet/",))]:
    with zipfile.ZipFile(path) as archive:
        for name in archive.namelist():
            assert (WORKSPACE/name).resolve().is_relative_to(WORKSPACE.resolve())
            assert name.startswith(prefixes)
        archive.extractall(WORKSPACE)
model_folder = OUTPUT / "efficientdet_d0"
assert sha(model_folder/"initial.pt") == "dd0a92a8261e15f8776deba607c6ef0b9bab4389f3705a656c4a8050b45c6fce"
assert (model_folder/"execution_profile.json").is_file(), "Shared execution profile missing"
command = [sys.executable, "-u", "-m", "parallel_effdet.screen", "--root", str(WORKSPACE),
    "--output", str(OUTPUT), "--first", "{first}", "--last", "{last}", "--workers", str(WORKERS)]
process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
for line in process.stdout:
    print(line, end="", flush=True)
if process.wait():
    raise subprocess.CalledProcessError(process.returncode, command)
print("Assigned screening finished. Combine all 16 trials before finalist selection.")
'''
        notebook['cells'][5]['source'] = source.splitlines(keepends=True)
        notebook['cells'][-1]['source'] = ['## Resume\n',
            f'Reopen this same notebook on {account}, select T4, and Run all after a session ends. Results remain in Agri Vision/grid_search/grid_v1_20261017/efficientdet_d0/trials/. Only trials {first}-{last} are eligible.\n',
            '\n',
            'Do not run the original unbounded 71gmail search once it reaches trial 12 while 4gmail owns trials 12-16. Use the supplied bounded 71gmail notebook on its next restart. After both ranges complete, copy trials 12-16 with their configurations and metrics into the canonical 71gmail model folder, verify hashes and identities, then use the original continuation notebook for global selection and final training.\n']
        for cell in notebook['cells']:
            if cell['cell_type'] == 'code':
                ast.parse(''.join(cell['source']))
        destination = ROOT/name
        destination.write_text(json.dumps(notebook, indent=2)+'\n', encoding='utf-8')
        outputs.append(dict(account=account, file=name, first=first, last=last,
            sha256=hashlib.sha256(destination.read_bytes()).hexdigest()))
    (REPORT/'package.json').write_text(json.dumps(dict(bundle=bundle.name, sha256=checksum,
        notebooks=outputs, training_started=False), indent=2)+'\n')
    print(json.dumps(dict(bundle=str(bundle), sha256=checksum, notebooks=outputs), indent=2))


if __name__ == '__main__':
    main()
