"""Create an EfficientDet-only Colab resume notebook without changing runner code."""
import ast
import json
from pathlib import Path

root = Path(__file__).resolve().parents[1]
source = root / 'AgriVision_Comparison_10Round.ipynb'
notebook = json.loads(source.read_text(encoding='utf-8'))
changes = 0
for cell in notebook['cells']:
    if cell['cell_type'] != 'code':
        continue
    text = ''.join(cell['source'])
    if 'RESUME_RUNS = {}' in text:
        text = text.replace('MODE = "pilot"', 'MODE = "main"')
        text = text.replace('MODELS = ["yolo_v8n", "ssdlite", "ssd_mobilenetv3", "efficientdet_d0"]', 'MODELS = ["efficientdet_d0"]')
        text = text.replace('RESUME_RUNS = {}', 'RESUME_RUNS = {"efficientdet_d0": DRIVE_ROOT / "comparison_results" / "efficientdet_d0_non_chili_seed42_20261009_224152_449800"}')
        # DRIVE_ROOT is defined before RESUME_RUNS in the original settings cell.
        cell['source'] = text.splitlines(keepends=True)
        changes += 1
    cell['outputs'] = []
    cell['execution_count'] = None
assert changes == 1
settings = next(''.join(c['source']) for c in notebook['cells'] if c['cell_type'] == 'code' and 'RESUME_RUNS =' in ''.join(c['source']))
ast.parse(settings)
assert 'MODE = "main"' in settings and 'MODELS = ["efficientdet_d0"]' in settings
assert 'ROUNDS = 10' in settings and 'LOCAL_EPOCHS = 1' in settings
destination = root / 'AgriVision_EfficientDet_Resume.ipynb'
destination.write_text(json.dumps(notebook, indent=1, ensure_ascii=False), encoding='utf-8')
print(destination)
