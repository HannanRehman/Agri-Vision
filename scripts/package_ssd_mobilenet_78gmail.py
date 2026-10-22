"""Prepare the separate 78gmail SSD-MobileNetV3 notebook without training."""
import ast
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
NAME = 'AgriVision_GridSearch_SSD_MobileNetV3_78gmail.ipynb'


def main():
    notebook = json.loads((ROOT/'AgriVision_GridSearch_SSD_Pair.ipynb').read_text(encoding='utf-8'))
    notebook['metadata']['colab']['name'] = NAME
    notebook['cells'][0]['source'] = [
        '# Agri Vision: SSD-MobileNetV3 grid search - 78gmail\n',
        '\n',
        'Sign in as hannanrehman78@gmail.com. Keep the supplied Agri Vision folder in My Drive. Select T4 GPU and Run all. This notebook runs only ssd_mobilenetv3.\n',
        '\n',
        'The original search settings are preserved: 16 combinations of learning rate, weight decay, batch size and image resolution; five federated rounds per trial; the top two continue to ten rounds; the winner starts a fresh 30-round run. Each round trains three clients for one local epoch and aggregates their weights. Selection uses pooled validation mAP50-95.\n',
        '\n',
        'This is a fresh search for this architecture. Existing SSDLite trials remain on 40gmail. Reopen this notebook and Run all after a disconnection: committed rounds are skipped and an unfinished round repeats.\n',
    ]
    for cell in notebook['cells']:
        cell['metadata'] = {}
        if cell['cell_type'] == 'code':
            source = ''.join(cell['source'])
            source = source.replace("MODELS = ['ssdlite', 'ssd_mobilenetv3']", "MODELS = ['ssd_mobilenetv3']")
            source = source.replace('subprocess.run([sys.executable, "-m", "tuning.search",', 'subprocess.run([sys.executable, "-u", "-m", "tuning.search",')
            ast.parse(source)
            cell['source'] = source.splitlines(keepends=True)
            cell['outputs'] = []
            cell['execution_count'] = None
    notebook['cells'][-1]['source'] = [
        '## Resume and results\n',
        'Run all again with the same T4 setup and unchanged uploads after a session ends. Results are stored under Agri Vision/grid_search/grid_v1_20261017/ssd_mobilenetv3/. The search_status.json file is in grid_v1_20261017; trial status.json shows current activity and latest.json identifies the last committed checkpoint.\n',
        '\n',
        'screening_results.csv ranks the 16 trials. finalist_results.csv compares the two finalists. selected_configuration.json records the winning settings. final_training/ holds the fresh 30-round run, with best and latest checkpoints. Training and validation use the shared study dataset; held-out testing is not part of the search.\n',
    ]
    output = ROOT/NAME
    output.write_text(json.dumps(notebook, indent=2, ensure_ascii=False)+'\n', encoding='utf-8')
    report = ROOT/'reports/comparison/ssd_mobilenet_78gmail_20261021'
    report.mkdir(parents=True, exist_ok=True)
    readme = report/'START_HERE_SSD_MobileNetV3_78gmail.md'
    readme.write_text(
        '# SSD-MobileNetV3 grid search on 78gmail\n\n'
        'Account: hannanrehman78@gmail.com\n\n'
        'Open AgriVision_GridSearch_SSD_MobileNetV3_78gmail.ipynb in Colab, select a T4 GPU, and choose Run all. Authorize Google Drive as 78gmail. The supplied Agri Vision folder must be in My Drive under that exact name.\n\n'
        'Required uploads: this notebook, agrivision_grid_search_v1_code.zip, and non_chili_v1_upload_parts/ containing parts.json plus all 14 dataset parts. The notebook verifies dataset and code SHA256 before execution.\n\n'
        'Only SSD-MobileNetV3 runs here. Sixteen five-round screening trials are followed automatically by two ten-round confirmations and a fresh 30-round run of the selected configuration. Reopen the same notebook and Run all to resume. Keep the uploads and settings unchanged.\n\n'
        '40gmail continues SSDLite separately. Its original SSD pair notebook also queues SSD-MobileNetV3 after SSDLite finishes all stages; stop that session before it starts the queued model to avoid a duplicate experiment. No changes were made to its active session.\n\n'
        'This preparation does not launch GPU training. No prior SSD-MobileNetV3 grid-search checkpoints existed on 40gmail when the separate notebook was prepared. The 30-round baseline checkpoints are separate and are not search initialization.\n', encoding='utf-8')
    print(json.dumps(dict(notebook=NAME, bytes=output.stat().st_size,
        sha256=hashlib.sha256(output.read_bytes()).hexdigest(), models=['ssd_mobilenetv3'],
        gpu_training_started=False, readme=str(readme)), indent=2))


if __name__ == '__main__':
    main()
