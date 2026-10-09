"""Record the explicit per-model results layout; never trains a model."""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

def main():
    old = ('Use checkpoints/ for best and last global weights, metrics/ for round/client CSVs and summaries, '
           'logs/ for progress, figures/ for charts, predictions/ for selected examples and evaluations/ for final evaluation exports.')
    new = ('Use checkpoints/ for best and last global weights and logs/ for progress. '
           'Each model has a separate results/ folder: results/metrics/ for round/client CSVs and summaries, '
           'results/figures/ for charts, results/predictions/ for selected examples and results/evaluations/ for final evaluator exports.')
    for path in sorted((ROOT/'configs/comparison').glob('*.json')):
        if path.name == 'protocol.json':
            continue
        spec = json.loads(path.read_text())
        folder = ROOT/spec['run_directory']
        for name in ['checkpoints','logs','results/metrics','results/figures','results/predictions','results/evaluations']:
            assert (folder/name).is_dir(), str(folder/name)
        spec['results_directory'] = spec['run_directory']+'/results'
        spec['checkpoints_directory'] = spec['run_directory']+'/checkpoints'
        spec['logs_directory'] = spec['run_directory']+'/logs'
        path.write_text(json.dumps(spec,indent=2)+'\n',encoding='utf-8')
        readme = folder/'README.md'
        text = readme.read_text(encoding='utf-8')
        assert old in text or new in text
        readme.write_text(text.replace(old,new),encoding='utf-8')
        (folder/'results/README.md').write_text(
            f'# Results: {spec["display_name"]}\n\n**Status: empty scaffold; no results yet.**\n\n'
            'metrics/ holds round/client scores and final summaries; figures/ holds charts; '
            'predictions/ holds annotated examples; evaluations/ holds evaluator exports. '
            'These outputs belong only to this model/run. Checkpoints and progress logs are in the parent run folder.\n',
            encoding='utf-8')
    print('Verified and documented independent result folders for all four models.')

if __name__ == '__main__':
    main()
