"""Create a thesis comparison table only from compatible completed main runs."""
import argparse
import csv
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--runs',nargs='+',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args = parser.parse_args()
    records = []; contracts = []; seen = set()
    for run in args.runs:
        settings = json.loads((run/'settings.json').read_text())
        result = json.loads((run/'results/evaluations/best_test.json').read_text())
        config = settings['config']; model = config['model']
        if result['diagnostic_only'] or settings['diagnostic_only']:
            raise ValueError('Pilot metrics are not final thesis results')
        if model in seen:
            raise ValueError('Duplicate model in comparison')
        seen.add(model)
        contracts.append((config['dataset_checksums_sha256'],config['seed'],config['rounds'],config['local_epochs'],
                          settings['hardware']['gpu'],result['hardware'],config['augmentation'],config['normalization_policy'],
                          config['evaluator'],config['implementation_sha256'],settings['hardware']['torch'],
                          tuple(sorted(settings['hardware']['packages'].items()))))
        metrics = result['metrics']; runtime = result['runtime']; size = result['model']
        records.append(dict(model=model,map_50_95=metrics['map_50_95'],map50=metrics['map50'],
                            precision=metrics['precision'],recall=metrics['recall'],f1=metrics['f1'],
                            parameters=size['parameters'],input_size=size['input_size'],
                            weights_mb=size['tensor_state_bytes']/1e6,batch1_median_ms=runtime['median_ms'],
                            selected_round=result['selected_round'],test_images=metrics['images']))
    if len(set(contracts))!=1:
        raise ValueError('Dataset/seed/budget/GPU/code/packages/augmentation/normalization/evaluation contracts differ')
    args.output.parent.mkdir(parents=True,exist_ok=True)
    with args.output.open('w',newline='',encoding='utf-8') as stream:
        writer = csv.DictWriter(stream,fieldnames=list(records[0])); writer.writeheader(); writer.writerows(records)
    print(f'Saved compatible main-run comparison: {args.output}')


if __name__=='__main__':
    main()
