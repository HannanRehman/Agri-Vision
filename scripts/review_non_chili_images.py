"""Create labelled audit sheets for manual review; does not alter dataset images."""
from collections import defaultdict
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--classes-only', action='store_true')
    args = parser.parse_args()
    manifest = ROOT / 'manifests/non_chili_v1'
    summary = json.loads((manifest / 'audit.json').read_text())
    records = {r['key']: r for r in map(json.loads, (manifest / 'inventory.jsonl').read_text().splitlines())}
    output = ROOT / 'reports/non_chili_v1'
    output.mkdir(exist_ok=True)
    names = {int(k): v for k, v in summary['source_classes'].items()}

    def draw(ax, row, title):
        with Image.open(ROOT / row['source_image']) as image:
            ax.imshow(image.convert('RGB'))
        for box in row['boxes']:
            x, y, w, h = box['coords']
            width, height = row['width'], row['height']
            ax.add_patch(Rectangle(((x-w/2)*width, (y-h/2)*height), w*width, h*height,
                                   fill=False, edgecolor='#00ff70', linewidth=1.6))
        ax.set_title(title, fontsize=9)
        ax.axis('off')

    for split in ['train', 'valid', 'test']:
        by_class = defaultdict(list)
        for row in records.values():
            if row['reason'] is None and row['split'] == split:
                for class_id in {b['class_id'] for b in row['boxes']}:
                    by_class[class_id].append(row)
        fig, axes = plt.subplots(6, 4, figsize=(16, 22))
        for ax, old in zip(axes.flat, sorted(map(int, summary['class_map']))):
            if by_class[old]:
                sample = sorted(by_class[old], key=lambda r: r['key'])[0]
                draw(ax, sample, names[old].replace('_', ' ') + f'\n{split} / {sample["client"]}')
            else:
                ax.set_title(names[old] + ': MISSING')
                ax.axis('off')
        fig.suptitle(f'non_chili_v1: {split} annotation samples (all 24 classes)', fontsize=18)
        fig.tight_layout(rect=(0, 0, 1, .975))
        fig.savefig(output / f'{split}_class_samples.jpg', dpi=110)
        plt.close(fig)
    pairs = [] if args.classes_only else summary['near_duplicate_candidate_pairs']
    for page in range((len(pairs)+5)//6):
        fig, axes = plt.subplots(6, 2, figsize=(10, 22))
        for index, (left, right) in enumerate(pairs[page*6:(page+1)*6]):
            for column, key in enumerate([left, right]):
                row = records[key]
                draw(axes[index, column], row, f'{key[:65]}\n{row["source_identity"][:65]}')
        for ax in axes.flat[len(pairs[page*6:(page+1)*6])*2:]:
            ax.axis('off')
        fig.tight_layout()
        fig.savefig(output / f'near_duplicate_review_{page+1:03d}.jpg', dpi=100)
        plt.close(fig)
    print(f'Wrote 3 class sheets and {(len(pairs)+5)//6} near-duplicate sheets to {output}')


if __name__ == '__main__':
    main()
