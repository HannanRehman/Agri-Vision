"""Export saved final-round class metrics; no training or fresh evaluation."""
import csv
import hashlib
import json
from pathlib import Path
from zipfile import ZipFile, ZIP_DEFLATED

HERE = Path(__file__).resolve().parent
RUNS = [
    ('yolo_v8n', 30, '1YLkn0B-rKZu3mCbmlpQay5Owzy8jmShI'),
    ('ssdlite', 30, '13O2JpB0a0BRFBIK_0dqRXOkhVrTpjZfi'),
    ('ssd_mobilenetv3', 30, '1QNb-q_fFRhlXcdGJckdGAjl_KPM0W_78'),
    ('efficientdet_d0', 15, '10JHDqnTAJHRNCqGqwTsqbbDNrObjaHS0'),
]
FIELDS = ['model', 'round', 'class_id', 'class_name', 'precision', 'recall',
          'f1', 'map50', 'map_50_95', 'true_positives', 'false_positives',
          'false_negatives', 'ground_truth_boxes']

def write_csv(path, rows):
    with path.open('w', encoding='utf-8-sig', newline='') as output:
        writer = csv.DictWriter(output, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    with path.open(encoding='utf-8-sig', newline='') as saved:
        assert len(list(csv.DictReader(saved))) == len(rows)

combined = []
manifest = []
class_mapping = None
for model, round_number, drive_id in RUNS:
    source = HERE / 'sources' / f'{model}_round_{round_number:03}.json'
    data = json.loads(source.read_text(encoding='utf-8'))
    pooled = data['pooled_validation']
    assert data['round'] == round_number
    classes = pooled['per_class']
    assert len(classes) == 24
    mapping = {row['class_id']: row['class_name'] for row in classes}
    assert set(mapping) == set(range(24))
    if class_mapping is None:
        class_mapping = mapping
    assert mapping == class_mapping
    assert pooled['images'] == 4157 and pooled['ground_truth_boxes'] == 7210
    for key in ('true_positives', 'false_positives', 'false_negatives'):
        assert sum(row[key] for row in classes) == pooled[key]
    rows = []
    for row in sorted(classes, key=lambda item: item['class_id']):
        exported = {'model': model, 'round': round_number, **row,
                    'ground_truth_boxes': row['true_positives'] + row['false_negatives']}
        for metric in ('precision', 'recall', 'f1', 'map50', 'map_50_95'):
            assert 0 <= exported[metric] <= 1
        rows.append(exported)
    assert sum(row['ground_truth_boxes'] for row in rows) == 7210
    dest = HERE / f'{model}_round_{round_number:03}_per_class.csv'
    write_csv(dest, rows)
    combined.extend(rows)
    manifest.append({'model': model, 'round': round_number, 'class_rows': len(rows),
                     'csv': dest.name, 'source': source.relative_to(HERE).as_posix(),
                     'source_drive_url': f'https://drive.google.com/file/d/{drive_id}/view',
                     'saved_source_sha256': hashlib.sha256(source.read_bytes()).hexdigest(),
                     'csv_sha256': hashlib.sha256(dest.read_bytes()).hexdigest(),
                     'validation_images': pooled['images'],
                     'precision_recall_confidence': pooled['precision_recall_confidence'],
                     'precision_recall_iou': pooled['precision_recall_iou']})
    print(f'{dest.name}: {len(rows)} class rows')

combined_file = HERE / 'all_models_final_round_per_class.csv'
write_csv(combined_file, combined)
(HERE / 'manifest.json').write_text(json.dumps({
    'fetched_date': '2026-10-13', 'scope': 'final completed global round; pooled validation',
    'training_started': False, 'fresh_evaluation_started': False,
    'metric_scale': '0 to 1; multiply by 100 for percentages',
    'ground_truth_boxes': 'derived per class as true_positives + false_negatives',
    'runs': manifest,
}, indent=2) + '\n', encoding='utf-8')
with ZipFile(HERE / 'Agri_Vision_Final_Round_Per_Class_CSVs.zip', 'w', ZIP_DEFLATED) as archive:
    for path in sorted(HERE.glob('*.csv')):
        archive.write(path, path.name)
    for path in [HERE / 'README.md', HERE / 'manifest.json', *sorted((HERE / 'sources').glob('*.json'))]:
        archive.write(path, path.relative_to(HERE).as_posix())
print('Combined CSV: 96 rows; verified shared class mapping and pooled counts.')
