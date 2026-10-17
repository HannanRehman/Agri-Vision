"""Build final/best and per-class comparison tables from saved validation results."""
import csv
import hashlib
import json
import math
from pathlib import Path

HERE = Path(__file__).resolve().parent
MODELS = [('yolo_v8n', 'YOLOv8n'), ('ssdlite', 'SSDLite'),
          ('ssd_mobilenetv3', 'SSD-MobileNetV3'), ('efficientdet_d0', 'EfficientDet-D0')]
SCORES = {'precision': 'precision_percent', 'recall': 'recall_percent',
          'f1': 'f1_percent', 'map50': 'map50_percent', 'map_50_95': 'map50_95_percent'}

def percent_scores(row):
    return {column: f'{float(row[key])*100:.2f}' for key, column in SCORES.items()}

def write_csv(filename, fields, rows):
    path = HERE / filename
    with path.open('w', encoding='utf-8-sig', newline='') as output:
        writer = csv.DictWriter(output, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    with path.open(encoding='utf-8-sig', newline='') as saved:
        assert len(list(csv.DictReader(saved))) == len(rows)
    return path

summary, per_class, evidence = [], [], []
mapping = None
for model, label in MODELS:
    history_path = HERE / 'sources' / f'{model}_history.csv'
    detail_path = HERE / 'sources' / f'{model}_round_030.json'
    with history_path.open(newline='', encoding='utf-8-sig') as saved:
        history = list(csv.DictReader(saved))
    history.sort(key=lambda row: int(row['round']))
    assert [int(row['round']) for row in history] == list(range(1, 31))
    final = history[-1]
    best = max(history, key=lambda row: float(row['map_50_95']))
    details = json.loads(detail_path.read_text(encoding='utf-8'))
    pooled = details['pooled_validation']
    assert details['round'] == 30
    assert pooled['images'] == 4157 and pooled['ground_truth_boxes'] == 7210
    for score in SCORES:
        assert math.isclose(float(final[score]), pooled[score], rel_tol=1e-12)
    for kind, row in [('round_30', final), ('best_round', best)]:
        assert int(row['images']) == 4157
        summary.append(dict(model=label, selection=kind, round=int(row['round']),
                            **percent_scores(row), validation_images=int(row['images']),
                            selection_metric='mAP50-95'))
    classes = pooled['per_class']
    assert len(classes) == 24
    current_mapping = {row['class_id']: row['class_name'] for row in classes}
    assert set(current_mapping) == set(range(24))
    if mapping is None:
        mapping = current_mapping
    assert current_mapping == mapping
    for count in ('true_positives', 'false_positives', 'false_negatives'):
        assert sum(row[count] for row in classes) == pooled[count]
    for row in sorted(classes, key=lambda row: row['class_id']):
        assert all(0 <= row[score] <= 1 for score in SCORES)
        per_class.append(dict(model=label, round=30, class_id=row['class_id'], class_name=row['class_name'],
                              **percent_scores(row), true_positives=row['true_positives'],
                              false_positives=row['false_positives'], false_negatives=row['false_negatives'],
                              ground_truth_boxes=row['true_positives']+row['false_negatives']))
    evidence.append(dict(model=label, final_round=30, best_round=int(best['round']),
                         history=history_path.relative_to(HERE).as_posix(),
                         history_sha256=hashlib.sha256(history_path.read_bytes()).hexdigest(),
                         detail=detail_path.relative_to(HERE).as_posix(),
                         detail_sha256=hashlib.sha256(detail_path.read_bytes()).hexdigest()))
    print(label, 'round 30:', percent_scores(final), 'best round:', best['round'])

summary_path = write_csv('All_Models_Round30_and_Best.csv',
                         ['model', 'selection', 'round', *SCORES.values(), 'validation_images', 'selection_metric'], summary)
classes_path = write_csv('All_Models_Round30_Per_Class.csv',
                         ['model', 'round', 'class_id', 'class_name', *SCORES.values(),
                          'true_positives', 'false_positives', 'false_negatives', 'ground_truth_boxes'], per_class)
assert len(summary) == 8 and len(per_class) == 96
(HERE/'validation.json').write_text(json.dumps(dict(
    date='2026-10-15', split='pooled validation', score_scale='percent 0-100, rounded to two decimals',
    best_selection='highest unrounded pooled mAP50-95 among completed rounds 1-30; first occurrence breaks ties',
    summary_rows=len(summary), per_class_rows=len(per_class), classes_per_model=24,
    validation_images=4157, ground_truth_boxes=7210,
    precision_recall_confidence=.25, precision_recall_iou=.5,
    source_round30_json_matches_history=True, class_counts_match_pooled=True,
    training_started=False, fresh_evaluation_started=False, sources=evidence,
    outputs=[dict(file=p.name, sha256=hashlib.sha256(p.read_bytes()).hexdigest())
             for p in (summary_path,classes_path)]), indent=2)+'\n', encoding='utf-8')
print('Saved comparison: 8 rows. Saved round-30 class metrics: 96 rows.')
