"""Independently verify prepared files, frozen manifests and original provenance."""
import csv
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import os
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]


def digest(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def canonical(path):
    # These freshly copied files have no symlinks. Normalize '..' without repeated filesystem resolution.
    return Path(os.path.abspath(path))


def verify(root=ROOT):
    manifest = root / 'manifests/non_chili_v1'
    audit = json.loads((manifest / 'audit.json').read_text())
    inventory = {r['key']: r for r in map(json.loads, (manifest / 'inventory.jsonl').read_text().splitlines())}
    with (manifest / 'retained.csv').open(newline='', encoding='utf-8') as stream:
        rows = list(csv.DictReader(stream))
    assert {r['key'] for r in rows} == {k for k, r in inventory.items() if r['reason'] is None}
    assert len({r['key'] for r in rows}) == len(rows)
    names = {int(k): v for k, v in audit['new_classes'].items()}
    mapping = {int(k): int(v) for k, v in audit['class_map'].items()}
    assert sorted(names) == list(range(24)) and not any(n.startswith('Chili_') for n in names.values())
    checksums = json.loads((manifest / 'checksums.json').read_text())
    for path, expected in checksums.items():
        assert digest(root / path) == expected, f'Manifest changed: {path}'

    def check_file(row):
        original = inventory[row['key']]
        image, label = root / row['image'], root / row['label']
        assert digest(image) == row['image_sha256'] == digest(root / original['source_image'])
        source_bytes = (root / original['source_label']).read_bytes()
        target_bytes = label.read_bytes()
        assert hashlib.sha256(target_bytes).hexdigest() == row['label_sha256']
        assert hashlib.sha256(source_bytes).hexdigest() == original['label_sha256']
        source_tokens = [line.split() for line in source_bytes.decode('utf-8').splitlines() if line.strip()]
        target_tokens = [line.split() for line in target_bytes.decode('utf-8').splitlines() if line.strip()]
        assert len(source_tokens) == len(target_tokens) == int(row['instances'])
        ids = set()
        for old, new in zip(source_tokens, target_tokens):
            assert int(old[0]) in mapping and int(new[0]) == mapping[int(old[0])]
            assert old[1:] == new[1:], f'Coordinates changed: {row["key"]}'
            ids.add(int(new[0]))
        return ids

    with ThreadPoolExecutor(max_workers=8) as pool:
        class_sets = []
        for index,ids in enumerate(pool.map(check_file, rows),1):
            class_sets.append(ids)
            if index % 5000 == 0:
                print(f'VERIFY {index}/{len(rows)} images',flush=True)
    print('Image and label hashes checked; checking split and client coverage.',flush=True)
    groups, pixels, sources = defaultdict(set), defaultdict(set), defaultdict(set)
    for row in rows:
        original = inventory[row['key']]
        groups[row['group_id']].add((row['split'], row['client']))
        pixels[original['decoded_sha256']].add((row['split'], row['client']))
        sources[original['source_identity']].add((row['split'], row['client']))
    assert all(len(locations) == 1 for collection in [groups, pixels, sources] for locations in collection.values())
    assert len(pixels) == len(rows), 'Identical decoded images remain'
    visual_path = manifest / 'visual_review.json'
    if visual_path.exists():
        decisions = json.loads(visual_path.read_text())['decisions']
        assert {tuple(d['pair']) for d in decisions} == {tuple(pair) for pair in audit['near_duplicate_candidate_pairs']}
        for decision in decisions:
            if decision['decision'] == 'same_source':
                left,right = [inventory[key] for key in decision['pair']]
                assert left['group_id'] == right['group_id']
                assert left['reason'] is not None or right['reason'] is not None or (left['split'],left['client']) == (right['split'],right['client'])
    client_counts = {}
    for client in ['client1', 'client2', 'client3']:
        directory = root / 'dataset_non_chili' / client
        config = yaml.safe_load((directory / 'data.yaml').read_text())
        assert {int(k): v for k, v in config['names'].items()} == names and config['nc'] == 24
        client_counts[client] = {}
        for split, yaml_key in [('train', 'train'), ('valid', 'val'), ('test', 'test')]:
            assert config[yaml_key] == f'../../original_dataset_non_chili/{client}_{split}.txt'
            list_path = canonical(directory / config[yaml_key])
            paths = [canonical(list_path.parent / line) for line in list_path.read_text().splitlines()]
            expected = {canonical(root / r['image']) for r in rows if r['client'] == client and r['split'] == split}
            assert len(paths) == len(set(paths)) and set(paths) == expected and all(p.is_file() for p in paths)
            client_counts[client][split] = len(paths)
    for split in ['train', 'valid', 'test']:
        selected = [r for r in rows if r['split'] == split]
        directory = root / 'original_dataset_non_chili' / split
        assert {canonical(p) for p in (directory / 'images').iterdir()} == {canonical(root / r['image']) for r in selected}
        assert {canonical(p) for p in (directory / 'labels').iterdir()} == {canonical(root / r['label']) for r in selected}
        assert set().union(*(ids for r, ids in zip(rows, class_sets) if r['split'] == split)) == set(range(24))
    report = dict(status='passed', images=len(rows), split_counts=dict(Counter(r['split'] for r in rows)),
                  client_counts=client_counts, classes=24, coordinate_tokens_preserved=True,
                  retained_original_files_unchanged=True, exact_and_inferred_source_group_overlap=0,
                  near_duplicate_candidates=len(audit['near_duplicate_candidate_pairs']),
                  visual_review=audit.get('visual_review'),
                  near_duplicate_review='Reviewed decisions enforced when present; this check does not certify semantic labels or exhaustively detect near duplicates.',
                  training_started=False)
    (manifest / 'verification.json').write_text(json.dumps(report, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(report, indent=2))
    return report


if __name__ == '__main__':
    verify()
