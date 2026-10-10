"""Audit and build a separate, versioned non-chili dataset; never trains."""
import argparse
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import hashlib
import io
import json
import math
import os
from pathlib import Path
import re
import shutil

import yaml
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
CLIENTS = ('client1', 'client2', 'client3')
SPLITS = ('train', 'valid', 'test')
SUFFIXES = {'.jpg', '.jpeg', '.png', '.bmp', '.webp'}
PRIORITY = {'train': 0, 'valid': 1, 'test': 2}

def sha_file(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()

def names_from(path):
    raw = yaml.safe_load(path.read_text(encoding='utf-8'))['names']
    names = dict(enumerate(raw)) if isinstance(raw, list) else {int(k): v for k, v in raw.items()}
    if sorted(names) != list(range(len(names))):
        raise ValueError(f'Non-consecutive classes: {path}')
    return names

def parse_labels(text, class_count):
    rows = []
    for line in text.splitlines():
        if not line.strip():
            continue
        values = line.split()
        if len(values) != 5:
            raise ValueError('Expected class ID and four box coordinates')
        class_id = int(values[0])
        coords = list(map(float, values[1:]))
        if not 0 <= class_id < class_count or not all(math.isfinite(v) for v in coords):
            raise ValueError('Invalid class ID or non-finite coordinates')
        x, y, w, h = coords
        if not (0 <= x <= 1 and 0 <= y <= 1 and 0 < w <= 1 and 0 < h <= 1):
            raise ValueError('Invalid normalized box')
        if min(x-w/2, y-h/2) < -0.0001 or max(x+w/2, y+h/2) > 1.0001:
            raise ValueError('Box exceeds image boundaries')
        rows.append({'class_id': class_id, 'coords': coords, 'tokens': values[1:]})
    return rows

def label_signature(rows):
    return sorted((r['class_id'], *(round(v, 5) for v in r['coords'])) for r in rows)

def source_identity(filename, crops):
    stem = re.sub(r'\.rf\.[0-9a-f]+$', '', Path(filename).stem, flags=re.I)
    return '|'.join(sorted(crops)) + ':' + stem.casefold()

def remap_text(rows, mapping):
    return ''.join(str(mapping[r['class_id']])+' '+' '.join(r['tokens'])+'\n' for r in rows)

class Groups:
    def __init__(self, size):
        self.parent = list(range(size))
    def find(self, i):
        while self.parent[i] != i:
            self.parent[i] = self.parent[self.parent[i]]
            i = self.parent[i]
        return i
    def join(self, a, b):
        self.parent[self.find(b)] = self.find(a)

def choose_records(records, reviewed_pairs=()):
    """Group source variants; retain held-out copies, quarantine conflicts."""
    candidates = [r for r in records if r['reason'] is None]
    groups = Groups(len(candidates))
    for field in ['image_sha256', 'decoded_sha256', 'source_identity']:
        seen = {}
        for i, row in enumerate(candidates):
            key = row[field]
            if key in seen:
                groups.join(i, seen[key])
            else:
                seen[key] = i
    by_key = {r['key']:i for i,r in enumerate(candidates)}
    for left,right in reviewed_pairs:
        if left not in by_key or right not in by_key:
            raise ValueError('Reviewed source pair is not a valid non-chili candidate')
        groups.join(by_key[left],by_key[right])
    buckets = defaultdict(list)
    for i, row in enumerate(candidates):
        buckets[groups.find(i)].append(row)
    changes, conflicts = [], []
    for bucket in buckets.values():
        keys = sorted(r['key'] for r in bucket)
        group_id = hashlib.sha256('\n'.join(keys).encode()).hexdigest()[:20]
        exact = defaultdict(list)
        for row in bucket:
            row['group_id'] = group_id
            exact[row['decoded_sha256']].append(row)
        bad = False
        for copies in exact.values():
            if len({json.dumps(label_signature(r['boxes'])) for r in copies}) > 1:
                conflicts.append({'group_id': group_id, 'files': [r['key'] for r in copies]})
                bad = True
        if bad:
            for row in bucket:
                row['reason'] = 'conflicting_annotations_on_identical_image'
            continue
        selected_split = max((r['split'] for r in bucket), key=PRIORITY.get)
        kept = []
        seen_pixels = set()
        for row in sorted(bucket, key=lambda r: r['key']):
            if row['split'] != selected_split:
                row['reason'] = 'source_group_cross_split_overlap'
            elif row['decoded_sha256'] in seen_pixels:
                row['reason'] = 'duplicate_image_within_split'
            else:
                seen_pixels.add(row['decoded_sha256'])
                kept.append(row)
        ownership = Counter(r['original_client'] for r in kept)
        client = sorted(ownership, key=lambda c: (-ownership[c], c))[0]
        for row in kept:
            row['client'] = client
            if row['original_client'] != client:
                changes.append({'key': row['key'], 'old_client': row['original_client'],
                                'new_client': client, 'group_id': group_id})
    return changes, conflicts

def audit(root, workers):
    source, clients = root/'original_dataset', root/'dataset'
    names = names_from(source/'data.yaml')
    excluded = {i for i, name in names.items() if str(name).startswith('Chili_')}
    if excluded != set(range(13, 19)) or len(names) != 30:
        raise ValueError('Expected audited 30-class mapping with chili IDs 13–18')
    mapping = {old: new for new, old in enumerate(i for i in sorted(names) if i not in excluded)}
    owners = {}
    tasks = []
    orphan_labels = []
    for client in CLIENTS:
        if names_from(clients/client/'data.yaml') != names:
            raise ValueError(f'Client class mapping differs: {client}')
    for split in SPLITS:
        original_images = sorted(p for p in (source/split/'images').iterdir() if p.suffix.lower() in SUFFIXES)
        stems = [p.stem for p in original_images]
        if len(set(stems)) != len(stems):
            raise ValueError(f'Conflicting image stems in {split}')
        original_names = {p.name for p in original_images}
        for client in CLIENTS:
            for p in (clients/client/split/'images').iterdir():
                if p.suffix.lower() not in SUFFIXES:
                    continue
                key = (split, p.name)
                if key in owners:
                    raise ValueError(f'Image assigned to multiple clients: {key}')
                owners[key] = client
        if {name for (part, name) in owners if part == split} != original_names:
            raise ValueError(f'Client coverage differs from source: {split}')
        stem_set = set(stems)
        orphan_labels.extend(str(p.relative_to(root)) for p in (source/split/'labels').glob('*.txt') if p.stem not in stem_set)
        tasks.extend((split, path, owners[(split, path.name)]) for path in original_images)

    def inspect(task):
        split, path, owner = task
        label = source/split/'labels'/f'{path.stem}.txt'
        row = {'key': f'{split}/{path.name}', 'split': split, 'filename': path.name,
               'source_image': path.relative_to(root).as_posix(),
               'source_label': label.relative_to(root).as_posix(), 'original_client': owner,
               'client': owner, 'reason': None, 'boxes': [], 'group_id': None}
        try:
            raw = path.read_bytes()
            row['image_sha256'] = hashlib.sha256(raw).hexdigest()
            row['bytes'] = len(raw)
            if sha_file(clients/owner/split/'images'/path.name) != row['image_sha256']:
                raise RuntimeError('Client image differs from source')
            with Image.open(io.BytesIO(raw)) as image:
                image.load()
                rgb = image.convert('RGB')
                row['width'], row['height'] = rgb.size
                row['decoded_sha256'] = hashlib.sha256(str(rgb.size).encode()+rgb.tobytes()).hexdigest()
                small = rgb.convert('L').resize((9,8),Image.Resampling.LANCZOS)
                pixels = list(small.get_flattened_data())
                row['dhash'] = sum((pixels[y*9+x] > pixels[y*9+x+1]) << (y*8+x) for y in range(8) for x in range(8))
            text = label.read_text(encoding='utf-8')
            row['label_sha256'] = sha_file(label)
            row['boxes'] = parse_labels(text, len(names))
            client_rows = parse_labels((clients/owner/split/'labels'/label.name).read_text(encoding='utf-8'), len(names))
            if label_signature(client_rows) != label_signature(row['boxes']):
                raise RuntimeError('Client annotations differ from source')
            class_ids = {r['class_id'] for r in row['boxes']}
            row['source_identity'] = source_identity(path.name, {names[i].split('_')[0] for i in class_ids})
            if not class_ids:
                row['reason'] = 'empty_annotation_unknown_crop'
            elif class_ids & excluded:
                row['reason'] = 'mixed_chili_and_non_chili' if class_ids-excluded else 'chili_crop'
        except RuntimeError:
            raise  # Inconsistent client/source data must be resolved, not hidden.
        except Exception as error:
            row['reason'] = 'missing_or_invalid_image_or_annotation'
            row['error'] = f'{type(error).__name__}: {error}'
        return row

    records = []
    with ThreadPoolExecutor(max_workers=workers) as pool:
        for i, row in enumerate(pool.map(inspect, tasks), 1):
            records.append(row)
            if i % 500 == 0:
                print(f'AUDIT {i}/{len(tasks)} images', flush=True)
    changes, conflicts = choose_records(records)
    retained = [r for r in records if r['reason'] is None]
    missing_classes = {}
    for split in SPLITS:
        present = {b['class_id'] for r in retained if r['split'] == split for b in r['boxes']}
        missing_classes[split] = sorted(set(mapping)-present)
    # Perceptual candidates are review evidence, never automatically deleted.
    bands = defaultdict(list)
    pairs = set()
    for row in retained:
        crop = row['source_identity'].split(':',1)[0]
        for band in range(4):
            key = (crop,band,(row['dhash'] >> (16*band)) & 65535)
            for other in bands[key][-80:]:
                if row['split'] != other['split'] and (row['dhash'] ^ other['dhash']).bit_count() <= 2:
                    pairs.add(tuple(sorted((row['key'],other['key']))))
            bands[key].append(row)
    summary = {'created_at_utc': datetime.now(timezone.utc).isoformat(),
               'source_classes': names, 'class_map': mapping,
               'new_classes': {mapping[i]: names[i] for i in mapping},
               'source_counts': dict(Counter(r['split'] for r in records)),
               'retained_counts': {s:sum(r['split']==s for r in retained) for s in SPLITS},
               'exclusions': dict(Counter(r['reason'] for r in records if r['reason'])),
               'client_reassignments': changes, 'conflicting_image_groups': conflicts,
               'orphan_source_labels': orphan_labels, 'missing_retained_classes': missing_classes,
               'near_duplicate_candidate_pairs': sorted(pairs),
               'near_duplicate_scope': 'Same crop, 64-bit dHash distance <=2, across retained splits; bounded bucket search. Candidates require visual review.',
               'group_policy': 'Exact bytes, identical decoded RGB pixels and inferred Roboflow source identity; test > validation > train; preserve chosen split; majority client per group.',
               'original_data_changed': False, 'training_started': False}
    return records, summary

def apply_review(records, summary, path):
    review = json.loads(path.read_text(encoding='utf-8'))
    expected = {tuple(pair) for pair in summary['near_duplicate_candidate_pairs']}
    decisions = review['decisions']
    if {tuple(d['pair']) for d in decisions} != expected or len(decisions) != len(expected):
        raise ValueError('Visual decisions do not cover exactly the audited candidate pairs')
    if any(d['decision'] not in ['same_source','different_source'] for d in decisions):
        raise ValueError('Unresolved visual decision')
    derived = {'source_group_cross_split_overlap','duplicate_image_within_split',
               'conflicting_annotations_on_identical_image'}
    for row in records:
        if row['reason'] in derived:
            row['reason'] = None
        row['client'] = row['original_client']
    confirmed = [d['pair'] for d in decisions if d['decision']=='same_source']
    changes,conflicts = choose_records(records,confirmed)
    retained = [r for r in records if r['reason'] is None]
    mapping = {int(k):int(v) for k,v in summary['class_map'].items()}
    summary.update(retained_counts={s:sum(r['split']==s for r in retained) for s in SPLITS},
                   exclusions=dict(Counter(r['reason'] for r in records if r['reason'])),
                   client_reassignments=changes,conflicting_image_groups=conflicts,
                   visual_review={'pairs_reviewed':len(decisions),'same_source_pairs':len(confirmed),
                                  'different_source_pairs':len(decisions)-len(confirmed),
                                  'decision_file':path.name,'decision_sha256':sha_file(path)},
                   missing_retained_classes={s:sorted(set(mapping)-{b['class_id'] for r in retained if r['split']==s for b in r['boxes']}) for s in SPLITS})
    summary['group_policy'] += ' Visually confirmed perceptual candidate pairs also share a source group.' if 'Visually confirmed' not in summary['group_policy'] else ''

def write_csv(path, fieldnames, rows):
    import csv
    with path.open('w',newline='',encoding='utf-8') as stream:
        writer = csv.DictWriter(stream,fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

def save_audit(manifests, records, summary):
    manifests.mkdir(parents=True,exist_ok=True)
    (manifests/'audit.json').write_text(json.dumps(summary,indent=2)+'\n',encoding='utf-8')
    with (manifests/'inventory.jsonl').open('w',encoding='utf-8') as stream:
        for row in records:
            stream.write(json.dumps(row)+'\n')
    fields = ['key','reason','error','original_client','group_id']
    write_csv(manifests/'excluded.csv',fields,
              [{k:r.get(k,'') for k in fields} for r in records if r['reason']])
    (manifests/'class_map.json').write_text(json.dumps({'old_to_new':summary['class_map'], 'names':summary['new_classes']},indent=2)+'\n')

def build(root, manifests, records, summary, workers):
    final_prepared = root/'original_dataset_non_chili'
    final_clients = root/'dataset_non_chili'
    staging = root/'dataset_staging_non_chili_v1'
    if final_prepared.exists() or final_clients.exists() or staging.exists():
        raise FileExistsError('Prepared output already exists; refuse to overwrite a dataset version')
    if any(summary['missing_retained_classes'].values()):
        raise ValueError('A retained split lacks classes; inspect audit before building')
    if summary['near_duplicate_candidate_pairs'] and not summary.get('visual_review'):
        raise ValueError('Perceptual candidate review must be recorded before freezing')
    retained = [r for r in records if r['reason'] is None]
    for client in CLIENTS:
        for split in SPLITS:
            if not any(r['client'] == client and r['split'] == split for r in retained):
                raise ValueError(f'Empty client split: {client}/{split}')
    mapping = {int(k):int(v) for k,v in summary['class_map'].items()}
    prepared = staging/'original_dataset_non_chili'
    clients = staging/'dataset_non_chili'
    final_manifests = manifests
    manifests = staging/'manifests'
    manifests.mkdir(parents=True)
    for split in SPLITS:
        (prepared/split/'images').mkdir(parents=True)
        (prepared/split/'labels').mkdir(parents=True)
    def copy(row):
        image = prepared/row['split']/'images'/row['filename']
        source = root/row['source_image']
        if sha_file(root/row['source_label']) != row['label_sha256']:
            raise ValueError(f'Source label changed after audit: {row["key"]}')
        shutil.copy2(source,image)
        if sha_file(image) != row['image_sha256']:
            raise ValueError(f'Source image changed or copy failed: {row["key"]}')
        label = prepared/row['split']/'labels'/f'{image.stem}.txt'
        label.write_text(remap_text(row['boxes'],mapping),encoding='utf-8')
        return {'key':row['key'],'split':row['split'],'client':row['client'],
                'original_client':row['original_client'],'group_id':row['group_id'],
                'image':(final_prepared/image.relative_to(prepared)).relative_to(root).as_posix(),
                'label':(final_prepared/label.relative_to(prepared)).relative_to(root).as_posix(),
                'image_sha256':row['image_sha256'],'label_sha256':sha_file(label),
                'width':row['width'],'height':row['height'],'instances':len(row['boxes'])}
    output = []
    with ThreadPoolExecutor(max_workers=workers) as pool:
        for i, row in enumerate(pool.map(copy,retained),1):
            output.append(row)
            if i % 500 == 0:
                print(f'BUILD {i}/{len(retained)} images',flush=True)
    config = {'train':'train/images','val':'valid/images','test':'test/images','nc':24,
              'names':summary['new_classes']}
    (prepared/'data.yaml').write_text(yaml.safe_dump(config,sort_keys=False),encoding='utf-8')
    write_csv(manifests/'retained.csv',list(output[0]),output)
    for split in SPLITS:
        selected = [r for r in output if r['split']==split]
        write_csv(manifests/f'{split}.csv',list(output[0]),selected)
    for client in CLIENTS:
        directory = clients/client
        directory.mkdir(parents=True)
        for split in SPLITS:
            selected = [r for r in output if r['split']==split and r['client']==client]
            if not selected:
                raise ValueError(f'Empty client split: {client}/{split}')
            # Ultralytics replaces every './' substring. Avoid '../' inside image lists.
            # Keep lists beside the shared split folders; YAML paths may safely use '..'.
            lines = ['./'+os.path.relpath(root/r['image'],final_prepared).replace('\\','/') for r in selected]
            (prepared/f'{client}_{split}.txt').write_text('\n'.join(lines)+'\n',encoding='utf-8')
            write_csv(manifests/f'{client}_{split}.csv',list(output[0]),selected)
        cfg = {'train':f'../../original_dataset_non_chili/{client}_train.txt',
               'val':f'../../original_dataset_non_chili/{client}_valid.txt',
               'test':f'../../original_dataset_non_chili/{client}_test.txt','nc':24,'names':summary['new_classes']}
        (directory/'data.yaml').write_text(yaml.safe_dump(cfg,sort_keys=False),encoding='utf-8')
    counts = []
    for old,new in mapping.items():
        row = {'class_id':new,'original_class_id':old,'class_name':summary['source_classes'][str(old)] if str(old) in summary['source_classes'] else summary['source_classes'][old]}
        for split in SPLITS:
            matches = [r for r in retained if r['split']==split]
            row[f'{split}_images'] = sum(any(b['class_id']==old for b in r['boxes']) for r in matches)
            row[f'{split}_instances'] = sum(sum(b['class_id']==old for b in r['boxes']) for r in matches)
        counts.append(row)
    write_csv(manifests/'class_distribution.csv',list(counts[0]),counts)
    prepared.rename(final_prepared)
    clients.rename(final_clients)
    for path in manifests.iterdir():
        path.replace(final_manifests/path.name)
    manifests.rmdir()
    staging.rmdir()
    manifests = final_manifests
    checksums = {p.relative_to(root).as_posix():sha_file(p) for p in sorted(manifests.glob('*.csv'))}
    for path in [manifests/'class_map.json', manifests/'audit.json', manifests/'inventory.jsonl',
                 manifests/'visual_review.json',
                 final_prepared/'data.yaml', *final_prepared.glob('*.txt'), *final_clients.glob('*/*')]:
        if path.is_file():
            checksums[path.relative_to(root).as_posix()] = sha_file(path)
    (manifests/'checksums.json').write_text(json.dumps(checksums,indent=2)+'\n')
    print('Prepared dataset and shared-image client manifests created.',flush=True)

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--workers',type=int,default=8)
    parser.add_argument('--apply',action='store_true')
    parser.add_argument('--from-audit',action='store_true')
    args = parser.parse_args()
    if args.workers < 1:
        parser.error('workers must be positive')
    manifests = ROOT/'manifests/non_chili_v1'
    if (ROOT/'original_dataset_non_chili').exists() or (ROOT/'dataset_non_chili').exists():
        raise FileExistsError('Dataset version already frozen; use verify_non_chili_dataset.py instead of overwriting its audit')
    if args.from_audit:
        summary = json.loads((manifests/'audit.json').read_text())
        records = [json.loads(line) for line in (manifests/'inventory.jsonl').read_text().splitlines()]
    else:
        records,summary = audit(ROOT,args.workers)
        save_audit(manifests,records,summary)
    review_path = manifests/'visual_review.json'
    if review_path.exists():
        apply_review(records,summary,review_path)
        save_audit(manifests,records,summary)
    print(json.dumps({k:summary[k] for k in ['source_counts','retained_counts','exclusions','missing_retained_classes']}),flush=True)
    print(f'Near-duplicate candidates needing review: {len(summary["near_duplicate_candidate_pairs"])}',flush=True)
    if args.apply:
        build(ROOT,manifests,records,summary,args.workers)
    else:
        print('Read-only audit complete; no source data changed. Use --from-audit --apply after reviewing.',flush=True)

if __name__ == '__main__':
    main()
