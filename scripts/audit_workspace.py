"""Read-only duplicate audit of project artifacts; never alters training data."""
import argparse
from collections import defaultdict
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shutil

ROOT = Path(__file__).resolve().parents[1]
EXCLUDED = {'.git', '.venv', 'venv', '__pycache__', 'original_dataset', 'dataset',
            'original_dataset_non_chili', 'dataset_non_chili', 'PENYAKIT PADA CABAI.v1i.yolov8'}

def sha256(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()

def folder_fingerprint(path):
    return {p.relative_to(path).as_posix(): {'bytes': p.stat().st_size, 'sha256': sha256(p)}
            for p in sorted(path.rglob('*')) if p.is_file()}

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    sizes = defaultdict(list)
    files = []
    for directory, children, names in os.walk(ROOT):
        children[:] = [name for name in children if name not in EXCLUDED
                       and not name.startswith('dataset_backup_')
                       and not name.startswith('dataset_staging_')
                       and name not in {'legacy_environment', 'legacy_datasets', 'imported_datasets'}]
        for name in names:
            path = Path(directory)/name
            if path.resolve() == args.output.resolve():
                continue
            size = path.stat().st_size
            files.append({'path': path.relative_to(ROOT).as_posix(), 'bytes': size})
            if size:
                sizes[size].append(path)
    groups = []
    for size, candidates in sizes.items():
        if len(candidates) < 2:
            continue
        hashes = defaultdict(list)
        for path in candidates:
            hashes[sha256(path)].append(path.relative_to(ROOT).as_posix())
        for digest, paths in hashes.items():
            if len(paths) > 1:
                groups.append({'bytes_each': size, 'sha256': digest, 'paths': paths})
    a = ROOT/'saved_models/test_evaluation_round_035'
    b = ROOT/'saved_models/test_evaluation_round_035-2'
    duplicate_folder = None
    if a.is_dir() and b.is_dir():
        first, second = folder_fingerprint(a), folder_fingerprint(b)
        duplicate_folder = {'keep': a.relative_to(ROOT).as_posix(),
                            'candidate': b.relative_to(ROOT).as_posix(),
                            'identical': bool(first) and first == second,
                            'keep_files': first, 'candidate_files': second}
    usage = shutil.disk_usage(ROOT)
    payload = {'created_at_utc': datetime.now(timezone.utc).isoformat(),
               'root': str(ROOT), 'free_bytes': usage.free,
               'scope': 'Artifact files only; data partitions, backups, Git, environments and caches excluded',
               'files': sorted(files, key=lambda item: item['path']),
               'duplicate_groups': sorted(groups, key=lambda item: -item['bytes_each']),
               'round_035_evaluation_folder_comparison': duplicate_folder}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2)+'\n', encoding='utf-8')
    print(f'Artifact files: {len(files)}; duplicate groups: {len(groups)}')
    print(f'Round-35 evaluation folders identical: {duplicate_folder and duplicate_folder["identical"]}')
    for group in payload['duplicate_groups'][:12]:
        print(group['bytes_each'], ' | '.join(group['paths']))

if __name__ == '__main__':
    main()
