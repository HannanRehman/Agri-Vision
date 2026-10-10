"""Package the verified shared dataset and portable client lists for Colab; never train."""
from datetime import datetime, timezone
import csv
import hashlib
import json
from pathlib import Path
import zipfile

ROOT = Path(__file__).resolve().parents[1]


def main():
    manifests = ROOT / 'manifests/non_chili_v1'
    verification = json.loads((manifests / 'verification.json').read_text())
    if verification['status'] != 'passed':
        raise ValueError('Run the dataset verifier first')
    expected = json.loads((manifests/'checksums.json').read_text())
    with (manifests/'retained.csv').open(newline='',encoding='utf-8') as stream:
        for row in csv.DictReader(stream):
            expected[row['image']] = row['image_sha256']
            expected[row['label']] = row['label_sha256']
    destination = ROOT / 'non_chili_v1_colab_dataset.zip'
    if destination.exists():
        raise FileExistsError('Refuse to overwrite an existing upload bundle')
    paths = sorted(p for directory in [ROOT/'original_dataset_non_chili', ROOT/'dataset_non_chili', manifests]
                   for p in directory.rglob('*') if p.is_file())
    entries = {}
    staging = destination.with_suffix('.zip.partial')
    if staging.exists():
        raise FileExistsError('Interrupted bundle exists; inspect it before retrying')
    with zipfile.ZipFile(staging,'w',compression=zipfile.ZIP_DEFLATED,compresslevel=1) as archive:
        for index,path in enumerate(paths,1):
            name = path.relative_to(ROOT).as_posix()
            payload = path.read_bytes()
            entries[name] = hashlib.sha256(payload).hexdigest()
            if name in expected and entries[name] != expected[name]:
                raise ValueError(f'Frozen file changed before packaging: {name}')
            compression = zipfile.ZIP_STORED if path.suffix.lower() in ['.jpg','.jpeg','.png','.webp'] else zipfile.ZIP_DEFLATED
            archive.writestr(name,payload,compress_type=compression)
            if index % 5000 == 0:
                print(f'PACKAGE {index}/{len(paths)} files',flush=True)
        archive.writestr('non_chili_v1_package_checksums.json',json.dumps(entries,indent=2)+'\n')
        archive.writestr('NON_CHILI_DATASET_README.txt',
                        'Verified non_chili_v1 dataset, 24 classes, prepared 10 October 2026.\n'
                        'Extract the ZIP into one directory, preserving its directory structure.\n'
                        'The original_dataset_non_chili folder stores images once.\n'
                        'dataset_non_chili/clientN/data.yaml references portable image lists.\n'
                        'Client image lists are stored in original_dataset_non_chili/clientN_split.txt.\n'
                        'Foreground IDs are 0-23. TorchVision detection adapters must add 1 for background 0.\n'
                        'The original_dataset paths in inventory.jsonl record local provenance only.\n'
                        'No pretrained weights, training runner or historical notebook are included.\n'
                        'The existing AgriVision_Colab.ipynb runs historical 30-class training; use a new comparison runner.\n')
    with zipfile.ZipFile(staging) as archive:
        if archive.testzip() is not None:
            raise ValueError('ZIP integrity check failed')
    staging.rename(destination)
    with destination.open('rb') as stream:
        digest = hashlib.file_digest(stream,'sha256').hexdigest()
    report = dict(created_at_utc=datetime.now(timezone.utc).isoformat(),
                  filename=destination.name,bytes=destination.stat().st_size,sha256=digest,
                  dataset_files=len(paths),dataset_images=verification['images'],zip_crc_check='passed',
                  training_code_included=False)
    (ROOT/'reports/non_chili_v1/upload_package.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2),flush=True)


if __name__ == '__main__':
    main()
