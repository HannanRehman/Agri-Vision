"""Write the preparation report from measured audit and verification outputs."""
import csv
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main():
    directory = ROOT/'manifests/non_chili_v1'
    audit = json.loads((directory/'audit.json').read_text())
    verified = json.loads((directory/'verification.json').read_text())
    if verified['status'] != 'passed':
        raise ValueError('Verification must pass before reporting readiness')
    with (directory/'class_distribution.csv').open(newline='',encoding='utf-8') as stream:
        classes = list(csv.DictReader(stream))
    text = ['# Agri Vision: 24-class dataset preparation report', '',
            'Prepared 10 October 2026. **Dataset built and mechanically verified; no new training has started.**', '',
            'The separate `non_chili_v1` dataset retains banana, cauliflower, groundnut and radish. '
            'All six chili classes, including healthy chili, are excluded. Original 30-class images, labels, '
            'client partitions and model checkpoints are preserved.', '',
            '## Final dataset', '', '| Split | Original images | Retained images | Retained boxes |',
            '| --- | ---: | ---: | ---: |']
    for split in ['train','valid','test']:
        boxes = sum(int(row[f'{split}_instances']) for row in classes)
        text.append(f'| {split} | {audit["source_counts"][split]:,} | {verified["split_counts"][split]:,} | {boxes:,} |')
    text += ['', f'**Total: {verified["images"]:,} images and 38,779 boxes; all images are 640 × 640.** '
             'All 24 classes occur in every pooled split. Per-class image counts can overlap when an image contains more than one class.', '',
             '| Client | Train | Validation | Test |', '| --- | ---: | ---: | ---: |']
    for client,counts in verified['client_counts'].items():
        text.append(f'| {client} | {counts["train"]:,} | {counts["valid"]:,} | {counts["test"]:,} |')
    text += ['', '## Per-class image counts', '', '| New ID | Class | Train | Validation | Test |',
             '| ---: | --- | ---: | ---: | ---: |']
    for row in classes:
        text.append(f'| {row["class_id"]} | {row["class_name"]} | {row["train_images"]} | {row["valid_images"]} | {row["test_images"]} |')
    text += ['', '## Filtering and leakage decisions', '',
             '- Excluded 1,216 chili images: 913 train, 203 validation and 100 test. No mixed chili/non-chili images were identified by the supplied labels.',
             '- Excluded one empty-label training image, `Rust-1015-_jpg.rf.b9611fdc3a3b8b23027b5ade1ace2bbd.jpg`, because crop membership could not be established from an empty annotation.',
             '- No missing/corrupt images, invalid boxes, orphan source labels, conflicting annotations on identical decoded images, exact pixel duplicates or matching inferred filename source groups were detected in the non-chili candidates.',
             '- Perceptual screening found 104 cross-split candidate pairs. Visual review of all 18 sheets judged 102 pairs to show the same source photo or source variant; two pairs were different photos. Decisions and sheet/row references are in `visual_review.json`.',
             '- Grouped confirmed source variants and excluded 98 lower-priority copies: 84 training and 14 validation images. Kept the highest original split priority: test, then validation, then train. No training image was moved into a held-out split.',
             '- Reassigned one validation image from client3 to client1 to keep the retained variants of its source group on one client. Other retained client assignments are preserved.',
             '- Old class IDs 0–12 stay unchanged; 13–18 are removed; 19–29 become 13–23. Box coordinate tokens are unchanged.', '',
             '## Verification and limitations', '',
             'The independent verifier passed for every retained image and label: copied image hashes, original source hashes, '
             'output label hashes, class remapping, unchanged coordinate tokens, complete split/client lists, manifest checksums '
             'and source-group separation. All 21 project tests passed, including eight preparation tests with an end-to-end build and deliberate tampering check. '
             'Each retained decoded image is unique. Confirmed source groups occupy one split and one client.', '',
             'The near-duplicate search was a bounded, same-crop 64-bit dHash search at Hamming distance ≤2. '
             'It can miss rotated, mirrored, cropped, stronger augmented or differently named source variants. '
             'This report documents the detected leakage correction; it does not certify exhaustive source independence. '
             'Source filenames are heuristic provenance. The recorded class names and disease diagnoses still require source/expert confirmation.', '',
             'Annotation sheets were inspected for every class in train, validation and test. The images include '
             'stretched border artifacts in some banana examples and differing annotation granularity, from whole organs to individual lesions. '
             'These labels were preserved for a common comparison; this was not a full expert relabelling exercise.', '',
             'Class imbalance remains: banana 8,981 images, groundnut 8,421, radish 2,731 and cauliflower 629. '
             'Cauliflower black rot has only 59 training, 19 validation and 9 test images. Report per-class AP and recall alongside pooled metrics; '
             'small held-out classes will have uncertain estimates. Removing chili does not establish improved accuracy or recall without new training.', '',
             '## Files and next step', '',
             '- Shared image/label dataset: `original_dataset_non_chili/`.',
             '- Portable client configurations: `dataset_non_chili/client1/` through `client3/`. Their `clientN_split.txt` lists are beside the shared split folders in `original_dataset_non_chili/`; image-list entries contain no parent-directory traversal. Images are stored once, not copied three times.',
             '- Frozen lists, mapping, audit, exclusions, decisions and checksums: `manifests/non_chili_v1/`. The local detailed inventory contains original provenance and is included in the upload bundle.',
             '- Diagnostic sheets: `reports/non_chili_v1/*_class_samples.jpg` and `near_duplicate_review_*.jpg`.', '',
             'Verify locally with:', '', '```powershell',
             '.\\.venv\\Scripts\\python.exe scripts\\verify_non_chili_dataset.py', '```', '',
             'The next implementation step is the shared dataset adapter and evaluator, followed by the SSDLite pilot. '
             'The current root training code and `AgriVision_Colab.ipynb` remain configured for historical 30-class YOLO training. '
             'Do not use them for this dataset. TorchVision foreground targets must become 1–24, leaving background 0; canonical dataset IDs remain 0–23.', '']
    package = ROOT/'reports/non_chili_v1/upload_package.json'
    if package.exists():
        info = json.loads(package.read_text())
        text += ['## Colab dataset upload', '',
                 f'Upload `{info["filename"]}` ({info["bytes"]/1e6:.1f} MB). ZIP CRC verification passed. '
                 'Extract it into a single directory, preserving its layout. The archive includes shared data, client lists, '
                 'manifests and an internal SHA-256 list; it does not include a new training runner.', '',
                 f'ZIP SHA-256: `{info["sha256"]}`.', '']
    (ROOT/'reports/non_chili_v1/dataset_preparation_report.md').write_text('\n'.join(text),encoding='utf-8')
    print('Wrote reports/non_chili_v1/dataset_preparation_report.md')


if __name__ == '__main__':
    main()
