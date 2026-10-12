# EfficientDet-D0: verified completion through round 15

Checked 12 October 2026 against the 4gmail Drive run `efficientdet_d0_non_chili_seed42_20261009_224152_449800_to15`. This review reads saved results only; it starts no training or model evaluation.

The run's final status is `complete`, with 15 completed global rounds. `latest.json` commits round 15 and retains round 11 as the validation-best checkpoint. The complete CSV contains exactly one row for each round 1–15. Round-11 and round-15 metric JSONs both contain three client records and agree with the CSV and pooled detection counts.

| Validation metric | Round 10 | Best checkpoint: round 11 | Final round 15 |
|---|---:|---:|---:|
| Precision | 66.05% | 66.40% | 64.23% |
| Recall | 78.02% | 77.43% | 77.27% |
| mAP50 | 84.99% | 85.28% | 84.95% |
| mAP50–95 | 59.56% | 59.70% | 59.47% |
| F1 | 71.54% | 71.49% | 70.15% |

Checkpoint selection uses the highest pooled validation mAP50–95, so round 11 is selected even though other rounds can have higher values for other metrics. These are full pooled validation results on 4,157 images and 7,210 ground-truth boxes. Precision and recall use confidence 0.25 and class-aware IoU 0.50 matching; they are not held-out test scores or classification accuracy.

Round 15 is 0.23 percentage points below the best mAP50–95. The five added rounds produced a small new best at round 11 but no continuing upward trend thereafter. Their saved timers total 99.3 minutes, excluding notebook setup and interruption downtime. Preserve the validation-best checkpoint instead of automatically extending unchanged training further.

## Saved evidence

- `global_validation_metrics.csv`: all 15 committed round summaries.
- `status.json`: complete, 15 rounds, round-11 best checkpoint.
- `latest.json`: round-15 commit, checkpoint SHA256 and best checkpoint path.
- `round_011.json` and `round_015.json`: detailed pooled and per-class validation metrics, client training counts, losses, rates and timing.

The round-11 and round-15 checkpoints were present in Drive during verification. The round-11 checkpoint was also downloaded locally and loaded with `weights_only=True`: its round, class mapping and configuration metadata were checked. Its local SHA256 is `86f4fc50bb5ad26d7e8c8791215a36a2d225b807c5f8b4d93b1cec0f29398db4`. Round-15 bytes were not independently downloaded or hashed for this review; the commit record stores expected SHA256 `3815ef3cc290b35d529ecd6b2d626a61f1a3ec66ae56e0fd7f2991681c5e391d`.

Checkpoint binaries stay local or in Drive rather than Git. This folder contains no poster materials, portraits, PowerPoint files, authenticated download links or credentials.

Source folder: [4gmail EfficientDet continuation](https://drive.google.com/drive/folders/1v3NPBJjJKh9rpKSgrK1NjlWnpQDSEBSg).
