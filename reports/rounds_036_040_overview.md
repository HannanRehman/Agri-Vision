# Rounds 36–40: Validation Overview

Source: `saved_models/20261004_200656_355a3a-20261004T205120Z-1-001.zip`.

The saved status confirms successful completion through round 40. All five global checkpoints, global metric JSON files, per-client validation records, and the combined validation CSV are present. Training continued from round 35 on a Tesla T4, with one local epoch per client per round, batch 8, and image size 640.

These are validation-image-weighted client summaries, not held-out test results.

| Round | Precision | Recall | mAP50 | mAP50–95 |
| ---: | ---: | ---: | ---: | ---: |
| 36 | 67.19% | 60.19% | 61.12% | 39.87% |
| 37 | 65.27% | 59.24% | 60.23% | 39.63% |
| 38 | 64.74% | 58.38% | 59.90% | 39.36% |
| 39 | 63.94% | 59.13% | 59.54% | 38.87% |
| 40 | 66.27% | 57.43% | 59.55% | 38.81% |

Round 36 is strongest on all four metrics within this five-round continuation. Performance briefly recovered and then declined again. Relative to round 35, round 40 has 1.62 percentage points higher precision, 1.39 points lower recall, 0.51 points lower mAP50, and 0.29 points lower mAP50–95.

Across the reviewed rounds 26–40, round 30 still has the highest mAP50–95 (40.61%). Round 33 still has the highest mAP50 (61.22%) and recall (61.90%). The results do not support continuing the same training setup merely to increase the round count. Select a checkpoint using the declared validation criterion and preserve the others for comparison.

The numerical exports are [global validation metrics](rounds_036_040_global_metrics.csv) and [per-client global-model validation metrics](rounds_036_040_client_validation_metrics.csv). The latter records each client's scores and validation-image count; it is distinct from the local models' training CSVs.
