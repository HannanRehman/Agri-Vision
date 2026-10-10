# Agri Vision comparison: saved validation results

Snapshot: 2026-10-10T07:55:26.004702+00:00.

| Model | Saved rounds | Best round | Precision | Recall | F1 | mAP50 | mAP50–95 |
|---|---:|---:|---:|---:|---:|---:|---:|
| efficientdet_d0 | 2 | 2 | 56.75% | 68.36% | 62.02% | 69.44% | 46.68% |
| ssd_mobilenetv3 | 10 | 10 | 57.92% | 47.61% | 52.26% | 51.14% | 30.30% |
| ssdlite | 10 | 10 | 66.44% | 51.17% | 57.81% | 57.91% | 39.06% |
| yolo_v8n | 10 | 8 | 0.00% | 0.00% | 0.00% | 1.40% | 0.80% |

These scores come from the same 4,157-image pooled validation split with 24 non-chili classes. They are not held-out test scores. Best checkpoints are selected by validation mAP50–95. Precision/recall use confidence 0.25 and class-aware IoU 0.5; AP sweeps scores above 0.001.

One local epoch means each client visits its training split once in each global round. The completed models ran 10 global rounds, each with three clients and one local epoch.

EfficientDet committed round 2 and stopped during round 3 pooled validation. The cause of the Colab disconnect is unconfirmed. Resume with the identical code, settings, package versions and Tesla T4; round 3 will repeat from the round-2 checkpoint. Keep the total budget at 10 rounds. The EfficientDet-only resume notebook is prepared.

YOLO is abnormal: all ten rounds have zero true positives and zero false positives at confidence 0.25. This means no predictions survived that cutoff. Its nonzero AP comes from lower-confidence predictions. Do not use this run to claim YOLO is inherently inferior. The custom YOLO training/initialization and prediction path need investigation. A lower reporting cutoff alone will not repair its very low AP.

SSDLite improved from 10.98% to 39.06% mAP50–95. SSD–MobileNetV3 improved from 1.18% to 30.30%. EfficientDet improved from 29.93% to 46.68% in two committed rounds. Its lead is provisional until the common round budget is completed and selected checkpoints are evaluated on the held-out test set.

The experiment shares data, clients, seed, round budget, augmentation and evaluator. Input sizes, optimizers, batch sizes and pretraining/head initialization differ by model. These are practical training configurations, not a controlled architecture-only ablation.

Local backup: 187 files verified by byte size and SHA-256; 29 files still missing or incomplete. Private transfer URLs and the cleanup audit stay in the ignored local archive directory.

## Completed storage cleanup

All saved checkpoints, settings, metrics and client logs are backed up in the four local experiment folders. All initial and final checkpoint SHA-256 values match their recorded settings/commit values. Some raw prediction exports are also local; 29 raw exports remain unbacked-up locally and were retained on Drive because their downloads stalled.

Removed 26 redundant intermediate checkpoints from the three completed Drive runs after independently checking local byte size and SHA-256 and refreshing each Drive file parent and size. This reclaimed 397,735,770 bytes (397.74 MB). Every intermediate checkpoint remains in its local run folder. Initial, best and final checkpoints remain on Drive; all EfficientDet files and training inputs remain on Drive. Post-cleanup folder listings confirmed the retained checkpoint files. The private cleanup audit is in archive/comparison_download_20261010/drive_cleanup_audit.json.

Drive had about 0.5 GB free according to the user; its live account quota is unavailable through the connector. The cleanup adds approximately 0.40 GB to that reported headroom. Local F: has about 11.7 GB free. No training or new evaluation was started locally.
