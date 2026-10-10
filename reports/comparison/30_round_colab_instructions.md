# Run the separate YOLO and alternative-detector notebooks

Prepared 10 October 2026. No GPU training was started during preparation. Local checks exercise synthetic checkpoint recovery, original checkpoint metadata and native inference. The actual GPU learning pilot and training must run in Colab; accuracy improvement is not established yet.

User update: fresh 24-class YOLO is active again. The YOLO notebook, code bundle and all fourteen dataset parts plus manifest have been placed in the shared 19gmail Agri Vision folder. Use 19gmail when opening Colab and authorizing Drive. Ownership transfer was verified: 19gmail owns the notebook, code bundle, fourteen dataset parts and manifest (17 files). The connected 4gmail account retains writer access. GPU training has not been started by the assistant.

## New YOLO account

1. In that account's Drive create `MyDrive/Agri Vision/`.
2. Upload the existing `non_chili_v1_colab_dataset.zip` and the new `agrivision_30_round_code.zip` into that folder. Alternatively copy/upload the complete `non_chili_v1_upload_parts/` folder, including `parts.json` and all fourteen parts. Sharing access alone does not make files appear at this mounted MyDrive path; make them accessible at the exact configured location before running.
3. Open `AgriVision_YOLO_Native_30Rounds.ipynb` with that account, select a T4 GPU, then Run all.
4. The first execution runs a training-only learning pilot: 48 images, 12 small epochs, classification-loss decrease, training AP50, adapting classifier BatchNorm and save/reload checks. Failed checks block main training. Pilot scores are diagnostics and its weights are discarded.
5. A passed pilot starts a fresh 24-class YOLOv8n run through round 30 using native Ultralytics local training. Output: `comparison_results/yolo_v8n_native_non_chili_seed42_30round/`.

The historical local method is restored: Ultralytics 8.4.48, optimizer auto, native warmup, batch 8, 640 input, one local epoch, two workers, native augmentation, adaptive BatchNorm and the original floating-state NumPy FedAvg calculation. The 24-class dataset and declared seed 42 differ from the historical experiment. The native trainer verifies it received the intended global weights before any optimizer update.

The previous custom YOLO loss loop is not used. Inference uses a separate model copy because native prediction can fuse layers in place. Checkpoint state remains unfused and resumable. Native training restores EMA weights as the historical fit method did.

## Current account: the other three models

The updated `AgriVision_Comparison.ipynb` and the descriptively named `AgriVision_Alternatives_Continue_30Rounds.ipynb` contain the same continuation workflow. Use either one, not both simultaneously.

Keep the existing dataset parts and original result folders. Add only the new small `agrivision_30_round_code.zip` to `MyDrive/Agri Vision/`. Select T4 and Run all. The notebook validates all three sources before training anything, then runs EfficientDet, SSDLite and SSD–MobileNetV3 sequentially.

| Model | Last verified committed round | Next round | Initial/base LR | LR for rounds 11–30 |
| --- | --- | --- | --- | --- |
| EfficientDet-D0 | 2 | 3 | 0.0001 | 0.00001 |
| SSDLite | 10 | 11 | 0.001 | 0.0001 |
| SSD–MobileNetV3 | 10 | 11 | 0.001 | 0.0001 |

EfficientDet's actual committed round is read at execution. It completes its original cosine schedule through round 10. All three use the terminal LR thereafter, with per-client optimizer reset unchanged. This schedule avoids a sudden LR restart; whether it improves validation scores must be measured. The thesis should distinguish continued ten-round runs from the fresh native-YOLO thirty-round experiment.

Each creates a linked `_to30` folder beside its original run. Original settings and checkpoints are validated and remain untouched. Initial, current and best tensors are re-encoded under the new continuation configuration, with original hashes/lineage recorded. Existing metrics through the source committed round are copied. The unchanged original `comparison/*.py` implementation remains compatible with the earlier ten-round resume notebook and bundle.

## Resume, metrics and storage

- Reopen the same notebook on the same account, select T4 and Run all. No round number needs editing. Completed models skip training; partial global rounds repeat from the latest committed global checkpoint.
- `status.json` shows the current phase. `latest.json` records the completed round and selected best checkpoint. Training output prints each round/client; validation metrics are committed after validation succeeds.
- Pooled CSV: `results/metrics/global_validation_metrics.csv`. YOLO also writes `native_validation_metrics.csv`, containing historical-style validation-image-weighted native client summaries. Keep these definitions separate in the thesis.
- Round JSONs include per-class metrics and client details. Select the best pooled validation AP50–95, not automatically the last round. No test-set tuning or automatic held-out evaluation occurs.
- Save initial/best/latest checkpoints. Prune only intermediate weights generated in the new run after the round commit. Do not retain bulky local YOLO weights or all-round raw prediction files on Drive; native local arguments and CSVs are retained.
- Budget, code, dataset, GPU and core package changes cause resume checks to stop. On the current account the exact original torch/torchvision and recorded detector packages are required. The notebook preserves the runtime's CUDA pair and reports any mismatch; it does not silently substitute a different environment.

The new bundle is shared by both workflows, but the accounts keep independent results. The original `AgriVision_Comparison_10Round.ipynb` is retained locally, and `AgriVision_EfficientDet_Resume.ipynb` still implements the older EfficientDet-only ten-round recovery. Use the new alternatives notebook for the agreed thirty-round continuation.

See [package checksums](30_round_upload_package.json), [agreed plan](next_30_round_plan.md), [YOLO diagnosis](yolo_diagnosis_20261010/README.md) and [earlier results](main_run_review_20261010/README.md).
