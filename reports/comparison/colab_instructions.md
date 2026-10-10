# Agri Vision: new 24-class comparison in Colab

Prepared 10 October 2026. The shared pipeline and four adapters are implemented and locally checked. **GPU compatibility, full-round runtime and main results are still pending.** The local environment has CPU-only PyTorch. No full-dataset main training has run.

## Upload these two files

1. `F:\Agri Vision\non_chili_v1_colab_dataset.zip` — the verified dataset, approximately 926 MB. Reuse this exact ZIP.
2. `F:\Agri Vision\agrivision_comparison_code.zip` — the new comparison code bundle.

Place both under `MyDrive/Agri Vision/`. Open [AgriVision_Comparison.ipynb](../../AgriVision_Comparison.ipynb) in Colab. **Use the new comparison notebook.** `AgriVision_Colab.ipynb` remains the historical 30-class continuation notebook.

The Drive connector has a 512 MiB upload limit, and large transfers can time out. The connector upload therefore uses **14 smaller parts** and `parts.json` inside `MyDrive/Agri Vision/non_chili_v1_upload_parts/`. Keep the code ZIP and notebook directly in `MyDrive/Agri Vision/`. The notebook reads the manifest, checks every part's presence and size, joins the parts on Colab's local disk, checks the original dataset SHA-256, and extracts it. It also accepts the original complete ZIP if uploaded manually, or the earlier two-part layout directly in the Agri Vision folder. These parts are portions of the same archive, not separate datasets.

Bundle hashes are in [colab_upload_package.json](colab_upload_package.json) and embedded in the new notebook. If comparison code/configurations change, rebuild with:

```powershell
.\.venv\Scripts\python.exe scripts\package_comparison_colab.py
```

Replace the uploaded code ZIP and reopen the regenerated notebook together. The dataset ZIP does not need rebuilding for code-only changes.

## Start with the GPU pilot

Select a GPU runtime. Prefer the same GPU type for all main models. Leave `MODE = "pilot"` and run the notebook. It will:

- Verify both upload hashes and extract the common dataset to Colab's local disk.
- Preserve Colab's CUDA-enabled Torch/TorchVision pair, install comparison dependencies and run the correctness tests.
- Check pretrained adapters for all four models, and run an eight-step SSDLite overfit check using two training-only images.
- Run one limited SSDLite federated round with two batches per client and 24 fixed validation images.
- Save checkpoints, pooled and per-client validation metrics, predictions, CSV, settings and progress to Drive.
- Verify resume without retraining a committed round, then save `gpu_pilot_passed.json`.

The pilot is deliberately small. Its AP, precision and recall are **engineering diagnostics**, not thesis results. The GPU pilot must pass before main runs; CPU checks do not establish Colab compatibility or model quality.

## Main runs after the pilot

Review the GPU pilot outputs. Resolve batch sizes, initialization, augmentation and optimizer settings before the main comparison. Defaults are starting candidates, not a completed hyperparameter search. The chosen implementations use:

| Model | Input | Default batch | Candidate optimizer / initial LR |
| --- | ---: | ---: | --- |
| YOLOv8n | 640 | 8 | SGD / 0.005 |
| SSDLite MobileNetV3-Large | 320 | 8 | SGD / 0.001 |
| Standard SSD MobileNetV3-Large | 320 | 8 | SGD / 0.001 |
| TF EfficientDet-D0 | 512 | 4 | AdamW / 0.0001 |

Set `MODE = "main"`. The starting budget is **10 global rounds, one local epoch per client, seed 42**. Three local epochs would reuse each client's partition three times; it does not require additional datasets. We have not selected that change.

The models run serially in the agreed order: fresh YOLOv8n, SSDLite, standard SSD–MobileNetV3, EfficientDet-D0. Main runs use all frozen client training images and all 4,157 validation images. Each run starts from its own recorded pretrained initialization, not from the pilot's trained weights or the old 30-class checkpoint. Every client starts from the same global weights in a round.

The shared augmentation is a training-only horizontal flip with probability 0.5. Adapters perform their recorded resize and normalization. All models freeze BatchNorm running statistics while training affine parameters. Matching pretrained classification feature/BN tensors are preserved where compatible; new class projections are initialized separately. The standard SSD head is entirely new, including regression, while SSDLite retains compatible COCO regression weights. Therefore this is a comparison of practical detector configurations with initialization differences, not a controlled head-only ablation.

EfficientDet's new class projection uses standard deviation 0.001 and a 1% focal prior to avoid saturated initial logits after transferring classification feature/BN tensors. Its five-step CPU training-only check decreased loss from 1.5374 to 1.4218. This is a stability check, not an accuracy result.

Each client's optimizer resets per round, but the learning rate follows the declared global-round cosine schedule. Gradient clipping is 10; mixed precision is disabled initially. Native NMS is 0.55 for YOLO/SSD/SSDLite and 0.50 for EfficientDet, recorded in settings. Predictions use score floor 0.001 and at most 100 detections/image. Shared AP uses the score sweep; micro precision/recall/F1 use confidence 0.25 and IoU 0.5. Global AP comes from pooled predictions, not averaged client AP.

Do not silently extend a 10-round checkpoint to a different budget: the current runner rejects that resume because it changes the declared schedule. Decide a larger matched budget before starting those runs, or implement and document an explicit extension schedule applied to every model. A short pilot cannot establish whether 10 rounds is sufficient for final convergence.

## Recover an interruption

Outputs are unique folders under `MyDrive/Agri Vision/comparison_results/`. `settings.json`, `status.json`, `latest.json`, per-client logs, round checkpoints and `results/metrics/global_validation_metrics.csv` show progress. A `main_runs_*.json` file lists completed model folders.

Set `RESUME_RUNS` in the notebook to the existing run directories for models you want to recover. Keep the model, dataset, code, package versions, GPU type, seed, batch, learning rate, round budget and local epochs unchanged. The runner starts from the latest committed global round. If interruption happened during training/validation before the commit, it repeats that whole round. A completed run resumes without repeating training. If setup stopped before `settings.json` was created, start a new uniquely named run.

Re-running `MODE="main"` with an empty `RESUME_RUNS` starts fresh experiments. Populate the map to reuse completed runs when requesting final evaluation later.

## Final held-out evaluation

Keep `FINAL_TEST = False` while resolving settings and selecting checkpoints. Once all chosen main runs finish, use their `RESUME_RUNS` paths and enable final test evaluation. The evaluator selects each model's highest pooled validation mAP50–95 within the declared budget and evaluates the full 2,091-image held-out test split.

It saves full test predictions, pooled/per-client metrics and per-class AP/P/R/F1. It measures batch-1 latency after five warmup iterations on validation images, including transfer, preprocessing and NMS, while excluding disk loading. It also records parameter counts, tensor-state size, input resolution and peak CUDA allocation. The comparison table rejects pilot results or mismatched dataset/seed/budget/GPU/augmentation/normalization/evaluator contracts.

Once test results exist, they are preserved; repeated final evaluation will not overwrite them. Do not use test scores to tune settings. Collect new experimental repeats transparently if more analysis is needed.

## Evidence and remaining work

Local checks cover target IDs and box conversion, loss/gradients, pretrained initialization, frozen BN buffers, save/reload predictions, pooled AP versus fixed-threshold P/R, empty predictions, weighted aggregation and interrupted-round recovery. See [local validation record](local_validation.json), [adapter checks](pretrained_adapter_checks.json) and [SSDLite overfit check](ssdlite_pretrained_overfit_check.json).

The Colab Tesla T4 pilot passed. YOLO, SSDLite and SSD–MobileNetV3 completed the initial 10-round main budget; EfficientDet committed round 2 before an interruption during round 3 validation. See the [saved results review](main_run_review_20261010/README.md) and [EfficientDet resume instructions](main_run_review_20261010/resume_instructions.md). Remaining: resume EfficientDet to the original round-10 budget, investigate YOLO's abnormally low-confidence predictions and AP, resolve configurations with a comparable tuning allowance, then perform final held-out evaluation and interpret per-class results and convergence. The [dataset preparation report](../non_chili_v1/dataset_preparation_report.md) documents residual imbalance, annotation artifacts and the limits of the near-duplicate audit.

Implementation references: [TorchVision SSDLite documentation](https://docs.pytorch.org/vision/master/models/generated/torchvision.models.detection.ssdlite320_mobilenet_v3_large.html), [TorchVision detector source](https://github.com/pytorch/vision/blob/main/torchvision/models/detection/ssdlite.py), and [EfficientDet source](https://github.com/rwightman/efficientdet-pytorch).
