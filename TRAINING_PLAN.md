# Agri Vision: training and evaluation plan

Updated 10 October 2026. **Current agreed plan:** fresh native-Ultralytics YOLO rounds 1?30 on a separate Colab account; continue SSDLite and SSD from round 10 and EfficientDet from its last verified round 2 on the current account/Drive, sequentially, toward round 30. See [the account split and continuation plan](reports/comparison/next_30_round_plan.md). The sections below preserve the original ten-round design and preparation history; their pending-run status, training order and budget are superseded. The thirty-round runners and continuation schedule still need implementation.

Original implementation record, 10 October 2026. **Status: separate 24-class data and comparison pipeline implemented; local engineering checks passed; GPU pilot and main runs pending.** Four adapters, pooled evaluator, recoverable FedAvg runner and final-evaluation/table tools now exist in `comparison/`. A limited CPU SSDLite pilot is diagnostic, not a main result. Use [the new Colab instructions](reports/comparison/colab_instructions.md). Thesis writing starts after 17 October; that is not a submission deadline. Dataset evidence is in [the preparation report](reports/non_chili_v1/dataset_preparation_report.md) and [verification record](manifests/non_chili_v1/verification.json).

## What we will compare

We will train four separate one-stage object detectors on the same verified 24-class, non-chili dataset using three-client FedAvg.

| Model | Proposed starting configuration | Main purpose |
| --- | --- | --- |
| YOLOv8n | Fresh pretrained initialization; candidate image size 640 | New baseline for the revised dataset |
| SSDLite320 MobileNetV3-Large | Pretrained detector; replace the class head; 320 input | Lightweight detection alternative and first pilot |
| SSD MobileNetV3-Large, standard head | Separate verified SSD configuration; candidate 320 input | Compare standard and lightweight heads using the same backbone family |
| EfficientDet-D0 | Verified pretrained D0 implementation; replace the class head; candidate 512 input | Separate efficient multiscale detector |

SSDLite and SSD–MobileNetV3 will be distinct models, not the same model trained twice under different names. The latter needs custom integration; it is not an existing ready-to-run model in this workspace. Faster R-CNN is excluded. The old 30-class YOLO checkpoints remain historical results, not the initialization for the new fair comparison.

## 1. Prepare one common dataset

1. Audit source labels, crop provenance, missing images, invalid boxes, class imbalance and cross-split duplicates/source variants.
2. Exclude all six chili classes, including healthy chili. Keep banana, cauliflower, groundnut and radish. Use labels and provenance rather than filenames alone. Exclude mixed chili/non-chili images for this four-crop study; record all exclusions. Resolve empty labels explicitly.
3. Remap IDs: 0–12 unchanged; 13–18 removed; 19–29 become 13–23.
4. Preserve existing train/validation/test and client membership where valid. If leakage is found, correct source-group assignments before freezing the dataset and document that decision.
5. Save class map, retained/excluded image lists, per-class/per-client image and box counts, source grouping and checksums under `manifests/non_chili_v1/`.
6. Inspect labelled overlays from every retained class. Numeric validity alone does not confirm correct disease labels.
7. Prepare one shared image/annotation export and one Colab upload package. Avoid one complete data copy per model. Preserve the original 30-class datasets.

**Pass condition:** four crops, 24 contiguous class IDs, valid boxes, reproducible client assignments, a recorded leakage audit and frozen manifests. The final retained image counts will be measured, not guessed from class totals.

Completed 10 October: built the separate `non_chili_v1` data with 14,514 train, 4,157 validation and 2,091 test images. Excluded 1,216 chili images, one empty-label image and 98 lower-priority copies of visually confirmed cross-split source variants. All 24 classes remain in every pooled split. Original merged datasets still contain chili and are preserved for historical reproduction. Near-duplicate screening has documented limitations; see the preparation report.

## 2. Build the shared training and evaluation pipeline

The historical root runner remains YOLO-specific. The new `comparison/` implementation supplies separate adapters and a common interface for initialization, local training, prediction, state exchange, saving and resume. Local checks now cover gradients, save/reload, pooled metrics, integer-buffer handling and interrupted-round recovery; GPU validation is next.

- Canonical dataset IDs: 0–23. TorchVision SSD/SSDLite foreground targets: 1–24, with background 0. Verify EfficientDet's conventions explicitly.
- Convert YOLO normalized boxes to each model's required coordinates and preserve correct boxes through augmentations and resizing.
- Convert every model's predictions back to original-image boxes and canonical class IDs before shared evaluation.
- All clients start a round from the same global state. Aggregate floating updates by the number of training images used; handle non-floating buffers explicitly.
- Use one explicit learning-rate schedule across the intended round budget. Do not accidentally restart a complete one-epoch schedule or warmup on every client call.
- Initial proposal: reset local optimizer state at each round while preserving the global round-based learning-rate schedule. Apply and document the same state-handling rule for every model.
- Specify normalization handling before training: which statistics are frozen, learned locally or aggregated. Do not average integer counters as though they were floating parameters. Check that batch sizes work with the model's normalization layers.
- Validate the global model using pooled predictions over the frozen validation images. Also retain separately labelled client metrics. Do not substitute a weighted average of client AP for pooled AP.

**Pass condition:** correct target/output conversions; finite gradients; identical initial global weights for clients; valid aggregation; save/reload produces consistent predictions; resume preserves the recorded schedule.

## 3. Verify the environment and pretrained initialization

Use the same GPU type for the main runs, preferably the Tesla T4 already used historically if available. Record actual hardware and compatible pinned PyTorch/TorchVision/core package versions. Record model-specific EfficientDet dependencies; do not assume the local Python 3.14 environment is the Colab environment.

Initialize each model from its recorded pretrained source and save the weight hash. Pilots must not give only one model extra training before its main run; restart main experiments from their specified initialization.

**Important SSD–MobileNetV3 limitation:** a standard SSD head cannot simply inherit differently shaped SSDLite head weights. Reuse verified compatible pretrained components, initialize incompatible layers explicitly, and report which components are new. If the goal is a controlled head-only ablation, add a separate paired experiment with identical backbone/feature maps, resolution, data and matched head initialization policy. Otherwise the main table compares practical configurations with initialization differences, not only head architecture.

## 4. Start with an SSDLite pilot

SSDLite is the first GPU pilot. Its CPU gradient, overfit, save/reload and limited three-client round checks passed. The separate standard-head SSD and EfficientDet/YOLO adapters also passed pretrained local checks. The new notebook defaults to the SSDLite GPU pilot and checks all four adapters before main training.

Pilot sequence:

1. Show labelled overlays and verify IDs, shapes and original-image coordinates.
2. Train one batch and run prediction on a validation batch.
3. Perform a brief overfit check on a small training-only subset to catch conversion or gradient problems. Its metrics are not final results.
4. Save/reload a checkpoint and test interrupted-run recovery.
5. Profile loader speed, peak GPU memory and a short client-local training segment. Candidate starting batch: 8; reduce if needed and record changes.
6. Complete one three-client federated round and pooled validation. Confirm all clients used identical round-start weights and that metrics/checkpoints reach Drive.

**Pass condition:** working predictions/boxes, finite losses, correct aggregation, recoverable checkpoints and measured full-round runtime. If a check fails, fix it before a long run. One pilot round is not enough to judge final architecture quality.

## 5. Main training order and budget

After the first pilot, verify the EfficientDet dependency/adapter early so its setup problems do not appear only after the other runs finish. Train serially on a single GPU.

Recommended main-run order:

1. Fresh 24-class YOLOv8n baseline.
2. SSDLite.
3. SSD–MobileNetV3 with its standard head.
4. EfficientDet-D0.

| Setting | Initial comparison protocol |
| --- | --- |
| Dataset | Same frozen `non_chili_v1` manifests |
| Clients | Same three fixed partitions |
| Aggregation | Training-image-weighted FedAvg |
| Main seed | 42 |
| First benchmark | 10 global rounds for every model |
| Local epochs | 1 per client per round |
| Validation | Pooled global validation after every round; client metrics saved separately |
| Checkpoint selection | Highest validation mAP50–95 within the declared budget |
| Test use | Final evaluation after model configurations/checkpoints are selected |

Ten rounds means each client's images receive ten local epochs. It does not mean every image receives thirty epochs merely because there are three clients.

Choose model-appropriate optimizer settings through a small equal tuning allowance, using training/validation only. Freeze resolved batch sizes, augmentation, learning rates, schedule and trainable layers before the main runs. Do not assume identical learning-rate numbers or architecture defaults make the comparison fair.

Review progress after round 5 without tuning against test results. Complete matched 10-round results first. If validation curves justify more training, extend all compared models to the next common budget, such as 20 rounds, using a documented continuation schedule. Do not automatically require 40 rounds or assume more rounds improve performance.

With available compute, repeat the strongest models with additional seeds and report mean/spread; prefer repeating the same-task comparison over declaring a winner from one noisy run. If only one seed is completed, state that limitation.

## 6. Resolution and fairness

The candidate configurations use different resolutions: YOLO640, EfficientDet512, SSDLite320 and SSD–MobileNetV3 320. Their main comparison measures practical accuracy/resource tradeoffs; it is not an isolated architecture-only test.

Keep the two MobileNetV3-backed SSD variants at the same resolution. Add a separate YOLO320 control if compute permits, especially if small lesions perform poorly on 320-input models. Report exact resizing/letterboxing and resolution for every model. Do not silently change input size during a run or attribute all score differences to the backbone.

## 7. Evaluation and analysis

- Main quality metric: shared COCO-style bounding-box mAP50–95. Also report mAP50 and per-class/crop results with image/instance counts.
- Precision, recall and F1: class-aware one-to-one matching at IoU 0.5 and the declared fixed confidence cutoff, initially 0.25. Report macro and micro values and the zero-prediction policy. AP calculation must retain predictions across the score curve; the precision/recall reporting cutoff is not its AP filter.
- Use the same category inclusion and maximum detections per image, initially 100. Freeze any evaluator implementation details before final results.
- Save validation curves, precision–recall plots and labelled examples of missed detections, false positives and localization errors.
- Compare training time, checkpoint size, parameter count, peak VRAM and batch-1 inference latency on the same hardware. Warm up inference, synchronize GPU timing and distinguish model-only from full pipeline latency.
- Select checkpoints only on validation. The historical test set has been examined before; disclose this and avoid additional test-driven tuning.

A higher score after removing chili is not by itself evidence of a better detector: the dataset scope and class average have changed. Keep historical 30-class, historical 24-class-subset analysis and new 24-class training results in distinct tables.

## 8. Independent result folders

Each model already has its own scaffold:

```text
experiments/<model>_non_chili_seed42_run01/
  README.md
  checkpoints/          best, last and required recovery checkpoints
  logs/                 progress and run status
  results/
    README.md
    metrics/            global/client CSVs and final summaries
    figures/            learning curves and comparison charts
    predictions/        annotated examples
    evaluations/        evaluator exports and per-class analysis
```

Use the same run IDs on Drive. Before training, save the resolved configuration, environment/package versions, manifest checksum, pretrained hash, class map, hardware and initialization details. Save completed global checkpoints and metrics atomically after every round. Test restore before long training. Preserve required resume state and random states; a partial federated round restarts from its last completed global state unless full mid-round recovery has been explicitly implemented.

To control storage, retain best/last/recovery global checkpoints and intentional milestone checkpoints. Local-client checkpoints need not all be retained once a global round and its metadata are verified, unless they are needed for analysis; declare that retention policy before training. Store training data locally in the Colab runtime and persistent results on Drive.

## 9. Timing and immediate work

Use measured pilot time to schedule jobs:

`estimated main time ≈ rounds × (three client-training times + pooled validation + saving)`, with extra allowance for data setup and interruptions.

The provisional sequence before 17 October is data preparation → common adapter/evaluator → SSDLite pilot → verified main runs → pooled evaluation and error analysis. The exact dates and completion estimates depend on GPU access and profiling. Writing begins after 17 October; experiments can continue when justified.

**The immediate next implementation task is the audited 24-class dataset and shared annotation adapter, followed by the SSDLite pilot.** This plan does not start training and the existing historical Colab notebook must not be used unchanged for the new comparison.

## Sources and related files

- [Architecture comparison plan](ARCHITECTURE_COMPARISON_PLAN.md)
- [Shared draft protocol](configs/comparison/protocol.json)
- [Experiment folder index](experiments/README.md)
- [Dataset manifest status](manifests/non_chili_v1/README.md)
- [PyTorch SSD/SSDLite architecture and training explanation](https://pytorch.org/blog/torchvision-ssdlite-implementation/)
- [EfficientDet PyTorch implementation and pretrained models](https://github.com/rwightman/efficientdet-pytorch)
- [TorchVision custom-data detection tutorial](https://docs.pytorch.org/tutorials/intermediate/torchvision_tutorial.html)
