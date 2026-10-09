# Agri Vision: revised architecture comparison and 24-class implementation plan

Prepared 9 October 2026. Status: plan only; no new training started and no dataset changed.

This replaces the architecture recommendation in the earlier fifteen-day plan. The supervisor requested EfficientDet, MobileNetV3 and SSDLite as separate experiments alongside the existing YOLO model, and explicitly excluded Faster R-CNN because it is a two-stage detector. The user clarified that MobileNetV3 means a MobileNetV3-based one-stage object detector, not image classification. Faster R-CNN is removed from this implementation plan. The poster will be revised using the supplied stacked-section layout.

## Recommended direction

Build one verified four-crop, 24-class dataset and compare four separate one-stage detection models: YOLOv8n, EfficientDet-D0, SSD with MobileNetV3-Large and a standard detection head, and SSDLite with MobileNetV3-Large and a lightweight detection head. The separate SSD–MobileNetV3 configuration must be implemented and verified; do not count the same SSDLite model twice. Retrain a 24-class YOLO baseline under the same experimental protocol as the other detectors. Workspace scaffolds and draft configurations now exist; no training or dataset filtering has started.

For the requested comparable environment, use the same three-client federated protocol, retained source-image partitions and GPU type for the confirmed detection experiments. Run short local smoke tests to verify each implementation and estimate runtime before its federated run. These local pilots are engineering checks, not the final architecture comparison. A centralized alternative and the historical federated YOLO are separate experiments, not a fair architecture comparison by themselves. Shared hardware does not make classification and detection scores directly comparable.

Timing correction: 17 October is not a submission deadline. The user intends to start writing the thesis after 17 October and use the time before then for implementation and experiments. No final submission deadline has been specified. Aim for a usable initial comparison by 17 October, without treating that date as a forced end to training. The earlier deadline-driven schedules no longer apply.

## Exact models to compare

| Supervisor's name | Proposed implementation | What the experiment tests |
| --- | --- | --- |
| Existing YOLO | YOLOv8n, 24-class head | Familiar lightweight detection baseline |
| MobileNetV3 | SSD with MobileNetV3-Large and a standard detection head; custom configuration to verify | Standard versus lightweight detection head using the same backbone family |
| SSDLite | SSDLite320 + MobileNetV3-Large, TorchVision `ssdlite320_mobilenet_v3_large` | Lightweight one-stage detector; different detection head using the same backbone family |
| EfficientDet | EfficientDet-D0 | Separate complete one-stage detector; do not substitute an EfficientNet classifier |

MobileNetV3 alone is an image classifier/backbone. Its classification accuracy cannot be compared directly with YOLO detection mAP. The complete model names and tasks must appear in the thesis. SSDLite can internally use MobileNetV3; that does not replace the requested separate MobileNetV3 experiment. EfficientDet likewise has an internal backbone, but it is trained and reported as the complete EfficientDet detector.

Clarification received: MobileNetV3-based one-stage object detection, proposed as SSD with a standard detection head. All four experiments should output bounding boxes, disease classes and scores on full images and share the detection evaluation protocol. Report complete model names. Do not use Faster R-CNN, substitute classification, or duplicate the SSDLite experiment under a MobileNetV3 label. Check pretrained initialization carefully: replacing SSDLite's lightweight head with a standard SSD head does not supply pretrained standard-head weights. Record newly initialized components and this experimental limitation.

Standalone MobileNetV3 and SSDLite have documented TorchVision implementations. EfficientDet has an additional dependency and compatibility burden; use a tested, version-pinned implementation with pretrained weights rather than writing its architecture from scratch. No model is guaranteed to outperform YOLO on this dataset.

## Dataset change: remove all six chili classes

Assumption based on the request: remove the complete chili crop group, including healthy chili, rather than selecting classes according to their scores. The remaining scope is banana, cauliflower, groundnut and radish, with 24 classes.

The records do not show zero chili images. For example, the distribution report lists 55 training images for leaf curl, 59 for whitefly and 67 for leaf spot; some chili test classes have only seven images. The merge audit also records excluded duplicate and contradictory external annotations. This supports an audit and narrower scope, but does not prove every chili annotation is wrong. Preserve the original 30-class results and explain the revised scope transparently.

Implementation order:

1. Audit labels, crop/source metadata and a visual sample before filtering. Record missing/corrupt images, invalid boxes, class imbalance and exact duplicates across splits. Group source variants when identifiable; check near-duplicates where practical.
2. Create a separate `original_dataset_non_chili/` and `dataset_non_chili/`. Preserve the original datasets, client partitions and saved checkpoints.
3. Exclude chili images from train, validation and test. Do not infer crop membership from filenames alone. If an image contains both chili and retained labels, exclude it from this four-crop study and record that decision. Resolve empty-label images using provenance; do not silently treat unknown crop images as valid non-chili backgrounds.
4. Remap IDs: old 0–12 stay 0–12; old 13–18 are excluded; old 19–29 become 13–23. Save a machine-readable old-to-new map and the class names.
5. Preserve retained images' existing split and client membership where valid. If leakage is found, fix source-group assignments before freezing the new split; document the correction and apply the same final manifest to every model.
6. Produce manifests, checksums, image/box counts per class and per client, and exclusions with reasons. Freeze these before the main runs. Do not guess final image totals from per-class counts, because one image can have several labels.

Do not continue round 40 as round 41 with a 24-class head. The label space and experiment have changed. Start a new experiment from pretrained weights. A historical 30-class model may be scored separately on the retained test set, with its output labels mapped and chili predictions excluded, but it remains a historical reference.

The previous non-chili report (69.81% precision, 62.73% recall, 65.16% mAP50, 43.44% mAP50–95) averaged 24 classes of the existing 30-class checkpoint while retaining all original test images. It is not the score of a retrained 24-class model and is not directly the new benchmark.

## Work packages

| Component | Proposed responsibility | Acceptance condition |
| --- | --- | --- |
| Dataset preparation | Separate filtering script, class-map export and frozen manifests | Only four crop groups; contiguous IDs; valid labels; leakage audit recorded |
| Dataset adapter | Read YOLO labels; convert normalized centres/sizes to pixel boxes; provide COCO annotations | Boxes overlay correctly; round-trip conversion preserves geometry |
| Model factory | Build SSDLite and the separate MobileNetV3-based one-stage detector | Distinct detection heads; correct class/box outputs; save/reload preserves predictions |
| EfficientDet adapter | Build pretrained D0; replace class head; handle its box and target conventions explicitly | Finite training loss and verified boxes in original-image coordinates |
| Training runner | Local pilot and per-client training, validation, checkpoint selection and resume | Best/last weights and complete logs saved to Drive |
| Shared evaluator | Convert each model's outputs to one canonical class/box format | Same evaluator, dataset and metric settings for all models |
| Federated runner | Add model-independent local training and aggregation | All clients receive identical global state at each round start |
| Reporting | Metrics CSV, per-class results, plots, runtime and error examples | A comparison table traceable to immutable run settings |

Keep new comparison code isolated from the current YOLO pipeline until verified. The existing `model.py` and client paths are YOLO-specific; this is more than swapping a model name.

TorchVision SSDLite and standard SSD foreground labels must be 1–24 because label 0 is background; YOLO's canonical IDs stay 0–23. Verify the custom SSD–MobileNetV3 configuration follows this convention. EfficientDet class conventions must also be checked against the chosen implementation. Conversion and label offsets need meaningful tests because mistakes can produce plausible but wrong metrics.

## First implementation sequence

1. Prepare and audit the 24-class manifests.
2. Implement and verify SSD–MobileNetV3's standard detection head before its training. Do not build or train Faster R-CNN.
3. For confirmed detection work, build the annotation adapter and inspect representative box overlays, including multi-object images and healthy classes.
4. Pilot SSDLite or EfficientDet as a separately named experiment with pretrained weights and the appropriate 24-class detection head. This does not count as the separate MobileNetV3 run.
5. Check one training batch, one validation batch, checkpoint save/reload and a brief overfit check on training samples. Profile throughput and VRAM on the actual Colab GPU.
6. Start matched federated runs only after the pilot passes. Start from recorded pretrained initialization, without giving only one model extra pilot training.

Workspace organization has been authorized and completed. No detector training has started. MobileNetV3's detection task and proposed standard SSD head are specified; verify its distinct implementation before running it.

## Fair comparison protocol

- Same frozen training, validation and test image lists, 24 classes and evaluator.
- Fresh pretrained initialization for every benchmark model. Save the old 40-round YOLO experiment as historical evidence.
- Main-run starting target: ten federated rounds per model, three clients and one local epoch per round, after profiling confirms a practical runtime. Inspect results after five rounds as a progress check. Across all clients ten rounds gives ten passes through the partitioned training data, not thirty passes per image. Plan the schedule from measured runtime, not an assumed submission deadline. Use one seed for the initial comparison; add repeat seeds and longer matched budgets when resources permit. Ten rounds is an initial benchmark, not evidence that every architecture has converged.
- Use appropriate optimizers for each detector, with a small declared tuning allowance. Identical learning rates across different models are not automatically fair. Record effective batch sizes, schedules, augmentation and frozen layers. Keep the same GPU type and compatible pinned core environment; record model-specific dependencies rather than forcing every detector into Ultralytics. Train one model at a time on a single GPU.
- Use a modest common geometric augmentation policy where supported, with correctly transformed boxes. Record implementation differences and do not attribute all results to backbone choice.
- Select the best checkpoint by validation mAP50–95. Keep last checkpoint for resume. Do not choose checkpoints or confidence thresholds from test performance.
- SSDLite's standard implementation uses 320×320; EfficientDet-D0 typically uses 512×512; YOLO can use larger inputs. Report exact sizes. The main comparison measures practical model configurations, not architecture alone. Add a YOLO320 control if time permits to help interpret SSDLite's resolution tradeoff.
- Evaluate final selected checkpoints on the same held-out test set after configurations are frozen. The old test set has already been examined historically, so disclose this and avoid further tuning against it.

Metric protocol:

- Primary: common COCO-style bounding-box mAP50–95; also report mAP50, per-class AP/recall and crop-group summaries.
- Precision, recall and F1: class-aware one-to-one box matching at IoU 0.5; report macro and micro values at a declared fixed confidence threshold (initial proposal 0.25). This score cutoff is separate from AP calculation, which needs low-score predictions across the curve. Model-calibrated operating points, if used, must be chosen on validation and labelled separately.
- Use the same maximum detections per image, category inclusion and evaluation settings. Preserve original-image coordinates after resizing and letterboxing.
- Resource comparison: training time, inference latency on the same hardware at batch 1 with warmup and GPU synchronization, checkpoint size, parameter count and peak VRAM. Distinguish model-only latency from preprocessing/postprocessing.
- Object-detection mAP is the main quality metric; conventional image-classification accuracy does not describe this task.

The current runner reports validation-image-weighted client AP/P/R. A weighted average of client AP values is not the AP obtained from pooled predictions. For the new comparison, use one pooled validation evaluator over the frozen validation set, alongside separately labelled client metrics. Do not silently rename the historical metric.

## Main federated comparison

Recommended detection comparison: YOLOv8n, EfficientDet, the separate MobileNetV3-based one-stage detector and SSDLite, each using three fixed non-overlapping client partitions, initially ten rounds and one local epoch per round, subject to measured runtime. Use equal declared data-pass budgets and fresh matched experiment starts. Do not compare a new ten-round alternative to the old forty-round YOLO as evidence of architecture superiority. Review validation curves at matched round counts; if improvements justify it, agree on and run an extended common budget, such as twenty rounds. Save validation-best checkpoints even when later rounds decline.

The new runner should retain explicit learning-rate progression across rounds, reset or retain local optimizer state according to one documented rule, and record local and global metrics. Aggregate floating tensors by client training-image count; handle integer buffers explicitly. Specify a consistent normalization policy before the run, since BatchNorm behavior differs between detectors and can influence aggregation. Log actual communication payload bytes per round.

If measured runtime or implementation problems make the initial comparison impractical by 17 October, adjust the timeline rather than treating the date as a submission cutoff. Keep completed matched-budget results and clearly mark unfinished work. Do not silently switch just one model to centralized training or put classifier and detector scores into one ranking. Local pilots verify correctness and predict runtime; they cannot establish federated performance.

## Schedule and scope control

The user clarified that thesis writing starts after 17 October and there is no stated submission deadline. The following is a provisional work sequence to check against GPU profiling, not a guarantee that all training jobs will finish on those dates.

| Date | Priority | Deliverable |
| --- | --- | --- |
| 9 October | Specify distinct MobileNetV3 detector; data audit and 24-class preparation | Settled model definitions and verified data |
| 10–11 October | First alternative pilot/run; fresh YOLO baseline; shared evaluation | First same-task runs and measured budgets |
| 12–13 October | Remaining detector adapter/run; separate MobileNetV3 detection pipeline | Three requested alternatives implemented as separate experiments if checks pass |
| 14–16 October | Complete initial matched runs; review validation curves and investigate errors | Initial comparable results and decisions about extensions |
| 17 October | Review available comparison with supervisor; organize results and experiment records | Stable methodology and an honest status for each model |
| After 17 October | Start thesis writing; finish justified extensions or repeat seeds alongside writing | Thesis grounded in completed experiments; revised poster and presentation as needed |

Before thesis writing starts, keep experiment notes, configurations, sources, figures and explanations ready so the methodology can be written from accurate records. Profile runs first and schedule serially on a single GPU; two heavy training jobs on one T4 are not assumed to speed up completion. Do not invent a final submission date or impose an experiment freeze based on the earlier misunderstanding.

EfficientDet is the highest integration risk. If its dependency or training pipeline cannot pass a correct pilot within one focused setup day, record the blocker and revisit scope with the supervisor rather than silently claiming completion. Keep a small result from a clearly labelled pilot separate from the full benchmark.

## Saving results reliably

Each run gets a unique directory on Drive with `config.json`, class map, dataset-manifest checksum, pinned package versions, `metrics.csv`, per-class CSV, best/last checkpoints, resume state, timing and selected prediction/error examples. Record losses, validation metrics, learning rate, epoch/round, elapsed time and GPU use. Save atomically after each completed epoch/round and test resume before long training.

Detection table: full detector name, regime, input size, data-pass budget, precision, recall, F1, mAP50, mAP50–95, latency, model size and training time. Include four distinct one-stage detectors; missing experiments remain marked incomplete. Keep historical 30-class results distinct from the new 24-class comparison and state the scope reduction to four crops.

## Local evidence and architecture sources

- [Current class distribution](reports/dataset_class_distribution.md)
- [Chili merge audit](reports/chili_dataset_merge.md)
- [Historical round-35 non-chili analysis](reports/round_035_non_chili_report.md)
- [TorchVision standalone MobileNetV3 documentation](https://docs.pytorch.org/vision/stable/models/generated/torchvision.models.mobilenet_v3_large.html)
- [TorchVision SSDLite documentation](https://docs.pytorch.org/vision/stable/models/generated/torchvision.models.detection.ssdlite320_mobilenet_v3_large.html)
- [TorchVision detection fine-tuning tutorial](https://docs.pytorch.org/tutorials/intermediate/torchvision_tutorial.html)
- [EfficientDet original paper](https://arxiv.org/abs/1911.09070)
- [EfficientDet PyTorch implementation and pretrained model configurations](https://github.com/rwightman/efficientdet-pytorch)
