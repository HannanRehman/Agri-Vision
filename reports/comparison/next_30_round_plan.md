# Agri Vision: account split and 30-round training plan

Agreed 10 October 2026. Implementation now prepared: [native YOLO and alternative continuation instructions](30_round_colab_instructions.md). Local engineering checks are complete; GPU learning verification is performed by the new YOLO notebook before main training. This supersedes the earlier main-run order and 10-round target. No GPU training was started during preparation.

## Accounts and resume points

| Model | Colab account and storage | Next round | Target global round |
| --- | --- | --- | --- |
| YOLOv8n | Separate account; dedicated accessible dataset and results folder | 1, fresh pretrained initialization | 30 |
| SSDLite320 MobileNetV3-Large | Current account and existing Agri Vision Drive | 11, from committed round 10 | 30 |
| SSD MobileNetV3-Large, standard head | Current account and existing Agri Vision Drive | 11, from committed round 10 | 30 |
| EfficientDet-D0 | Current account and existing Agri Vision Drive | 3, from last verified committed round 2 | 30 |

Recheck EfficientDet's latest committed checkpoint before execution; do not assume it reached round 10. An incomplete round repeats from the preceding committed global checkpoint.

YOLO can run alongside one alternative on the other account when both have GPU access. Run the three alternatives sequentially on the current account, each in its own notebook with only that model selected. Separate notebooks do not guarantee separate GPUs. Use authorized accounts within service limits; GPU access and runtime duration are not guaranteed.

## Shared experiment conditions

Use the frozen non_chili_v1 dataset: 24 classes, 14,514 training images, 4,157 validation images and 2,091 test images. Preserve its three client partitions, seed 42, training-image-weighted FedAvg and one local epoch per client per global round. Thirty global rounds means thirty local epochs per client's images, not ninety.

Use the same GPU type where available and record actual hardware. Keep separate model results, initialization records, package versions, configuration hashes and dataset checksums. Use pooled validation with the common evaluator and best validation mAP50-95 checkpoint selection. Keep native YOLO validation metrics separately labelled; historical weighted client summaries are not the common comparison metric. Reserve the held-out test for the selected configurations and checkpoints.

## YOLO: restore the historical method before attempting improvements

Restore native Ultralytics 8.4.48 local training rather than the custom comparison SGD loss loop. Match the verified historical settings: batch 8, image size 640, one local epoch, two workers, optimizer auto, native warmup, native augmentation and adapting BatchNorm statistics. Preserve the historical client training and aggregation behavior after auditing state transfer; do not introduce a custom head initializer into this baseline. Start from pretrained YOLOv8n with a canonical 24-class head, not the old 30-class round-40 checkpoint.

The dataset and class count differ from the historical experiment, so this reproduces the training method, not identical weights or guaranteed scores. Save the resolved arguments and actual optimizer, learning-rate progression, transferred/new tensor inventory and normalization policy.

Before the main 30-round run, verify learning on a small training-only sample and then a short federated pilot. Check that loss falls, detections become meaningful, save/reload works and local training versus aggregation does not collapse predictions. Restart the main run from its declared pretrained initialization after the pilot. Do not use pilot weights to give YOLO extra unreported training.

Improvements follow a working baseline as separately named validation-only experiments. Change one factor at a time, such as augmentation, learning-rate handling or imbalance treatment; apply a documented tuning allowance and never promise an accuracy gain in advance.

## Alternatives: checkpoint continuation requires an explicit schedule

Keep the successful SSD/SSDLite checkpoints and existing EfficientDet progress. Their original run settings declared ten rounds and a global-round cosine learning-rate schedule. Merely changing --rounds to 30 changes the configuration and is not supported by the current strict resume checks.

Before execution, implement a documented continuation mechanism that validates the original configuration and checkpoint lineage, preserves rounds already completed, and records a separately specified learning-rate schedule for the extension. Choose the schedule using training/validation evidence before launching; avoid silently recomputing earlier learning rates as if the experiment had always targeted thirty rounds. Record optimizer reset behavior and original versus continuation settings. Test recovery using synthetic state without starting real training.

Implemented schedule: EfficientDet resumes from committed round 2, completes its original ten-round schedule, then extends with its terminal LR through round 30. SSDLite and SSD extend from their committed round 10 at their terminal LR. Separate linked continuation folders preserve the originals and record the new schedule.

The final table must describe the alternatives as continued ten-round experiments and YOLO as a fresh thirty-round native-training experiment. Matching total rounds alone does not make training schedules or preprocessing identical; report practical detector configurations, resolutions, initialization and these differences. Show matching round checkpoints where useful.

## Preparation order

1. Preserve the current bundles, notebooks, source implementation and checkpoint hashes for recovery.
2. Implement and verify the separate native-YOLO 24-class runner, common-evaluator export and learning pilot notebook.
3. Implement and verify explicit alternative-model continuation without weakening existing resume checks.
4. Prepare one notebook per model, dedicated output paths and account-specific dataset access; verify Drive storage before launch.
5. Run the YOLO learning pilot when GPU access returns, then launch its fresh thirty-round run. On the current account resume EfficientDet, then extend SSDLite and SSD sequentially according to the declared continuation policy.
6. Save every completed round atomically; retain initial/best/latest recovery checkpoints and metrics. Clean intermediate Drive weights only after verified local backup, never before saving a resumable state.

AgriVision_Comparison now implements the three-model continuation workflow toward round 30; AgriVision_YOLO_Native_30Rounds runs the separate native YOLO experiment. The old comparison notebook is retained as AgriVision_Comparison_10Round. AgriVision_EfficientDet_Resume still restores the original EfficientDet run toward round 10. Original comparison implementation and original bundles remain unchanged; use the distinct agrivision_30_round_code.zip for the new workflows.

Related evidence: [saved main-run review](main_run_review_20261010/README.md), [YOLO diagnosis](yolo_diagnosis_20261010/README.md), and [training plan](../../TRAINING_PLAN.md).
