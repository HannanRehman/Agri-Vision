# Colab GPU pilot review — 10 October 2026

Reviewed the files in Agri Vision/comparison_results and retained copies of the small JSON reports locally. No training or Drive changes were performed during this review.

## Decision

The engineering pilot passed on a Tesla T4. Proceed to the first full-dataset main run. Start with YOLOv8n alone so its duration and full validation curve can be reviewed before the remaining matched architecture runs.

Use MODE="main", MODELS=["yolo_v8n"], ROUNDS=10, LOCAL_EPOCHS=1, SEED=42, FINAL_TEST=False. Use the same T4 type, package versions and frozen dataset for every architecture. This is the agreed initial comparison budget, not a claim that ten rounds establish optimal convergence. Existing main-run round budgets cannot be changed on resume.

## Verified evidence

- gpu_pilot_passed.json records phase=passed, GPU=Tesla T4, local_epochs=1.
- The recorded implementation SHA-256 matches the current workspace: c6d295dc79d706a18d73090edf91532cda74fee23a9f4d3e737e0470915fff71.
- status.json records phase=complete, completed_rounds=1, mode=pilot, diagnostic_only=true. The notebook writes the pass marker only after its completed-run resume check succeeds.
- The initial and global_round_001 checkpoints are present, and latest.json records the committed checkpoint and its hash. Checkpoint binary hashes were not independently recalculated during this review.
- All three clients trained on 16 images each, with two optimizer steps per client. Their round-start hashes all match the initial global state.
- SSDLite's separate eight-step train-only loss check decreased from 16.6706 to 13.6200; finite gradients, frozen BN buffers and save/reload prediction consistency passed.
- SSD–MobileNetV3, YOLOv8n and EfficientDet-D0 each passed one-step GPU gradient/inference/save-reload checks. Those checks establish execution compatibility, not full-run convergence or accuracy.
- Colab used Python 3.13.15, Torch 2.11.0+cu130, TorchVision 0.26.0+cu130, Ultralytics 8.4.48, pycocotools 2.0.11, effdet 0.4.1 and timm 1.0.27.

## Diagnostic metrics

The single federated pilot round trained on 48 images total and evaluated 24 images containing 60 annotated boxes. Pooled mAP50=3.3976%, mAP50–95=2.4889%, and fixed-threshold precision/recall/F1=0 at confidence 0.25 and IoU 0.5. AP uses a lower score floor and a score sweep, so positive AP can coexist with zero precision/recall at the fixed cutoff.

These are engineering diagnostics and must not be presented as the final 24-class model results. Full main training will use all 14,514 training images and all 4,157 validation images; the 2,091-image held-out test remains separate.

The pilot's six-second round duration cannot be extrapolated directly to full epochs. Its saved peak CUDA allocation follows the completed-resume path and should not be used as a training-memory benchmark.

Keep the pilot folder and pass marker. Run the main notebook once on the same GPU type; rerunning main with an empty RESUME_RUNS creates another fresh run. If Colab interrupts, use the generated run folder in RESUME_RUNS and preserve all settings.
