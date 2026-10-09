# EfficientDet-D0: non-chili comparison

Run ID: `efficientdet_d0_non_chili_seed42_run01`. **Status: planned; no model trained.**

Model specification: [configuration](../../configs/comparison/efficientdet_d0.json). Shared settings: [protocol](../../configs/comparison/protocol.json). Dataset: [non_chili_v1](../../manifests/non_chili_v1/README.md), not yet prepared.

Use checkpoints/ for best and last global weights and logs/ for progress. Each model has a separate results/ folder: results/metrics/ for round/client CSVs and summaries, results/figures/ for charts, results/predictions/ for selected examples and results/evaluations/ for final evaluator exports. Save actual environment versions, initialization, manifest checksum, hardware and timing before training. Draft configs are specifications, not supported CLI arguments for the existing YOLO runner.
