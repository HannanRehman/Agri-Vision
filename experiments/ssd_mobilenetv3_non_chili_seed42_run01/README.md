# SSD MobileNetV3-Large (standard head): non-chili comparison

Run ID: `ssd_mobilenetv3_non_chili_seed42_run01`. **Status: planned; no model trained.**

Model specification: [configuration](../../configs/comparison/ssd_mobilenetv3.json). Shared settings: [protocol](../../configs/comparison/protocol.json). Dataset: [non_chili_v1](../../manifests/non_chili_v1/README.md), not yet prepared.

Use checkpoints/ for best and last global weights and logs/ for progress. Each model has a separate results/ folder: results/metrics/ for round/client CSVs and summaries, results/figures/ for charts, results/predictions/ for selected examples and results/evaluations/ for final evaluator exports. Save actual environment versions, initialization, manifest checksum, hardware and timing before training. Draft configs are specifications, not supported CLI arguments for the existing YOLO runner.
