# All four models: round 30, best round and per-class results

- `All_Models_Round30_and_Best.csv`: eight rows, showing each model's round-30 result and validation-best round across rounds 1–30. When round 30 is also best, both labelled rows intentionally show the same result.
- `All_Models_Round30_Per_Class.csv`: 96 rows, covering all 24 classes for every model at round 30.

Precision, recall, F1, mAP50 and mAP50–95 are percentages (0–100), rounded to two decimals. Best-round selection uses the original unrounded pooled validation mAP50–95. These are object-detection validation results, not classification accuracy or final held-out test results. The shared validation split contains 4,157 images and 7,210 annotated boxes. Precision/recall use confidence 0.25 and class-aware IoU 0.50; AP uses a confidence sweep.

Per-class TP, FP and FN are retained. The class ground-truth box count equals TP + FN. Do not average per-class precision/recall to recreate the micro-pooled summary values. The source JSONs and history CSVs preserve unrounded metrics under `sources/`; the validation record contains their checksums and the selected rounds. No training or new model evaluation was performed for this export.

Saved YOLOv8n, SSDLite and SSD–MobileNetV3 histories come from `reports/comparison/validation_metrics_20261012/`; their round-30 detail records were fetched from Drive for `reports/comparison/final_round_per_class_20261013/`. EfficientDet history and round-30 details were fetched from the completed 4gmail `_to30_from15` run and cached under `archive/comparison_download_efficientdet_round30/`.

Regenerate with `.venv/Scripts/python.exe reports/comparison/all_models_round30_and_best_20261015/build_csvs.py`.
