# Final-round per-class validation metrics

Fetched from saved Drive results on 13 October 2026. Each separate CSV contains 24 disease/pest/healthy class rows from the latest completed global round: YOLOv8n round 30, SSDLite round 30, SSD–MobileNetV3 round 30 and EfficientDet-D0 round 15. The combined CSV contains 96 rows.

These are final-round results, rather than each model's validation-best checkpoint. The poster selects the best checkpoint, so its model-level values can differ. These tables are pooled validation detection metrics on the same 4,157 images and 7,210 annotated boxes; they are not held-out test scores or classification accuracy.

Precision, recall, F1, mAP50 and mAP50–95 retain their original 0–1 values. Multiply by 100 to display percentages. Precision and recall use confidence 0.25 and IoU 0.50; AP uses the evaluator's confidence sweep. TP, FP and FN are per-class counts. The ground-truth box count is derived as TP + FN. Do not average per-class precision/recall to reproduce pooled micro precision/recall.

The `sources/` JSON files preserve the saved detailed round results, including pooled and client metrics. `manifest.json` records original Drive links and checksums. Each class mapping, row count and sum of TP/FP/FN was checked against the corresponding pooled result. No training or fresh inference/evaluation was started to create these CSVs.

Open `Agri_Vision_Final_Round_Per_Class_CSVs.zip` to obtain all CSVs and their source evidence together.
