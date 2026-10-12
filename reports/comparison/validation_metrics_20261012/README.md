# Saved comparison validation metrics — 12 October 2026

These CSVs retain the 30 completed pooled-validation rounds for YOLOv8n, SSDLite and SSD–MobileNetV3. They preserve the model result evidence independently of the local seminar-material folder. Copying these tables involved no training or new evaluation.

- [YOLOv8n](yolo_v8n.csv)
- [SSDLite](ssdlite.csv)
- [SSD–MobileNetV3](ssd_mobilenetv3.csv)
- [EfficientDet-D0 through round 15](../efficientdet_round15_review_20261012/global_validation_metrics.csv)

All rows describe the same 4,157-image validation split. Precision and recall use the fixed confidence 0.25 and IoU 0.50 operating point; AP measures use the saved pooled evaluator. These are detection metrics, not classification accuracy or held-out test results.
