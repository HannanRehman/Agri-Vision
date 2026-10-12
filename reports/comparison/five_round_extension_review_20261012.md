# Five-round extension assessment — 12 October 2026

This review reads the locally verified saved validation CSVs, including the subsequently verified EfficientDet completion records. It runs no training or new model inference, and makes no changes to notebooks, checkpoints or training settings. Completed rounds: YOLOv8n 30, SSDLite 30, SSD–MobileNetV3 30, EfficientDet-D0 15.

## Recommendation

Do not add five unchanged rounds to every model. EfficientDet has now completed rounds 11–15; preserve its validation-best round-11 checkpoint rather than automatically adding another five rounds. Consider separate SSDLite and SSD–MobileNetV3 extensions through round 35. Keep YOLO's completed 30-round baseline and its validation-best round-12 checkpoint. If YOLO improvement is pursued, use a separately recorded controlled tuning experiment rather than treating additional rounds alone as the fix.

The primary selection measure is pooled validation mAP50–95. Precision, recall, mAP50 and F1 are secondary checks. All reported changes below are percentage points, not relative percentages.

| Model | Interval | mAP50–95 at start | mAP50–95 at end | Change | Current validation-best round | Decision |
|---|---|---:|---:|---:|---:|---|
| YOLOv8n | 25 → 30 | 43.19% | 42.69% | −0.50 | 12: 45.34% | Do not extend unchanged |
| SSDLite | 25 → 30 | 44.01% | 45.20% | +1.19 | 30: 45.20% | Five-round extension is justified as an experiment |
| SSD–MobileNetV3 | 25 → 30 | 33.91% | 35.20% | +1.29 | 30: 35.20% | Five-round extension is justified as an experiment |
| EfficientDet-D0 | 10 → 15 | 59.56% | 59.47% | −0.10 | 11: 59.70% | Target of 15 completed; no automatic further extension |

## Curve interpretation

YOLO: the round-30 score is 2.65 points below its round-12 optimum. Its mean mAP50–95 is 42.76% over rounds 21–25 and 42.88% over rounds 26–30, a small 0.11-point fluctuation rather than sustained recovery. Precision falls 5.80 points from round 25 to round 30, while recall increases 1.93 points and F1 falls 1.70 points. The existing loss review supports a generalization problem, but does not establish one exclusive cause. Repeated one-epoch optimizer/warmup resets and persistent mosaic are confirmed implementation characteristics whose causal effects require ablations.

SSDLite: mean mAP50–95 increases from 43.68% over rounds 21–25 to 44.56% over rounds 26–30, so the positive conclusion is supported by windows as well as endpoints. From round 25 to round 30, precision improves 1.56 points, recall 0.75 points and mAP50 1.57 points. Gains are gradual; another five rounds may produce a modest improvement, but no exact gain is predictable.

SSD–MobileNetV3: mean mAP50–95 increases from 33.59% over rounds 21–25 to 34.72% over rounds 26–30. From round 25 to round 30, recall improves 1.37 points while precision drops 1.95 points and F1 drops 0.22 points. Round 30 is best for mAP50–95, but round 29 has higher mAP50. Further rounds are justified by the localization trend; improved precision is not guaranteed.

EfficientDet: the completed extension reaches its best mAP50–95 at round 11 (59.70%), only 0.14 points above round 10. Round 15 ends at 59.47%, 0.23 points below the best and 0.10 points below round 10. Mean mAP50–95 improves from 59.16% over rounds 6–10 to 59.59% over rounds 11–15, but the individual rounds do not show continuing improvement after round 11. Precision decreases 1.83 points, recall 0.75 points and F1 1.39 points from round 10 to round 15. This supports preserving the best checkpoint and declining an automatic further unchanged extension; it does not establish the cause of the fluctuations.

These are observations from one run per model. Adjacent rounds are correlated, and no repeated-seed study establishes statistical significance or predicts extension gains.

## Estimated five-round runtime

Estimates extrapolate the mean saved round duration over the last five completed rounds. They cover local training, pooled validation and checkpoint work recorded by the runner. Dataset assembly, dependency installation, Drive startup, interruptions and different GPU performance can add time.

| Model | Mean recent time per round | Five additional rounds |
|---|---:|---:|
| YOLOv8n | 9.85 min | 49.3 min, not recommended unchanged |
| SSDLite | 8.25 min | 41.2 min |
| SSD–MobileNetV3 | 8.00 min | 40.0 min |
| EfficientDet-D0 | 19.86 min | 99.3 min, not recommended automatically |

The two possible SSD extensions total about 1 h 21 min if executed serially, excluding setup and interruptions. EfficientDet's completed rounds 11–15 took 99.3 minutes according to the saved round timers. Parallel account execution changes wall-clock duration, not the total recorded compute requirement.

## Continuation and reporting rules

- Preserve the completed baseline results and original best/latest checkpoint lineage. Begin each unchanged extension from its last committed round. Do not silently restart from an earlier best round and call it the next consecutive round.
- Keep the same dataset, clients, local epochs, evaluator, resolutions and model settings during an unchanged extension. Alternative-model continuations retain their original schedule followed by the terminal learning rate; do not restart the cosine schedule merely by changing the target.
- The existing source/configuration validation rejects unrecorded budget changes. Use an audited continuation branch rather than editing a completed run's target in place. Any chained extension must explicitly preserve the original ten-round learning-rate schedule; reusing a thirty-round continuation budget as a new cosine horizon would alter the training method.
- Select the best validation mAP50–95 checkpoint over the available baseline and extension; the latest checkpoint need not be best. Keep precision/recall and per-class behavior visible, especially for SSD–MobileNetV3.
- Assess the five-round extension against the previous best, not only against the last checkpoint. Stop further budget increases if the extension establishes no new best and shows no sustained positive trend. Do not implement this as an automatic stopping rule without separately agreeing on the criterion.
- Treat extensions as additional experiments in the thesis. A matched-round-10 table is currently available for all four models; round-30 results are available for three models. EfficientDet at 15 and SSD variants at 35 still represent different training budgets. Report those budgets rather than claiming equal training.
- Keep the held-out test split out of extension decisions. Perform final test evaluation on the selected checkpoints after configuration selection.

## Sources

- [YOLO saved pooled validation](validation_metrics_20261012/yolo_v8n.csv)
- [SSDLite saved pooled validation](validation_metrics_20261012/ssdlite.csv)
- [SSD–MobileNetV3 saved pooled validation](validation_metrics_20261012/ssd_mobilenetv3.csv)
- [EfficientDet saved pooled validation through round 15](efficientdet_round15_review_20261012/global_validation_metrics.csv)
- [EfficientDet completion and checkpoint review](efficientdet_round15_review_20261012/README.md)
- [Existing YOLO loss and implementation analysis](yolo_native_decline_20261011/README.md)
- [Continuation implementation](../../scripts/continue_comparison.py)
- [Ultralytics guidance on validation monitoring and early stopping](https://docs.ultralytics.com/guides/model-training-tips/)
