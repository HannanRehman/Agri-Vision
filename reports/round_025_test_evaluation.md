# Round 25 Held-Out Test Evaluation

The round-25 global checkpoint was evaluated at image size 640 on the unchanged held-out test split: 2,191 images containing 4,305 labelled objects across 30 classes.

| Checkpoint | Precision | Recall | mAP50 | mAP50-95 |
| --- | ---: | ---: | ---: | ---: |
| Round 20 | 61.7% | 50.6% | 50.51% | 33.53% |
| Round 25 | 66.3% | 55.3% | 57.01% | 38.58% |
| Change | +4.6 pp | +4.7 pp | +6.50 pp | +5.05 pp |

Round 25 is a clear overall improvement over round 20 on the same test split. Its mAP50 improved by 12.9% relative to the round-20 value, while the stricter mAP50-95 improved by 15.1% relative.

## Chili classes

| Class | Test instances | Round-20 mAP50 | Round-25 mAP50 | Change |
| --- | ---: | ---: | ---: | ---: |
| Chili anthracnose | 89 | 2.07% | 9.64% | +7.57 pp |
| Chili healthy | 54 | 62.00% | 74.50% | +12.50 pp |
| Chili leaf curl | 15 | 0.00% | 0.03% | +0.03 pp |
| Chili leaf spot | 95 | 3.33% | 3.07% | -0.26 pp |
| Chili whitefly | 34 | 0.51% | 1.35% | +0.84 pp |
| Chili yellowish | 7 | 17.00% | 22.30% | +5.30 pp |

The extra training data helped chili anthracnose, healthy, whitefly, and yellowish, but leaf curl, leaf spot, and whitefly remain unreliable. The 100% precision reported for leaf curl and whitefly is not evidence of good detection: their recall is zero, meaning the model made no true detections at the evaluation threshold. These weak results are consistent with the very small number of independent examples for several chili classes and should be reported honestly.

The complete per-class figures are in `reports/round_025_test_metrics.csv`. Evaluation plots and prediction samples are stored locally in `saved_models/test_evaluation_round_025/`.
