# Round 35: Existing Model Performance Excluding Chili Classes

The existing 30-class checkpoint was evaluated at image size 640 on the held-out original test split. No weights or datasets were modified. Chili classes are excluded from the reported 24-class average, not from the model or test images.

| Group | Classes | Labelled instances | Precision | Recall | mAP50 | mAP50–95 |
|---|---:|---:|---:|---:|---:|---:|
| all_30_classes | 30 | 4305 | 63.72% | 54.81% | 55.72% | 37.26% |
| non_chili_24_classes | 24 | 4011 | 69.81% | 62.73% | 65.16% | 43.44% |
| chili_6_classes | 6 | 294 | 39.37% | 23.14% | 17.94% | 12.53% |

These are macro class averages: every included class has equal weight. Precision and recall use the operating point selected for the original 30-class model, rather than a newly optimized non-chili threshold. Conventional classification accuracy is not computed for this detection task.

All test images remain included. Predictions of non-chili classes on chili images can count as false positives. This reports how the existing model performs on non-chili classes; it does not predict performance after retraining without chili.

| Crop group | Classes | Precision | Recall | mAP50 | mAP50–95 |
|---|---:|---:|---:|---:|---:|
| Banana | 9 | 62.92% | 57.31% | 58.29% | 38.67% |
| Cauliflower | 4 | 72.97% | 63.40% | 61.79% | 51.87% |
| Groundnut | 6 | 80.02% | 75.69% | 77.05% | 54.88% |
| Radish | 5 | 67.41% | 56.38% | 65.97% | 31.54% |

## Comparison with round 25 on the same held-out test split

Round-25 values are recomputed from its rounded per-class CSV, so they are approximate. Precision/recall operating points can differ between checkpoints.

| Non-chili metric | Round 25 | Round 35 | Change (percentage points) |
|---|---:|---:|---:|
| precision | 68.77% | 69.81% | +1.03 |
| recall | 64.79% | 62.73% | -2.07 |
| mAP50 | 66.64% | 65.16% | -1.47 |
| mAP50_95 | 44.80% | 43.44% | -1.37 |

## Non-chili per-class scores

| Class | Instances | Precision | Recall | mAP50 | mAP50–95 |
|---|---:|---:|---:|---:|---:|
| Banana_bract_mosaic_virus | 40 | 92.70% | 77.50% | 90.51% | 74.29% |
| Banana_cordana | 241 | 63.38% | 61.83% | 61.97% | 28.13% |
| Banana_healthy | 108 | 83.06% | 92.59% | 90.76% | 71.55% |
| Banana_insect_pest | 320 | 40.42% | 17.19% | 14.87% | 6.66% |
| Banana_moko | 51 | 65.06% | 76.47% | 81.23% | 55.62% |
| Banana_panama | 134 | 79.55% | 61.94% | 66.76% | 54.51% |
| Banana_pestalotiopsis | 148 | 35.82% | 31.76% | 23.93% | 9.87% |
| Banana_sigatoka | 472 | 40.18% | 39.00% | 33.14% | 11.23% |
| Banana_yellow_and_black_sigatoka | 434 | 66.10% | 57.51% | 61.45% | 36.17% |
| Cauliflower_black_rot | 9 | 35.44% | 100.00% | 99.50% | 91.43% |
| Cauliflower_bacterial_spot | 16 | 100.00% | 0.00% | 6.86% | 2.62% |
| Cauliflower_downy_mildew | 22 | 92.49% | 56.04% | 66.78% | 53.03% |
| Cauliflower_healthy | 20 | 63.94% | 97.55% | 74.01% | 60.40% |
| Groundnut_early_leaf_spot | 790 | 34.92% | 11.39% | 15.84% | 4.32% |
| Groundnut_early_rust | 147 | 78.87% | 89.80% | 85.48% | 51.37% |
| Groundnut_healthy | 41 | 97.81% | 100.00% | 99.50% | 86.64% |
| Groundnut_late_leaf_spot | 222 | 84.91% | 76.03% | 82.10% | 55.53% |
| Groundnut_nutrition_deficiency | 200 | 91.25% | 77.50% | 80.58% | 53.75% |
| Groundnut_rust | 173 | 92.37% | 99.42% | 98.79% | 77.65% |
| Radish_black_leaf_spot | 53 | 65.31% | 54.72% | 68.26% | 34.53% |
| Radish_downy_mildew | 159 | 69.29% | 77.36% | 81.18% | 29.63% |
| Radish_flea_beetle | 51 | 61.16% | 39.22% | 46.46% | 17.54% |
| Radish_healthy | 61 | 97.57% | 72.13% | 93.64% | 60.99% |
| Radish_mosaic_virus | 99 | 43.74% | 38.49% | 40.33% | 15.02% |

Evaluation plots and prediction samples: `F:\Agri Vision\saved_models\round_035_non_chili_analysis`.
