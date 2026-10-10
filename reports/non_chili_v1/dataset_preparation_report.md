# Agri Vision: 24-class dataset preparation report

Prepared 10 October 2026. **Dataset built and mechanically verified; no new training has started.**

The separate `non_chili_v1` dataset retains banana, cauliflower, groundnut and radish. All six chili classes, including healthy chili, are excluded. Original 30-class images, labels, client partitions and model checkpoints are preserved.

## Final dataset

| Split | Original images | Retained images | Retained boxes |
| --- | ---: | ---: | ---: |
| train | 15,512 | 14,514 | 27,558 |
| valid | 4,374 | 4,157 | 7,210 |
| test | 2,191 | 2,091 | 4,011 |

**Total: 20,762 images and 38,779 boxes; all images are 640 × 640.** All 24 classes occur in every pooled split. Per-class image counts can overlap when an image contains more than one class.

| Client | Train | Validation | Test |
| --- | ---: | ---: | ---: |
| client1 | 4,837 | 1,379 | 698 |
| client2 | 4,837 | 1,395 | 695 |
| client3 | 4,840 | 1,383 | 698 |

## Per-class image counts

| New ID | Class | Train | Validation | Test |
| ---: | --- | ---: | ---: | ---: |
| 0 | Banana_bract_mosaic_virus | 279 | 80 | 40 |
| 1 | Banana_cordana | 379 | 111 | 56 |
| 2 | Banana_healthy | 755 | 216 | 108 |
| 3 | Banana_insect_pest | 478 | 139 | 71 |
| 4 | Banana_moko | 364 | 98 | 48 |
| 5 | Banana_panama | 785 | 224 | 111 |
| 6 | Banana_pestalotiopsis | 387 | 114 | 58 |
| 7 | Banana_sigatoka | 1097 | 173 | 88 |
| 8 | Banana_yellow_and_black_sigatoka | 1953 | 558 | 280 |
| 9 | Cauliflower_black_rot | 59 | 19 | 9 |
| 10 | Cauliflower_bacterial_spot | 118 | 34 | 16 |
| 11 | Cauliflower_downy_mildew | 118 | 34 | 17 |
| 12 | Cauliflower_healthy | 144 | 41 | 20 |
| 13 | Groundnut_early_leaf_spot | 701 | 340 | 170 |
| 14 | Groundnut_early_rust | 1022 | 293 | 147 |
| 15 | Groundnut_healthy | 278 | 77 | 41 |
| 16 | Groundnut_late_leaf_spot | 1382 | 396 | 200 |
| 17 | Groundnut_nutrition_deficiency | 1162 | 332 | 166 |
| 18 | Groundnut_rust | 1197 | 344 | 173 |
| 19 | Radish_black_leaf_spot | 366 | 105 | 53 |
| 20 | Radish_downy_mildew | 411 | 118 | 60 |
| 21 | Radish_flea_beetle | 327 | 93 | 47 |
| 22 | Radish_healthy | 425 | 121 | 61 |
| 23 | Radish_mosaic_virus | 382 | 107 | 55 |

## Filtering and leakage decisions

- Excluded 1,216 chili images: 913 train, 203 validation and 100 test. No mixed chili/non-chili images were identified by the supplied labels.
- Excluded one empty-label training image, `Rust-1015-_jpg.rf.b9611fdc3a3b8b23027b5ade1ace2bbd.jpg`, because crop membership could not be established from an empty annotation.
- No missing/corrupt images, invalid boxes, orphan source labels, conflicting annotations on identical decoded images, exact pixel duplicates or matching inferred filename source groups were detected in the non-chili candidates.
- Perceptual screening found 104 cross-split candidate pairs. Visual review of all 18 sheets judged 102 pairs to show the same source photo or source variant; two pairs were different photos. Decisions and sheet/row references are in `visual_review.json`.
- Grouped confirmed source variants and excluded 98 lower-priority copies: 84 training and 14 validation images. Kept the highest original split priority: test, then validation, then train. No training image was moved into a held-out split.
- Reassigned one validation image from client3 to client1 to keep the retained variants of its source group on one client. Other retained client assignments are preserved.
- Old class IDs 0–12 stay unchanged; 13–18 are removed; 19–29 become 13–23. Box coordinate tokens are unchanged.

## Verification and limitations

The independent verifier passed for every retained image and label: copied image hashes, original source hashes, output label hashes, class remapping, unchanged coordinate tokens, complete split/client lists, manifest checksums and source-group separation. All 21 project tests passed, including eight preparation tests with an end-to-end build and deliberate tampering check. Each retained decoded image is unique. Confirmed source groups occupy one split and one client.

The near-duplicate search was a bounded, same-crop 64-bit dHash search at Hamming distance ≤2. It can miss rotated, mirrored, cropped, stronger augmented or differently named source variants. This report documents the detected leakage correction; it does not certify exhaustive source independence. Source filenames are heuristic provenance. The recorded class names and disease diagnoses still require source/expert confirmation.

Annotation sheets were inspected for every class in train, validation and test. The images include stretched border artifacts in some banana examples and differing annotation granularity, from whole organs to individual lesions. These labels were preserved for a common comparison; this was not a full expert relabelling exercise.

Class imbalance remains: banana 8,981 images, groundnut 8,421, radish 2,731 and cauliflower 629. Cauliflower black rot has only 59 training, 19 validation and 9 test images. Report per-class AP and recall alongside pooled metrics; small held-out classes will have uncertain estimates. Removing chili does not establish improved accuracy or recall without new training.

## Files and next step

- Shared image/label dataset: `original_dataset_non_chili/`.
- Portable client configurations: `dataset_non_chili/client1/` through `client3/`. Their `clientN_split.txt` lists are beside the shared split folders in `original_dataset_non_chili/`; image-list entries contain no parent-directory traversal. Images are stored once, not copied three times.
- Frozen lists, mapping, audit, exclusions, decisions and checksums: `manifests/non_chili_v1/`. The local detailed inventory contains original provenance and is included in the upload bundle.
- Diagnostic sheets: `reports/non_chili_v1/*_class_samples.jpg` and `near_duplicate_review_*.jpg`.

Verify locally with:

```powershell
.\.venv\Scripts\python.exe scripts\verify_non_chili_dataset.py
```

The next implementation step is the shared dataset adapter and evaluator, followed by the SSDLite pilot. The current root training code and `AgriVision_Colab.ipynb` remain configured for historical 30-class YOLO training. Do not use them for this dataset. TorchVision foreground targets must become 1–24, leaving background 0; canonical dataset IDs remain 0–23.

## Colab dataset upload

Upload `non_chili_v1_colab_dataset.zip` (925.8 MB). ZIP CRC verification passed. Extract it into a single directory, preserving its layout. The archive includes shared data, client lists, manifests and an internal SHA-256 list; it does not include a new training runner.

ZIP SHA-256: `7a2920e4a4e0405ee157e9b40ac6306b54d6933b20a306b44a4b7c6871f20868`.
