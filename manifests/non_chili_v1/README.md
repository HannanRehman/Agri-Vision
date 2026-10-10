# non_chili_v1 dataset manifests

**Status: built 10 October 2026; mechanical verification is recorded in `verification.json`. No full comparison training has started; limited CPU engineering pilots are recorded separately.**

The four crops are banana, cauliflower, groundnut and radish, with 24 foreground classes. Original class IDs 13–18 (all chili classes, including healthy chili) are excluded. IDs 0–12 stay unchanged; IDs 19–29 become 13–23.

| Split | Images | Bounding boxes |
| --- | ---: | ---: |
| Train | 14,514 | 27,558 |
| Validation | 4,157 | 7,210 |
| Test | 2,091 | 4,011 |

All 24 classes occur in each pooled split. The original 30-class source and client partitions are preserved. Exclusions: 1,216 chili images, one empty-label image and 98 lower-priority copies of visually confirmed cross-split source photos/variants.

- [Audit and policies](audit.json), [exclusions](excluded.csv), [visual decisions](visual_review.json).
- [Class mapping](class_map.json), [per-class image/box counts](class_distribution.csv).
- [All retained images](retained.csv), [train](train.csv), [validation](valid.csv), [test](test.csv); separate `clientN_split.csv` files freeze each client partition.
- [Checksums](checksums.json), [verification](verification.json).
- Local `inventory.jsonl` stores detailed original provenance and label tokens. It is excluded from Git and included in the Colab dataset bundle; regenerate it from unchanged source data using the preparation script if needed.

Images are stored once under `original_dataset_non_chili/`. Portable image lists are beside the split folders there, named `clientN_train.txt`, `clientN_valid.txt` and `clientN_test.txt`. Each `dataset_non_chili/clientN/data.yaml` references those lists. This avoids a relative-path replacement issue in Ultralytics when image-list entries contain `../`.

Read the [preparation report](../../reports/non_chili_v1/dataset_preparation_report.md) for the leakage review, class imbalance and annotation limitations. Perceptual screening is bounded and does not guarantee exhaustive source independence or expert disease-label validation.

Verify from the project root:

```powershell
.\.venv\Scripts\python.exe scripts\verify_non_chili_dataset.py
```

The preparation script refuses to overwrite an existing frozen dataset version. The historical notebook and root training code still use 30 classes; a new comparison runner is required.
