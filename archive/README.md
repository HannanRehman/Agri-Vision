# Historical archive

Organized 9 October 2026. Items were moved within `F:\Agri Vision` and retained for reference/recovery. Moving them did not reclaim their disk space. Active source data, client partitions, unique checkpoints and current Colab bundles keep their working paths.

| Archived item | Previous location | Reason |
| --- | --- | --- |
| [FIFTEEN_DAY_THESIS_PLAN.md](documents/FIFTEEN_DAY_THESIS_PLAN.md) | Project root | Superseded timing and architecture recommendation |
| [Architecture review DOCX](documents/Agri_Vision_Non_YOLO_Architecture_Review.docx) and [PDF](documents/Agri_Vision_Non_YOLO_Architecture_Review.pdf) | `reports/` | Historical supervisor discussion; its Faster R-CNN recommendation is excluded from the current plan |
| `legacy_environment/venv/` | `venv/` | Older environment points to `E:\python.exe`; active environment is `.venv/` |
| `legacy_datasets/dataset_backup_20260922_173524/` | Project root | Earlier client partition retained for recovery |
| `legacy_datasets/dataset_backup_20260929_chili_archive/` | Project root | Contains an explicitly named incomplete archive; retained as historical evidence, not validated recovery data |
| `legacy_bundles/AgriVision_dataset.zip` | Project root | Older bundle; current historical Colab workflow uses `original_dataset_colab.zip` |
| [Chili source ZIP](<imported_datasets/PENYAKIT PADA CABAI.v1i.yolov8.zip>) | Project root | Compressed historical source retained; all 1,557 extracted files matched it before the unused folder was deleted at the user's request |
| `legacy_documents/structure.txt` | Project root | Large outdated generated tree listing |
| `legacy_documents/architecture_review_pages.png` | `reports/.document_preview/` | Preview of superseded review |

Legacy environments, datasets, imports, bundles and generated preview material are ignored by Git. The superseded documents remain tracked. Their contents reflect the original recommendations; consult the [current comparison plan](../ARCHITECTURE_COMPARISON_PLAN.md) for new work.

See the [cleanup manifest](../reports/workspace_cleanup_20261009.json) for exact moves and hash-verified duplicate deletions. The archive is not a second-drive backup. Verify contents and dependencies before removing anything unique from it.

The removed chili source folder can be restored from its ZIP if historical merge reproduction is needed. Supply that restored location with `merge_chili_dataset.py --source`; no automatic extraction is performed. The original 30-class training dataset still includes previously merged chili images until the separate non-chili preparation step is implemented.
