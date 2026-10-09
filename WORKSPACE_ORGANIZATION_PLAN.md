# Agri Vision workspace organization record

Updated 9 October 2026. The user authorized organization, README updates and removal of verified duplicate files. This first organization pass is complete; dataset filtering and model training are separate work.

## Completed

- Rewrote the README to identify completed 30-class YOLO experiments, the planned 24-class comparison and the current timing. Thesis writing starts after 17 October; no submission deadline is specified.
- Added a [workspace index](WORKSPACE_INDEX.md) with authoritative reports, checkpoint archives and links to new-work scaffolds.
- Created draft shared/per-model configurations, four separate experiment folders and a reserved manifest folder.
- Moved superseded documents, the older environment, old dataset backups, the earlier bundle, historical chili source folder/ZIP and obsolete generated listing/preview into `archive/`.
- Moved three report/evaluation/document scripts from `reports/` to `scripts/`. Updated their checkpoint/output references and documentation links.
- Added Git ignore rules for archived bulk material and new experiment runtime outputs.
- Removed two redundant dataset-distribution CSVs and the six-file duplicate `test_evaluation_round_035-2/` directory only after SHA-256 comparisons. Kept the canonical report and evaluation folder.
- Subsequently deleted the unused extracted chili source folder at the user's request. Verified all 1,557 source files against the retained compressed ZIP before individual file removal; recorded the verification and deletion list. Merged training images remain unchanged.
- Recorded before/after audits and the exact move/deletion manifest under `reports/`.

## Current structure

```text
Agri Vision/
  README.md                         current project status
  WORKSPACE_INDEX.md                authoritative-file map
  ARCHITECTURE_COMPARISON_PLAN.md    current experiment plan
  model.py, colab_train.py, ...      existing working YOLO code
  scripts/                          packaging, audit and analysis helpers
  tests/                            existing verification suite
  configs/comparison/               draft shared and model specifications
  manifests/non_chili_v1/            reserved; data preparation pending
  experiments/                      four planned one-stage detector runs
  original_dataset/, dataset/       current 30-class source/client data
  saved_models/, runs/              historical checkpoints and run outputs
  reports/                          reports, metrics, figures and poster
  archive/                          indexed historical material
  .venv/                            working local environment
```

## Preservation decisions

The current source/client dataset paths and unique checkpoints/ZIPs remain available. Images copied into client partitions are required inputs, not disposable duplicates. Identical ground-truth samples or tables in separate run records were retained where they preserve context. Old backups and bundles were archived rather than declared identical to the current data. The older environment was retained in the archive rather than deleted as a supposed duplicate.

The compressed source chili ZIP is under `archive/imported_datasets/`; its unused extracted folder was removed. Reproducing the historical merge requires explicitly restoring that source and passing `--source`. Existing training inputs remain unchanged. Timestamped result ZIPs retain their names because current reporting scripts reference them; their round ranges are documented in the workspace index.

## Storage and scope

Before organization, F had approximately **14.26 GB free**, confirming the user's updated estimate. Only verified duplicate deletions reclaimed space; moving files inside F did not. Continue to use one shared verified 24-class dataset rather than create one dataset copy per architecture.

This work concerns the local workspace, not Google Drive. The new experiment folders/configurations are explicitly marked planned, not completed results. The old Colab notebook still reproduces rounds 36–40 and must not be mistaken for the future comparison notebook.

## Next implementation work

Prepare and audit the 24-class non-chili dataset, populate its immutable manifests and class map, implement the common evaluator, and prepare the SSDLite pilot. SSD–MobileNetV3 with a standard head is a separate custom configuration; verify its initialization and report differences from pretrained detector alternatives. No training has been started by this organization pass.

## Audit trail

- [Workspace index](WORKSPACE_INDEX.md)
- [Archive index](archive/README.md)
- [Before-cleanup audit](reports/workspace_audit_before_20261009.json)
- [Exact cleanup manifest](reports/workspace_cleanup_20261009.json)
- [After-cleanup audit](reports/workspace_audit_after_20261009.json)
- [Verification results](reports/workspace_verification_20261009.json)
