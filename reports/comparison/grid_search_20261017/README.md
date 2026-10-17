# Prepared grid-search workflows

See the root [grid-search plan](../../../GRID_SEARCH_PLAN.md) for the full grid, fixed training policies, screening/confirmation/final budgets, account assignment, storage and recovery.

- `package.json`: code ZIP and notebook hashes/sizes.
- `cpu_adapter_checks.json`: finite loss/backward/prediction on one synthetic image at both SSD and EfficientDet resolutions; no real-data training.
- `cpu_yolo_checks.json`: native CPU prediction at 512 and 640 with the requested predictor size.
- `validation.json`: local unit tests, notebook/package/extraction checks and Drive verification.
- `drive_upload.json`: uploaded notebook/code file IDs and account folders, without authenticated download URLs.

The new modules live under `tuning/`; the original `comparison/` implementation is byte-for-byte unchanged. Use `scripts/package_grid_search.py` to regenerate the isolated ZIP and three notebooks. Changes to a prepared implementation require a reviewed new search version before resuming an existing run.
