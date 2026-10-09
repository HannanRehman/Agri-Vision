# Agri Vision: Federated Crop-Disease Detection

Agri Vision is an FYP prototype for locating crop-disease and pest symptoms using bounding boxes and class predictions. The completed experiment uses a **30-class YOLOv8n model** and three-client sample-weighted FedAvg. The Colab implementation simulates the clients sequentially on one GPU.

## Current status — 9 October 2026

| Work | Status |
| --- | --- |
| Historical YOLO training through round 40 | Completed; archived run files and reports retained |
| Round-35 held-out test and non-chili class analysis | Completed on the original 30-class checkpoint |
| Workspace organization | Separate experiment scaffolds, current file index and cleanup audit added |
| Four-crop, 24-class dataset | Planned; original data has not yet been filtered or relabelled |
| YOLO / EfficientDet / SSDLite / SSD–MobileNetV3 comparison | Planned; draft configurations exist, no new model training has started |
| Seminar poster | Earlier A1 draft retained; revision to the supplied stacked layout is pending |

The new comparison excludes all six chili classes and retains banana, cauliflower, groundnut and radish. The agreed alternatives are **EfficientDet-D0**, **SSDLite with MobileNetV3-Large**, and **SSD with MobileNetV3-Large using a standard detection head**. The latter is a separate custom configuration to implement and verify. Faster R-CNN is excluded. All models will be evaluated as one-stage object detectors; standalone MobileNetV3 classification is not this experiment.

Thesis writing begins **after 17 October**. That date is not a submission deadline or a forced end to training.

Start with the [workspace index](WORKSPACE_INDEX.md), [training plan](TRAINING_PLAN.md), [architecture comparison plan](ARCHITECTURE_COMPARISON_PLAN.md) and [organization record](WORKSPACE_ORGANIZATION_PLAN.md). Historical recommendations in `archive/documents/` are superseded by the current comparison plan.

## Workspace layout

| Location | Purpose |
| --- | --- |
| `model.py`, `federated_client.py`, `client1.py`–`client3.py`, `server.py` | Existing YOLO/Flower implementation |
| `colab_train.py` | Existing sequential three-client Colab FedAvg runner |
| `evaluate.py`, `predict.py` | Existing YOLO checkpoint evaluation and inference |
| `split_dataset.py`, `merge_chili_dataset.py` | Historical data partitioning and audited chili import |
| `original_dataset/`, `dataset/client1/`–`client3/` | Current 30-class source data and client partitions; paths retained for compatibility |
| `configs/comparison/` | Shared experiment protocol and per-model draft specifications |
| `manifests/non_chili_v1/` | Reserved location for audited 24-class image lists, class map and checksums |
| `experiments/` | Separate new-model run folders, each with its own `results/`, `checkpoints/` and `logs/`; READMEs identify their planned status |
| `saved_models/`, `runs/` | Historical checkpoints, result archives and local outputs |
| `reports/` | Reviewed metrics, figures, reports and poster artifacts |
| `scripts/` | Packaging, workspace audit, historical analysis and document tools |
| `archive/` | Superseded documents, earlier bundle, dataset backups and old environment |
| `.venv/` | Working local Python environment |

Model weights, raw datasets and runtime artifacts are excluded from Git where configured. The workspace index records the authoritative reports and the round ranges inside timestamped ZIPs. Do not delete source images shared by several data partitions merely because their bytes match.

## Historical results

- [Rounds 26–35 detailed report](reports/rounds_026_035_detailed_report.md) and [validation CSV](reports/rounds_026_035_global_metrics.csv).
- [Rounds 36–40 overview](reports/rounds_036_040_overview.md) and [validation CSV](reports/rounds_036_040_global_metrics.csv).
- [Round-35 non-chili analysis](reports/round_035_non_chili_report.md).
- [Dataset class distribution](reports/dataset_class_distribution.md) and [primary CSV](reports/dataset_class_distribution.csv).
- [Historical chili merge audit](reports/chili_dataset_merge.md).

The round-35 non-chili analysis averages 24 classes of a **30-class checkpoint while retaining all original test images**. It does not establish the performance of a newly trained 24-class model. Historical global validation metrics are validation-image-weighted client averages; they are not pooled-prediction COCO AP. The new comparison will use a shared evaluator and explicitly labelled metrics.

## Existing Colab workflow — historical rounds 36–40

[AgriVision_Colab.ipynb](AgriVision_Colab.ipynb) remains configured to reproduce the completed round-35-to-40 continuation. It is **not configured for the new 24-class architecture comparison**. Running it starts historical YOLO training again.

For intentional reproduction, see the [exact upload instructions](reports/rounds_036_040_upload_instructions.md). Required local artifacts are `original_dataset_colab.zip`, `agrivision_colab_code.zip` and `saved_models/global_round_035.pt`. Upload the checkpoint directly as `MyDrive/Agri Vision/global_round_035.pt`. The notebook verifies SHA-256 checksums, extracts data locally and builds non-overlapping client partitions with seed 42.

The recorded continuation uses five rounds, one local epoch per client, batch 8, image size 640 and two loader workers. Results go into unique folders under `MyDrive/Agri Vision/colab_results/`. Completed checkpoints, per-client metrics, the combined validation CSV, settings and progress status are saved. An interrupted partial round must restart.

If the bundled training code changes, regenerate the bundle and notebook checksums with:

```powershell
.\.venv\Scripts\python.exe scripts\prepare_colab_rounds_036_040.py
```

That command prepares artifacts; it does not train. New comparison notebooks and bundles should have distinct names and use the experiment IDs in `experiments/`.

## Local environment and checks

Run from `F:\Agri Vision` using `.venv`. Its currently recorded interpreter is **Python 3.14.4**. This does not establish compatibility for future detector packages or reproduce the Colab environment; record and pin the tested comparison environment when preparing it. The older `venv/`, referencing `E:\python.exe`, is archived under `archive/legacy_environment/` and is not the active environment.

```powershell
.\.venv\Scripts\Activate.ps1
python -c "import torch, flwr, ultralytics; print(torch.__version__, torch.cuda.is_available())"
python split_dataset.py
```

The partition checker validates the existing 30-class client data, not the future 24-class dataset. Expected totals are 15,512 training, 4,374 validation and 2,191 test images. It checks client overlap, coverage, class mappings and label values. Rebuild only when needed:

```powershell
python split_dataset.py --rebuild --seed 42
```

Rebuilding creates a new copy and keeps the previous partition as a timestamped backup. Confirm storage before running it. Existing archived backups remain available for recovery. Class names were reconstructed from label/file evidence; confirm disease-name semantics against source provenance, since numeric validation cannot establish them.

The unused extracted external chili source folder was deleted at the user's request after all 1,557 files were verified against its ZIP. The compressed ZIP is retained under `archive/imported_datasets/`. Current training reads the merged data in `original_dataset/` and `dataset/`; those images and labels were preserved. To reproduce the historical merge, restore the source ZIP into a separate folder and pass that folder with `--source`. The import script does not automatically recreate the deleted folder.

`requirements.txt` specifies the existing YOLO/Flower dependencies. New detector dependencies are not installed by the draft configurations. A fresh environment can be created using a Python version supported by the selected, tested PyTorch packages; use the [official PyTorch installer](https://pytorch.org/get-started/locally/) for CUDA installation.

## Existing local YOLO training

These commands run the current **30-class** pipeline and start training. They are retained for intentional historical use:

```powershell
# Separate terminals, each using .venv:
python server.py --rounds 5 --local-epochs 1 --batch 8 --imgsz 640
python client1.py
python client2.py
python client3.py
```

Each round sends identical global weights to all clients, trains each client locally and aggregates by training-image count. On one GPU, prefer the sequential Colab runner; concurrent clients can compete for VRAM. Local outputs go to `runs/federated/clientN/round_XXX/` and server checkpoints default to `saved_models/`. Use unique output locations to avoid confusing new runs with earlier results.

The project checkpoint contains `model_state_dict`, `class_names` and `round`; it is not a native Ultralytics checkpoint. Evaluate or predict with the project loader:

```powershell
python evaluate.py --checkpoint saved_models/global_round_035.pt --device cpu
python predict.py --checkpoint saved_models/global_round_035.pt --source path\to\image.jpg --device cpu
```

Choose checkpoints/settings on validation, not repeated test-set trials. The historical Flower loss value is `1 - mAP50`, an error proxy, rather than the model's training loss.

## Analysis and maintenance tools

| Tool | Behavior |
| --- | --- |
| `scripts/audit_workspace.py --output reports/workspace_audit_latest.json` | Read-only inventory and duplicate detection; excludes datasets and runtime environments |
| `scripts/analyze_rounds_026_035.py` | Reads the archived run and regenerates its report/figures; does not train |
| `scripts/evaluate_round_035_non_chili.py` | Runs a real detector evaluation; invoke only when deliberately reevaluating |
| `scripts/create_architecture_review.py` | Regenerates the superseded architecture review under `archive/documents/`; requires python-docx |
| `scripts/organize_workspace.ps1` | One-time organization script already executed; not an idempotent routine to rerun |
| `scripts/initialize_comparison_workspace.py` | Creates missing scaffold folders/files and preserves existing configurations; does not train |
| `scripts/remove_verified_chili_source.ps1` | One-time, user-authorized removal already executed; verifies ZIP copies before deleting source files |

The cleanup manifest records moved paths and SHA-256-verified duplicate deletions. Original datasets, unique checkpoints and run archives are retained. Matching ground-truth images or log content in distinct experiments may be retained to keep each experiment's record complete.

## Verification and limitations

```powershell
python -m unittest discover -s tests -v
```

Local tests and syntax checks do not establish GPU runtime, training convergence or model accuracy. FedAvg parameter exchange is not a formal privacy guarantee. The completed Colab experiment is a sequential client simulation, not a deployed network of independent devices.
