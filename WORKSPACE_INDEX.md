# Agri Vision workspace index

Updated 9 October 2026. Use this index to find current plans, historical evidence and authoritative copies.

## Current work

- [README and project status](README.md)
- [Architecture comparison plan](ARCHITECTURE_COMPARISON_PLAN.md)
- [Training and evaluation plan](TRAINING_PLAN.md)
- [Workspace organization record](WORKSPACE_ORGANIZATION_PLAN.md)
- [Draft common protocol and model configurations](configs/comparison/README.md)
- [Four separate planned experiments](experiments/README.md)
- [24-class manifest preparation status](manifests/non_chili_v1/README.md)

The planned detectors are YOLOv8n, EfficientDet-D0, SSDLite–MobileNetV3 and SSD–MobileNetV3 with a standard head. The new data and training pipeline are not implemented yet. Thesis writing starts after 17 October; no final submission deadline is specified.

## Working code and environment

Existing training code remains at the project root. The working environment is [.venv](.venv). Historical report/evaluation helpers now live in [scripts](scripts). Existing dataset, checkpoint and Colab bundle paths remain compatible with the current runner.

## Authoritative data and reports

| Artifact | Primary location |
| --- | --- |
| Original 30-class dataset | [original_dataset](original_dataset) |
| Existing client partitions | [dataset](dataset) |
| Data class distribution | [reports/dataset_class_distribution.csv](reports/dataset_class_distribution.csv) and [report](reports/dataset_class_distribution.md) |
| Historical chili provenance | [merge audit](reports/chili_dataset_merge.md); compressed source ZIP under `archive/imported_datasets/`; unused extracted folder deleted after byte verification |
| Rounds 26–35 metrics | [validation CSV](reports/rounds_026_035_global_metrics.csv), [client CSV](reports/rounds_026_035_client_metrics.csv), [report](reports/rounds_026_035_detailed_report.md) |
| Rounds 36–40 metrics | [validation CSV](reports/rounds_036_040_global_metrics.csv), [client CSV](reports/rounds_036_040_client_validation_metrics.csv), [overview](reports/rounds_036_040_overview.md) |
| Round-35 test/non-chili analysis | [report](reports/round_035_non_chili_report.md), [summary](reports/round_035_non_chili_summary.json), [per-class CSV](reports/round_035_test_per_class.csv) |
| Round-35 evaluation samples | [test_evaluation_round_035](saved_models/test_evaluation_round_035); the identical `-2` folder was removed |
| Original non-chili analysis outputs | [round_035_non_chili_analysis](saved_models/round_035_non_chili_analysis) |
| Seminar poster | [artifact notes](reports/seminar_poster/README.md); historical draft, stacked-layout revision pending |

Some ground-truth samples and exported tables repeat across separate experiment records. Those copies retain context and are not removed indiscriminately. The current class names require provenance confirmation before class-specific claims.

## Historical checkpoints and Colab bundles

| Item | Location and interpretation |
| --- | --- |
| Round-35 checkpoint | [saved_models/global_round_035.pt](saved_models/global_round_035.pt), used by the historical loader and packaging tool |
| Rounds 26–35 run | [20261002_121640_89aa17-20261003T102507Z-1-001.zip](saved_models/20261002_121640_89aa17-20261003T102507Z-1-001.zip); directory verified to contain all ten round checkpoints |
| Rounds 36–40 run | [20261004_200656_355a3a-20261004T205120Z-1-001.zip](saved_models/20261004_200656_355a3a-20261004T205120Z-1-001.zip); directory verified to contain all five round checkpoints |
| Earlier checkpoints and ZIPs | [saved_models](saved_models); retained as historical evidence |
| Current historical upload data | [original_dataset_colab.zip](original_dataset_colab.zip) |
| Current historical upload code | [agrivision_colab_code.zip](agrivision_colab_code.zip) |
| Historical Colab notebook | [AgriVision_Colab.ipynb](AgriVision_Colab.ipynb), configured for already completed rounds 36–40 |

ZIPs retain their original names and locations because packaging/report tools reference them. Archive directory checks identify checkpoint filenames, not their load validity. Large artifacts are local and ignored by Git.

## Archive and cleanup

- [Archive index](archive/README.md) lists moved historical material and original paths.
- [Before-cleanup duplicate audit](reports/workspace_audit_before_20261009.json)
- [Exact move/deletion manifest](reports/workspace_cleanup_20261009.json)
- [After-cleanup audit](reports/workspace_audit_after_20261009.json)
- [Verification results](reports/workspace_verification_20261009.json)

These audits describe the local `F:\Agri Vision` workspace. Google Drive has not been reorganized. No new detector training was run as part of organization.
