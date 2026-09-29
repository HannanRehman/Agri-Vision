# Chili Dataset Merge Audit

Merge date: 2026-09-29

Source: [PENYAKIT PADA CABAI v1](https://universe.roboflow.com/tugas-akhr-pemtek-analisis-penyakit-pada-cabai-merah/penyakit-pada-cabai/dataset/1)  
License: CC BY 4.0  
Export format: YOLOv8

## Decisions

- The export contains 777 processed images derived from 318 independent source images.
- Only one processed variant per source identity was considered. The variant with the
  smallest black rotation border was selected, preventing three rotated copies from
  artificially inflating the dataset.
- Three source identities with empty annotations were rejected.
- Eighty-nine source identities already represented in AgriVision were rejected.
- Twenty-four source identities with labels contradicting identical AgriVision images
  were rejected. For example, some externally labelled leaf-curl images are identical
  to current images showing and labelled as whitefly. Contradictory labels were not mixed.
- The remaining 202 images were added only to `original_dataset/train` and distributed
  between the three clients. Existing validation and held-out test data were untouched.

## Accepted images

| AgriVision ID | Class | Added images | Added boxes |
|---:|---|---:|---:|
| 13 | Chili anthracnose | 139 | 311 |
| 14 | Chili healthy | 22 | 40 |
| 15 | Chili leaf curl | 5 | 20 |
| 16 | Chili leaf spot | 9 | 21 |
| 17 | Chili whitefly | 0 | 0 |
| 18 | Chili yellowish | 27 | 42 |
| | **Total** | **202** | **434** |

Client training additions were balanced according to the pre-existing shard sizes:

- Client 1: 67 images
- Client 2: 68 images
- Client 3: 67 images

The final physical dataset contains 15,512 training, 4,374 validation, and 2,191 test
images (22,077 total). The class distribution is recorded in
`reports/dataset_class_distribution.md` and `.csv`.

## Class mapping

| Source ID | Source class | AgriVision ID | AgriVision class |
|---:|---|---:|---|
| 0 | Cabai-Sehat | 14 | Chili_healthy |
| 1 | Cabai-Terkena-Daun-Berkerut | 15 | Chili_leaf_curl |
| 2 | Cabai-Terkena-Penyakit-Anthacnose | 13 | Chili_anthracnose |
| 3 | Cabai-Terkena-Penyakit-Kutu-Putih | 17 | Chili_whitefly |
| 4 | Cabai-Terkena-Penyakit-Leaf-Spot | 16 | Chili_leaf_spot |
| 5 | Cabai-Terkena-Penyakit-Yellowish | 18 | Chili_yellowish |

`Kutu-Putih` samples were visually checked and contain whiteflies. Nevertheless, none
were added because all independent source images in that source class were already present
or excluded by the consistency audit.
