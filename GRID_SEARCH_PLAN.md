# Four-detector hyperparameter grid search

Prepared 17 October 2026 following the supervisor's request. This is a separate tuning experiment. Completed baseline checkpoints, notebooks, metrics and the original code ZIP are retained. No Flower integration or held-out test evaluation is introduced.

## Grid and fixed settings

| Detector | Initial learning rate | Weight decay | Batch size | Actual input resolution |
| --- | --- | --- | --- | --- |
| YOLOv8n | 0.000357 / 0.0001785 | 0.0005 / 0.001 | 4 / 8 | 512 / 640 |
| SSDLite | 0.001 / 0.0005 | 0.0005 / 0.001 | 4 / 8 | 320 / 512 |
| SSD–MobileNetV3 | 0.001 / 0.0005 | 0.0005 / 0.001 | 4 / 8 | 320 / 512 |
| EfficientDet-D0 | 0.0001 / 0.00005 | 0.0001 / 0.0005 | 2 / 4 | 512 / 640 |

All four binary dimensions are crossed exhaustively: 16 configurations per architecture, 64 total. Fixed: seed 42, the same 24-class frozen splits and three clients, one local epoch, image-count-weighted FedAvg, model-specific baseline augmentation/normalization, optimizer reset per client/round, and the shared pooled evaluator. Learning-rate ranges are architecture-specific. These candidates are a bounded search space, not guaranteed improvements.

The two SSD variants retain SGD; EfficientDet retains AdamW. YOLO explicitly uses AdamW, with the baseline's resolved momentum 0.9 and bias warmup LR 0, to prevent `optimizer=auto` from overriding searched values. Its native three-epoch local warmup restarts per fit, as in the baseline. Native nominal batch size `nbs=64` is retained; the grid changes actual microbatch size, and gradient accumulation/effective batch policy is logged rather than silently disabled. Native adaptive BatchNorm and historical floating-weight NumPy aggregation are retained. Other adapters retain frozen running BatchNorm statistics and their existing FedAvg implementation.

The original scheduling policies are preserved across the intended 30-round experiment: SSD/EfficientDet use the original 10-round cosine followed by the terminal rate for rounds 11–30; native YOLO repeats its existing local schedule at the searched base rate each fit. Short screening runs take prefixes of those policies. They do not squeeze a cosine schedule into five rounds or restart the candidate's global schedule at confirmation.

Each architecture receives one cached, freshly COCO-transferred initialization shared across every grid candidate and the selected final run. Preflight weights are discarded. Existing trained crop checkpoints are not used to seed the search. Changing resolution modifies the actual SSD transform or EfficientDet model/anchor configuration, and inference uses the corresponding resolution with boxes mapped back to original images.

## Automatic stages

1. GPU preflight executes one training batch and prediction per batch/resolution pair on a small training-only sample. It checks actual resolution and finite operations and records GPU memory. Failures stop execution rather than silently reducing batch/resolution.
2. Train all 16 configurations for five full federated rounds, with complete pooled validation after each round. Select the two highest best validation mAP50–95 values within rounds 1–5; trial ID breaks ties deterministically.
3. Continue those two candidates through round 10 from their own round-5 commits. Compare their best validation mAP50–95 across rounds 1–10.
4. Train the selected configuration in a new `final_training/` folder from the common untrained initialization through round 30. Preserve its best validation checkpoint and latest checkpoint; report P/R/F1/mAP50 alongside mAP50–95.

This is an exhaustive four-parameter grid at the screening budget with finalist confirmation, not exhaustive 30-round training of all 64 candidates. The full-budget optimum may differ from the short-budget ranking. Candidate ranges, budgets and selection are recorded before evaluation; no test data informs selection.

## Work assignment and running

| Drive label | Notebook | Models |
| --- | --- | --- |
| 19gmail | `AgriVision_GridSearch_YOLO.ipynb` | YOLOv8n |
| 4gmail | `AgriVision_GridSearch_EfficientDet.ipynb` | EfficientDet-D0 |
| 40gmail | `AgriVision_GridSearch_SSD_Pair.ipynb` | SSDLite then SSD–MobileNetV3, sequentially |

Open the matching notebook, select T4 GPU and Run all. It assembles the existing fourteen dataset parts on Colab's disk, verifies hashes, extracts the new isolated code ZIP, checks dependencies, runs the local engineering checks, then GPU preflight/search/confirmation/final training automatically. The earlier code ZIP and dataset uploads remain in place. No manual model switching or hyperparameter editing is needed.

There are 80 screening rounds + 10 additional confirmation rounds + 30 final rounds per model: 120 global rounds per model, 480 total, plus small diagnostic probes. Historical default-setting timers imply roughly 87 total GPU hours for that work at unchanged speed. Different batch sizes/resolutions can materially increase time, and free GPU availability is not guaranteed; this is not a one-week completion promise.

## Results, storage and recovery

Each Drive uses `Agri Vision/grid_search/grid_v1_20261017/`, with separate model and trial folders. Root `search_status.json` identifies the active stage/model/trial. Model `screening_results.csv`, `finalist_results.csv`, `selected_configuration.json` and `final_best_metrics.json` provide the decisions. Full trial/final metrics are in `results/metrics/global_validation_metrics.csv`; per-class results are under each round JSON's `pooled_validation.per_class`. Scores in raw CSVs are 0–1.

Screening retains one latest checkpoint per trial plus the shared initial checkpoint. Final training keeps best/latest checkpoints. Completed metric histories are retained. No all-round raw-prediction export is written. Only obsolete checkpoints inside new trial folders are pruned; original model runs are untouched.

After interruption, reopen the same notebook and Run all. Committed rounds/trials are skipped; the incomplete round repeats from its last global commit. Finalist ranking is restricted to the declared budget, so confirming a candidate does not change the original five-round shortlist on restart. Configuration, initial/checkpoint hashes, code, data, environment and loader workers are checked. Environment mismatches stop execution rather than silently changing the study.

CPU engineering checks and synthetic state/recovery tests are documented in `reports/comparison/grid_search_20261017/`. Actual T4 memory checks and GPU training happen only when the user runs the notebooks. At preparation time, no grid training or final held-out evaluation has been run.
