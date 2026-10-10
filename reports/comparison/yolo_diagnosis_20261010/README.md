# Why the new YOLO run differs from the successful round-40 model

The new experiment started fresh from COCO weights. It did not continue or damage the historical round-40 crop checkpoint. The historical model learned 30 classes over 40 global rounds; the new comparison learned 24 non-chili classes over 10 global rounds using a different trainer.

| Item | Historical round 40 | New 24-class run |
|---|---|---|
| Reported precision | 66.27% | 0% at confidence 0.25 |
| Reported recall | 57.43% | 0% at confidence 0.25 |
| Reported mAP50–95 | 38.81% | 0.80% at best round 8 |
| Initialization | Continued crop-trained weights through round 40 | Fresh 80-class COCO weights adapted to 24 classes |
| Training | Native Ultralytics `model.train()` | Custom shared SGD loop, initial LR 0.005 |
| BatchNorm running statistics | Updated during native local training | Frozen on every layer from the beginning |
| Metrics | Validation-image-weighted client summaries of Ultralytics per-class metrics | Pooled COCO AP; micro P/R at confidence 0.25 and IoU 0.5 |

These percentages are not a direct before/after measurement of one model. The dataset, training exposure, initialization, training procedure and metric definitions changed. The removal of chili happened alongside these changes; it is not a controlled explanation for the performance collapse.

## What the checks establish

All eleven saved YOLO states, including initialization, contain finite tensors. Metadata and class maps agree. Each global state hash agrees with the next round’s logged starting state, and all clients received the same global initialization. Each completed round visited all 14,514 training images. BN buffers are unchanged across every checkpoint, confirming the freeze policy. Initial and final file hashes match their saved records.

Replaying the best round’s full saved validation predictions exactly reproduces its AP and P/R. Native Ultralytics prediction on 24 fixed validation images agrees with the custom adapter: the counts and labels match, maximum score error is below 2e-7 and maximum box error is below 0.0002 pixels. This strongly rules out a simple adapter confidence, NMS or box-rescaling defect on that sample. It does not independently certify the entire native training procedure.

The round-8 export includes all 4,157 validation images and 21,803 detections above the 0.001 export floor. Its highest confidence is 0.0632302. Round 1 had no detections above even 0.001. Low confidence explains zero fixed-threshold precision/recall, while poor score-ranked detections explain the low AP.

| Confidence cutoff | Precision | Recall | True positives | False positives |
|---:|---:|---:|---:|---:|
| 0.25 | 0.00% | 0.00% | 0 | 0 |
| 0.1 | 0.00% | 0.00% | 0 | 0 |
| 0.05 | 9.55% | 11.01% | 794 | 7520 |
| 0.01 | 9.55% | 11.01% | 794 | 7520 |
| 0.005 | 6.50% | 11.25% | 811 | 11660 |
| 0.001 | 3.85% | 11.65% | 840 | 20963 |

The threshold table is a diagnostic on validation data. It does not replace the shared confidence-0.25 comparison or constitute a new test result. Lowering confidence does not restore AP or make this model comparable to the historical result.

## The main implementation weakness

YOLOv8n chooses its classification feature width using the number of classes. COCO has width 80; the new 24-class constructor has width 64. Consequently 36 of its 42 classification state tensors cannot transfer from COCO. The only six compatible tensors are scalar BatchNorm counters. Six newly created classification BN layers begin with zero means and unit variances, and our common freeze policy prevents them adapting throughout training. This is a larger initialization change than the metadata phrase “new output tensors” suggests.

The historical 30-class constructor also replaces incompatible classifier tensors, but its native trainer lets the new BN layers learn statistics. The combination of fresh classification features, permanently frozen statistics and a different optimizer/training schedule is a strong candidate cause. This read-only audit does not prove which change caused how much of the collapse.

Native training also supplies warmup, parameter grouping and its own augmentation/optimization behavior. The new loop has no warmup, applies SGD weight decay uniformly and uses only horizontal-flip augmentation. Historical archived settings and local args/CSV extracts are saved separately; an `optimizer=auto` argument is not proof of a particular actual optimizer.

The historical round-36 local CSVs report a final learning rate of 0.0000978485 for all three parameter groups, while the new loop starts SGD at 0.005. These values belong to different optimizer/schedule configurations and should not be treated as equivalent update magnitudes. The historical round-40 classifier BN layers have learned nonzero running means and nonunit variances, confirming that their statistics adapted.

A four-image feature probe found a sharp loss of image-specific variation in deeper layers. For the final neck output, mean inter-image standard deviation fell from 0.1561 at initialization to 0.00210 at round 8, while overall activation RMS rose from 0.3200 to 4.4654. This supports feature degeneration in the trained network. It is a small diagnostic sample, not a dataset-wide feature-quality metric or causal proof. Client model weights were not retained, so we cannot isolate local-training damage from additional effects of FedAvg.

“Improve image features” therefore means restoring useful learned representations in the model. It does not mean your existing crop photographs suddenly became poor quality.

## A concrete candidate checked without training

An isolated prototype retains the pretrained COCO classification feature width of 80 and all 36 feature/BN state tensors, replacing only the six final projection tensors for 24 outputs. Its loss forward is finite, output IDs are valid and inference/loss checks leave model tensors unchanged. No backward pass, optimizer step or training run was performed. Its parameter count is 3,143,592.

This prototype is a custom COCO-width YOLOv8n classification head, rather than the canonical nc=24 width-64 graph. Its initialization check is not evidence of improved accuracy. The alternative is canonical YOLOv8n with an explicit trainable-BN policy, which would require a declared change to the comparison protocol.

## Recommended sequence when GPU access returns

1. Resume EfficientDet with its existing code and settings to complete the original budget.
2. Before any new full YOLO run, perform a small training-only ablation comparing the original setup with a corrected initialization/normalization setup. Require finite losses, correct target IDs, unchanged intended frozen buffers, and recovery of class-correct boxes on a fixed training sanity sample; decreasing loss alone is insufficient.
3. Add a short federated check that measures each client model before aggregation and the global model after aggregation. This will distinguish local learning failure from averaging-related degeneration.
4. If the sanity checks pass, start a separately versioned YOLO experiment on the same frozen 24-class data, clients, seed and declared round budget. Preserve this failed run as diagnostic evidence. Record head width, normalization and optimizer changes explicitly, and confirm the comparison contract before interpreting the final table.
5. Evaluate selected valid models on the untouched held-out test set only after configuration selection. Never tune using test scores.

The current comparison modules, package ZIP, dataset manifests and Drive checkpoints were not edited. The implementation hash remains `c6d295dc79d706a18d73090edf91532cda74fee23a9f4d3e737e0470915fff71`, so the prepared EfficientDet resume remains compatible.

## Evidence

- [Full audit and replay](audit.json)
- [Feature probe](activations.json)
- [Untrained initialization candidate checks](initialization_candidate.json)
- [Historical archived training settings and CSV excerpts](historical_training_comparison.json)
- [Ultralytics 8.4.48 detection head implementation](https://github.com/ultralytics/ultralytics/blob/v8.4.48/ultralytics/nn/modules/head.py)
- [Ultralytics 8.4.48 shape-matched weight loading](https://github.com/ultralytics/ultralytics/blob/v8.4.48/ultralytics/nn/tasks.py)
- [Ultralytics training documentation](https://docs.ultralytics.com/modes/train)
