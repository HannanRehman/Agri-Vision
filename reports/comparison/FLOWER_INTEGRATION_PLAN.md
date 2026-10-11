# Flower integration plan for the current Agri Vision experiments

Saved 11 October 2026. Status: proposed integration; not implemented or GPU-tested. No training is authorized or started by this plan.

## Current implementation

The new Colab experiments implement sequential three-client, sample-weighted FedAvg directly in Python. They do not use Flower for client/server orchestration. The earlier `server.py` and `federated_client.py` use Flower, but depend on older dataset paths, model construction and checkpoint formats. They cannot be substituted into the new experiments without adaptation.

Existing checkpoints and detection metrics remain usable. Adding Flower now cannot retroactively make the previous runs Flower-based. The original FYP proposal named Flower, so the final documentation must distinguish the completed comparison experiments from additional framework integration evidence.

## Implementation steps

1. Create separate Flower integration files, notebook and result folders. Preserve baseline runs, their settings, checkpoints, metric definitions and uploaded bundles.
2. Wrap YOLOv8n, EfficientDet-D0, SSDLite and standard-head SSD-MobileNetV3 as Flower clients. Reuse the frozen 24-class dataset and the same three client partitions. Each client receives global weights, trains one local epoch with the established detector-specific settings, and returns updated weights plus its unique training-image count.
3. Implement a Flower server/strategy initialized from a recorded saved checkpoint. Preserve the existing model-specific aggregation rules, floating-tensor order, integer-buffer handling and normalization policy. Keep client ordering explicit where it affects floating-point accumulation. Flower must actually dispatch work and collect client responses; importing Flower around an unchanged manual loop is insufficient.
4. Run three logical clients sequentially on one T4 using Flower simulation resource scheduling. Allocate one GPU to each active client so clients do not train simultaneously on the same GPU. Record the Flower version and dependency environment; protect the existing CUDA-enabled PyTorch/torchvision pair.
5. Use the existing pooled validation evaluator after aggregation: the same 4,157 validation images, confidence/IoU settings and COCO AP definitions. Retain separately labelled native YOLO metrics where applicable. Do not use the held-out test to tune the integration.
6. Save Flower orchestration logs, client participation, initialization/transfer hashes, image counts, validation metrics and atomic best/latest checkpoints. Record source checkpoint SHA, source experiment/global round, Flower round number, settings, code hashes and environment. Keep the additional experiment's round numbering separate from the baseline history.

## Verification sequence

- First perform local synthetic tests of serialization, client mapping, exact received model states, aggregation and checkpoint recovery. No detector training is needed for these checks.
- Test one full Flower-managed round with SSDLite once GPU access and explicit training authorization are available. Confirm all three clients participate, all use the same intended initial global state, the returned counts match their data, and evaluation/checkpoint saves succeed.
- If successful, run one full integration round for each of the other three detectors. Use documented saved starting checkpoints, and preserve the baseline selected checkpoints even if new validation performance declines.

Estimated model work for one full round of all four detectors is about 45 minutes, based on prior T4 timings: YOLO about 9.7 minutes, EfficientDet about 18.5 minutes, and the two SSD detectors about 8 minutes each. Flower installation, orchestration, serialization, setup and saving add overhead. This is a planning estimate; Flower runtime has not been measured. Partial synthetic tests do not count as a full-data training experiment.

## Thesis and presentation reporting

Report the architecture comparison as the existing sequential FedAvg experiments. Report the new Flower-managed rounds as separate integration experiments initialized from recorded checkpoints. Do not claim the completed baseline training used Flower, or claim formal privacy, secure aggregation, TLS/authentication or deployed farm devices unless those features are independently implemented and verified.

If the supervisor requires the architecture comparison itself to be conducted entirely through Flower, one-round integration evidence is not sufficient. A Flower-based comparison must be designed and run with an agreed matched budget and documented settings. That requirement should determine the scope before further GPU training.

## References and relevant workspace files

- [Flower advanced PyTorch example: custom strategy and checkpoint saving](https://flower.ai/docs/examples/advanced-pytorch.html)
- [Flower simulation resource scheduling](https://flower.ai/docs/framework/how-to-run-simulations.html)
- Earlier implementation: `server.py`, `federated_client.py`, `model.py`.
- Current native YOLO workflow: `scripts/yolo_native_train.py`.
- Current alternatives: `comparison/runner.py`, `comparison/models.py`, `scripts/continue_comparison.py`.
- Shared data/evaluation: `comparison/data.py`, `comparison/evaluation.py`, `configs/comparison/protocol.json`.
- Proposal scope review: `reports/fyp1_scope_review/scope_comparison.md`.
