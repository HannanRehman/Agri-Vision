# Comparison configurations

**Status: implemented adapters/runner; candidate hyperparameters; Colab GPU validation pending.**

protocol.json defines the frozen common dataset, clients, budget and evaluation. The four model JSON files specify distinct one-stage detectors and candidate resolutions. `comparison.runner` consumes the protocol and records the model specification hash; optimizer defaults currently come from its explicit candidate table. Finalize settings after GPU pilots and record the resolved configuration in each run. The historical model.py/colab_train.py do not read these files.

The `non_chili_v1` dataset was built and verified on 10 October 2026. The protocol records `dataset_ready: true`, dataset/client YAMLs, measured split counts and the SHA-256 of the frozen checksum manifest. Four adapters, shared evaluator, recoverable runner and final test/profile/table tools are implemented. CPU checks establish correctness on that environment; the [GPU workflow](../../reports/comparison/colab_instructions.md) still needs execution. See [the preparation report](../../reports/non_chili_v1/dataset_preparation_report.md).
