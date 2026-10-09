# Comparison configurations

**Status: draft specifications, not runnable training configurations.**

protocol.json defines the intended common dataset, clients, budget and evaluation. The four model JSON files specify distinct one-stage detectors and candidate resolutions. The separate SSD–MobileNetV3 configuration still needs implementation and initialization checks. Finalize settings after pilots and record the resolved configuration in each run. The existing model.py/colab_train.py do not read these files.
