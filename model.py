"""The YOLO detector shared by all federated clients and the server."""

from copy import deepcopy
from pathlib import Path

import numpy as np
import torch
import yaml
from ultralytics import YOLO
from ultralytics.nn.tasks import DetectionModel


PROJECT_ROOT = Path(__file__).resolve().parent
PRETRAINED_WEIGHTS = PROJECT_ROOT / "yolov8n.pt"
CLASS_CONFIG = PROJECT_ROOT / "dataset" / "client1" / "data.yaml"


def class_names(config=CLASS_CONFIG):
    """Read the common class IDs, rejecting gaps or inconsistent ordering."""
    with Path(config).open(encoding="utf-8") as stream:
        names = yaml.safe_load(stream)["names"]
    names = {int(index): name for index, name in names.items()}
    if sorted(names) != list(range(len(names))):
        raise ValueError("Class IDs must be consecutive and start at zero")
    return names


def create_model():
    """Build the same 23-class architecture on every participant.

    YOLOv8n.pt has an 80-class COCO head. Training silently replaces that head;
    building it explicitly here keeps round-zero and later parameter shapes equal.
    """
    if not PRETRAINED_WEIGHTS.is_file():
        raise FileNotFoundError(f"Missing pretrained weights: {PRETRAINED_WEIGHTS}")
    names = class_names()
    model = YOLO(str(PRETRAINED_WEIGHTS))
    pretrained = model.model
    detector = DetectionModel(deepcopy(pretrained.yaml), nc=len(names), verbose=False)
    detector.load(pretrained)
    detector.names = names
    model.model = detector
    return model


def get_parameters(model):
    """Transmit trainable values and floating buffers, not integer counters."""
    return [
        value.detach().cpu().numpy().copy()
        for value in model.model.state_dict().values()
        if torch.is_floating_point(value)
    ]


def set_parameters(model, parameters):
    """Load federated floating tensors, including batch-normalization statistics."""
    current = model.model.state_dict()
    floating = [(key, value) for key, value in current.items() if torch.is_floating_point(value)]
    if len(parameters) != len(floating):
        raise ValueError(f"Expected {len(floating)} tensors, received {len(parameters)}")
    updated = dict(current)
    for (key, reference), array in zip(floating, parameters):
        array = np.asarray(array)
        if tuple(array.shape) != tuple(reference.shape):
            raise ValueError(f"Wrong shape for {key}: {array.shape} != {tuple(reference.shape)}")
        updated[key] = torch.as_tensor(array, dtype=reference.dtype, device=reference.device)
    model.model.load_state_dict(updated, strict=True)
