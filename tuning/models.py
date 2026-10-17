"""Resolution-aware adapters for the hyperparameter-search experiments."""
from copy import deepcopy

import torch
from torch import nn
from comparison.models import Adapter, EfficientDetAdapter, TorchSSDAdapter, transfer_matching


class SizedEfficientDet(EfficientDetAdapter):
    def __init__(self, device, root, size, pretrained=True):
        from effdet import create_model
        from effdet.bench import DetBenchTrain, DetBenchPredict
        Adapter.__init__(self, 'efficientdet_d0', device, root, pretrained)
        self.net = create_model('tf_efficientdet_d0', pretrained=pretrained,
                                pretrained_backbone=False, max_det_per_image=100,
                                image_size=(size, size))
        original = self.net.class_net.state_dict()
        self.net.reset_head(num_classes=24)
        if pretrained:
            transfer_matching(self.net.class_net, original)
        projection = self.net.class_net.predict.conv_pw
        nn.init.normal_(projection.weight, std=.001)
        nn.init.constant_(projection.bias, -4.59511985013459)
        self.net.to(self.device)
        self.training_bench = DetBenchTrain(self.net, create_labeler=True).to(self.device)
        self.prediction_bench = DetBenchPredict(self.net).to(self.device)
        self.size = size
        self.metadata = dict(architecture='tf_efficientdet_d0', input_size=size,
                             anchors_image_size=list(self.net.config.image_size),
                             normalization='frozen BN running statistics; train affine parameters')


def create_adapter(model, device, root, size, pretrained=True):
    if model == 'efficientdet_d0':
        return SizedEfficientDet(device, root, size, pretrained)
    if model in ('ssdlite', 'ssd_mobilenetv3'):
        adapter = TorchSSDAdapter(model, device, root, pretrained)
        # The torchvision builder ignores a size kwarg. Change the actual transform.
        adapter.net.transform.fixed_size = (size, size)
        adapter.net.transform.min_size = (size,)
        adapter.net.transform.max_size = size
        adapter.metadata.update(input_size=size, transform_fixed_size=[size, size])
        return adapter
    raise ValueError(model)


class NativePredictor:
    def __init__(self, model, device, size):
        from ultralytics import YOLO
        self.model = YOLO('yolov8n.yaml', task='detect', verbose=False)
        self.model.model = deepcopy(model.model).float()
        self.model.overrides = deepcopy(model.overrides)
        self.device, self.size = torch.device(device), size

    @torch.inference_mode()
    def predict(self, images):
        arrays = [image.mul(255).round().byte().permute(1, 2, 0).numpy()[:, :, ::-1].copy() for image in images]
        results = self.model.predict(source=arrays, imgsz=self.size, device=str(self.device),
                                     conf=.001, iou=.55, max_det=100, verbose=False, save=False)
        return [dict(boxes=r.boxes.xyxy.cpu(), labels=r.boxes.cls.long().cpu(),
                     scores=r.boxes.conf.cpu()) for r in results]
