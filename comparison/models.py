"""Distinct detector adapters with canonical 0-based predictions and original-image boxes."""
from functools import partial
from pathlib import Path
from types import SimpleNamespace

import torch
from torch import nn
import torch.nn.functional as F


def freeze_batchnorm(model):
    # Train affine parameters; freeze running statistics on all models/clients.
    for module in model.modules():
        if isinstance(module,nn.modules.batchnorm._BatchNorm):
            module.eval()


def transfer_matching(module,source):
    current = module.state_dict()
    matched = {key:value for key,value in source.items()
               if key in current and value.shape==current[key].shape}
    module.load_state_dict({**current,**matched},strict=True)
    return len(matched)


def clean_prediction(boxes,labels,scores,image,max_detections=100):
    height,width = image.shape[-2:]
    boxes = boxes.clone()
    boxes[:,[0,2]] = boxes[:,[0,2]].clamp(0,width)
    boxes[:,[1,3]] = boxes[:,[1,3]].clamp(0,height)
    keep = torch.isfinite(boxes).all(dim=1) & torch.isfinite(scores) & (scores>=.001)
    keep &= (labels>=0) & (labels<24) & ((boxes[:,2:]-boxes[:,:2])>0).all(dim=1)
    order = torch.nonzero(keep).flatten()
    order = order[scores[order].argsort(descending=True)[:max_detections]]
    return dict(boxes=boxes[order],labels=labels[order].long(),scores=scores[order])


class Adapter:
    def __init__(self,model_id,device,root,pretrained=True):
        self.model_id,self.device,self.root = model_id,torch.device(device),Path(root)
        self.pretrained = pretrained

    def train_mode(self):
        self.net.train()
        freeze_batchnorm(self.net)

    def state(self):
        return {key:value.detach().cpu().clone() for key,value in self.net.state_dict().items()}

    def load(self,state):
        self.net.load_state_dict(state,strict=True)


class TorchSSDAdapter(Adapter):
    def __init__(self,model_id,device,root,pretrained=True):
        super().__init__(model_id,device,root,pretrained)
        from torchvision.models.detection import ssdlite320_mobilenet_v3_large,SSDLite320_MobileNet_V3_Large_Weights
        from torchvision.models.detection.ssdlite import SSDLiteClassificationHead
        from torchvision.models.detection.ssd import SSDHead
        from torchvision.models.detection._utils import retrieve_out_channels
        weights = SSDLite320_MobileNet_V3_Large_Weights.COCO_V1 if pretrained else None
        self.net = ssdlite320_mobilenet_v3_large(weights=weights,weights_backbone=None,
                                                trainable_backbone_layers=6 if pretrained else None,
                                                score_thresh=.001,nms_thresh=.55,detections_per_img=100)
        self.net.backbone.eval()
        with torch.no_grad():
            channels = retrieve_out_channels(self.net.backbone,(320,320))
        anchors = self.net.anchor_generator.num_anchors_per_location()
        if model_id == 'ssdlite':
            norm = partial(nn.BatchNorm2d,eps=.001,momentum=.03)
            original_classifier = self.net.head.classification_head.state_dict()
            self.net.head.classification_head = SSDLiteClassificationHead(channels,anchors,25,norm)
            if pretrained:
                transfer_matching(self.net.head.classification_head,original_classifier)
            components = 'COCO backbone/extra features/regression head and compatible classification feature/BN tensors; new 25-class projection tensors'
        elif model_id == 'ssd_mobilenetv3':
            self.net.head = SSDHead(channels,anchors,25)
            components = 'COCO MobileNetV3/extra features; new standard SSD classification AND regression heads'
        else:
            raise ValueError(model_id)
        self.net.to(self.device)
        self.metadata = dict(architecture=model_id,input_size=320,background_id=0,target_label_offset=1,
                             pretrained_source=weights.url if weights else None,
                             initialization=components if pretrained else 'random all components',
                             nms_iou=.55,score_floor=.001,trainable_backbone_layers=6,
                             normalization='frozen BN running statistics; affine parameters trainable')

    def loss(self,images,targets):
        self.train_mode()
        shifted = [{k:v.to(self.device) for k,v in target.items() if k in ['boxes','labels']} for target in targets]
        for target in shifted:
            target['labels'] = target['labels']+1
        losses = self.net([image.to(self.device) for image in images],shifted)
        return sum(losses.values()),{k:float(v.detach()) for k,v in losses.items()}

    @torch.inference_mode()
    def predict(self,images):
        self.net.eval()
        outputs = self.net([image.to(self.device) for image in images])
        return [clean_prediction(p['boxes'],p['labels']-1,p['scores'],image)
                for p,image in zip(outputs,images)]


class YoloAdapter(Adapter):
    def __init__(self,model_id,device,root,pretrained=True):
        super().__init__(model_id,device,root,pretrained)
        from ultralytics import YOLO
        from ultralytics.nn.tasks import DetectionModel
        from ultralytics.utils import DEFAULT_CFG_DICT
        self.net = DetectionModel('yolov8n.yaml',nc=24,verbose=False)
        if pretrained:
            local = self.root/'yolov8n.pt'
            source = YOLO(str(local) if local.exists() else 'yolov8n.pt').model
            if len(source.names)!=80:
                raise ValueError('YOLO initialization must be the COCO model, not an old crop checkpoint')
            self.net.load(source,verbose=False)
        self.net.args = SimpleNamespace(**DEFAULT_CFG_DICT)
        self.net.to(self.device)
        self.size = 640
        self.metadata = dict(architecture='YOLOv8n',input_size=640,target_label_offset=0,
                             pretrained_source='Ultralytics yolov8n.pt COCO' if pretrained else None,
                             initialization='matching COCO tensors transferred; new 24-class output tensors' if pretrained else 'random all components',
                             nms_iou=.55,score_floor=.001,
                             normalization='frozen BN running statistics; affine parameters trainable')

    def batch(self,images):
        return torch.stack([F.interpolate(image[None].to(self.device),size=(self.size,self.size),
                                          mode='bilinear',align_corners=False)[0] for image in images])

    def loss(self,images,targets):
        self.train_mode()
        boxes,classes,indices = [],[],[]
        for index,(image,target) in enumerate(zip(images,targets)):
            height,width = image.shape[-2:]
            b = target['boxes'].to(self.device)
            boxes.append(torch.cat(((b[:,:2]+b[:,2:])/2,b[:,2:]-b[:,:2]),dim=1)/b.new_tensor([width,height,width,height]))
            classes.append(target['labels'].to(self.device).float().reshape(-1,1))
            indices.append(torch.full((len(b),),index,device=self.device,dtype=torch.float32))
        batch = dict(img=self.batch(images),bboxes=torch.cat(boxes),cls=torch.cat(classes),batch_idx=torch.cat(indices))
        loss,items = self.net.loss(batch)
        return loss.sum()/len(images),dict(box=float(items[0]),classification=float(items[1]),dfl=float(items[2]))

    @torch.inference_mode()
    def predict(self,images):
        from ultralytics.utils.nms import non_max_suppression
        self.net.eval()
        raw = self.net(self.batch(images))
        predicted = non_max_suppression(raw[0] if isinstance(raw,tuple) else raw,
                                        conf_thres=.001,iou_thres=.55,max_det=100,nc=24,max_time_img=10.)
        output = []
        for prediction,image in zip(predicted,images):
            boxes = prediction[:,:4].clone()
            height,width = image.shape[-2:]
            boxes *= boxes.new_tensor([width/self.size,height/self.size,width/self.size,height/self.size])
            output.append(clean_prediction(boxes,prediction[:,5].long(),prediction[:,4],image))
        return output


class EfficientDetAdapter(Adapter):
    def __init__(self,model_id,device,root,pretrained=True):
        super().__init__(model_id,device,root,pretrained)
        from effdet import create_model
        from effdet.bench import DetBenchTrain,DetBenchPredict
        self.net = create_model('tf_efficientdet_d0',pretrained=pretrained,
                                pretrained_backbone=False,max_det_per_image=100)
        original_classifier = self.net.class_net.state_dict()
        self.net.reset_head(num_classes=24)
        if pretrained:
            transfer_matching(self.net.class_net,original_classifier)
        # Transferred class features may have large activations. Start disease projections
        # near zero with the standard 1% focal prior instead of saturating random logits.
        projection = self.net.class_net.predict.conv_pw
        nn.init.normal_(projection.weight,std=.001)
        nn.init.constant_(projection.bias,-4.59511985013459)
        self.net.to(self.device)
        self.training_bench = DetBenchTrain(self.net,create_labeler=True).to(self.device)
        self.prediction_bench = DetBenchPredict(self.net).to(self.device)
        self.size = 512
        self.metadata = dict(architecture='tf_efficientdet_d0',input_size=512,target_label_offset=1,
                             pretrained_source=self.net.config.url if pretrained else None,
                             initialization='COCO D0 and matching classification feature/BN tensors; new 24-class projection tensors' if pretrained else 'random all components',
                             class_projection_init_std=.001,class_projection_prior=.01,
                             nms_iou=.5,score_floor=.001,
                             normalization='frozen BN running statistics; affine parameters trainable')

    def batch(self,images):
        batch = torch.stack([F.interpolate(image[None].to(self.device),size=(self.size,self.size),
                                          mode='bilinear',align_corners=False)[0] for image in images])
        return (batch-batch.new_tensor([.485,.456,.406])[None,:,None,None])/batch.new_tensor([.229,.224,.225])[None,:,None,None]

    def loss(self,images,targets):
        self.training_bench.train(); freeze_batchnorm(self.net)
        count = max(max(len(t['boxes']) for t in targets),1)
        boxes = torch.full((len(targets),count,4),-1.,device=self.device)
        labels = torch.full((len(targets),count),-1.,device=self.device)
        for index,(image,target) in enumerate(zip(images,targets)):
            height,width = image.shape[-2:]
            scaled = target['boxes'].to(self.device)*boxes.new_tensor([self.size/width,self.size/height,self.size/width,self.size/height])
            boxes[index,:len(scaled)] = scaled[:,[1,0,3,2]]  # effdet anchors expect YXYX
            labels[index,:len(scaled)] = target['labels'].to(self.device)+1
        result = self.training_bench(self.batch(images),dict(bbox=boxes,cls=labels))
        return result['loss'],{k:float(result[k].detach()) for k in ['class_loss','box_loss']}

    @torch.inference_mode()
    def predict(self,images):
        self.prediction_bench.eval()
        detections = self.prediction_bench(self.batch(images))
        output = []
        for detection,image in zip(detections,images):
            height,width = image.shape[-2:]
            boxes = detection[:,:4]*detection.new_tensor([width/self.size,height/self.size,width/self.size,height/self.size])
            output.append(clean_prediction(boxes,detection[:,5].long()-1,detection[:,4],image))
        return output


def create_adapter(model_id,device,root,pretrained=True):
    if model_id in ('ssdlite','ssd_mobilenetv3'):
        return TorchSSDAdapter(model_id,device,root,pretrained)
    if model_id == 'yolo_v8n':
        return YoloAdapter(model_id,device,root,pretrained)
    if model_id == 'efficientdet_d0':
        return EfficientDetAdapter(model_id,device,root,pretrained)
    raise ValueError(f'Unknown detector: {model_id}')
