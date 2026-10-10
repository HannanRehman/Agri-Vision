"""Check a proposed pretrained-feature-preserving head without training or saving weights."""
import copy
import json
import math
from pathlib import Path
from types import SimpleNamespace
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import torch
from torch import nn
from ultralytics import YOLO
from ultralytics.utils import DEFAULT_CFG_DICT
from comparison.data import CropDataset
from comparison.models import create_adapter
from comparison.runner import state_hash,implementation_hash,seed_all

torch.set_num_threads(4)
seed_all(42)
source=YOLO(str(ROOT/'yolov8n.pt')).model.float()
candidate=copy.deepcopy(source)
head=candidate.model[-1]
for branch,stride in zip(head.cv3,head.stride):
    previous=branch[-1]
    branch[-1]=nn.Conv2d(previous.in_channels,24,kernel_size=1)
    nn.init.constant_(branch[-1].bias,math.log(5/24/(640/float(stride))**2))
head.nc=24;head.no=24+head.reg_max*4
candidate.yaml=copy.deepcopy(candidate.yaml);candidate.yaml['nc']=24
dataset=CropDataset(ROOT,'train',limit=2,seed=42)
candidate.names=dataset.names
candidate.args=SimpleNamespace(**DEFAULT_CFG_DICT)
if hasattr(candidate,'criterion'):del candidate.criterion
for name,parameter in candidate.named_parameters():
    parameter.requires_grad_('.dfl.conv.' not in name)
adapter=create_adapter('yolo_v8n','cpu',ROOT,pretrained=False)
adapter.net=candidate
original=source.state_dict(); adapted=candidate.state_dict()
feature_keys=[k for k in adapted if k.startswith('model.22.cv3.') and k.split('.')[4]!='2']
assert len(feature_keys)==36
assert all(torch.equal(adapted[k],original[k]) for k in feature_keys)
assert all(branch[-1].out_channels==24 and branch[-1].in_channels==80 for branch in head.cv3)
images,targets=zip(*[dataset[i] for i in range(2)])
before=state_hash(adapter.state())
with torch.no_grad():
    loss,components=adapter.loss(images,targets)
assert torch.isfinite(loss)
assert state_hash(adapter.state())==before
outputs=adapter.predict(images)
assert state_hash(adapter.state())==before
assert all(not len(p['labels']) or (p['labels'].min()>=0 and p['labels'].max()<24) for p in outputs)
result=dict(status='initialization_and_forward_checks_passed',training_performed=False,optimizer_steps_performed=0,
    performance_improvement_proven=False,hidden_classification_width=80,output_classes=24,
    preserved_classification_feature_and_bn_state_tensors=len(feature_keys),new_class_projection_tensors=6,
    scalar_forward_loss=float(loss),loss_components=components,model_state_unchanged=True,
    parameter_count=sum(p.numel() for p in candidate.parameters()),
    implementation_hash=implementation_hash(),
    caveat='Custom COCO-width classification branches; not identical to the canonical nc=24 width-64 graph. Record this explicitly if used in a new experiment.')
out=ROOT/'reports/comparison/yolo_diagnosis_20261010/initialization_candidate.json'
out.write_text(json.dumps(result,indent=2))
print(json.dumps(result))
