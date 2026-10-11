"""Prepare reproducible picture-demo assets from saved checkpoints; no training."""
import csv
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
import torch
from PIL import Image, ImageDraw, ImageFont
from comparison.data import CropDataset
from comparison.models import create_adapter
from scripts.yolo_native_train import create_yolo, NativePredictor

OUT = ROOT / 'reports/seminar_poster/assets'
torch.set_num_threads(4)
FONT = ImageFont.truetype('C:/Windows/Fonts/arial.ttf', 16)
MODELS = {
    'yolo_v8n': ('YOLOv8n', ROOT/'saved_models/yolo_native_decline_20261011/global_round_012.pt', 'yolo'),
    'ssdlite': ('SSDLite', ROOT/'saved_models/seminar_demo/ssdlite_round030.pt', 'ssdlite'),
    'ssd_mobilenetv3': ('SSD-MobileNetV3', ROOT/'saved_models/seminar_demo/ssd_mobilenetv3_round030.pt', 'ssd'),
    'efficientdet_d0': ('EfficientDet-D0', ROOT/'saved_models/seminar_demo/efficientdet_d0_round010.pt', 'eff'),
}
CROPS = [('Banana', 'sigatoka'), ('Cauliflower', 'black_rot'),
         ('Groundnut', 'early_leaf_spot'), ('Radish', 'black_leaf_spot')]

def json_prediction(p):
    keep = p['scores'] >= .25
    return {k: p[k][keep].cpu().tolist() for k in ('boxes', 'labels', 'scores')}

def visual(path, prediction, names, dest, truth=False):
    im = Image.open(path).convert('RGB')
    # Preserve the complete photograph: padding/layout happens in the documents.
    d = ImageDraw.Draw(im)
    for index, (box, label) in enumerate(zip(prediction['boxes'], prediction['labels'])):
        color = '#E6B849' if truth else '#13A87B'
        d.rectangle(box, outline=color, width=3)
        caption = names[int(label)].replace('_', ' ')
        if not truth:
            caption += f" {prediction['scores'][index]:.0%}"
        width = d.textlength(caption, font=FONT) + 10
        x = max(0, min(box[0], im.width-width))
        y = max(0, min(box[1]-22, im.height-22))
        d.rectangle((x,y,x+width,y+22), fill=color)
        d.text((x+5,y+2), caption, fill='white' if not truth else '#1D352F', font=FONT)
    dest.parent.mkdir(parents=True,exist_ok=True)
    im.save(dest)
