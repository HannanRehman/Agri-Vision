"""Frozen manifests and canonical original-image XYXY annotations, labels 0-23."""
import csv
import hashlib
import json
from pathlib import Path

import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset


def sha256(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def load_protocol(root):
    root = Path(root)
    protocol = json.loads((root/'configs/comparison/protocol.json').read_text())
    if not protocol['dataset_ready'] or protocol['foreground_classes'] != 24:
        raise ValueError('A verified 24-class dataset is required')
    checksum_path = root/protocol['dataset_checksums']
    if sha256(checksum_path) != protocol['dataset_checksums_sha256']:
        raise ValueError('Frozen checksum manifest differs from the protocol')
    for filename, expected in json.loads(checksum_path.read_text()).items():
        if sha256(root/filename) != expected:
            raise ValueError(f'Frozen manifest/config changed: {filename}')
    return protocol


class CropDataset(Dataset):
    def __init__(self, root, split, client=None, augment=False, limit=None, seed=42):
        if split not in ('train','valid','test') or (augment and split != 'train'):
            raise ValueError('Only training data may be augmented')
        self.root = Path(root)
        self.augment = augment
        directory = self.root/'manifests/non_chili_v1'
        class_map = json.loads((directory/'class_map.json').read_text())
        self.names = {int(k):v for k,v in class_map['names'].items()}
        if sorted(self.names) != list(range(24)):
            raise ValueError('Non-contiguous foreground labels')
        with (directory/'retained.csv').open(newline='',encoding='utf-8') as stream:
            all_rows = list(csv.DictReader(stream))
        # Image IDs are stable across pooled/client/subset views.
        self.rows = [{**row,'image_id':index+1} for index,row in enumerate(all_rows)
                     if row['split']==split and (client is None or row['client']==client)]
        if limit is not None:
            if limit < 1:
                raise ValueError('Subset limit must be positive')
            generator = np.random.default_rng(seed)
            selected = sorted(generator.choice(len(self.rows),min(limit,len(self.rows)),replace=False))
            self.rows = [self.rows[index] for index in selected]
        if not self.rows:
            raise ValueError(f'Empty dataset: {split}/{client}')

    def __len__(self):
        return len(self.rows)

    def __getitem__(self,index):
        row = self.rows[index]
        with Image.open(self.root/row['image']) as image:
            rgb = np.array(image.convert('RGB'),copy=True)
        height,width = rgb.shape[:2]
        boxes,labels = [],[]
        for line in (self.root/row['label']).read_text().splitlines():
            if not line.strip():
                continue
            raw = line.split()
            label = int(raw[0]); x,y,w,h = map(float,raw[1:])
            if label not in self.names or min(w,h) <= 0:
                raise ValueError(f'Invalid annotation: {row["key"]}')
            boxes.append([(x-w/2)*width,(y-h/2)*height,(x+w/2)*width,(y+h/2)*height])
            labels.append(label)
        boxes = torch.tensor(boxes,dtype=torch.float32).reshape(-1,4)
        # Advanced indexing returns a copy: explicitly assign clipped columns.
        boxes[:,[0,2]] = boxes[:,[0,2]].clamp(0,width)
        boxes[:,[1,3]] = boxes[:,[1,3]].clamp(0,height)
        if not torch.all((boxes[:,2:] - boxes[:,:2]) > 0):
            raise ValueError('Degenerate box after clipping')
        tensor = torch.from_numpy(rgb).permute(2,0,1).float()/255
        if self.augment and torch.rand(()) < .5:
            tensor = tensor.flip(-1)
            left = width-boxes[:,2].clone()
            boxes[:,2] = width-boxes[:,0]
            boxes[:,0] = left
        target = dict(boxes=boxes,labels=torch.tensor(labels,dtype=torch.int64),
                      image_id=torch.tensor(row['image_id']),
                      original_size=torch.tensor([height,width]),key=row['key'])
        return tensor,target


def collate(batch):
    images,targets = zip(*batch)
    return list(images),list(targets)
