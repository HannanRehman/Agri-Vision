"""Verified project-checkpoint inference. No downloads, training or image storage."""
import hashlib
import json
import math
from pathlib import Path
import threading
import time

import numpy as np
from PIL import Image
import torch

ROOT = Path(__file__).resolve().parents[1]


def sha256(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def names_list(names):
    if isinstance(names, dict):
        if {int(k) for k in names} != set(range(24)):
            raise ValueError('Expected contiguous IDs 0-23')
        return [names[i] if i in names else names[str(i)] for i in range(24)]
    if isinstance(names, list) and len(names) == 24:
        return names
    raise ValueError('Expected the frozen 24-class mapping')


def describe(name):
    crop, condition = name.split('_', 1)
    return crop, condition.replace('_', ' ').capitalize()


class Detector:
    def __init__(self, config_path=None):
        self.config_path = Path(config_path or Path(__file__).with_name('config.json'))
        self.config = json.loads(self.config_path.read_text(encoding='utf-8'))
        self.lock = threading.Lock()
        self.adapter = None
        self.ready = False
        self.failed = False
        self.names = []
        self.checkpoint_round = None

    def load(self):
        cfg = self.config
        mapping = ROOT/cfg['class_map']
        checkpoint = ROOT/cfg['checkpoint']
        if sha256(mapping) != cfg['class_map_sha256'] or sha256(checkpoint) != cfg['checkpoint_sha256']:
            raise ValueError('Checkpoint or class mapping SHA256 mismatch')
        self.names = names_list(json.loads(mapping.read_text())['names'])
        saved = torch.load(checkpoint, map_location='cpu', weights_only=True)
        if names_list(saved['names']) != self.names:
            raise ValueError('Checkpoint class order differs')
        if saved['config_hash'] != cfg['checkpoint_config_hash']:
            raise ValueError('Checkpoint training identity differs')
        if not 0 < cfg['display_threshold'] < 1 or cfg['device'] not in ('cpu', 'cuda'):
            raise ValueError('Invalid inference configuration')
        torch.set_num_threads(cfg['threads'])
        if cfg['model'] == 'yolo_v8n':
            raise ValueError('Native YOLO requires its matching loader; do not silently use a different preprocessing path')
        # Reuse the exact baseline adapter for its baseline checkpoint. Tuned
        # checkpoints require tuning.models.create_adapter and matching manifest.
        from comparison.models import create_adapter
        self.adapter = create_adapter(cfg['model'], cfg['device'], ROOT, pretrained=False)
        if self.adapter.metadata['input_size'] != cfg['input_size']:
            raise ValueError('Checkpoint preprocessing size mismatch')
        self.adapter.load(saved['model_state'])
        self.adapter.net.eval()
        self.checkpoint_round = saved['round']
        # One warm-up; no gradient calculation, optimizer, or training data access.
        self.adapter.predict([torch.zeros(3, 256, 256)])
        self.ready = True

    def public_info(self):
        cfg = self.config
        return dict(ready=self.ready, state='ready' if self.ready else ('unavailable' if self.failed else 'loading'),
            model=cfg['display_name'], provisional=cfg['provisional'],
            threshold=cfg['display_threshold'], threshold_validated=cfg['threshold_validated'],
            device=cfg['device'], classes=[dict(id=i, name=n, crop=describe(n)[0],
                condition=describe(n)[1]) for i, n in enumerate(self.names)],
            supported_crops=['Banana', 'Cauliflower', 'Groundnut', 'Radish'],
            max_upload_bytes=cfg['max_upload_bytes'], max_pixels=cfg['max_pixels'])

    @torch.inference_mode()
    def predict(self, image: Image.Image):
        # The route owns this nonblocking lock. Busy callers get a retry message,
        # rather than building an unbounded queue of decoded images.
        array = np.array(image, dtype=np.uint8, copy=True)
        tensor = torch.from_numpy(array).permute(2, 0, 1).float().div_(255)
        start = time.perf_counter()
        output = self.adapter.predict([tensor])[0]
        if self.config['device'] == 'cuda':
            torch.cuda.synchronize()
        milliseconds = (time.perf_counter()-start)*1000
        detections = []
        width, height = image.size
        for box, label, score in zip(output['boxes'].tolist(), output['labels'].tolist(), output['scores'].tolist()):
            if not math.isfinite(score) or score < self.config['display_threshold'] or not 0 <= label < len(self.names):
                continue
            if not all(math.isfinite(v) for v in box):
                continue
            x1, y1, x2, y2 = box
            box = [max(0, min(width, x1)), max(0, min(height, y1)),
                   max(0, min(width, x2)), max(0, min(height, y2))]
            if box[2] <= box[0] or box[3] <= box[1]:
                continue
            crop, condition = describe(self.names[label])
            detections.append(dict(class_id=label, class_name=self.names[label], crop=crop,
                condition=condition, confidence=score, box=box))
        return dict(status='detected' if detections else 'no_reliable_detection',
            detections=detections, width=width, height=height, inference_ms=round(milliseconds, 1),
            model=self.config['display_name'], provisional=self.config['provisional'],
            threshold=self.config['display_threshold'], threshold_validated=self.config['threshold_validated'])
