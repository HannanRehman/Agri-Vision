"""Pooled COCO bbox AP and explicit micro P/R/F1 at fixed confidence and IoU."""
from contextlib import redirect_stdout
import io

import numpy as np
import torch
from torchvision.ops import box_iou


class DetectionEvaluator:
    def __init__(self,names,confidence=.25,iou=.5,max_detections=100):
        self.names = names
        self.confidence,self.iou,self.max_detections = confidence,iou,max_detections
        self.targets,self.predictions = {},{}

    def update(self,targets,predictions):
        if len(targets) != len(predictions):
            raise ValueError('Prediction/target batch size differs')
        for target,prediction in zip(targets,predictions):
            image_id = int(target['image_id'])
            if image_id in self.targets:
                raise ValueError('Image evaluated twice')
            clean = {k:prediction[k].detach().cpu() for k in ['boxes','labels','scores']}
            boxes,labels,scores = clean['boxes'],clean['labels'],clean['scores']
            if boxes.shape != (len(scores),4) or labels.shape != scores.shape:
                raise ValueError('Invalid prediction shapes')
            if not torch.isfinite(boxes).all() or not torch.isfinite(scores).all():
                raise ValueError('Non-finite prediction')
            if len(labels) and (labels.min()<0 or labels.max()>=len(self.names)):
                raise ValueError('Prediction class convention is wrong')
            order = scores.argsort(descending=True)[:self.max_detections]
            self.targets[image_id] = {k:target[k].detach().cpu() for k in ['boxes','labels','original_size']}
            self.predictions[image_id] = {k:v[order] for k,v in clean.items()}

    def precision_recall(self):
        counts = {i:[0,0,0] for i in self.names}  # TP, FP, FN
        for image_id,target in self.targets.items():
            prediction = self.predictions[image_id]
            for cls in self.names:
                truth = target['boxes'][target['labels']==cls]
                mask = (prediction['labels']==cls) & (prediction['scores']>=self.confidence)
                detected = prediction['boxes'][mask]
                matched = set(); tp = 0
                overlaps = box_iou(detected,truth)
                for row in overlaps:
                    # Match the highest-IoU unmatched GT, not an already used GT.
                    eligible = [(float(value),index) for index,value in enumerate(row)
                                if index not in matched and value >= self.iou]
                    if eligible:
                        _,index = max(eligible)
                        matched.add(index); tp += 1
                counts[cls][0] += tp
                counts[cls][1] += len(detected)-tp
                counts[cls][2] += len(truth)-tp
        def metrics(tp,fp,fn):
            precision = tp/(tp+fp) if tp+fp else 0.
            recall = tp/(tp+fn) if tp+fn else 0.
            return dict(precision=precision,recall=recall,f1=2*precision*recall/(precision+recall) if precision+recall else 0.,
                        true_positives=tp,false_positives=fp,false_negatives=fn)
        pooled = metrics(*np.sum(list(counts.values()),axis=0).tolist())
        return pooled,{i:metrics(*values) for i,values in counts.items()}

    def compute(self):
        from pycocotools.coco import COCO
        from pycocotools.cocoeval import COCOeval
        if not self.targets:
            raise ValueError('No images evaluated')
        images,annotations,detections = [],[],[]
        for image_id,target in self.targets.items():
            height,width = target['original_size'].tolist()
            images.append(dict(id=image_id,height=height,width=width))
            for box,label in zip(target['boxes'].tolist(),target['labels'].tolist()):
                x1,y1,x2,y2 = box; bbox = [x1,y1,x2-x1,y2-y1]
                annotations.append(dict(id=len(annotations)+1,image_id=image_id,category_id=label+1,
                                        bbox=bbox,area=bbox[2]*bbox[3],iscrowd=0))
            pred = self.predictions[image_id]
            for box,label,score in zip(pred['boxes'].tolist(),pred['labels'].tolist(),pred['scores'].tolist()):
                x1,y1,x2,y2 = box
                if x2>x1 and y2>y1:
                    detections.append(dict(image_id=image_id,category_id=label+1,
                                           bbox=[x1,y1,x2-x1,y2-y1],score=score))
        with redirect_stdout(io.StringIO()):
            truth = COCO()
            truth.dataset = dict(images=images,annotations=annotations,
                                 categories=[dict(id=i+1,name=name) for i,name in self.names.items()],info={})
            truth.createIndex()
            if detections:
                detected = truth.loadRes(detections)
            else:
                detected = COCO(); detected.dataset = {**truth.dataset,'annotations':[]}; detected.createIndex()
            evaluator = COCOeval(truth,detected,'bbox')
            evaluator.params.imgIds = sorted(self.targets)
            evaluator.params.catIds = [i+1 for i in self.names]
            evaluator.params.maxDets = [1,10,self.max_detections]
            evaluator.evaluate(); evaluator.accumulate(); evaluator.summarize()
        pooled,per_class_pr = self.precision_recall()
        precision = evaluator.eval['precision']  # IoU, recall, category, area, maxDets
        per_class = []
        for index,cls in enumerate(self.names):
            ap = precision[:,:,index,0,-1]
            ap50 = precision[0,:,index,0,-1]
            valid,valid50 = ap[ap>=0],ap50[ap50>=0]
            per_class.append(dict(class_id=cls,class_name=self.names[cls],
                                  map_50_95=float(valid.mean()) if len(valid) else None,
                                  map50=float(valid50.mean()) if len(valid50) else None,**per_class_pr[cls]))
        return dict(map_50_95=float(evaluator.stats[0]),map50=float(evaluator.stats[1]),
                    coco_average_recall_100=float(evaluator.stats[8]),**pooled,
                    images=len(images),ground_truth_boxes=len(annotations),
                    precision_recall_confidence=self.confidence,precision_recall_iou=self.iou,
                    precision_recall_aggregation='micro pooled class-aware greedy matching',
                    ap_confidence_filter='adapter score floor 0.001; AP uses score sweep, not P/R cutoff',
                    per_class=per_class)
