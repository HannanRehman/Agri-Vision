"""Combine read-only evidence and contrast it with the historical YOLO run."""
import csv
import json
from pathlib import Path
import sys
import zipfile
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import torch
from ultralytics import YOLO
from comparison.runner import implementation_hash
from comparison.data import CropDataset
torch.set_num_threads(4)
OUT=ROOT/'reports/comparison/yolo_diagnosis_20261010'
RUN=ROOT/'experiments/yolo_v8n_non_chili_seed42_20261009_224152_449800'
audit=json.loads((OUT/'audit.json').read_text())
initial=torch.load(RUN/'checkpoints/initial.pt',map_location='cpu',weights_only=True)
names_checks=[]
for number in range(11):
    path=RUN/('checkpoints/initial.pt' if number==0 else f'checkpoints/global_round_{number:03d}.pt')
    state=torch.load(path,map_location='cpu',weights_only=True)
    names_checks.append(state['names']==initial['names'])
source=YOLO(str(ROOT/'yolov8n.pt')).model.float().state_dict()
non_classifier=[k for k in initial['model_state'] if not k.startswith('model.22.cv3.')]
audit['checkpoint_names_match']=all(names_checks)
audit['non_classifier_pretrained_transfer']=dict(tensors=len(non_classifier),
    shape_compatible=sum(k in source and source[k].shape==initial['model_state'][k].shape for k in non_classifier),
    exact_values=sum(k in source and torch.equal(initial['model_state'][k],source[k]) for k in non_classifier))
dataset=CropDataset(ROOT,'valid')
dimensions={}
for row in dataset.rows:
    size=f"{row['width']}x{row['height']}";dimensions[size]=dimensions.get(size,0)+1
audit['validation_dimensions']=dimensions
assert audit['checkpoint_names_match']
assert audit['original_implementation_hash']==audit['recorded_implementation_hash']==implementation_hash()
assert audit['replay_matches_original'] and audit['checkpoints_unchanged']
assert all(x['finite'] and x['metadata_matches'] and not x['changed_bn_buffers'] for x in audit['checkpoint_integrity'])
assert all(x['next_client_start_hash_matches'] is not False for x in audit['checkpoint_integrity'])
assert all(x['same_count'] and x['same_labels'] for x in audit['native_predictor_comparison'])
(OUT/'audit.json').write_text(json.dumps(audit,indent=2))
old_archive=ROOT/'saved_models/20261004_200656_355a3a-20261004T205120Z-1-001.zip'
historical={}
with zipfile.ZipFile(old_archive) as z:
    settings=next(n for n in z.namelist() if n.endswith('/settings.json'))
    historical['settings']=json.loads(z.read(settings))
    configs=[n for n in z.namelist() if n.endswith('/args.yaml') and 'round_036/' in n]
    import yaml
    historical['round_36_local_training_args']=[{k:v for k,v in yaml.safe_load(z.read(n)).items() if k in ['epochs','batch','imgsz','optimizer','lr0','lrf','warmup_epochs','warmup_momentum','warmup_bias_lr','mosaic','hsv_h','hsv_s','hsv_v','scale','translate','freeze','nbs','weight_decay']} for n in configs]
    historical['round_36_local_csv_examples']=[]
    for name in z.namelist():
        if name.endswith('/results.csv') and 'round_036/' in name:
            rows=list(csv.DictReader(z.read(name).decode('utf-8').splitlines()))
            historical['round_36_local_csv_examples'].append({'member':name,'first':rows[0],'last':rows[-1]})
    old_checkpoint=next(n for n in z.namelist() if n.endswith('/global_round_040.pt'))
    with z.open(old_checkpoint) as stream:
        old_state=torch.load(stream,map_location='cpu',weights_only=True)['model_state_dict']
    old_bn=[k for k in old_state if k.startswith('model.22.cv3.') and k.endswith('running_mean')]
    historical['round40_classification_bn']=dict(layers=len(old_bn),
        means_nonzero=sum(torch.count_nonzero(old_state[k]).item()>0 for k in old_bn),
        variances_nonunit=sum(not torch.equal(old_state[k.replace('running_mean','running_var')],torch.ones_like(old_state[k])) for k in old_bn))
(OUT/'historical_training_comparison.json').write_text(json.dumps(historical,indent=2))
activations=json.loads((OUT/'activations.json').read_text())
candidate=json.loads((OUT/'initialization_candidate.json').read_text())
with (ROOT/'reports/rounds_036_040_global_metrics.csv').open() as stream:
    old=list(csv.DictReader(stream))[-1]
lines=['# Why the new YOLO run differs from the successful round-40 model','',
'The new experiment started fresh from COCO weights. It did not continue or damage the historical round-40 crop checkpoint. The historical model learned 30 classes over 40 global rounds; the new comparison learned 24 non-chili classes over 10 global rounds using a different trainer.','',
'| Item | Historical round 40 | New 24-class run |','|---|---|---|',
f"| Reported precision | {100*float(old['precision']):.2f}% | 0% at confidence 0.25 |",
f"| Reported recall | {100*float(old['recall']):.2f}% | 0% at confidence 0.25 |",
f"| Reported mAP50–95 | {100*float(old['mAP50_95']):.2f}% | 0.80% at best round 8 |",
'| Initialization | Continued crop-trained weights through round 40 | Fresh 80-class COCO weights adapted to 24 classes |',
'| Training | Native Ultralytics `model.train()` | Custom shared SGD loop, initial LR 0.005 |',
'| BatchNorm running statistics | Updated during native local training | Frozen on every layer from the beginning |',
'| Metrics | Validation-image-weighted client summaries of Ultralytics per-class metrics | Pooled COCO AP; micro P/R at confidence 0.25 and IoU 0.5 |','',
'These percentages are not a direct before/after measurement of one model. The dataset, training exposure, initialization, training procedure and metric definitions changed. The removal of chili happened alongside these changes; it is not a controlled explanation for the performance collapse.','',
'## What the checks establish','',
'All eleven saved YOLO states, including initialization, contain finite tensors. Metadata and class maps agree. Each global state hash agrees with the next round’s logged starting state, and all clients received the same global initialization. Each completed round visited all 14,514 training images. BN buffers are unchanged across every checkpoint, confirming the freeze policy. Initial and final file hashes match their saved records.','',
'Replaying the best round’s full saved validation predictions exactly reproduces its AP and P/R. Native Ultralytics prediction on 24 fixed validation images agrees with the custom adapter: the counts and labels match, maximum score error is below 2e-7 and maximum box error is below 0.0002 pixels. This strongly rules out a simple adapter confidence, NMS or box-rescaling defect on that sample. It does not independently certify the entire native training procedure.','',
'The round-8 export includes all 4,157 validation images and 21,803 detections above the 0.001 export floor. Its highest confidence is 0.0632302. Round 1 had no detections above even 0.001. Low confidence explains zero fixed-threshold precision/recall, while poor score-ranked detections explain the low AP.','',
'| Confidence cutoff | Precision | Recall | True positives | False positives |','|---:|---:|---:|---:|---:|']
for row in audit['threshold_sweep']:
    lines.append(f"| {row['confidence']} | {100*row['precision']:.2f}% | {100*row['recall']:.2f}% | {row['true_positives']} | {row['false_positives']} |")
lines += ['',
'The threshold table is a diagnostic on validation data. It does not replace the shared confidence-0.25 comparison or constitute a new test result. Lowering confidence does not restore AP or make this model comparable to the historical result.','',
'## The main implementation weakness','',
'YOLOv8n chooses its classification feature width using the number of classes. COCO has width 80; the new 24-class constructor has width 64. Consequently 36 of its 42 classification state tensors cannot transfer from COCO. The only six compatible tensors are scalar BatchNorm counters. Six newly created classification BN layers begin with zero means and unit variances, and our common freeze policy prevents them adapting throughout training. This is a larger initialization change than the metadata phrase “new output tensors” suggests.','',
'The historical 30-class constructor also replaces incompatible classifier tensors, but its native trainer lets the new BN layers learn statistics. The combination of fresh classification features, permanently frozen statistics and a different optimizer/training schedule is a strong candidate cause. This read-only audit does not prove which change caused how much of the collapse.','',
'Native training also supplies warmup, parameter grouping and its own augmentation/optimization behavior. The new loop has no warmup, applies SGD weight decay uniformly and uses only horizontal-flip augmentation. Historical archived settings and local args/CSV extracts are saved separately; an `optimizer=auto` argument is not proof of a particular actual optimizer.','',
'The historical round-36 local CSVs report a final learning rate of 0.0000978485 for all three parameter groups, while the new loop starts SGD at 0.005. These values belong to different optimizer/schedule configurations and should not be treated as equivalent update magnitudes. The historical round-40 classifier BN layers have learned nonzero running means and nonunit variances, confirming that their statistics adapted.','',
'A four-image feature probe found a sharp loss of image-specific variation in deeper layers. For the final neck output, mean inter-image standard deviation fell from 0.1561 at initialization to 0.00210 at round 8, while overall activation RMS rose from 0.3200 to 4.4654. This supports feature degeneration in the trained network. It is a small diagnostic sample, not a dataset-wide feature-quality metric or causal proof. Client model weights were not retained, so we cannot isolate local-training damage from additional effects of FedAvg.','',
'“Improve image features” therefore means restoring useful learned representations in the model. It does not mean your existing crop photographs suddenly became poor quality.','',
'## A concrete candidate checked without training','',
'An isolated prototype retains the pretrained COCO classification feature width of 80 and all 36 feature/BN state tensors, replacing only the six final projection tensors for 24 outputs. Its loss forward is finite, output IDs are valid and inference/loss checks leave model tensors unchanged. No backward pass, optimizer step or training run was performed. Its parameter count is 3,143,592.','',
'This prototype is a custom COCO-width YOLOv8n classification head, rather than the canonical nc=24 width-64 graph. Its initialization check is not evidence of improved accuracy. The alternative is canonical YOLOv8n with an explicit trainable-BN policy, which would require a declared change to the comparison protocol.','',
'## Recommended sequence when GPU access returns','',
'1. Resume EfficientDet with its existing code and settings to complete the original budget.',
'2. Before any new full YOLO run, perform a small training-only ablation comparing the original setup with a corrected initialization/normalization setup. Require finite losses, correct target IDs, unchanged intended frozen buffers, and recovery of class-correct boxes on a fixed training sanity sample; decreasing loss alone is insufficient.',
'3. Add a short federated check that measures each client model before aggregation and the global model after aggregation. This will distinguish local learning failure from averaging-related degeneration.',
'4. If the sanity checks pass, start a separately versioned YOLO experiment on the same frozen 24-class data, clients, seed and declared round budget. Preserve this failed run as diagnostic evidence. Record head width, normalization and optimizer changes explicitly, and confirm the comparison contract before interpreting the final table.',
'5. Evaluate selected valid models on the untouched held-out test set only after configuration selection. Never tune using test scores.','',
'The current comparison modules, package ZIP, dataset manifests and Drive checkpoints were not edited. The implementation hash remains `c6d295dc79d706a18d73090edf91532cda74fee23a9f4d3e737e0470915fff71`, so the prepared EfficientDet resume remains compatible.','',
'## Evidence','',
'- [Full audit and replay](audit.json)',
'- [Feature probe](activations.json)',
'- [Untrained initialization candidate checks](initialization_candidate.json)',
'- [Historical archived training settings and CSV excerpts](historical_training_comparison.json)',
'- [Ultralytics 8.4.48 detection head implementation](https://github.com/ultralytics/ultralytics/blob/v8.4.48/ultralytics/nn/modules/head.py)',
'- [Ultralytics 8.4.48 shape-matched weight loading](https://github.com/ultralytics/ultralytics/blob/v8.4.48/ultralytics/nn/tasks.py)',
'- [Ultralytics training documentation](https://docs.ultralytics.com/modes/train)','']
(OUT/'README.md').write_text('\n'.join(lines),encoding='utf-8')
print(json.dumps({'non_classifier_transfer':audit['non_classifier_pretrained_transfer'],'validation_dimensions':dimensions,
                  'historical_args':historical['round_36_local_training_args'],'report':str(OUT/'README.md')}))
