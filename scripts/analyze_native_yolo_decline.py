"""Read saved native-YOLO evidence and generate a report; never train a model."""
import csv
import hashlib
import json
from pathlib import Path
import statistics

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'reports/comparison/yolo_native_decline_20261011'


def main():
    records = [json.loads(p.read_text()) for p in sorted((OUT/'rounds').glob('round_*.json'))]
    best = max(records, key=lambda r: r['pooled_validation']['map_50_95'])
    last = records[-1]
    a, b = best['pooled_validation'], last['pooled_validation']
    losses = []
    for n in [3, 9, 12, 15, 18, 21]:
        rows = [list(csv.DictReader(p.open()))[-1] for p in sorted((OUT/'client_logs').glob(f'*round_{n:03}.csv'))]
        assert len(rows) == 3
        losses.append(dict(round=n, **{k:statistics.mean(float(r[k]) for r in rows) for k in rows[0] if 'loss' in k}))
    changes = [dict(class_name=x['class_name'], ap_best=x['map_50_95'], ap_latest=y['map_50_95'],
                    delta_pp=100*(y['map_50_95']-x['map_50_95']),
                    validation_boxes=x['true_positives']+x['false_negatives'])
               for x, y in zip(a['per_class'], b['per_class'])]
    changes.sort(key=lambda r:r['delta_pp'])
    checks = dict(completed_rounds=[r['round'] for r in records],
                  images=sorted({r['pooled_validation']['images'] for r in records}),
                  ground_truth_boxes=sorted({r['pooled_validation']['ground_truth_boxes'] for r in records}),
                  all_transfers_verified=all(c['native_trainer_transfer_verified'] for r in records for c in r['clients']),
                  client_starts_match=all(len({c['round_start_sha256'] for c in r['clients']}) == 1 for r in records),
                  actual_optimizers=sorted({c['actual_optimizer'] for r in records for c in r['clients']}),
                  final_learning_rates=sorted({tuple(c['final_learning_rates']) for r in records for c in r['clients']}))
    assert checks['completed_rounds'] == list(range(1, 22))
    for r in records:
        m = r['pooled_validation']
        assert m['true_positives']+m['false_negatives'] == 7210
        assert abs(m['precision']-m['true_positives']/(m['true_positives']+m['false_positives'])) < 1e-12
        assert abs(m['recall']-m['true_positives']/7210) < 1e-12
        assert sum(c['unique_training_images'] for c in r['clients']) == 14514
        assert m['precision_recall_confidence'] == .25 and m['precision_recall_iou'] == .5
    state_audit = {}
    import torch
    for n in [12, 21]:
        p = ROOT/f'saved_models/yolo_native_decline_20261011/global_round_{n:03}.pt'
        d = torch.load(p, map_location='cpu', weights_only=True)
        digest = hashlib.sha256(p.read_bytes()).hexdigest()
        state = d['model_state']
        state_audit[n] = dict(round=d['round'], sha256=digest, config_hash=d['config_hash'], classes=len(d['names']),
                             all_floating_finite=all(torch.isfinite(v).all().item() for v in state.values() if v.is_floating_point()),
                             bn_variance_min=min(v.min().item() for k,v in state.items() if k.endswith('running_var')),
                             bn_variance_max=max(v.max().item() for k,v in state.items() if k.endswith('running_var')))
        assert d['round'] == n and len(d['names']) == 24 and state_audit[n]['all_floating_finite']
    assert state_audit[12]['config_hash'] == state_audit[21]['config_hash']
    assert state_audit[21]['sha256'] == 'c25c95199d3e3cf4c234a63ff44b7ba0bf6a8428baa45ef5b03075384ed62409'
    data = dict(best_round=best['round'],latest_round=last['round'],checks=checks,checkpoint_audit=state_audit,
                losses=losses,class_changes=changes,top_four_decline_contribution_pp=-sum(c['delta_pp'] for c in changes[:4])/24,
                changed_training=False,optimizer_steps=0)
    (OUT/'analysis_summary.json').write_text(json.dumps(data,indent=2)+'\n')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,2,figsize=(12,4.5))
    for k,label in [('map50','mAP50'),('map_50_95','mAP50-95'),('precision','Precision'),('recall','Recall')]:
        axes[0].plot([r['round'] for r in records],[100*r['pooled_validation'][k] for r in records],label=label)
    axes[0].axvline(best['round'],color='gray',linestyle='--',alpha=.6)
    axes[0].set(xlabel='Global round',ylabel='Validation score (%)',title='Pooled validation: fixed evaluation settings')
    axes[0].legend();axes[0].grid(alpha=.2)
    for k,label in [('train/box_loss','Training box'),('val/box_loss','Local validation box'),('train/dfl_loss','Training DFL'),('val/dfl_loss','Local validation DFL')]:
        axes[1].plot([r['round'] for r in losses],[r[k] for r in losses],marker='o',label=label)
    axes[1].set(xlabel='Global round',ylabel='Mean loss across three clients',title='Training fit improves; validation localization worsens')
    axes[1].legend();axes[1].grid(alpha=.2)
    fig.tight_layout();fig.savefig(OUT/'validation_and_loss_trends.png',dpi=180);plt.close(fig)
    lines=['# Native YOLO decline analysis: rounds 1-21','',
           'Evidence: 21 full Drive round JSONs, 18 client training CSVs at rounds 3/9/12/15/18/21, frozen dataset manifests/labels, exact native runner, Ultralytics 8.4.48 source, and downloaded best/latest checkpoint tensors. No training, evaluation rerun or hyperparameter changes were performed. The held-out test was not used.','',
           '## Finding','',
           'The strongest supported diagnosis is deteriorating validation generalization, with an important localization component. This is consistent with overfitting after the useful early learning phase. A single exclusive causal explanation cannot be established from observational logs; optimizer schedule, augmentation and BatchNorm alternatives require controlled ablations.','',
           f'Best pooled AP50-95 is round {best["round"]}: {a["map_50_95"]*100:.2f}%. Latest round {last["round"]}: {b["map_50_95"]*100:.2f}%, a decline of {(a["map_50_95"]-b["map_50_95"])*100:.2f} percentage points. AP50 drops {a["map50"]*100:.2f}% to {b["map50"]*100:.2f}%. Precision barely changes ({a["precision"]*100:.2f}% to {b["precision"]*100:.2f}%) and recall slightly increases ({a["recall"]*100:.2f}% to {b["recall"]*100:.2f}%). Thus the post-peak decline is not primarily a continuing collapse of precision/recall.','',
           '## Loss evidence','',
           '| Round | Train box | Local val box | Train classification | Local val classification | Train DFL | Local val DFL |',
           '| --- | ---: | ---: | ---: | ---: | ---: | ---: |']
    for r in losses:
        lines.append('| '+str(r['round'])+' | '+' | '.join(f'{r[k]:.3f}' for k in ['train/box_loss','val/box_loss','train/cls_loss','val/cls_loss','train/dfl_loss','val/dfl_loss'])+' |')
    lines += ['', 'Local validation losses come from individual client fits before server averaging; they are not pooled global-model losses. Training uses augmented images while validation uses unaugmented images, so absolute loss levels are not directly comparable. Their opposing trends under unchanged settings support a generalization problem. Classification validation loss is not monotonically worsening; box/DFL validation losses show the clearer divergence.','',
              '## Classes behind the decline','',
              '| Class | AP50-95 at best round | AP50-95 at round 21 | Change (points) | Validation boxes |',
              '| --- | ---: | ---: | ---: | ---: |']
    for c in changes:
        lines.append(f'| {c["class_name"]} | {100*c["ap_best"]:.2f}% | {100*c["ap_latest"]:.2f}% | {c["delta_pp"]:+.2f} | {c["validation_boxes"]} |')
    lines += ['',f'The four largest declines account for {data["top_four_decline_contribution_pp"]:.2f} points of the net {(a["map_50_95"]-b["map_50_95"])*100:.2f}-point macro AP decline. Small classes have equal weight in macro AP; small validation counts increase sensitivity to prediction changes. Validation images are fixed, so this is not sampling a different validation set every round.','',
              '## Confirmed training design limitations','',
              '- Each client receives the same current global floating tensors; transfer callbacks passed for all 63 completed client fits. All 14,514 training images are represented every global round.',
              '- Every local fit is a fresh one-epoch native trainer. AdamW moments, scheduler, EMA state and local early-stopping history restart each fit. Learned global model weights do continue; the model is not restarting from COCO every round.',
              '- Actual optimizer is AdamW. Auto selects initial LR 0.000357 for 24 classes; the configured lr0=0.01 is ignored. Three-epoch warmup lasts about 1,815 batches, while a one-epoch client fit lasts about 605 batches. The full warmup never finishes. Each fit ends at 0.0001188033 in all parameter groups; this is repeated every round, with no global-round LR decay. A lower nominal lr0 alone would not work while optimizer=auto stays enabled.',
              '- Seed 42 is reset per native fit. This repeats the random initialization of the data pipeline; potentially repeating shuffle/augmentation patterns reduces their diversity. Exact augmented-image equality was not recorded, so it is not asserted.',
              '- Mosaic probability is 1.0 and close_mosaic=10, but the local trainer sees epochs=1. The shutoff condition is local epoch -9 and never fires; the final global rounds do not get the intended unaugmented refinement stage. Persistent mosaic is a plausible contributor to train/validation localization mismatch, not proven to be the dominant cause.',
              '- Native local patience=100 cannot stop a one-epoch fit, and no global early stopping is implemented. Therefore the run can continue after its validation optimum. Best checkpoint selection protects the useful model.',
              '- Floating BatchNorm running statistics are averaged with model weights using the historical method. This can be a federated normalization concern; neither harmful BN mismatch nor client drift is established here. Client class image counts are relatively similar, so extreme class segregation is not a defensible diagnosis.',
              '', '## Dataset and false-positive burden','',
              'At round 21, Banana_sigatoka contributes 960 false positives and Groundnut_early_leaf_spot 335: together 1,295 of 2,668 false positives (48.54%). Their AP50-95 values are 14.33% and 6.16%. This is a persistent performance bottleneck, distinct from the rare classes causing much of the post-peak macro AP decline.',
              '', 'Frozen label statistics show different localization scales: median validation box area is 1.63% of image area for groundnut early leaf spot and 4.78% for banana sigatoka, versus 79.77% for cauliflower healthy. Training groundnut early-leaf-spot boxes are below 1% of image area in 36.61% of cases, versus 19.72% in validation. These differences show scale/domain difficulty; they do not establish incorrect labels. Mosaic can further shrink lesions. Inspect actual annotations before changing labels, resolution or removing a class.',
              '', 'Class imbalance is substantial: cauliflower black rot has 59 training images, versus 1,953 for banana yellow-and-black sigatoka. Every class is represented in all three clients, so wholesale class segregation is absent. The earlier duplicate/leakage audit was enforced; semantic annotation quality remains a separate question.',
              '', 'At rounds 12 and 21, individual local-client AP50-95 values in training CSVs are below the aggregated global model native AP on the corresponding client splits. These observations do not support claiming that averaging is universally damaging this run. Client drift and BN mismatch remain unisolated hypotheses.',
              '', '## Evaluation and integrity checks','',
              '- Same 4,157 images and 7,210 ground-truth boxes every round; micro precision/recall at confidence 0.25, IoU 0.50. AP uses confidence floor 0.001, NMS IoU 0.55 and max 100 predictions. These settings are unchanged. Counts reproduce reported precision/recall exactly.',
              '- Both pooled COCO AP and native client AP decline after the peak; a fixed-threshold precision artifact alone does not explain this. Native precision uses a different aggregation/operating-point definition and must not be substituted into the pooled comparison table.',
              '- Checkpoints 12 and 21 load with weights_only=True, have 24 class names, matching config hashes, finite floating tensors and positive BN variances. Round 21 file SHA matches latest.json. Round 12 SHA is recorded locally; its original expected SHA was not independently available in the metric record.',
              '- Label syntax, class remapping and split-leakage checks were previously passed. This does not certify disease-name correctness, complete annotation coverage or consistent box tightness. Raw all-round predictions and client weight vectors were not retained, so no exact misclassification confusion audit or client-update angle audit was possible.',
              '', '## Recommended action','',
              'Keep the current baseline unchanged if completing the promised 30-round experiment, and choose the best pooled validation checkpoint across all completed rounds. Do not replace round 12 with the latest checkpoint automatically. GPU termination during round 22 cannot explain the earlier round-13-to-21 decline; it only determines what is saved.',
              '', 'For a separate improvement branch starting from the validation-best model: first test a deliberately lower explicit AdamW rate with a global-round schedule and one-time warmup policy, against an unchanged control. Then independently test final-stage mosaic reduction and round-varying deterministic seeds. Inspect boxes and labels in the worst affected classes using validation data. Do not combine all changes at once or promise a performance gain. Test BatchNorm recalibration on training images only if the simpler ablations leave a localization gap. Use exactly the same pooled evaluator; reserve the test split for final reporting.',
              '', '## Sources','',
              '- [Ultralytics 8.4.48 trainer source](https://github.com/ultralytics/ultralytics/blob/v8.4.48/ultralytics/engine/trainer.py): optimizer auto-selection, warmup, mosaic shutoff and final-epoch validation.',
              '- [Ultralytics validation definitions](https://docs.ultralytics.com/modes/val/): native summary precision/recall operating point.',
              '- [SCAFFOLD](https://proceedings.mlr.press/v119/karimireddy20a.html): heterogeneous-client drift is a general risk, not proof for this run.',
              '- [FedBN](https://arxiv.org/abs/2102.07623): normalization under feature-shift heterogeneity is a general risk, not proof for this run.',
              '', '![Validation and loss trends](validation_and_loss_trends.png)', '']
    (OUT/'README.md').write_text('\n'.join(lines),encoding='utf-8')
    print(json.dumps({'report':str(OUT/'README.md'),'checks':checks,'checkpoint_audit':state_audit,'best_round':best['round'],'decline_pp':100*(a['map_50_95']-b['map_50_95'])},indent=2))


if __name__ == '__main__':
    main()
