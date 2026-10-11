"""Select strong real predictions from a deterministic validation candidate pool."""
import csv
import hashlib
import json
import random
import sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
sys.path.insert(0,str(Path(__file__).resolve().parent))
import torch
from torchvision.ops import box_iou
from comparison.data import CropDataset
from comparison.models import create_adapter
from scripts.yolo_native_train import create_yolo,NativePredictor
from prediction_assets import MODELS,CROPS,json_prediction,visual,OUT

torch.set_num_threads(4)

def quality(pred,gt):
    if not pred['boxes']:
        return dict(tp=0,fp=0,fn=1,iou=0.,confidence=0.,clean=False)
    b=torch.tensor(pred['boxes']); labels=torch.tensor(pred['labels'])
    overlaps=box_iou(b,torch.tensor(gt['boxes']))[:,0]
    overlaps[labels!=gt['labels'][0]]=0
    best=int(overlaps.argmax());iou=float(overlaps[best])
    tp=int(iou>=.5)
    return dict(tp=tp,fp=len(b)-tp,fn=1-tp,iou=iou,
                confidence=pred['scores'][best] if iou>0 else 0.,clean=bool(tp and len(b)==1))

def main():
    ds=CropDataset(ROOT,'valid');rng=random.Random(42)
    cache=OUT/'selection_candidates.json'
    candidates=json.loads(cache.read_text()) if cache.exists() else []
    labels=[]
    for i,row in enumerate(ds.rows):
        ann=[line.split() for line in (ROOT/row['label']).read_text().splitlines() if line.strip()]
        if len(ann)==1:labels.append((i,int(ann[0][0])))
    searches=CROPS if not candidates else [('Cauliflower','downy_mildew'),('Groundnut','rust')]
    seen={c['key'] for c in candidates}
    for crop,disease in searches:
        cls=next(k for k,v in ds.names.items() if v.lower()==f'{crop}_{disease}'.lower())
        pool=[i for i,l in labels if l==cls and ds.rows[i]['key'] not in seen]
        selected=sorted(rng.sample(pool,min(24,len(pool))))
        for i in selected:
            tensor,target=ds[i]
            candidates.append(dict(crop=crop,disease=disease,index=i,image=ds.rows[i]['image'],
                label=ds.rows[i]['label'],key=ds.rows[i]['key'],
                ground_truth={k:target[k].tolist() for k in ('boxes','labels')},predictions={},quality={}))
        print(crop,'validation candidates:',len(selected),'of',len(pool),flush=True)
    provenance=dict(snapshot='2026-10-12',split='validation',confidence=.25,
        illustrations_not_test_evaluation=True,
        selection='Deterministic single-object validation candidate samples (seed 42), including additional cauliflower downy-mildew and groundnut rust examples. Prefer correct class and IoU >= 0.5 without extra predictions across models. Poster examples maximize correct single-box overlap times confidence across the reviewed candidate pool; booklet comparison images favor consistency across all models. Selected illustrations are not representative performance estimates.',
        models={},names=ds.names)
    for mid,(label,path,csv_name) in MODELS.items():
        saved=torch.load(path,map_location='cpu',weights_only=False)
        assert {int(k):v for k,v in saved['names'].items()}==ds.names
        if mid=='yolo_v8n':
            model=create_yolo(ROOT,ds.names);model.model.load_state_dict(saved['model_state'],strict=True)
            adapter=NativePredictor(model,'cpu')
        else:
            adapter=create_adapter(mid,'cpu',ROOT,pretrained=False);adapter.load(saved['model_state'])
        missing=[item for item in candidates if mid not in item['predictions']]
        for start in range(0,len(missing),4):
            batch=missing[start:start+4]
            tensors=[ds[item['index']][0] for item in batch]
            with torch.inference_mode():outputs=adapter.predict(tensors)
            for item,p in zip(batch,outputs):
                pred=json_prediction(p);item['predictions'][mid]=pred
                item['quality'][mid]=quality(pred,item['ground_truth'])
            if start%24==0:print(label,'screened additional',min(start+4,len(missing)),'/',len(missing),flush=True)
        with (OUT/'metrics'/f'{csv_name}.csv').open(newline='',encoding='utf-8') as f:rows=list(csv.DictReader(f))
        metric=next(r for r in rows if int(r['round'])==saved['round'])
        provenance['models'][mid]=dict(label=label,checkpoint=str(path.relative_to(ROOT)),
            sha256=hashlib.sha256(path.read_bytes()).hexdigest(),round=saved['round'],
            completed_rounds=max(int(r['round']) for r in rows),metrics=metric,config_hash=saved.get('config_hash'))
        del adapter
    chosen=[];poster_examples=[]
    for crop,_ in CROPS:
        pool=[c for c in candidates if c['crop']==crop]
        def rank(item):
            qs=list(item['quality'].values())
            return (sum(q['clean'] for q in qs),sum(q['tp'] for q in qs),
                    min(q['iou']*q['confidence'] for q in qs),sum(q['iou']*q['confidence'] for q in qs))
        item=max(pool,key=rank)
        # Display the strongest correct single-object prediction on the poster.
        mids=[mid for mid,q in item['quality'].items() if q['tp']]
        assert mids,'No correct detection available for '+crop
        item['poster_model']=max(mids,key=lambda mid:(item['quality'][mid]['clean'],item['quality'][mid]['iou']*item['quality'][mid]['confidence']))
        item['poster_prediction_path']=f"assets/images/{crop}_{item['poster_model']}.png"
        visual(ROOT/item['image'],item['ground_truth'],ds.names,OUT/'images'/f'{crop}_annotation.png',True)
        for mid,p in item['predictions'].items():visual(ROOT/item['image'],p,ds.names,OUT/'images'/f'{crop}_{mid}.png')
        chosen.append(item)
        print('SELECTED',crop,item['poster_model'],item['quality'],flush=True)
        eligible=[(c,mid) for c in pool for mid,q in c['quality'].items() if q['clean']]
        assert eligible
        best,best_mid=max(eligible,key=lambda cm:cm[0]['quality'][cm[1]]['iou']*cm[0]['quality'][cm[1]]['confidence'])
        poster=json.loads(json.dumps(best));poster['poster_model']=best_mid
        poster['poster_prediction_path']=f'assets/images/{crop}_poster.png'
        visual(ROOT/poster['image'],poster['predictions'][best_mid],ds.names,OUT/'images'/f'{crop}_poster.png')
        poster_examples.append(poster)
        print('POSTER',crop,best_mid,poster['quality'][best_mid],flush=True)
    provenance['examples']=chosen
    provenance['poster_examples']=poster_examples
    (OUT/'selection_candidates.json').write_text(json.dumps(candidates,indent=2)+'\n',encoding='utf-8')
    (OUT/'predictions.json').write_text(json.dumps(provenance,indent=2)+'\n',encoding='utf-8')
    print('Verified selection saved.',flush=True)

if __name__=='__main__':main()
