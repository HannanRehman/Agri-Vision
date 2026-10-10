"""Read-only feature and parameter drift audit on saved YOLO checkpoints."""
import json
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import torch
from comparison.data import CropDataset
from comparison.models import create_adapter
from comparison.runner import state_hash
torch.set_num_threads(4)
run=ROOT/'experiments/yolo_v8n_non_chili_seed42_20261009_224152_449800'
out=ROOT/'reports/comparison/yolo_diagnosis_20261010'
audit=json.loads((out/'audit.json').read_text())
dataset=CropDataset(ROOT,'valid')
ids=[audit['inference_sample'][i]['image_id'] for i in [0,6,12,18]]
index={row['image_id']:i for i,row in enumerate(dataset.rows)}
images=[dataset[index[i]][0] for i in ids]
adapter=create_adapter('yolo_v8n','cpu',ROOT,pretrained=False)
initial=torch.load(run/'checkpoints/initial.pt',map_location='cpu',weights_only=True)['model_state']
reports=[]
for number in [0,1,8,10]:
    path=run/('checkpoints/initial.pt' if number==0 else f'checkpoints/global_round_{number:03d}.pt')
    state=torch.load(path,map_location='cpu',weights_only=True)['model_state']
    adapter.load(state); adapter.net.eval(); before=state_hash(adapter.state())
    stages=[]; hooks=[]
    def hook(name):
        def collect(module,args,value):
            if not isinstance(value,torch.Tensor) or value.ndim!=4: return
            v=value.detach().float()
            stages.append(dict(stage=name,mean=float(v.mean()),std=float(v.std()),max_abs=float(v.abs().max()),
                               inter_image_std=float(v.std(dim=0).mean()),rms=float(v.square().mean().sqrt())))
        return collect
    for name,module in adapter.net.model.named_children():
        hooks.append(module.register_forward_hook(hook('model.'+name)))
    for scale,branch in enumerate(adapter.net.model[-1].cv3):
        for position,module in enumerate(branch):
            hooks.append(module.register_forward_hook(hook(f'classification.{scale}.{position}')))
    with torch.inference_mode(): raw=adapter.net(adapter.batch(images))[0]
    for h in hooks:h.remove()
    group_drift={}
    for group in ['backbone','neck','classification','regression']:
        chosen=[]
        for name,p in adapter.net.named_parameters():
            layer=int(name.split('.')[1])
            actual='classification' if name.startswith('model.22.cv3.') else 'regression' if name.startswith('model.22.') else 'backbone' if layer<10 else 'neck'
            if actual==group:chosen.append(name)
        original=sum(float(initial[k].double().square().sum()) for k in chosen)
        drift=sum(float((state[k].double()-initial[k].double()).square().sum()) for k in chosen)
        group_drift[group]=dict(relative_l2_drift=(drift/original)**.5 if original else None,parameters=sum(state[k].numel() for k in chosen))
    reports.append(dict(round=number,stages=stages,parameter_drift=group_drift,
                        raw_score_max=float(raw[:,4:].max()),
                        raw_score_inter_image_std=float(raw[:,4:].std(dim=0).mean()),
                        inference_changed_state=state_hash(adapter.state())!=before))
    print('Activation audit round',number,flush=True)
(out/'activations.json').write_text(json.dumps(dict(training_performed=False,image_ids=ids,checkpoints=reports),indent=2))
print(json.dumps([dict(round=x['round'],drift=x['parameter_drift'],raw_score_max=x['raw_score_max'],raw_score_inter_image_std=x['raw_score_inter_image_std'],selected_stages=[s for s in x['stages'] if s['stage'] in ['model.0','model.9','model.21','classification.0.0','classification.0.1','classification.0.2']]) for x in reports]))
