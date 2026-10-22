"""Real-checkpoint upload/parity checks across crops and image formats."""
import csv
import io
import json
from pathlib import Path
import statistics
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import numpy as np
from PIL import Image
import torch
from fastapi.testclient import TestClient
from web_demo.app import create_app, decode_image
from web_demo.inference import Detector


def main():
    service=Detector(); service.load()
    examples=json.loads((ROOT/'reports/seminar_poster/assets/predictions.json').read_text())['examples']
    paths={x['crop']:ROOT/x['image'] for x in examples if '/valid/' in x['image']}
    with (ROOT/'manifests/non_chili_v1/valid.csv').open(newline='') as stream:
        row=next(row for row in csv.DictReader(stream) if 'Bacterial-Spot_' in row['image'])
    paths['Cauliflower']=ROOT/row['image']
    cases=[]
    for crop in ['Banana','Cauliflower','Groundnut','Radish']:
        with Image.open(paths[crop]) as image:
            cases.append((crop,'image/jpeg',paths[crop].read_bytes()))
            if crop=='Banana':
                for fmt,media in [('PNG','image/png'),('WEBP','image/webp')]:
                    output=io.BytesIO();image.save(output,fmt);cases.append((fmt,media,output.getvalue()))
                output=io.BytesIO();exif=Image.Exif();exif[274]=6
                image.resize((480,320)).save(output,'JPEG',exif=exif)
                cases.append(('EXIF rotated','image/jpeg',output.getvalue()))
    blank=io.BytesIO();Image.new('RGB',(640,640),'white').save(blank,'PNG')
    cases.append(('Blank image: rejection not guaranteed','image/png',blank.getvalue()))
    records=[]
    with TestClient(create_app(service,background_load=False)) as client:
        info=client.get('/api/health').json()
        assert len(info['classes'])==24
        assert [c['name'] for c in info['classes']]==service.names
        adapter_id=id(service.adapter)
        for name,media,data in cases:
            response=client.post('/api/predict',content=data,headers={'content-type':media})
            if name.startswith('Blank image'):
                assert response.status_code==422,response.text
                assert not service.lock.locked()
                records.append(dict(case=name,quality_rejected=True,status_code=422))
                continue
            assert response.status_code==200,response.text
            result=response.json()
            image=decode_image(data,service.config['max_pixels'])
            try:
                tensor=torch.from_numpy(np.array(image,copy=True)).permute(2,0,1).float()/255
                raw=service.adapter.predict([tensor])[0]
                keep=raw['scores']>=service.config['display_threshold']
                boxes,scores,labels=raw['boxes'][keep].tolist(),raw['scores'][keep].tolist(),raw['labels'][keep].tolist()
                assert len(result['detections'])==len(boxes)
                assert (result['width'],result['height'])==image.size
                if name=='EXIF rotated':assert image.size==(320,480)
                for detection,box,score,label in zip(result['detections'],boxes,scores,labels):
                    assert detection['class_id']==label
                    assert detection['class_name']==service.names[label]
                    assert abs(detection['confidence']-score)<1e-6
                    assert np.allclose(detection['box'],box,atol=1e-4)
                assert id(service.adapter)==adapter_id
                assert not service.lock.locked()
                records.append(dict(case=name,retained_detections=len(boxes),
                    inference_ms=result['inference_ms'],processing_ms=result['processing_ms'],adapter_parity=True))
            finally:image.close()
    report=dict(cases=records,total_cases=len(records),classes_verified=24,
        median_processing_ms=statistics.median(x['processing_ms'] for x in records if 'processing_ms' in x),
        minimum_processing_ms=min(x['processing_ms'] for x in records if 'processing_ms' in x),
        maximum_processing_ms=max(x['processing_ms'] for x in records if 'processing_ms' in x),
        model_reused=True,checkpoint_sha256=service.config['checkpoint_sha256'],
        scope='Local CPU, warm model; four validation images plus format/orientation/blank variants. Not accuracy or OOD evaluation.')
    (ROOT/'web_demo/qa/real_model_regression.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2))


if __name__=='__main__':main()
