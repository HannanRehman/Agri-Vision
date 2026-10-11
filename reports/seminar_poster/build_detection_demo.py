"""Eight-page local review PDF with genuine saved-model inference illustrations."""
import json
from io import BytesIO
from pathlib import Path
from PIL import Image
from reportlab.pdfgen import canvas
from reportlab.lib.colors import HexColor
from reportlab.lib.pagesizes import A4
from reportlab.lib.utils import ImageReader
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont

ROOT=Path(__file__).resolve().parents[2]
OUT=Path(__file__).resolve().parent
ASSETS=OUT/'assets'
DATA=json.loads((ASSETS/'predictions.json').read_text(encoding='utf-8'))
for family,file in [('Aptos','aptos.ttf'),('AptosBold','aptos-bold.ttf')]:
    path=Path('C:/Windows/Fonts')/file
    if not path.exists():path=Path('C:/Windows/Fonts')/('arialbd.ttf' if family.endswith('Bold') else 'arial.ttf')
    pdfmetrics.registerFont(TTFont(family,str(path)))
W,H=A4
BLUE='#165D40';GREEN='#226D4A';INK='#243747';MUTED='#526779';PALE='#EEF7F3'
DEST=OUT/'Agri_Vision_Detection_Demonstration.pdf'
c=canvas.Canvas(str(DEST),pagesize=A4)
c.setTitle('Agri Vision | Crop disease detection demonstration')
c.setAuthor('Hannan Rehman; Fuzail Hussain Talpur; Hasnain Ali Issani')
c.setCreator('Agri Vision Project Team')

def label(t,x,y,size=12,color=INK,bold=False):
    c.setFillColor(HexColor(color));c.setFont('AptosBold' if bold else 'Aptos',size);c.drawString(x,y,t)

def paragraph(t,x,y,width,size=11,color=INK,leading=None):
    leading=leading or size*1.45
    for line in t.split('\n'):
        words=line.split();current=''
        for word in words:
            trial=(current+' '+word).strip()
            if pdfmetrics.stringWidth(trial,'Aptos',size)>width and current:
                label(current,x,y,size,color);y-=leading;current=word
            else:current=trial
        if current:label(current,x,y,size,color);y-=leading
        y-=3
    return y

IMAGE_CACHE={}
def image_fit(path,x,y,w,h):
    key=str(path)
    if key not in IMAGE_CACHE:
        with Image.open(path) as original:
            if original.mode in ('RGBA','LA'):
                IMAGE_CACHE[key]=ImageReader(str(path))
            else:
                encoded=BytesIO()
                original.convert('RGB').save(encoded,format='JPEG',quality=97,subsampling=0,optimize=True)
                encoded.seek(0);IMAGE_CACHE[key]=ImageReader(encoded)
    img=IMAGE_CACHE[key];iw,ih=img.getSize();scale=min(w/iw,h/ih)
    c.drawImage(img,x+(w-iw*scale)/2,y+(h-ih*scale)/2,iw*scale,ih*scale,mask='auto')

def frame(page,title,subtitle):
    c.setFillColor(HexColor(BLUE));c.rect(0,H-91,W,91,fill=1,stroke=0)
    label('AGRI VISION',36,H-33,20,'#FFFFFF',True)
    label(title,36,H-57,13,'#FFFFFF',True)
    label(subtitle,36,H-76,9,'#FFFFFF')
    c.setStrokeColor(HexColor('#D8E8ED'));c.line(36,48,W-36,48)
    label('Crop disease detection | Validation examples',36,31,8,MUTED)
    label(f'{page} / 8',W-63,31,8,MUTED)

def card(mid,item,x,y,w=242,h=249):
    c.setFillColor(HexColor(PALE));c.roundRect(x,y,w,h,7,fill=1,stroke=0)
    m=DATA['models'][mid]
    label(m['label'],x+10,y+h-20,11,GREEN,True)
    caption='Disease labels, bounding boxes and confidence' if item['predictions'][mid]['boxes'] else 'No detections above the display threshold'
    label(caption,x+10,y+h-35,8,MUTED)
    image_fit(ASSETS/'images'/f'{item["crop"]}_{mid}.png',x+8,y+12,w-16,h-55)

def finish():c.showPage()

frame(1,'Four crops. Four detectors. One shared evaluation split.','A visual demonstration of multi-crop disease detection')
label('Crop disease detection, explained through pictures',36,711,19,GREEN,True)
y=paragraph('Agri Vision detects crop symptoms using a disease label, a confidence score and a bounding box. This booklet presents selected detections and compares four one-stage detectors on shared crop photographs.',36,681,W-72,12)
for i,item in enumerate(DATA['poster_examples']):
    x=36+i*132
    image_fit(OUT/item['poster_prediction_path'],x,475,124,124)
    label(item['crop'],x,458,11,GREEN,True)
y=paragraph('How to read the pictures\nGreen boxes and labels are predictions at confidence 0.25 or higher. Each label identifies a disease or symptom class and the confidence of that detection. The original image on each comparison page is shown as the input reference.',36,416,W-72,11)
y=paragraph('The pictures are selected examples of detector behaviour. Overall performance is measured on the complete validation split, rather than estimated from these illustrations. A confidence score belongs to one detection; it does not describe overall model accuracy.',36,y-8,W-72,11)
label('Prepared by',36,239,12,BLUE,True)
for person,roll,x in [('Hannan Rehman','22-21SW139',36),('Fuzail Hussain Talpur','22SW132(GL)',212),('Hasnain Ali Issani','22SW144',402)]:
    label(person,x,218,10,INK,True);label(roll,x,198,10,MUTED)
label('Supervisor: Dr. Rabeea Jaffari',36,151,10,MUTED)
paragraph('A final-year project on federated crop disease detection, combining disease recognition, symptom localisation and comparative detector evaluation.',36,119,W-72,10,MUTED)
finish()

for page,item in enumerate(DATA['examples'],2):
    frame(page,item['crop']+' | Same photograph, four detectors',item['disease'].replace('_',' ').title())
    image_fit(ROOT/item['image'],36,625,107,107)
    label('Original photograph',36,609,9,MUTED)
    label('Dataset annotation',160,715,12,GREEN,True)
    paragraph(DATA['names'][str(item['ground_truth']['labels'][0])].replace('_',' '),160,692,W-196,12)
    paragraph('Every detector receives the same photograph. The panels display the returned boxes, predicted labels and confidence scores.',160,653,W-196,10,MUTED)
    for j,mid in enumerate(DATA['models']):
        card(mid,item,36+(j%2)*281,326-(j//2)*263)
    finish()

frame(6,'Detection performance','Validation results across the shared 24-class dataset')
label('24 classes across banana, cauliflower, groundnut and radish',36,710,13,GREEN,True)
paragraph('Shared data: 14,514 training images, 4,157 validation images and 2,091 held-out test images across four crops and 24 classes.',36,684,W-72,11)
headers=['Detector','Precision','Recall','mAP50','mAP50–95']
xs=[40,246,326,402,480]
c.setFillColor(HexColor(BLUE));c.roundRect(36,600,W-72,34,4,fill=1,stroke=0)
for x,t in zip(xs,headers):label(t,x,613,9,'#FFFFFF',True)
for i,m in enumerate(DATA['models'].values()):
    yy=561-i*52;c.setFillColor(HexColor(PALE));c.rect(36,yy-13,W-72,45,fill=1,stroke=0)
    r=m['metrics'];vals=[m['label'],*[f"{float(r[k])*100:.2f}%" for k in ['precision','recall','map50','map_50_95']]]
    for x,t in zip(xs,vals):label(t,x,yy,9,INK,x==40)
y=paragraph('Precision measures how many returned detections match the annotated objects. Recall measures how many annotated objects are detected. A useful model should balance both, while localising the affected region accurately.',36,323,W-72,11)
y=paragraph('Precision and recall use the common pooled evaluator at confidence 0.25 and IoU 0.50. AP averages across classes; mAP50–95 tests stricter localization across IoU thresholds 0.50–0.95. Classification accuracy is not substituted for detection AP.',36,y-10,W-72,11)
y=paragraph('The table reports selected-model validation performance. Held-out test evaluation and consistent inference-speed measurements are planned. The detectors use different input resolutions and architecture-specific training settings.',36,y-10,W-72,11)
finish()

frame(7,'Selected crop detections','Strong verified examples across the four crops')
paragraph('Each example below has the correct annotated disease label and a predicted box that overlaps the annotation. Selection considers localisation overlap and confidence.',36,709,W-72,11)
for i,item in enumerate(DATA['poster_examples']):
    x=36+(i%2)*281;y=359-(i//2)*276
    c.setFillColor(HexColor(PALE));c.roundRect(x,y,242,257,7,fill=1,stroke=0)
    label(item['crop'],x+10,y+237,12,GREEN,True)
    label(DATA['models'][item['poster_model']]['label'],x+10,y+220,9,MUTED)
    image_fit(OUT/item['poster_prediction_path'],x+8,y+12,226,199)
finish()

frame(8,'Methodology and future work','Federated training, detection outputs and deployment evaluation')
blocks=[('Shared dataset','The study covers banana, cauliflower, groundnut and radish across 24 classes. Fixed training, validation and test splits support consistent evaluation across the detector experiments.'),
('Federated averaging','Three logical clients train locally on separate data partitions. The global model is formed by weighting client updates by their training-image counts. The implementation currently uses custom FedAvg; Flower integration is planned.'),
('Detection outputs','Each detector returns a disease or symptom class, a confidence score and a bounding box. Images in this booklet use actual saved-model inference, with a common display threshold of 0.25.'),
('Next research steps','Evaluate the selected models on the held-out test set, profile inference speed and model size under common hardware conditions, and assess the trade-off between detection quality and deployment cost.')]
y=713
for title,body in blocks:
    label(title,36,y,13,GREEN,True);y=paragraph(body,36,y-23,W-72,11)-19
label('Technologies used',36,252,13,GREEN,True)
for i,key in enumerate(['python','pytorch','ultralytics','colab','drive','vscode']):
    image_fit(ASSETS/'technology_symbols'/f'{key}.png',36+i*88,146,64,64)
finish()
c.save()
print(DEST)
