"""Build the current A1 seminar poster with configurable theme, supervisor and QR."""
import argparse
import json
from pathlib import Path

parser=argparse.ArgumentParser()
parser.add_argument('--theme',choices=['blue','green'],default='green')
parser.add_argument('--supervisor',default='Dr. Rabeea Jaffari')
parser.add_argument('--qr-url',default='https://drive.google.com/file/d/1-KmOBK5M1OJGyhUtk9fB3hayTWpxio0x/view?usp=drivesdk')
args=parser.parse_args()
HERE=Path(__file__).resolve().parent
# Reuse the original poster's editable shapes and reference-layout helpers.
template=(HERE/'poster_layout.py').read_text(encoding='utf-8')
exec(compile(template.split('# White institutional-header area:')[0],str(HERE/'poster_layout.py'),'exec'))
original_text=text
def text(value,x,y,w,h,size=27,color=INK,bold=False,align=PP_ALIGN.LEFT):
    # PowerPoint requires space for actual line height and paragraph margins.
    h=max(h,size*1.2/72+.035)
    return original_text(value,x,y,w,h,size,color,bold,align)
if args.theme=='green':
    BLUE='165D40';CYAN='62B890';LINE='D2E5D9'
DATA=json.loads((OUT/'assets/predictions.json').read_text(encoding='utf-8'))

text('FINAL YEAR PROJECT  /  SEMINAR',1.05,.48,9.1,.48,25,BLUE,True)
text('Multi-crop disease detection with federated averaging',1.05,1.1,11.2,.65,25,MUTED)
for i,name in enumerate(['BANANA','CAULIFLOWER','GROUNDNUT','RADISH']):
    x=12.25+i*2.47;leaf(x+.82,.39,.6)
    text(name,x,1.27,2.42,.35,16,GREEN,True,PP_ALIGN.CENTER)
shape(0,2.0,W,2.03,BLUE)
text('AGRI VISION',1.0,2.08,W-2,1.27,73,WHITE,True,PP_ALIGN.CENTER)
text('Federated multi-crop disease detection using one-stage object detectors',1.0,3.49,W-2,.45,25,WHITE,False,PP_ALIGN.CENTER)
team=[('Hannan Rehman • 22-21SW139',PP_ALIGN.LEFT),
      ('Fuzail Hussain Talpur • 22SW132(GL)',PP_ALIGN.CENTER),
      ('Hasnain Ali Issani • 22SW144',PP_ALIGN.RIGHT)]
for i,(person,alignment) in enumerate(team):
    text(person,1.0+i*(W-2)/3,4.15,(W-2)/3,.43,23,BLUE,True,alignment)
text('Supervisor: '+args.supervisor,1.0,4.61,W-2,.29,20,MUTED,False,PP_ALIGN.CENTER)

section('ABSTRACT',5.0,2.85);plant(1.45,5.67,.87)
text('Agri Vision studies crop disease and pest-symptom detection using bounding boxes and class labels. '
     'The dataset covers banana, cauliflower, groundnut and radish across 24 classes. '
     'Four one-stage detectors—YOLOv8n, SSDLite, SSD–MobileNetV3 and EfficientDet-D0—are trained through '
     'separate three-client federated-averaging simulations and assessed with a shared pooled evaluator.',
     4.0,5.82,17.75,1.82,26)

section('PROBLEM STATEMENT',8.10,2.55)
text('• Small lesions and similar symptoms make visual diagnosis difficult.\n'
     '• Uneven class representation and variable annotation quality affect detection.\n'
     '• A useful detector must balance disease recognition, localisation and computing cost.',
     1.35,8.89,19.8,1.51,26)

section('AIM AND OBJECTIVES',10.90,2.70);plant(1.4,11.53,.76)
text('Assess lightweight one-stage detectors for federated crop-symptom detection.\n'
     '• Use fixed 24-class train, validation and held-out test splits.\n'
     '• Compare four complete detectors through independent experiments.\n'
     '• Report precision, recall and AP; measure inference speed and model size.',
     4.0,11.63,17.75,1.92,25)

section('METHODOLOGY',13.85,5.30)
text('One shared dataset  →  fixed three-client partitions  →  separate detector experiments',
     1.1,14.58,W-2.2,.46,25,MUTED,False,PP_ALIGN.CENTER)
for i in range(3):
    x=2.0+i*7.0
    shape(x,15.25,5.3,.88,PALE,LINE,MSO_SHAPE.ROUNDED_RECTANGLE)
    text(f'CLIENT {i+1}  •  LOCAL TRAINING',x+.15,15.45,5.0,.45,24,GREEN,True,PP_ALIGN.CENTER)
    arrow(x+2.46,16.17)
shape(2.0,16.62,19.3,.67,BLUE,None,MSO_SHAPE.ROUNDED_RECTANGLE)
text('TRAINING-IMAGE-WEIGHTED FEDAVG  →  GLOBAL DETECTOR',2.25,16.73,18.8,.45,25,WHITE,True,PP_ALIGN.CENTER)
for i,name in enumerate(['YOLOv8n','SSDLite','SSD–MobileNetV3','EfficientDet-D0']):
    x=1.4+i*5.35
    shape(x,17.72,5.05,.64,PALE,LINE,MSO_SHAPE.ROUNDED_RECTANGLE)
    text(name,x+.08,17.82,4.89,.43,23,GREEN,True,PP_ALIGN.CENTER)
text('Local training  →  aggregate  →  pooled validation  →  save checkpoints  →  repeat',
     1.25,18.42,W-2.5,.45,24,BLUE,True,PP_ALIGN.CENTER)
text('Three logical clients simulated sequentially. Current runs use custom FedAvg; Flower integration is planned.',
     1.3,18.91,W-2.6,.23,16,MUTED,False,PP_ALIGN.CENTER)

section('RESULTS',19.45,8.85)
text('DETECTION PERFORMANCE  •  VALIDATION',1.25,20.26,11.8,.53,23,GREEN,True)
headers=['Detector','Precision','Recall','mAP50','mAP50–95']
widths=[3.70,1.73,1.73,1.75,2.21]
x0,y0=1.25,21.05
xx=x0
for title,width in zip(headers,widths):
    shape(xx,y0,width,.62,BLUE)
    text(title,xx+.04,y0+.16,width-.08,.36,20,WHITE,True,PP_ALIGN.LEFT if title=='Detector' else PP_ALIGN.CENTER)
    xx+=width
for i,m in enumerate(DATA['models'].values()):
    row=m['metrics'];values=[m['label'],
        *[f"{float(row[k])*100:.2f}%" for k in ['precision','recall','map50','map_50_95']]]
    yy=y0+.62+i*.72;xx=x0
    for j,(value,width) in enumerate(zip(values,widths)):
        shape(xx,yy,width,.72,PALE if i%2==0 else 'F6FAFC',WHITE)
        text(value,xx+.04,yy+.22,width-.08,.36,20,INK,j==0,PP_ALIGN.LEFT if j==0 else PP_ALIGN.CENTER)
        xx+=width
text('Each detector is evaluated on the same 24-class validation split. '
     'The comparison considers correct disease recognition, missed symptoms and bounding-box localisation.',
     1.25,24.75,11.12,1.18,23,MUTED)
shape(1.25,26.10,11.12,1.72,PALE,None,MSO_SHAPE.ROUNDED_RECTANGLE)
text('Same 4,157-image validation split. P/R: confidence 0.25, IoU 0.50. '
     'Selected model performance is reported using detection metrics. '
     'Held-out test evaluation and deployment profiling are planned.',
     1.48,26.31,10.65,1.26,22,MUTED)

# Use actual inference outputs; the strongest verified detector is selected per crop.
for i,item in enumerate(DATA['poster_examples']):
    x=13.2+(i%2)*4.55;y=20.54+(i//2)*3.77
    shape(x,y,3.91,3.50,INK,None,MSO_SHAPE.ROUNDED_RECTANGLE)
    shape(x+.10,y+.10,3.71,3.30,WHITE,None,MSO_SHAPE.ROUNDED_RECTANGLE)
    shape(x+1.48,y+.13,.95,.055,INK,None,MSO_SHAPE.ROUNDED_RECTANGLE)
    path=OUT/item['poster_prediction_path']
    with Image.open(path) as im:ratio=im.width/im.height
    pw,ph=3.49,2.26
    if ratio>pw/ph:ph=pw/ratio
    else:pw=ph*ratio
    s.shapes.add_picture(str(path),Inches(x+.21+(3.49-pw)/2),Inches(y+.36+(2.26-ph)/2),width=Inches(pw),height=Inches(ph))
    text(item['crop'],x+.16,y+2.65,3.59,.39,20,GREEN,True,PP_ALIGN.CENTER)
    text(DATA['models'][item['poster_model']]['label'],x+.16,y+3.09,3.59,.30,17,MUTED,False,PP_ALIGN.CENTER)
text('Selected model predictions • disease labels and confidence',12.75,28.00,9.2,.25,16,MUTED,False,PP_ALIGN.CENTER)

shape(.7,28.65,W-1.4,.025,CYAN)
text('TECHNOLOGIES USED',1.05,28.91,8.7,.5,25,BLUE,True)
for i,name in enumerate(['python','pytorch','ultralytics','colab','drive','vscode']):
    path=OUT/'assets/technology_symbols'/f'{name}.png'
    with Image.open(path) as icon:ratio=icon.width/icon.height
    iw,ih=1.65,1.12
    if ratio>iw/ih:ih=iw/ratio
    else:iw=ih*ratio
    x=1.65+(i%3)*2.78;y=29.64+(i//3)*1.45
    picture=s.shapes.add_picture(str(path),Inches(x+(1.65-iw)/2),Inches(y+(1.12-ih)/2),width=Inches(iw),height=Inches(ih))
    picture.name=name+' technology symbol'
shape(9.9,29.10,.02,2.96,LINE)
text('SUMMARY',10.3,28.91,8.6,.5,25,BLUE,True)
text('Agri Vision combines multi-crop disease localisation with three-client federated averaging. '
     'Four one-stage detectors are compared on a shared dataset. '
     'The linked picture demonstration presents actual predictions across all four crops.',10.3,29.63,8.1,2.39,23)

if args.qr_url:
    import qrcode
    qr=qrcode.QRCode(error_correction=qrcode.constants.ERROR_CORRECT_M,box_size=16,border=4)
    qr.add_data(args.qr_url);qr.make(fit=True)
    qr.make_image(fill_color='black',back_color='white').save(OUT/'assets/demo_qr.png')
    s.shapes.add_picture(str(OUT/'assets/demo_qr.png'),Inches(19.05),Inches(29.20),width=Inches(2.75),height=Inches(2.75))
    text('SCAN TO VIEW\nDETECTION EXAMPLES',18.9,31.99,3.05,.68,17,GREEN,True,PP_ALIGN.CENTER)
else:
    shape(19.10,29.37,2.7,2.7,PALE,LINE,MSO_SHAPE.ROUNDED_RECTANGLE)
    text('QR SPACE\n\nAwaiting approved\nPDF viewing link',19.23,29.70,2.43,1.98,21,GREEN,True,PP_ALIGN.CENTER)
    text('Review draft',19.08,32.17,2.74,.29,17,MUTED,False,PP_ALIGN.CENTER)
text('DATA: 14,514 train / 4,157 validation / 2,091 test  •  24 classes across four crops',
     1.05,32.35,17.5,.36,19,BLUE,True)
text('Three simulated clients  •  Four one-stage detectors  •  Multi-crop disease localisation',
     1.05,32.78,W-2.1,.30,15,MUTED,False,PP_ALIGN.CENTER)

prs.core_properties.title='Agri Vision — multi-crop disease detection'
prs.core_properties.author='Hannan Rehman; Fuzail Hussain Talpur; Hasnain Ali Issani'
prs.core_properties.last_modified_by='Agri Vision Project Team'
prs.core_properties.subject='A1 seminar poster; federated crop disease detection'
prs.save(OUT/'Agri_Vision_Seminar_Poster_A1.pptx')
sources=dict(format='A1 portrait, 594 × 841 mm',snapshot='2026-10-12',
    layout_reference='C:/Users/Hussaini Enterprise/Pictures/Screenshots/Screenshot 2026-10-09 115847.png',
    identities_source='22SW132_X-139_144.pptx, slide 2',supervisor=args.supervisor,
    theme=args.theme,theme_pending_confirmation=False,qr_url=args.qr_url,
    qr_status='QR linked and decoded; public Drive access must be verified' if args.qr_url else 'pending PDF upload; no fake QR generated',
    metrics_source='assets/predictions.json',examples_are_annotations=False,
    genuine_model_predictions='Agri_Vision_Detection_Demonstration.pdf',
    flower_status='planned integration; completed comparison uses custom FedAvg',
    validation_not_test=True,institutional_logos='not supplied')
(OUT/'sources.json').write_text(json.dumps(sources,indent=2,ensure_ascii=False)+'\n',encoding='utf-8')
print(OUT/'Agri_Vision_Seminar_Poster_A1.pptx')
