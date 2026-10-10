# Agri Vision: revised seminar poster

Updated 10 October 2026. A1 portrait, 594 ? 841 mm. The supplied screenshot determines the layout: white header, blue title band, stacked Abstract / Problem Statement / Aim and Objectives / Methodology / Results sections, four crop-image cards, and a two-column Technologies Used / Summary footer.

- [Print-ready A1 PDF](Agri_Vision_Seminar_Poster_A1.pdf)
- [Editable PowerPoint](Agri_Vision_Seminar_Poster_A1.pptx)
- [Preview](Agri_Vision_Seminar_Poster_A1_preview.png)
- [Sources and image provenance](sources.json)
- [Export verification](validation.json)

Team: Hannan Rehman, Fuzail Talpur, Ali. Supervisor: Dr Rabeea Jaffari. University, department and logos were not provided; no institutional branding has been invented. This is a local deliverable; the earlier Canva project has not been edited. Import the PPTX into Canva if needed and check its A1 size, fonts, table and line breaks after import.

## Current project content

The study now covers banana, cauliflower, groundnut and radish across 24 classes, using 14,514 training, 4,157 validation and 2,091 test images. The poster reflects the earlier decision to retain YOLOv8n as the historical round-40 baseline. Since this poster export, the user has approved a fresh 24-class native YOLO run on 19gmail; the training instructions contain the current plan. The exported poster has not yet been revised for that subsequent decision. SSDLite, SSD?MobileNetV3 with a standard head and EfficientDet-D0 are separate one-stage detector experiments, planned to continue toward round 30.

The alternative-model table contains saved best validation metrics: SSDLite and SSD at round 10, EfficientDet at round 2. These are provisional pooled-validation scores, not final held-out test results. The historical YOLO block uses 30-class, validation-image-weighted native client summaries and is deliberately separated from the new results. Its common 24-class evaluation remains pending. The poster does not claim a final architecture winner or comparable historical/new precision definitions.

Four photo cards show real labelled validation examples, not an implemented phone app or model predictions. Yellow rectangles are ground-truth boxes. Photographs are cropped proportionally within editable PPTX frames; displayed boxes are transformed and clipped to those frames. Original image files remain unchanged. The method is a sequential three-client single-GPU simulation, not an already deployed multi-device network or a formal privacy guarantee.

The previous 30-class draft and its builder/sources are preserved under `archive/documents/seminar_poster_30class/`. Run `python reports/seminar_poster/build_poster.py` to regenerate the current editable PPTX. Export using PowerPoint to PDF and check the render before printing.
