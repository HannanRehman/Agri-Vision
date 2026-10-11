# FYP-1 presentation versus current Agri Vision scope

Source: `22SW132_X-139_144.pptx`, 23 slides, reviewed 11 October 2026. Text and the embedded architecture/problem/early-results diagrams were inspected. The original presentation was not edited. Current status refers to saved runs verified during this conversation, not a new polling of all active runs.

The core remains multi-crop disease object detection with three-client federated averaging. The research scope has expanded from a YOLO-focused system proposal to a comparison of four one-stage detectors. The implemented deployment scope is currently a reproducible single-GPU simulation with image inference, rather than all the farm/device, communication-security and interface features shown in the proposal diagram.

| Topic | FYP-1 evidence | Current scope |
| --- | --- | --- |
| Title | Slide 1: crop prediction and disease detection | Work implements crop-disease/pest object detection. A separate crop recommendation, yield prediction or soil/weather prediction pipeline is not established; the old objectives do not describe such a pipeline either. |
| Architectures | Slides 8, 14, 15: YOLO/YOLOv8 | YOLOv8n, EfficientDet-D0, SSDLite-MobileNetV3 and standard-head SSD-MobileNetV3. These are separate detector experiments; standalone MobileNetV3 classification and Faster R-CNN are excluded. |
| Dataset | Multi-crop objective; slide 23 lists tomato Roboflow and Mendeley links, without a final crop/class inventory | Frozen non_chili_v1: banana, cauliflower, groundnut and radish; 24 classes; 14,514 train, 4,157 validation and 2,091 test images. The intermediate 30-class historical experiment included chili; the new comparison excludes all six chili classes. Deck references alone do not establish which source images were actually trained. |
| Federation | Slides 12-13 and 16 depict farms/devices and a central federated server; slide 20 lists Flower integration | Flower server/client code exists. Current comparison runs use sequential three-client FedAvg on one Colab GPU per experiment, not a deployed farm network. Different Gmail accounts host different model experiments; they are not the three federated clients. |
| Security/privacy | Architecture image promises SSL/TLS, client/server authentication, no raw-data sharing and privacy | Current code implements logical client partitions and parameter averaging. The inspected Flower entrypoints do not configure TLS/authentication; current Colab data are stored together. Formal privacy, secure aggregation and deployment security have not been demonstrated. Describe privacy as motivation and deployment design, not a proven guarantee. |
| Inference output | Architecture image lists disease name, confidence, bounding box and treatment suggestion | Detection labels, scores and boxes are the current output. No implemented treatment-recommendation module was found. |
| Interface | Architecture image includes phone inference and an admin/researcher dashboard; slide 20 lists interface development as pending | No implemented phone app/admin dashboard was found in the inspected workspace. Current deliverables are code/notebooks, inference outputs, CSV/JSON metrics and reports. A QR-linked image demonstration is planned. |
| Results | Slide 17: three-client preliminary metrics and YOLO prediction examples; slide 18: early training chart | Saved global-round curves and pooled validation metrics for four detectors, per-class analysis, best/latest checkpoints and an explicit final held-out evaluation plan. Old client scores and new pooled scores are not a direct before/after benchmark. |
| Checkpoint selection | Slide 15 ends with final global model generation | Select the validation-best checkpoint, which can precede the final round. Fresh YOLO completed 30 rounds with best at 12; SSDLite completed 30 with best at 30. EfficientDet and SSD-MobileNetV3 continuations are still incomplete. |
| Evaluation | Precision, recall, mAP50 and mAP50-95 in slide 20 | Same metric families, with declared pooled P/R at confidence 0.25/IoU 0.5 and COCO AP. Native client metrics are labelled separately. Held-out test results and consistent speed/size profiling are still pending for the new comparison. |
| Research gap | Slides 6-8 emphasize missing YOLO + federation | Reframe around comparative detection quality and resource tradeoffs under a shared multi-crop FedAvg protocol. Recheck original literature before asserting novelty or claiming other methods cannot localize symptoms. |
| Timeline | Slide 20 contains the original November 2025-October 2026 roadmap and statuses | Update around actual milestones: data freeze, pilots, completed/incomplete training, final test evaluation, profiling, demonstration, poster and thesis writing after 17 October. |

## What stayed the same

Crop-disease localization, multi-crop support, three logical clients, local training, sharing model parameters, sample-weighted FedAvg, global model generation, and precision/recall/AP evaluation remain central. Comparing more models extends that foundation.

## Priority changes for the next presentation/poster

1. Use a title such as **Agri Vision: Federated Multi-Crop Disease Detection Using One-Stage Object Detectors**.
2. State four crops/24 classes and the chili exclusion with the dataset audit; replace dataset links with the actual authoritative sources used.
3. Show the four detectors as separate experiments inside the same federated process; show three simulated clients, not three Gmail accounts as clients.
4. Label deployment/security/real-time/treatment features according to implemented versus proposed status. Measured inference latency is needed before making a quantitative real-time claim.
5. Present validation-selected checkpoint results, completed-round budgets and separate native/pooled definitions. After final selection, evaluate all candidates on the same test set and profile speed/model size under common conditions.
6. Replace early graphs and client tables with saved comparison curves and clearly labelled prediction examples. Add the requested QR link to a public image-demonstration PDF once inference images are generated.
7. Update the timeline and literature review. These changes have not yet been applied to the original PPT.

## Presentation identity details

Slide 2 spells the team as Fuzail Hussain Talpur (22SW132), Hasnain Ali Issani (22SW144), and Hannan Rehman (22-21SW139), and the supervisor as Dr. Rabia Jaffari. Earlier poster instructions used shorter names and Dr Rabeea Jaffari. Confirm official names/spelling before replacing identity information on final artifacts.
