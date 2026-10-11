# Current seminar materials — updated 12 October 2026

These current files supersede the 10 October poster. Older local exports and archived poster versions were removed on 12 October 2026; current sources and assets were retained.

- Poster: `Agri_Vision_Seminar_Poster_A1.pdf`
- Editable poster: `Agri_Vision_Seminar_Poster_A1.pptx`
- Poster preview: `Agri_Vision_Seminar_Poster_A1_preview.png`
- Eight-page picture booklet: `Agri_Vision_Detection_Demonstration.pdf`
- Poster sources: `sources.json`
- Prediction coordinates, checkpoint hashes and frozen metrics: `assets/predictions.json`
- Export checks: `validation.json`
- Flower plan: `../comparison/FLOWER_INTEGRATION_PLAN.md`

The poster preserves the supplied A1 portrait stacked layout. The user confirmed crop greens, Dr. Rabeea Jaffari and hosting on 19gmail. Both documents use the confirmed green palette and supervisor spelling. Full names and roll numbers follow slide 2 of `22SW132_X-139_144.pptx`, including `22SW132(GL)` and `22-21SW139`.

The comparison uses fresh 24-class YOLO results, replacing the historical 30-class block. Selected checkpoints are YOLOv8n round 12, SSDLite round 30, SSD–MobileNetV3 round 30 and EfficientDet-D0 round 10. These details are retained here and in the prediction provenance for reproducibility; the poster and booklet omit round labels as requested. YOLO and both SSD variants completed 30 rounds. EfficientDet has 10 saved rounds toward its target of 15. Metrics are common pooled validation results, not final held-out test results or classification accuracy.

The poster's crop cards now show actual predictions, replacing the earlier annotation pictures. A deterministic candidate search screened 139 validation photographs, including additional cauliflower downy-mildew and groundnut rust candidates. Each chosen poster example has the correct class, IoU at least 0.5 and no extra detection above the common display threshold. Selection ranks localization overlap times confidence across the four detectors; all four strongest poster examples happened to be YOLOv8n outputs. This is a selection of strong illustrations, not a representative performance sample. No box, label or score was fabricated. Original photographs and labels remain unchanged. Green confidence labels describe individual detections, not overall accuracy.

The eight-page booklet contains an introduction, four crop comparison pages, a metric table, a selected-detection gallery and a methodology/future-work page. Each comparison page gives every detector the same photograph and retains the actual returned outputs; empty outputs are labelled as no detections above threshold. Completed runs use custom sample-weighted federated averaging; Flower integration remains planned. Team order is Hannan Rehman on the left, Fuzail Hussain Talpur in the middle and Hasnain Ali Issani on the right. Technology names in the footer were replaced by publisher-sourced symbols for Python, PyTorch, Ultralytics, Colab, Google Drive and VS Code. Editing notes and review-draft text were removed from visible documents. Icon sources are recorded under `assets/technology_symbols/sources.json`.

## Drive copies and QR access

- Demonstration PDF: https://drive.google.com/file/d/1-KmOBK5M1OJGyhUtk9fB3hayTWpxio0x/view?usp=drivesdk
- Poster PDF: https://drive.google.com/file/d/1eGPppeTQe3e6q8bQ-DXLkCuaSVYIi5XW/view?usp=drivesdk
- Editable poster file ID: `1vJwTLhQ6tfjGdKY6yw8Wh-iRRuv61kga`
- Verified parent for all three: 19gmail's Agri Vision folder `18ath0ZIrihYILUQz-jKvIv9uvyXfj905`.

The real QR points to the uploaded demonstration PDF. The existing three Drive files were updated in place, preserving their IDs and the QR destination. Drive metadata verifies the replacements and Anyone/reader permission on the booklet. An anonymous download returned HTTP 200 and matched the updated local PDF's SHA256 exactly. The poster and editable file retain their original private sharing. The booklet is about 3.93 MB after high-quality JPEG embedding; original PNG prediction assets remain available locally. No extra public folder or training-file sharing was created.

Verification: eight booklet pages; one A1 poster page (593.98 × 840.99 mm); no poster shapes outside the slide; zero PowerPoint text overflow after adding the QR. The rendered green poster and booklet pages were inspected. No training or final test-set evaluation was performed to create these documents. Saved checkpoints were downloaded and used for CPU inference only.
