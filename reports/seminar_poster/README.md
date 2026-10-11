# Agri Vision seminar materials

This folder contains only the latest seminar deliverables. Older exports, archived poster versions and intermediate page previews were deleted at the user's request on 12 October 2026.

- [A1 poster PDF](Agri_Vision_Seminar_Poster_A1.pdf)
- [Editable poster PowerPoint](Agri_Vision_Seminar_Poster_A1.pptx)
- [Poster preview](Agri_Vision_Seminar_Poster_A1_preview.png)
- [Eight-page detection demonstration](Agri_Vision_Detection_Demonstration.pdf)
- [Sources and verification](SOURCES_AND_VALIDATION.md)

The poster uses the supplied A1 portrait layout, crop greens and Dr. Rabeea Jaffari. Team order is Hannan Rehman on the left, Fuzail Hussain Talpur in the middle and Hasnain Ali Issani on the right. Technology symbols include VS Code. Visible documents omit round numbers, drafting notes and dataset-revision/exclusion wording. Poster crop images are actual verified model predictions.

All three Drive documents were updated in place on 19gmail. The QR links to the publicly viewable [demonstration PDF](https://drive.google.com/file/d/1-KmOBK5M1OJGyhUtk9fB3hayTWpxio0x/view?usp=drivesdk). The [poster PDF](https://drive.google.com/file/d/1eGPppeTQe3e6q8bQ-DXLkCuaSVYIi5XW/view?usp=drivesdk) retains its existing sharing.

`build_poster.py` and `build_detection_demo.py` are the current builders. `poster_layout.py` and `prediction_assets.py` contain their supporting helpers. `assets/` retains only the current prediction pictures, technology symbols, QR, metrics and provenance needed to reproduce the documents. Checkpoint files remain in `saved_models/`.

Regenerate with `.venv/Scripts/python.exe reports/seminar_poster/build_poster.py` or `build_detection_demo.py`. Export the poster PowerPoint to PDF with PowerPoint, then check the render. No model retraining is needed to rebuild the existing documents.
