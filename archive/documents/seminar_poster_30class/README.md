# Agri Vision seminar poster

> Historical 30-class draft. Its two-column layout and proposed Faster R-CNN future-work text are superseded. A revision to the supplied stacked-section template is pending; use the [current comparison plan](../../ARCHITECTURE_COMPARISON_PLAN.md) for approved models. Do not present this draft as the new 24-class comparison results.

**Size:** A1 portrait, 594 × 841 mm. Team: Hannan Rehman, Fuzail Talpur, Ali. Supervisor: Dr Rabeea Jaffari.

- `Agri_Vision_Seminar_Poster_A1.pptx`: editable PowerPoint source. Upload this file to Canva using its presentation import, then check the page dimensions, fonts, chart and line breaks before printing. Canva may change the layout during import.
- `Agri_Vision_Seminar_Poster_A1.pdf`: print-ready, single-page A1 PDF.
- `Agri_Vision_Seminar_Poster_A1_preview.png`: poster preview.
- `build_poster.py`: reproducible poster builder using python-pptx.
- `sources.json`: exact local metric files and dataset photograph paths used.

The supplied Canva edit URL could not be accessed with the available tools. This poster is an independent design and has not been uploaded to or edited inside that Canva project. University, department, date and logos were not supplied, so they have not been invented.

## Evidence and interpretation

The four headline metrics are the round-35 held-out test macro averages across all 30 classes. The line chart uses client-weighted validation metrics from rounds 26–40. These are different evaluation sets and should not be compared as if they were the same run. Best-round statements refer only to the reviewed rounds 26–40. Faster R-CNN is future comparison work, not a completed experiment. The federated implementation is a sequential three-client simulation on a single GPU, not an already deployed multi-device network or a formal privacy guarantee.

The crop photographs are examples from the labelled test dataset, not predictions. The header is a decorative AI-generated illustration, not experimental evidence.

## Illustration provenance

Mode: built-in `image_gen.imagegen`, new-image generation, opaque background. Output copied into `assets/crop_illustration.png`.

Prompt:

> Use case: illustration-story. Asset type: an illustration for the header of an academic crop-disease detection poster called Agri Vision. Create a wide landscape botanical illustration of agricultural crop leaves: a broad banana leaf, a chilli plant with a small red chilli, cauliflower foliage, paired groundnut leaflets, and radish leaves with a glimpse of the radish root. Show some leaves with subtle brown disease spots and some healthy leaves, without assigning specific diagnoses. Elegant detailed botanical gouache and soft photographic texture, natural daylight, deep forest greens, sage, warm ivory, small earthy amber accents. Crop foliage concentrated across the lower and right portions with generous light ivory negative space at the upper left. Clean professional scientific-poster visual, calm and realistic rather than cartoonish. Wide 3:2 landscape composition with no frame, no text, no labels, no numbers, no bounding boxes, no logo, no watermark. This is an illustrative botanical asset, not a model prediction or empirical result.
