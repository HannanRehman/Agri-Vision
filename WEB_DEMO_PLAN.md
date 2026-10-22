# Agri Vision: single-page detection demo plan

Prepared 23 October 2026. Planning only; no application, deployment or training has been started by this document.

## Intended result

One crop-themed page accepts an image, runs one verified project checkpoint, and returns the actual detector output: bounding boxes, crop/class labels and confidence scores. The user can download the annotated image. The interface supports banana, cauliflower, groundnut and radish, using the frozen 24-class mapping. These classes include healthy plants, pests and a nutrient-deficiency category as well as diseases, so use “crop condition” where a generic label is needed.

Initial scope: one image per request, JPG/PNG/WebP, image preview, Detect button, processing state, results and Reset. No camera/video input, account system, model selector, treatment recommendations or automatic retraining is required for the first demo.

## Model choice and identity

Finish the tuned runs, then compare their validation-selected best checkpoints against the completed baseline checkpoints using the same pooled validation protocol. Prefer detection quality, including recall and class-level weaknesses, while measuring batch-one latency and RAM on the actual demo machine. Do not claim the fastest or most accurate deployed model before these measurements.

EfficientDet-D0 is the provisional quality-first candidate: the completed baseline's best pooled validation mAP50-95 is 59.75% and recall is 77.36%. These are historical validation results, not the final tuned winner or a guarantee on uploaded photographs. If the tuned runs do not improve quality, deploying the stronger baseline checkpoint remains valid. If the quality leader is too slow for the available hardware, document the accuracy/latency tradeoff before choosing another detector.

Freeze a deployment manifest containing architecture, experiment identity, checkpoint SHA256, dataset class-map SHA256, inference input resolution, preprocessing, NMS settings, display threshold policy and dependency versions. Preserve original checkpoint provenance separately from the page's user-facing content. Verify all 24 class IDs and their order before serving any prediction.

The checkpoints are project state dictionaries, not necessarily native Ultralytics model files. Do not pass an arbitrary project .pt directly to YOLO(path). Do not reuse the historical predict.py unchanged: it references the older dataset/model loader. Use the appropriate training-compatible adapter, including native YOLO preprocessing if YOLO wins. Reconstruct the selected architecture at its selected resolution, load weights strictly and disable pretrained downloads during application startup.

## What the page should show

Top: Agri Vision, a brief explanation and the four supported crops. Main card: drag-and-drop/upload, supported formats and a preview. Main action: Detect crop condition. During processing, disable duplicate submissions and show a progress indicator without inventing a completion percentage.

Result area: original image and annotated image, each actual retained detection's readable class name and confidence, and measured processing time with a clear definition. Support multiple detections and multiple classes in one photograph. Provide Download result and Upload another image. Use the model's boxes without manually shrinking them to look like lesion-level segmentation. Crop/leaf-level boxes remain possible where learned from the annotations. Show confidence as a model score, not “percentage certainty” or application accuracy.

Do not show a successful inference as a verified diagnosis: the server has no ground truth for a user upload. A successful result means that the model produced retained predictions. A missing detection does not establish a healthy plant; healthy is a separate learned class.

## Unrecognized images and error behavior

1. Invalid or corrupt image, unsupported format, oversized file or excessive decoded pixel count: show a clear input error before inference. Proposed initial limits are 10 MB and 20 million decoded pixels, adjustable after measuring memory. Check decoded image contents as well as MIME type and filename.
2. Valid image with no predictions above the selected threshold: return a normal empty result and display “No reliable detection. Try a clearer image of a supported crop. The condition may be outside the model's 24 classes.” Do not label this definitively as unknown disease or healthy.
3. One or more retained detections: render actual outputs. Low-confidence predictions can be withheld; multiple overlapping conflicting labels require a documented display policy consistent with the detector's class-aware NMS behavior.
4. Model unavailable or inference failure: show a processing error and allow retry. Do not expose stack traces or filesystem paths in the page.

The current detectors are closed-set. They can give high confidence to an unseen disease or an unrelated image. A confidence threshold is not a reliable unknown-class detector. Start with honest no-reliable-detection behavior and test unfamiliar crops, unseen conditions and unrelated images. If the supervisor requires reliable “outside supported classes” recognition, treat that as an additional evaluated open-set/OOD component or a separately trained rejection stage. Even then, report measured limitations. Do not add an arbitrary 25th output class without data and training.

Choose display thresholds on validation data to balance missed detections and false positives; do not assume the evaluation score floor of 0.001 is appropriate for the page. Consider per-class thresholds only if there is enough validation evidence. Fix the policy before final held-out assessment. Preserve study metrics computed with the original evaluator; a demo threshold is not a reason to rewrite comparison results.

## Implementation

Use one Python FastAPI service to serve the page and accept POST /api/predict image uploads. A simple HTML/CSS/JavaScript frontend is sufficient for the single page; no separate frontend hosting or framework is required. FastAPI UploadFile supports multipart image uploads: https://fastapi.tiangolo.com/tutorial/request-files/.

Load the model once at startup. Run inference in evaluation mode with torch.inference_mode(); evaluation mode remains necessary for BatchNorm/dropout behavior: https://docs.pytorch.org/docs/stable/generated/torch.autograd.grad_mode.inference_mode.html. Use a bounded inference queue or one active inference at a time initially, with image and pixel limits to control memory.

Decode image, apply EXIF orientation, convert to RGB and run the exact model preprocessing. Preserve the transform needed to map predictions to the displayed original dimensions. Retain class-ID mapping, score filtering and NMS consistently; do not apply an accidental second suppression pass. Return status, detections with original-coordinate xyxy boxes, scores, original dimensions and timing. Render the annotated image using those outputs and provide download in the browser. A canvas overlay with labels avoids persistent result-image storage.

Suggested application structure:

    web_demo/
      app.py                 # page and prediction route
      inference.py           # selected checkpoint loader and adapter
      config.json            # deployment model identity/settings
      static/index.html
      static/styles.css
      static/app.js
      README.md              # installation, startup and demo instructions

Keep application dependencies separate from the training environment. Store downloaded model weights locally in an ignored artifacts directory; do not commit them or the datasets. Freeze only the dependencies needed by the selected architecture. Record GPU/CPU details with the measured latency.

## Privacy and demonstration

Start with a local service on the presentation laptop and open it in the laptop's browser. Requests and image processing stay on that machine if the app makes no external calls; pre-cache model assets and use local fonts/scripts. Uploaded images are not added to training and are not retained in prediction logs. UploadFile may use temporary disk storage, so close temporary files in a finally block and do not claim that no bytes ever touch disk. Avoid recording image bytes or filenames in application logs.

The federated concept describes how the checkpoint was trained. It does not automatically make inference private, and a hosted upload service transmits the image to its server. If online access is requested later, explain server processing and retention honestly, use HTTPS, limit uploads and requests, and validate the chosen host's memory/CPU/GPU constraints. Browser-only inference could keep image processing on the user's device, but requires a separately verified export/runtime path and is not the initial plan.

If the local demo is shown to a supervisor, identify it as inference using the federated-trained global checkpoint. It is not live federated training or evidence that Flower was used in the newer custom FedAvg runs. No additional training is necessary merely to add the web interface.

## Build sequence and review gates

1. Lock the intended demo scope and prepare a verified inference loader using an available 24-class checkpoint. Model selection can remain provisional while tuning completes; use the same interface to swap the final verified artifact later.
2. Verify predictions on a small fixed set against the existing offline adapter: class IDs, scores, boxes, EXIF orientation and image dimensions. This resolves checkpoint-loading errors before styling the page.
3. Build the upload page, result rendering, downloadable annotation and the empty/error states. Keep the layout green and crop-focused, readable on a laptop and phone.
4. Validate input bounds and request cleanup. Check corrupted files, extreme dimensions, two sequential uploads, unsupported images and a request that produces no detections.
5. Choose the score policy on validation data. Assess class-level failures and unfamiliar-image behavior. Use saved training images only if clearly identified as training examples; do not present them as generalization evidence.
6. Measure end-to-end and inference-only latency on the intended demo device after warm-up. Choose CPU or GPU from measured behavior; do not promise a frame rate from Colab training times.
7. Swap in the final model and freeze its manifest. Conduct the planned held-out assessment without tuning on those results. Prepare real demo photographs and a short explanation of supported classes, confidence, limitations and federated training versus inference.
8. Provide one startup command, environment instructions and an offline demonstration checklist. Online publishing is a later, separate decision once the local application is reviewed.

Indicative effort: approximately three to four focused development days, depending on checkpoint portability and measured local inference speed. Work on the interface and loader can begin while search jobs finish. No extra training round is a prerequisite for the demo.

## Completion criteria

A genuine model checkpoint processes a valid upload; rendered boxes match the actual model outputs on the correctly oriented image; all 24 labels map correctly; an empty prediction has an honest message; invalid images fail cleanly; uploaded images are not retained after processing; repeated requests do not reload the model; result download works; artifact identity and measured demo latency are documented; no unknown-disease guarantee is claimed without separate evidence.
