# Agri Vision local web demo

Run from PowerShell in F:\Agri Vision:

```powershell
.\Start_AgriVision_Web.ps1
```

Then open http://127.0.0.1:8000. Keep the terminal open; Ctrl+C stops the server. If PowerShell blocks scripts, use the equivalent command directly:

```powershell
.\.venv\Scripts\python.exe -m uvicorn web_demo.app:app --host 127.0.0.1 --port 8000 --no-access-log
```

The page is local only. Select a JPG, PNG or still WebP under 10 MB and 20 million decoded pixels. Wait for “Local model ready,” then select Detect crop condition. Review the actual predictions, switch between original/detection views, and download the annotated PNG when detections are present. No detection means the model did not return a retained prediction, not that the plant is healthy or the condition is definitely unsupported.

## Current model

The provisional integration model is the existing EfficientDet-D0 24-class baseline checkpoint at saved_models/seminar_demo/efficientdet_d0_round011.pt, input size 512, CPU, four Torch threads. It is not claimed to be the best or final tuned model. The checkpoint and class-map SHA256 and training configuration identity are verified at startup, the state is loaded strictly, and the existing comparison.models adapter handles preprocessing, label offsets and NMS. No pretrained-weight download occurs.

The 0.50 display threshold is provisional, not calibrated rejection of unknown diseases. Confidence is not a guaranteed probability of correctness. All retained predictions are shown, including broad or incorrect boxes; no boxes are manually tightened or selected to imply better performance. No external diagnosis API or treatment advice is used.

Solid-color images with no visible detail are rejected before inference (422). This is an image-quality check, not a crop classifier or reliable unknown-disease detector. A non-uniform unrelated image can still be misclassified by the model.

An available local checkpoint makes this a genuine end-to-end demo while new training continues. Later model replacement requires the corresponding adapter, saved configuration, resolution, class ordering, hash and inference verification. Changing only the filename to another architecture is not supported. The loader deliberately rejects native YOLO until its native preprocessing loader is integrated.

## Local processing and API

The browser sends raw image bytes to POST /api/predict with the actual image Content-Type. This avoids multipart temporary-file spooling; uploaded images are decoded from bounded memory buffers and never saved by the application. The response contains original-oriented dimensions, original-coordinate xyxy boxes, scores, class/crop names, inference time and request-processing time. No user filename is sent. Downloads and drawing happen in browser canvas; object URLs are revoked when replaced/reset. Image objects and request buffers are released after processing; this is not a claim of secure RAM erasure.

GET /api/health reports readiness and the public model information. The model loads once in a background thread and stays in eval/inference mode. The page remains available during loading. Only one prediction request runs at a time; concurrent requests receive 429 rather than building an unbounded queue. Invalid images, byte/pixel overflow, animated images and unsupported formats return explicit errors. Inference failures return a generic message; details go only to the local terminal. No access logs or prediction history are written by the launcher. Bind remains 127.0.0.1 so this launcher does not expose a public service.

The initial local environment was provisioned with web_demo/requirements.txt alongside the existing project's CPU PyTorch, torchvision, effdet, timm, NumPy and Pillow. Install the web additions into an existing compatible inference environment with:

```powershell
.\.venv\Scripts\python.exe -m pip install -r web_demo/requirements.txt
```

For development checks, install httpx. Browser QA also uses the optional playwright package and an already installed Chromium binary. They are not runtime dependencies. No training environment files are edited for the demo. The local Python 3.14 effdet stack emits a TorchScript deprecation warning; the actual CPU inference path is checked separately.

## Scope and privacy

This is inference with a federated-trained global checkpoint, not live federated training. The newer runs use custom sequential FedAvg, so the web page does not claim Flower integration. No training loop or automatic model update runs in this application. All CSS, JavaScript and the SVG mark are served locally without CDNs or analytics. Running a future hosted version would send the image to that server and requires a separate deployment/retention decision.

See WEB_DEMO_PLAN.md for model selection, validation and deployment milestones. Model weights and uploaded photographs are not added to Git.

## Regression checks

Run `python -m unittest discover -s tests -p test_web_demo.py -v` for the 17 isolated API/model-identity tests. With the local server running on port 8000, `python scripts/test_web_demo_browser.py` exercises browser behavior using simulated responses for deterministic edge cases. `python scripts/test_web_demo_real_model.py` checks real-model parity across crop/format/orientation cases without using the web server. `python scripts/verify_web_demo.py` covers the actual end-to-end browser upload and download. Use the project's `.venv/Scripts/python.exe` in place of `python` if needed.

Restart the local server after Python/backend edits; it does not run with automatic reload. Refresh the browser after HTML/CSS/JavaScript edits. See VERIFICATION.md for results and limits.
