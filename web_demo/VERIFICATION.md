# Initial local verification — 23 October 2026

## Expanded regression check

After the initial verification below, 17 API/model-identity tests passed, covering byte and pixel limits (including chunked requests without Content-Length), corrupt files and actual file formats, grayscale/alpha/JPG/PNG/WebP, animated-image rejection, EXIF rotation, empty predictions, repeated requests, real request concurrency and recovery, loading/unavailable/failed-model states, generic internal-error responses and lock cleanup, model hash/class-ID guards, and the new solid-image quality gate.

Thirteen browser scenario groups passed: keyboard upload; multiple detections and healthy labels; original/detection view and download; disabled pending controls and old-box cleanup; empty detections and stale-download cleanup; seven HTTP errors and retry; non-JSON and network failures; invalid/corrupt/oversized uploads; same-file reselection, drag/drop and reset; portrait layout at 320/390/768/1024/1440/1920 px; EXIF orientation and all 24 displayed conditions; health-request timeout and recovery; intro exit with normal motion, reduced motion and JavaScript disabled. Browser scenarios simulate prediction responses to exercise edge cases deterministically. There were no page errors, unexpected console errors or requests to external origins. This is Chromium viewport testing, not physical-device or Safari/Firefox certification.

The separate real-model suite checked four validation crop photographs plus PNG, WebP and EXIF variants. All seven valid inference cases matched the offline adapter's labels, scores and boxes; all 24 public class names matched the checkpoint's ordering. The same adapter instance was reused across requests. One cauliflower photograph returned no detections at the provisional 0.50 threshold; this is an observed model limitation, not an integration failure.

A solid white image initially produced a false detection. The application now rejects exactly uniform RGB images before inference with a readable no-visible-detail message. The eighth real suite case confirms 422 for that input. This gate does not solve unknown-disease or unrelated-image rejection. Non-uniform unfamiliar images can still produce false positives. No threshold tuning, retraining or model swapping was performed to hide that limitation.

In the final warm-CPU seven-request check, request-processing times ranged from 390.8 to 471.3 ms, with median 422.5 ms. This is a small functional sample with a warm model; it excludes cold startup and does not establish representative latency or model accuracy.

Fixes made during the expanded check: remove old boxes immediately when starting a new request; allow selecting the same file again; bound health requests to five seconds so the readiness indicator can recover; provide readable non-JSON service errors; announce result-state changes for assistive technology; make the upload label the keyboard entry point; reject solid images before inference. Regression checks were rerun after these changes. The existing user-started server was left running; restart it to load changed Python routes, then refresh the page.

Raw reports remain locally under web_demo/qa/: browser_regression.json, real_model_regression.json and verification.json. Test scripts are retained under scripts/ and tests/. No uploaded images were added to source control, and no training or online deployment was started.

A temporary live server on port 8001 loaded the updated Python code. A real HTTP request and browser upload both confirmed the blank-image 422 message; a subsequent real banana image returned its detection and enabled download. The temporary server was stopped afterward. The existing port-8000 server was preserved.

## Earlier initial check

Eight API tests passed: unsupported/corrupt images, byte and pixel bounds, empty results, EXIF orientation, busy/unavailable service, repeated requests, error cleanup and local page assets/security headers. Tests use a stub detector to isolate request behavior; they do not establish detection accuracy.

A separate real CPU inference check loaded the verified EfficientDet baseline checkpoint (SHA256 86f4fc50bb5ad26d7e8c8791215a36a2d225b807c5f8b4d93b1cec0f29398db4) and used a frozen banana validation image. The API returned one detection. Its class ID, score and box matched the existing offline adapter within 1e-6 score and 1e-4 coordinate tolerance. The request used the full image; no bounding box was altered for display.

The first recorded warm-model check took 481.6 ms for model inference and 509.8 ms for request processing on the local CPU. This is one engineering example, not representative latency or an accuracy assessment.

Headless Chromium checked desktop (1440 px) and mobile (390 px) layouts, real upload/inference, original/detection toggles, annotated PNG download and image dimensions, reset behavior and horizontal overflow. No JavaScript page errors were recorded. Screenshots and the machine-readable result are retained locally under ignored web_demo/qa/. They include a validation example and are not new evaluation metrics. No held-out test images were used.

JavaScript syntax checking and pip dependency consistency passed. Real-model final selection, validation-calibrated display thresholds, a wider unfamiliar-image assessment and representative device latency profiling remain pending. No new training or online deployment was started.

The intro uses CSS opacity/transform animations with a 2.1-second total sequence; it does not delay page scripts or model initialization. Chromium checks at 1440 px and 390 px confirmed removal after the animation, no JavaScript errors, no horizontal overflow, and immediate visual suppression under prefers-reduced-motion. Frame sampling found one initial interval above 50 ms per run, with initial long tasks of 174 ms (desktop) and 134 ms (mobile viewport); this is not a claim of zero dropped frames or a physical-phone benchmark. The remaining sampled frames stayed below the 50 ms threshold. The intro has a CSS visibility fallback if JavaScript is unavailable.
