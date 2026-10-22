"""Check real inference parity and the local browser workflow; never train."""
import io
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import httpx
import numpy as np
from PIL import Image
import torch
from web_demo.inference import Detector
from web_demo.app import create_app
from fastapi.testclient import TestClient


def main():
    qa = ROOT/'web_demo/qa'
    qa.mkdir(exist_ok=True)
    source = ROOT/'original_dataset_non_chili/valid/images/cordana-557-_jpeg.rf.9227af2fb562a4a8e537be89ee198170.jpg'
    data = source.read_bytes()
    detector = Detector()
    detector.load()
    with TestClient(create_app(detector, background_load=False)) as client:
        response = client.post('/api/predict', content=data, headers={'content-type': 'image/jpeg'})
        assert response.status_code == 200, response.text
        result = response.json()
    with Image.open(io.BytesIO(data)) as image:
        tensor = torch.from_numpy(np.array(image.convert('RGB'), copy=True)).permute(2, 0, 1).float()/255
    raw = detector.adapter.predict([tensor])[0]
    keep = raw['scores'] >= detector.config['display_threshold']
    expected_boxes, expected_scores, expected_labels = raw['boxes'][keep], raw['scores'][keep], raw['labels'][keep]
    assert len(result['detections']) == len(expected_boxes) > 0
    for item, box, score, label in zip(result['detections'], expected_boxes.tolist(), expected_scores.tolist(), expected_labels.tolist()):
        assert item['class_id'] == label
        assert abs(item['confidence']-score) < 1e-6
        assert np.allclose(item['box'], box, atol=1e-4)
    assert not detector.lock.locked()
    report = dict(checkpoint_sha256=detector.config['checkpoint_sha256'],
        verification_image=str(source.relative_to(ROOT)), split='validation',
        actual_api_detections=len(result['detections']), offline_adapter_parity=True,
        inference_ms=result['inference_ms'], processing_ms=result['processing_ms'],
        note='Engineering parity on one validation example, not a model accuracy evaluation')
    # Real browser uses an existing Chromium installation; no browser is downloaded.
    from playwright.sync_api import sync_playwright
    chrome = Path.home()/'AppData/Local/ms-playwright/chromium-1248/chrome-win64/chrome.exe'
    errors = []
    with sync_playwright() as playwright:
        browser = playwright.chromium.launch(executable_path=str(chrome), headless=True)
        page = browser.new_page(viewport={'width': 1440, 'height': 1100}, device_scale_factor=1)
        page.on('pageerror', lambda error: errors.append(str(error)))
        page.goto('http://127.0.0.1:8000', wait_until='networkidle')
        page.get_by_text('Local model ready', exact=True).wait_for()
        page.screenshot(path=str(qa/'desktop_empty.png'), full_page=True)
        assert page.evaluate('document.documentElement.scrollWidth <= innerWidth')
        page.locator('#file-input').set_input_files(str(source))
        page.locator('#detect').wait_for(state='visible')
        with page.expect_response('**/api/predict', timeout=120000) as prediction:
            page.locator('#detect').click()
        assert prediction.value.status == 200
        page.locator('#result-title').filter(has_text='detected').wait_for()
        from playwright.sync_api import expect
        expect(page.locator('#download')).to_be_enabled()
        page.screenshot(path=str(qa/'desktop_result.png'), full_page=True)
        with page.expect_download() as download:
            page.locator('#download').click()
        downloaded = download.value
        downloaded.save_as(str(qa/'annotated_result.png'))
        with Image.open(qa/'annotated_result.png') as image:
            assert image.size == (result['width'], result['height'])
        page.locator('#original-tab').click()
        assert page.locator('#original-tab').get_attribute('aria-pressed') == 'true'
        page.locator('#result-tab').click()
        page.set_viewport_size({'width':390, 'height':844})
        page.screenshot(path=str(qa/'mobile_result.png'), full_page=True)
        assert page.evaluate('document.documentElement.scrollWidth <= innerWidth')
        page.locator('#reset').click()
        assert page.locator('#dropzone').is_visible()
        assert page.locator('#detect').is_disabled()
        assert page.locator('#image-canvas').evaluate('(c)=>c.width') == 1
        page.screenshot(path=str(qa/'mobile_empty.png'), full_page=True)
        browser.close()
    assert not errors, errors
    report.update(browser_errors=errors, desktop_and_mobile_checked=True,
        annotated_download_verified=True, reset_checked=True)
    (qa/'verification.json').write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
