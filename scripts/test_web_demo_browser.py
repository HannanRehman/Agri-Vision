"""Browser regression scenarios against a running local demo; no model training."""
import io
import json
from pathlib import Path
import statistics
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from PIL import Image
from playwright.sync_api import sync_playwright, expect


def main():
    qa=ROOT/'web_demo/qa'
    qa.mkdir(exist_ok=True)
    image=ROOT/'original_dataset_non_chili/valid/images/cordana-557-_jpeg.rf.9227af2fb562a4a8e537be89ee198170.jpg'
    errors=[]; console_errors=[]; external=[]; passed=[]
    result=dict(status='detected',width=640,height=640,inference_ms=300,processing_ms=350,
        detections=[dict(class_id=1,crop='Banana',condition='Cordana',confidence=.9,box=[10,10,200,200]),
                    dict(class_id=2,crop='Banana',condition='Healthy',confidence=.8,box=[300,300,600,600])])
    with sync_playwright() as p:
        chrome=Path.home()/'AppData/Local/ms-playwright/chromium-1248/chrome-win64/chrome.exe'
        browser=p.chromium.launch(executable_path=str(chrome),headless=True)
        context=browser.new_context(viewport={'width':1440,'height':1000},reduced_motion='reduce')
        page=context.new_page()
        page.on('pageerror',lambda e:errors.append(str(e)))
        page.on('console',lambda m:console_errors.append(m.text) if m.type=='error' else None)
        page.on('request',lambda r:external.append(r.url) if not r.url.startswith(('http://127.0.0.1:8000/','blob:','data:')) else None)
        page.goto('http://127.0.0.1:8000',wait_until='networkidle')
        expect(page.locator('#connection-text')).to_have_text('Local model ready')
        expect(page.locator('#detect')).to_be_disabled()
        assert not page.locator('#intro-screen').is_visible()
        page.locator('#dropzone').focus()
        with page.expect_file_chooser() as chooser:
            page.keyboard.press('Enter')
        chooser.value.set_files(str(image))
        expect(page.locator('#detect')).to_be_enabled()
        passed.append('keyboard_upload_and_initial_controls')
        page.route('**/api/predict',lambda route:route.fulfill(json=result))
        page.locator('#detect').click()
        expect(page.locator('.detection')).to_have_count(2)
        expect(page.locator('#download')).to_be_enabled()
        pixels=page.locator('#image-canvas').evaluate('(c)=>c.toDataURL()')
        page.locator('#original-tab').click()
        original=page.locator('#image-canvas').evaluate('(c)=>c.toDataURL()')
        assert original!=pixels
        # Download must stay annotated even when original view is selected.
        with page.expect_download() as saved:
            page.locator('#download').click()
        saved.value.save_as(str(qa/'browser_download.png'))
        with Image.open(qa/'browser_download.png') as annotated:
            assert annotated.size==(640,640)
        passed.append('multiple_detections_healthy_label_view_toggle_and_download')
        # Slow request exposes pending-state controls; this response is mocked
        # only to make error and timing scenarios deterministic.
        page.unroute('**/api/predict')
        page.route('**/api/predict',lambda route:route.fulfill(json=result))
        page.locator('#detect').click()
        expect(page.locator('#result-state')).to_have_text('Complete')
        page.unroute('**/api/predict')
        # Hold a request with an async route handler, observe UI, then release.
        captured=[]
        page.route('**/api/predict',lambda route:captured.append(route))
        page.locator('#detect').click()
        expect(page.locator('#processing-overlay')).to_be_visible()
        expect(page.locator('#detect')).to_be_disabled()
        expect(page.locator('#reset')).to_be_disabled()
        expect(page.locator('#replace')).to_be_disabled()
        assert page.locator('#original-tab').get_attribute('aria-pressed')=='true'
        assert page.locator('#image-canvas').evaluate('(c)=>c.toDataURL()')==original
        expect(page.locator('#download')).to_be_disabled()
        assert captured
        captured[0].fulfill(json=result)
        expect(page.locator('#result-state')).to_have_text('Complete')
        passed.append('pending_state_blocks_duplicate_actions_and_removes_old_boxes')
        page.unroute('**/api/predict')
        empty={**result,'status':'no_reliable_detection','detections':[]}
        page.route('**/api/predict',lambda route:route.fulfill(json=empty))
        page.locator('#detect').click()
        expect(page.locator('#result-title')).to_have_text('No reliable detection')
        expect(page.locator('#detections')).to_be_empty()
        expect(page.locator('#download')).to_be_disabled()
        assert 'not establish' in page.locator('#result-message').inner_text()
        passed.append('empty_detection_message_and_no_stale_download')
        page.unroute('**/api/predict')
        def error_handler(code):
            return lambda route: route.fulfill(status=code,json={'detail':f'Expected test response {code}'})
        for code in [400,413,415,422,429,500,503]:
            page.route('**/api/predict',error_handler(code))
            page.locator('#detect').click()
            expect(page.locator('#error')).to_have_text(f'Expected test response {code}')
            expect(page.locator('#detect')).to_be_enabled()
            expect(page.locator('#processing-overlay')).to_be_hidden()
            page.unroute('**/api/predict')
        passed.append('seven_server_error_states_allow_retry')
        page.route('**/api/predict',lambda route:route.fulfill(status=502,content_type='text/html',body='<h1>Gateway error</h1>'))
        page.locator('#detect').click()
        expect(page.locator('#error')).to_contain_text('unreadable response')
        expect(page.locator('#detect')).to_be_enabled()
        page.unroute('**/api/predict')
        passed.append('non_json_server_failure_has_readable_message')
        page.route('**/api/predict',lambda route:route.abort('failed'))
        page.locator('#detect').click()
        expect(page.locator('#error')).to_contain_text('could not be reached')
        page.unroute('**/api/predict')
        passed.append('network_failure_allows_retry')
        # Invalid uploads must not replace the last valid preview.
        page.locator('#file-input').set_input_files({'name':'not-a-photo.svg','mimeType':'image/svg+xml','buffer':b'<svg/>'})
        expect(page.locator('#error')).to_contain_text('Choose a JPG')
        page.locator('#file-input').set_input_files({'name':'broken.png','mimeType':'image/png','buffer':b'broken'})
        expect(page.locator('#error')).to_contain_text('could not be read')
        page.locator('#file-input').set_input_files({'name':'large.jpg','mimeType':'image/jpeg','buffer':b'x'*(10*1024*1024+1)})
        expect(page.locator('#error')).to_contain_text('smaller than 10 MB')
        expect(page.locator('#filename')).to_have_text(image.name)
        passed.append('client_rejects_invalid_corrupt_and_oversized_files')
        # Same-file replacement, then reset and drag/drop.
        with page.expect_file_chooser() as chooser:
            page.locator('#replace').click()
        chooser.value.set_files(str(image))
        expect(page.locator('#error')).to_be_hidden()
        page.locator('#reset').click()
        expect(page.locator('#dropzone')).to_be_visible()
        expect(page.locator('#detect')).to_be_disabled()
        assert page.locator('#image-canvas').evaluate('(c)=>c.width')==1
        page.locator('#dropzone').evaluate('''async (node)=>{const r=await fetch('/static/leaf.svg');const data=new DataTransfer();data.items.add(new File([await r.blob()],'not-image.svg',{type:'image/svg+xml'}));node.dispatchEvent(new DragEvent('drop',{bubbles:true,dataTransfer:data}));}''')
        expect(page.locator('#error')).to_contain_text('Choose a JPG')
        passed.append('same_file_reselection_reset_and_drop_validation')
        # A portrait image must preserve dimensions and never overflow a viewport.
        output=io.BytesIO();Image.new('RGB',(400,900),'green').save(output,'PNG')
        page.locator('#file-input').set_input_files({'name':'portrait.png','mimeType':'image/png','buffer':output.getvalue()})
        expect(page.locator('#dimensions')).to_have_text('400 × 900 px')
        for width in [320,390,768,1024,1440,1920]:
            page.set_viewport_size({'width':width,'height':900})
            assert page.evaluate('()=>document.documentElement.scrollWidth<=innerWidth'),width
        passed.append('portrait_preview_and_six_responsive_widths')
        # EXIF orientation in the browser must match the backend's normalisation.
        exif=Image.Exif();exif[274]=6
        output=io.BytesIO();Image.new('RGB',(80,40),'green').save(output,'JPEG',exif=exif)
        page.locator('#file-input').set_input_files({'name':'rotated.jpg','mimeType':'image/jpeg','buffer':output.getvalue()})
        expect(page.locator('#dimensions')).to_have_text('40 × 80 px')
        page.locator('summary').click()
        expect(page.locator('#class-list li')).to_have_count(24)
        assert 'Provisional' in page.locator('#model-info').inner_text()
        passed.append('browser_exif_orientation_and_all_24_condition_labels')
        context.close()
        ctx=browser.new_context(reduced_motion='reduce')
        tab=ctx.new_page()
        held=[]
        tab.route('**/api/health',lambda route:held.append(route))
        tab.goto('http://127.0.0.1:8000',wait_until='domcontentloaded')
        expect(tab.locator('#connection-text')).to_have_text('Service unreachable',timeout=8000)
        expect(tab.locator('#detect')).to_be_disabled()
        for route in held:
            try:
                route.abort('aborted')
            except Exception:
                pass  # The browser may already have canceled the timed-out request.
        tab.unroute('**/api/health')
        expect(tab.locator('#connection-text')).to_have_text('Local model ready',timeout=8000)
        ctx.close()
        passed.append('health_timeout_recovers_without_page_reload')
        # Intro is CSS only; disabling JS still reveals the page after its exit.
        for javascript,motion in [(True,'no-preference'),(False,'no-preference'),(True,'reduce')]:
            ctx=browser.new_context(java_script_enabled=javascript,reduced_motion=motion)
            tab=ctx.new_page();tab.goto('http://127.0.0.1:8000',wait_until='domcontentloaded')
            if motion=='reduce':expect(tab.locator('#intro-screen')).to_be_hidden()
            else:expect(tab.locator('#intro-screen')).to_be_hidden(timeout=4500)
            ctx.close()
        passed.append('intro_exit_reduced_motion_and_no_javascript_visibility_fallback')
        browser.close()
    # Expected failed-request console messages are emitted by Chromium, not page JS.
    unexpected_console=[e for e in console_errors if not ('Failed to load resource' in e or 'net::ERR_FAILED' in e)]
    assert not errors, errors
    assert not external,external
    assert not unexpected_console,unexpected_console
    report=dict(groups_passed=passed,groups_count=len(passed),page_errors=errors,
        unexpected_console_errors=unexpected_console,external_requests=external,
        simulated_prediction_responses=True,
        note='Deterministic browser state checks; real model parity is verified separately')
    (qa/'browser_regression.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2))


if __name__=='__main__':
    main()
