"""Request limits, orientation, empty detections and concurrency without a GPU."""
import io
import threading
from concurrent.futures import ThreadPoolExecutor
import unittest

from fastapi.testclient import TestClient
from PIL import Image, ImageDraw
from web_demo.app import create_app


class FakeDetector:
    def __init__(self):
        self.ready = True
        self.failed = False
        self.lock = threading.Lock()
        self.config = dict(max_upload_bytes=1024, max_pixels=10000)
        self.calls = 0

    def public_info(self):
        return dict(ready=self.ready, state='ready')

    def predict(self, image):
        self.calls += 1
        return dict(status='no_reliable_detection', detections=[], width=image.width,
                    height=image.height, inference_ms=1)


def picture(size=(20, 10), exif=None):
    output = io.BytesIO()
    image = Image.new('RGB', size, '#376b32')
    ImageDraw.Draw(image).rectangle((0,0,size[0]//2,size[1]//2),fill='#c4a775')
    if exif:
        image.save(output, 'JPEG', exif=exif)
    else:
        image.save(output, 'PNG')
    return output.getvalue()


class WebRequestTests(unittest.TestCase):
    def setUp(self):
        self.detector = FakeDetector()
        self.client = TestClient(create_app(self.detector, background_load=False))

    def post(self, data, media='image/png'):
        return self.client.post('/api/predict', content=data, headers={'content-type': media})

    def test_empty_prediction_is_success_not_healthy_or_unknown_diagnosis(self):
        response = self.post(picture())
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()['status'], 'no_reliable_detection')
        self.assertEqual(response.json()['detections'], [])
        self.assertEqual(response.json()['width'], 20)
        self.assertEqual(response.headers['cache-control'], 'no-store')
        self.assertFalse(self.detector.lock.locked())

    def test_reject_svg_even_if_renamed_as_png(self):
        self.assertEqual(self.post(b'<svg xmlns="http://www.w3.org/2000/svg"/>').status_code, 400)
        self.assertEqual(self.detector.calls, 0)
        self.assertFalse(self.detector.lock.locked())

    def test_enforce_byte_and_pixel_limits(self):
        self.assertEqual(self.post(b'x'*1025).status_code, 413)
        self.assertEqual(self.post(picture((101, 101))).status_code, 413)
        self.assertEqual(self.detector.calls, 0)

    def test_busy_and_unavailable_return_retryable_errors(self):
        self.detector.lock.acquire()
        self.assertEqual(self.post(picture()).status_code, 429)
        self.detector.lock.release()
        self.detector.ready = False
        self.assertEqual(self.post(picture()).status_code, 503)

    def test_orientation_applied_before_inference(self):
        exif = Image.Exif(); exif[274] = 6
        response = self.post(picture(exif=exif), 'image/jpeg')
        self.assertEqual(response.status_code, 200)
        self.assertEqual((response.json()['width'], response.json()['height']), (10, 20))

    def test_invalid_format_empty_and_multiple_requests(self):
        self.assertEqual(self.post(picture(), 'text/plain').status_code, 415)
        self.assertEqual(self.post(b'').status_code, 400)
        for _ in range(2):
            self.assertEqual(self.post(picture()).status_code, 200)
        self.assertEqual(self.detector.calls, 2)

    def test_internal_errors_do_not_escape_or_hold_lock(self):
        self.detector.predict = lambda image: (_ for _ in ()).throw(RuntimeError('private/internal/path'))
        with self.assertLogs('agrivision.web', level='ERROR'):
            response = self.post(picture())
        self.assertEqual(response.status_code, 500)
        self.assertNotIn('private', response.text)
        self.assertFalse(self.detector.lock.locked())

    def test_page_assets_local_and_security_headers(self):
        response = self.client.get('/')
        self.assertEqual(response.status_code, 200)
        self.assertIn('default-src', response.headers['content-security-policy'])
        self.assertEqual(self.client.get('/static/app.js').status_code, 200)

    def test_stream_without_content_length_cannot_bypass_byte_limit(self):
        response = self.client.post('/api/predict', content=iter([b'x'*700, b'y'*700]),
            headers={'content-type': 'image/png'})
        self.assertEqual(response.status_code, 413)
        self.assertFalse(self.detector.lock.locked())

    def test_real_concurrent_requests_reject_second_and_recover(self):
        entered, release = threading.Event(), threading.Event()
        original = self.detector.predict
        def slow(image):
            entered.set()
            if not release.wait(5):
                raise RuntimeError('Test worker timed out')
            return original(image)
        self.detector.predict = slow
        with ThreadPoolExecutor(max_workers=2) as pool:
            first = pool.submit(self.post, picture())
            self.assertTrue(entered.wait(5))
            try:
                self.assertEqual(self.post(picture()).status_code, 429)
            finally:
                release.set()
            self.assertEqual(first.result().status_code, 200)
        self.detector.predict = original
        self.assertEqual(self.post(picture()).status_code, 200)

    def test_supported_formats_and_grayscale_alpha_conversion(self):
        for mode, fmt, media in [('L','JPEG','image/jpeg'),('RGBA','PNG','image/png'),('RGB','WEBP','image/webp')]:
            output = io.BytesIO(); image=Image.new(mode,(10,20));ImageDraw.Draw(image).rectangle((0,0,5,10),fill='white');image.save(output,fmt)
            self.assertEqual(self.post(output.getvalue(), media).status_code, 200)

    def test_reject_animated_png(self):
        output = io.BytesIO()
        Image.new('RGB',(8,8),'green').save(output,'PNG',save_all=True,
            append_images=[Image.new('RGB',(8,8),'red')],duration=100,loop=0)
        self.assertEqual(self.post(output.getvalue()).status_code, 415)
        self.assertFalse(self.detector.lock.locked())

    def test_actual_file_content_checked_even_with_allowed_mime(self):
        output = io.BytesIO(); Image.new('RGB',(10,10)).save(output,'BMP')
        self.assertEqual(self.post(output.getvalue()).status_code, 415)

    def test_solid_image_rejected_before_model_call(self):
        output=io.BytesIO();Image.new('RGB',(20,20),'white').save(output,'PNG')
        self.assertEqual(self.post(output.getvalue()).status_code,422)
        self.assertEqual(self.detector.calls,0)
        self.assertFalse(self.detector.lock.locked())

    def test_startup_failure_keeps_page_available_with_safe_message(self):
        class Broken(FakeDetector):
            def __init__(self):
                super().__init__(); self.ready = False; self.done = threading.Event()
            def load(self):
                self.done.set(); raise ValueError('private checkpoint location')
        broken = Broken()
        with self.assertLogs('agrivision.web', level='ERROR'):
            with TestClient(create_app(broken)) as client:
                self.assertTrue(broken.done.wait(5))
                self.assertEqual(client.get('/').status_code,200)
                response = client.post('/api/predict',content=picture(),headers={'content-type':'image/png'})
                self.assertEqual(response.status_code,503)
                self.assertNotIn('private',response.text)


class ModelIdentityTests(unittest.TestCase):
    def test_bad_checkpoint_hash_rejected_before_weights_load(self):
        from unittest.mock import patch
        from web_demo.inference import Detector
        with patch('web_demo.inference.sha256', return_value='wrong'):
            with self.assertRaisesRegex(ValueError,'SHA256'):
                Detector().load()

    def test_noncontiguous_or_wrong_number_of_labels_rejected(self):
        from web_demo.inference import names_list
        for names in [['wrong']*23, {i:'wrong' for i in range(1,25)}]:
            with self.assertRaises(ValueError):
                names_list(names)


if __name__ == '__main__':
    unittest.main()
