"""Single-origin local inference service; uploaded bytes never persist to disk."""
from contextlib import asynccontextmanager
import io
import logging
from pathlib import Path
import threading
import time
import warnings

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from PIL import Image, ImageOps, UnidentifiedImageError
from starlette.concurrency import run_in_threadpool

from web_demo.inference import Detector

STATIC = Path(__file__).with_name('static')
logger = logging.getLogger('agrivision.web')
ALLOWED_TYPES = {'image/jpeg', 'image/png', 'image/webp'}
ALLOWED_FORMATS = {'JPEG', 'PNG', 'WEBP'}


def decode_image(data, max_pixels):
    try:
        with warnings.catch_warnings():
            warnings.simplefilter('error', Image.DecompressionBombWarning)
            with Image.open(io.BytesIO(data)) as image:
                if image.format not in ALLOWED_FORMATS:
                    raise HTTPException(415, 'Choose a JPG, PNG or WebP image.')
                if image.width*image.height > max_pixels:
                    raise HTTPException(413, 'Image dimensions are too large. Use an image under 20 megapixels.')
                if getattr(image, 'n_frames', 1) > 1:
                    raise HTTPException(415, 'Choose a still image rather than an animated image.')
                image.load()
                decoded = ImageOps.exif_transpose(image).convert('RGB')
                if all(low == high for low, high in decoded.getextrema()):
                    decoded.close()
                    raise HTTPException(422, 'This image has no visible detail. Upload a clear photograph of a supported crop.')
                return decoded
    except (Image.DecompressionBombWarning, Image.DecompressionBombError):
        raise HTTPException(413, 'Image dimensions are too large. Resize the image and try again.') from None
    except (UnidentifiedImageError, OSError, ValueError, SyntaxError):
        raise HTTPException(400, 'This image could not be read. Try a different JPG, PNG or WebP file.') from None


def create_app(detector=None, background_load=True):
    service = detector or Detector()

    @asynccontextmanager
    async def lifespan(app):
        def initialize():
            try:
                service.load()
            except Exception:
                service.failed = True
                logger.exception('Model initialization failed')
        if background_load:
            threading.Thread(target=initialize, name='agrivision-model', daemon=True).start()
        yield

    app = FastAPI(title='Agri Vision local detector', lifespan=lifespan, docs_url=None, redoc_url=None)
    app.state.detector = service

    @app.middleware('http')
    async def headers(request, call_next):
        response = await call_next(request)
        response.headers['Cache-Control'] = 'no-store'
        response.headers['X-Content-Type-Options'] = 'nosniff'
        response.headers['Referrer-Policy'] = 'no-referrer'
        response.headers['Content-Security-Policy'] = "default-src 'self'; script-src 'self'; style-src 'self'; img-src 'self' blob: data:; connect-src 'self'; object-src 'none'; base-uri 'none'; frame-ancestors 'none'"
        return response

    @app.get('/')
    def index():
        return FileResponse(STATIC/'index.html')

    @app.get('/api/health')
    def health():
        return service.public_info()

    @app.post('/api/predict')
    async def predict(request: Request):
        started = time.perf_counter()
        if not service.ready:
            raise HTTPException(503, 'The model is still loading. Try again shortly.' if not service.failed
                else 'The model is unavailable. Restart the local application or check its model configuration.')
        if request.headers.get('content-type', '').split(';')[0].strip().lower() not in ALLOWED_TYPES:
            raise HTTPException(415, 'Choose a JPG, PNG or WebP image.')
        maximum = service.config['max_upload_bytes']
        try:
            if int(request.headers.get('content-length', '0')) > maximum:
                raise HTTPException(413, 'Choose an image smaller than 10 MB.')
        except ValueError:
            raise HTTPException(400, 'Invalid upload size.') from None
        if not service.lock.acquire(blocking=False):
            raise HTTPException(429, 'Another image is being processed. Try again in a moment.')
        image = None
        payload = bytearray()
        try:
            async for chunk in request.stream():
                if len(payload)+len(chunk) > maximum:
                    raise HTTPException(413, 'Choose an image smaller than 10 MB.')
                payload.extend(chunk)
            if not payload:
                raise HTTPException(400, 'Choose an image before starting detection.')
            image = await run_in_threadpool(decode_image, payload, service.config['max_pixels'])
            result = await run_in_threadpool(service.predict, image)
            result['processing_ms'] = round((time.perf_counter()-started)*1000, 1)
            return result
        except HTTPException:
            raise
        except Exception:
            logger.exception('Image inference failed')
            raise HTTPException(500, 'The image could not be processed. Try again or choose a different image.') from None
        finally:
            payload.clear()
            if image is not None:
                image.close()
            service.lock.release()

    app.mount('/static', StaticFiles(directory=STATIC), name='static')
    return app


app = create_app()
