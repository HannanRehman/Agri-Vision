"""Download technology symbols from their publishers for the poster footer."""
import json
import urllib.request
import xml.etree.ElementTree as ET
from pathlib import Path
from io import BytesIO
from PIL import Image
import resvg_py

OUT=Path(__file__).resolve().parent/'assets/technology_symbols'
OUT.mkdir(parents=True,exist_ok=True)
SOURCES={
    'python':'https://s3.dualstack.us-east-2.amazonaws.com/pythondotorg-assets/media/files/python-logo-only.svg',
    'pytorch':'https://pytorch.org/wp-content/uploads/2024/10/logo.svg',
    'ultralytics':'https://raw.githubusercontent.com/ultralytics/assets/main/logo/Ultralytics-logomark-color.png',
    'colab':'https://colab.research.google.com/img/colab_favicon_256px.png',
    'drive':'https://ssl.gstatic.com/images/branding/product/2x/drive_2020q4_96dp.png',
    'vscode':'https://code.visualstudio.com/assets/branding/code-stable.png',
}
for key,url in SOURCES.items():
    raw=urllib.request.urlopen(url,timeout=30).read()
    if url.endswith('.svg'):
        (OUT/f'{key}_source.svg').write_bytes(raw)
        if key=='pytorch':
            # The publisher's first group is its standalone symbol. Preserve its
            # paths/colors and viewBox coordinates, excluding the separate wordmark.
            tree=ET.fromstring(raw);symbol=ET.Element('{http://www.w3.org/2000/svg}svg',{'viewBox':'0.6 539.9 80 87'})
            symbol.append(list(tree)[0]);raw=ET.tostring(symbol,encoding='utf-8')
        raw=resvg_py.svg_to_bytes(svg_string=raw.decode('utf-8'),width=700,dpi=96)
    im=Image.open(BytesIO(raw)).convert('RGBA')
    bbox=im.getchannel('A').getbbox()
    if bbox:im=im.crop(bbox)
    im.save(OUT/f'{key}.png')
    print(key,im.size,flush=True)
(OUT/'sources.json').write_text(json.dumps(SOURCES,indent=2)+'\n',encoding='utf-8')
