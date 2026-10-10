"""Download an authenticated Drive batch into a bounded local experiment snapshot."""
import concurrent.futures
import hashlib
import json
import os
from pathlib import Path
import sys
import time
import urllib.request

ROOT = Path(__file__).resolve().parents[1]
DEST = (ROOT / 'experiments').resolve()

def digest(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()

def download(item):
    path = Path(item['destination'])
    if not path.is_relative_to(DEST):
        raise ValueError('Destination outside experiment workspace')
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + '.download')
    expected = int(item['bytes'])
    row = {'id': item['id'], 'path': item['path'], 'bytes': expected}
    try:
        started = time.monotonic()
        request = urllib.request.Request(item['download_url'], headers={'User-Agent':'Mozilla/5.0'})
        with urllib.request.urlopen(request, timeout=90) as response, temporary.open('wb') as output:
            for block in iter(lambda: response.read1(64*1024), b''):
                if time.monotonic() - started > 600:
                    raise TimeoutError('Transfer exceeded bounded download time')
                output.write(block)
        if temporary.stat().st_size != expected:
            raise ValueError('Download size differs from fetched metadata')
        sha = digest(temporary)
        temporary.replace(path)
        row.update(status='saved', sha256=sha, local_path=str(path))
    except Exception as error:
        row.update(status='failed', error=type(error).__name__)
    return row

if __name__ == '__main__':
    tasks = json.loads(Path(sys.argv[1]).read_text())
    for item in tasks:
        destination = Path(os.path.abspath(DEST / item['path']))
        if not destination.is_relative_to(DEST):
            raise ValueError('Destination outside experiment workspace')
        destination.parent.mkdir(parents=True, exist_ok=True)
        if not destination.parent.resolve().is_relative_to(DEST):
            raise ValueError('Destination parent resolves outside experiment workspace')
        if destination.is_symlink():
            raise ValueError('Destination is a symbolic link')
        item['destination'] = str(destination)
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
        rows = list(pool.map(download, tasks))
    Path(sys.argv[2]).write_text(json.dumps(rows, indent=2))
    print(json.dumps({'saved':sum(r['status']=='saved' for r in rows), 'failed':sum(r['status']=='failed' for r in rows), 'bytes':sum(r['bytes'] for r in rows if r['status']=='saved')}))
