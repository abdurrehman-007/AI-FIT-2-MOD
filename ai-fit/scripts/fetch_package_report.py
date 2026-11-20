"""Download a pip report with bounded ranges and verify package hashes."""
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
import sys
import time
from urllib.parse import unquote, urlparse
import requests

ROOT = Path(__file__).resolve().parents[1]
destination = ROOT / '.external' / 'wheels'
destination.mkdir(parents=True, exist_ok=True)
report = json.loads(Path(sys.argv[1]).read_text())

def download(item):
    info = item['download_info']
    url = info['url']
    name = unquote(Path(urlparse(url).path).name)
    target = destination / name
    expected = info['archive_info']['hashes']['sha256']
    if target.exists() and hashlib.sha256(target.read_bytes()).hexdigest() == expected:
        return
    response = requests.head(url, timeout=30)
    response.raise_for_status()
    size = int(response.headers['Content-Length'])
    chunk = 256 * 1024
    count = (size + chunk - 1) // chunk
    parts = destination / (name + '.parts256k')
    parts.mkdir(exist_ok=True)
    def fetch(i):
        start, end = i * chunk, min(size, (i + 1) * chunk) - 1
        part = parts / str(i)
        if part.exists() and part.stat().st_size == end - start + 1:
            return
        for attempt in range(5):
            try:
                result = requests.get(url, headers={'Range': f'bytes={start}-{end}'}, timeout=90)
                result.raise_for_status()
                if len(result.content) != end - start + 1:
                    raise ValueError('Invalid package range')
                part.write_bytes(result.content)
                return
            except Exception:
                if attempt == 4:
                    raise
                time.sleep(2)
    with ThreadPoolExecutor(max_workers=12) as pool:
        list(pool.map(fetch, range(count)))
    digest = hashlib.sha256()
    with target.open('wb') as output:
        for i in range(count):
            data = (parts / str(i)).read_bytes()
            output.write(data)
            digest.update(data)
    if digest.hexdigest() != expected:
        raise RuntimeError('Package checksum mismatch: ' + name)
    print('Verified', name, flush=True)

with ThreadPoolExecutor(max_workers=5) as pool:
    list(pool.map(download, report['install']))
