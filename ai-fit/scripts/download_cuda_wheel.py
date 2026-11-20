"""Resume the official CUDA wheel in bounded HTTP range requests."""
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
from pathlib import Path
import re
import requests
import time

ROOT = Path(__file__).resolve().parents[1]
directory = ROOT / '.external' / 'wheels'
directory.mkdir(parents=True, exist_ok=True)
name = 'torch-2.5.1+cu118-cp311-cp311-win_amd64.whl'
url = 'https://download.pytorch.org/whl/cu118/' + name.replace('+', '%2B')
index = requests.get('https://download.pytorch.org/whl/cu118/torch/', timeout=30)
index.raise_for_status()
expected = re.search(r'torch-2\.5\.1(?:%2B|\+)cu118-cp311-cp311-win_amd64\.whl#sha256=([a-f0-9]+)', index.text).group(1)
probe = requests.get(url, headers={'Range': 'bytes=0-0'}, timeout=30)
probe.raise_for_status()
total = int(probe.headers['Content-Range'].split('/')[-1])
chunk = 8 * 1024 * 1024
count = (total + chunk - 1) // chunk

def fetch(i):
    start = i * chunk
    end = min(total, start + chunk) - 1
    part = directory / f'torch.part{i:04d}'
    if part.exists() and part.stat().st_size == end - start + 1:
        return
    for attempt in range(5):
        try:
            response = requests.get(url, headers={'Range': f'bytes={start}-{end}'}, timeout=90)
            response.raise_for_status()
            if response.status_code != 206 or len(response.content) != end - start + 1:
                raise ValueError('Invalid range response')
            part.write_bytes(response.content)
            return
        except Exception:
            if attempt == 4:
                raise
            time.sleep(2)

with ThreadPoolExecutor(max_workers=48) as pool:
    jobs = [pool.submit(fetch, i) for i in range(count)]
    for done, result in enumerate(as_completed(jobs), 1):
        result.result()
        if done % 10 == 0 or done == count:
            print(f'CUDA package: {done}/{count} chunks', flush=True)
digest = hashlib.sha256()
target = directory / name
with target.open('wb') as output:
    for i in range(count):
        data = (directory / f'torch.part{i:04d}').read_bytes()
        digest.update(data)
        output.write(data)
if digest.hexdigest() != expected:
    raise RuntimeError('Official wheel SHA256 mismatch')
print('Verified official wheel:', target, flush=True)
