"""Fetch the auxiliary assets required by the local DeCo service."""
from pathlib import Path
from huggingface_hub import snapshot_download, get_hf_file_metadata, hf_hub_url, list_repo_files, hf_hub_download
from concurrent.futures import ThreadPoolExecutor
import fnmatch
import hashlib
import requests
import time
import sys

ROOT = Path(__file__).resolve().parents[1]
cache = ROOT / '.models' / 'hf-cache'
for repo, patterns in [
    ('zhengchong/CatVTON', ['SCHP/exp-schp-201908261155-lip.pth']),
    ('stabilityai/sd-vae-ft-mse', ['config.json', 'diffusion_pytorch_model.safetensors']),
    ('stable-diffusion-v1-5/stable-diffusion-inpainting', [
        'scheduler/*', 'safety_checker/config.json',
        'safety_checker/model.fp16.safetensors',
        'feature_extractor/*']),
]:
    if len(sys.argv) > 1 and repo != sys.argv[1]:
        continue
    print('Downloading', repo, flush=True)
    for filename in list_repo_files(repo):
        if not any(fnmatch.fnmatch(filename, pattern) for pattern in patterns):
            continue
        url = hf_hub_url(repo, filename)
        metadata = get_hf_file_metadata(url)
        if metadata.size < 4 * 1024 * 1024:
            hf_hub_download(repo, filename, cache_dir=str(cache))
            continue
        repo_dir = cache / ('models--' + repo.replace('/', '--'))
        target = repo_dir / 'snapshots' / metadata.commit_hash / filename
        target.parent.mkdir(parents=True, exist_ok=True)
        if target.exists() and target.stat().st_size == metadata.size:
            continue
        parts = repo_dir / 'download-parts' / metadata.etag
        parts.mkdir(parents=True, exist_ok=True)
        chunk = 8 * 1024 * 1024
        count = (metadata.size + chunk - 1) // chunk
        def fetch(i):
            start = i * chunk
            end = min(metadata.size, start + chunk) - 1
            part = parts / str(i)
            if part.exists() and part.stat().st_size == end - start + 1:
                return
            for attempt in range(5):
                try:
                    response = requests.get(url, headers={'Range': f'bytes={start}-{end}'}, timeout=90)
                    response.raise_for_status()
                    if response.status_code != 206 or len(response.content) != end - start + 1:
                        raise ValueError('Invalid model range response')
                    part.write_bytes(response.content)
                    return
                except Exception:
                    if attempt == 4:
                        raise
                    time.sleep(2)
        with ThreadPoolExecutor(max_workers=48) as pool:
            for done, _ in enumerate(pool.map(fetch, range(count)), 1):
                if done % 10 == 0 or done == count:
                    print(filename, done, '/', count, flush=True)
        digest = hashlib.sha256()
        with target.open('wb') as output:
            for i in range(count):
                data = (parts / str(i)).read_bytes()
                output.write(data)
                digest.update(data)
        if len(metadata.etag) == 64 and digest.hexdigest() != metadata.etag:
            target.unlink()
            raise RuntimeError('Checkpoint checksum mismatch: ' + filename)
        (repo_dir / 'refs').mkdir(exist_ok=True)
        (repo_dir / 'refs' / 'main').write_text(metadata.commit_hash)
print('Downloading upstream clothing parser', flush=True)
snapshot_download('zhengchong/CatVTON', repo_type='space',
                  allow_patterns=['model/SCHP/**', 'model/__init__.py', 'LICENSE'],
                  local_dir=str(ROOT / '.external' / 'CatVTON'))
