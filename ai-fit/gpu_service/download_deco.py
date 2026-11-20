"""Download the pinned DeCo-VTON checkpoint, with resume and SHA256 checks."""
import hashlib
import time
from pathlib import Path
import requests

ROOT = Path(__file__).resolve().parents[1]
REVISION = '1cfe89ad5ef529d07a82fd5f3a6dbdb72e41fed3'
FILES = {
    'config.json': (1851, '6cef623329bcbcfc3b919cbd431ca4e11776e8e3555dbe0550b8a47ce9d24a1b'),
    'diffusion_pytorch_model.safetensors': (3438225152, '4ce41428749aad3fdb7e7bd38bbcd374e80967f6b41144a726a488f897afb2e3'),
}


def digest(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def download():
    folder = ROOT/'.models/deco/unet'
    folder.mkdir(parents=True, exist_ok=True)
    for name, (size, expected) in FILES.items():
        target = folder/name
        if target.exists() and target.stat().st_size == size and digest(target) == expected:
            print(name+': verified', flush=True)
            continue
        partial = folder/(name+'.partial')
        if partial.exists():
            if partial.stat().st_size > size or (
                    partial.stat().st_size == size and digest(partial) != expected):
                partial.unlink()
        offset = partial.stat().st_size if partial.exists() else 0
        if offset >= size:
            offset = 0
        url = f'https://huggingface.co/levinna/DeCo-VTON/resolve/{REVISION}/VITON-HD-512/unet/{name}'
        headers = {'Range': f'bytes={offset}-'} if offset else {}
        with requests.get(url, headers=headers, stream=True, timeout=(30, 90)) as response:
            response.raise_for_status()
            if offset and response.status_code == 206:
                if not response.headers.get('Content-Range', '').startswith(f'bytes {offset}-'):
                    raise RuntimeError('The server returned an unexpected resume range.')
            else:
                offset = 0
            count, last = offset, time.monotonic()
            with partial.open('ab' if offset else 'wb') as stream:
                for chunk in response.iter_content(1024*1024):
                    stream.write(chunk)
                    count += len(chunk)
                    if time.monotonic()-last >= 20:
                        print(f'{name}: {count*100/size:.1f}%', flush=True)
                        last = time.monotonic()
        if partial.stat().st_size != size or digest(partial) != expected:
            raise RuntimeError('Checkpoint verification failed. Rerun this setup command.')
        partial.replace(target)
        print(name+': downloaded and verified', flush=True)


if __name__ == '__main__':
    download()
