"""Fetch versioned preprocessing weights without changing the user's environment."""
import hashlib
from pathlib import Path
import requests

ROOT = Path(__file__).resolve().parents[1]
FILES = {
    'pose_landmarker_full.task': (
        'https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_full/float16/1/pose_landmarker_full.task',
        '5134a3aad27a58b93da0088d431f366da362b44e3ccfbe3462b3827a839011b1'),
    'u2netp.onnx': (
        'https://github.com/danielgatis/rembg/releases/download/v0.0.0/u2netp.onnx',
        '309c8469258dda742793dce0ebea8e6dd393174f89934733ecc8b14c76f4ddd8'),
}


def setup():
    folder = ROOT/'.models/preprocessing'
    folder.mkdir(parents=True, exist_ok=True)
    for name, (url, expected) in FILES.items():
        target = folder/name
        if target.exists() and hashlib.sha256(target.read_bytes()).hexdigest() == expected:
            continue
        response = requests.get(url, timeout=(20, 90))
        response.raise_for_status()
        if hashlib.sha256(response.content).hexdigest() != expected:
            raise RuntimeError(f'{name}: model checksum mismatch; file was not installed.')
        temporary = target.with_suffix(target.suffix+'.partial')
        temporary.write_bytes(response.content)
        temporary.replace(target)
        print(name+': downloaded and verified', flush=True)


if __name__ == '__main__':
    setup()
