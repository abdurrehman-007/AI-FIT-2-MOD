"""Identify inference code and live renderer configuration for private audits."""
import hashlib
import json
import os


def code_fingerprint(root):
    paths=sorted((root/'gpu_service').glob('*.py'))
    return hashlib.sha256(b''.join(p.name.encode()+p.read_bytes() for p in paths
                                  if not p.name.startswith('test_'))).hexdigest()


def runtime_fingerprint(root, loaded_code=None):
    calibration=root/'instance/garment-backends.json'
    config=calibration.read_bytes() if calibration.is_file() else b''
    engine=json.dumps({'engine':os.environ.get('AIFIT_ENGINE','auto'),
        'speed_profile':os.environ.get('AIFIT_SPEED_PROFILE','balanced'),
        'fashn_attention':os.environ.get('AIFIT_FASHN_ATTENTION','auto')},sort_keys=True).encode()
    return hashlib.sha256((loaded_code or code_fingerprint(root)).encode()+engine+config).hexdigest()
