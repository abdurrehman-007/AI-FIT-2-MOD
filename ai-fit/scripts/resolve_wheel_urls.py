"""Resolve direct dependency wheels without downloading their large bodies."""
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
from packaging.requirements import Requirement
from packaging.tags import sys_tags
from packaging.utils import parse_wheel_filename
from packaging.version import Version
import requests

ROOT = Path(__file__).resolve().parents[1]
tags = set(sys_tags())
def resolve(line):
    req = Requirement(line)
    exact = next((s.version for s in req.specifier if s.operator == '=='), None)
    endpoint = 'https://pypi.org/pypi/' + req.name + ('/' + exact if exact else '') + '/json'
    response = requests.get(endpoint, timeout=120)
    response.raise_for_status()
    data = response.json()
    releases = {exact: data['urls']} if exact else data['releases']
    versions = sorted((Version(v) for v in releases if req.specifier.contains(v) and not Version(v).is_prerelease), reverse=True)
    for version in versions:
        for entry in releases[str(version)]:
            if entry['filename'].endswith('.whl') and tags.intersection(parse_wheel_filename(entry['filename'])[3]):
                return {'download_info': {'url': entry['url'], 'archive_info': {'hashes': entry['digests']}}}
    raise RuntimeError('No wheel for ' + line)
lines = (ROOT / 'requirements-gpu-local.txt').read_text().splitlines()
with ThreadPoolExecutor(max_workers=8) as pool:
    items = list(pool.map(resolve, filter(None, lines)))
(ROOT / '.external' / 'packages.json').write_text(json.dumps({'install': items}))
print('Resolved', len(items), 'wheels', flush=True)
