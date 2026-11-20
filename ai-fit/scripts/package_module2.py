"""Build portable source/data archives from an explicit allowlist."""
import hashlib,json,zipfile
from datetime import date
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'deliverables';OUT.mkdir(exist_ok=True)
root_names={'app.py','analytics.py','test_analytics.py','test_full_ui.cjs','storage.py','sizing.py','train_size.py','requirements.txt','requirements-module2-audit.txt','README.md','Dockerfile','.dockerignore','.gitignore','.env.example','start.ps1','start-gpu.ps1','Open-AI-FIT.cmd','test_app.py','test_sizing.py','test_catalog.cjs','test_generate.cjs','test_module2_ui.cjs'}
directories={'gpu_service','static','templates','scripts','datasets','reports'}
source=[]
for p in ROOT.rglob('*'):
    if not p.is_file():continue
    rel=p.relative_to(ROOT)
    if any(part in {'__pycache__','.git','node_modules'} for part in rel.parts):continue
    if rel.as_posix() == 'reports/module2-ui-overall.png':continue
    if len(rel.parts)==1 and p.name in root_names or len(rel.parts)>1 and rel.parts[0] in directories:
        source.append(p)
assert not any(p.suffix in {'.key','.joblib','.onnx','.safetensors','.task','.sqlite3','.pyc'} for p in source)
assert not any('instance' in p.relative_to(ROOT).parts for p in source)
dataset=[p for p in source if p.relative_to(ROOT).parts[0]=='datasets']+[ROOT/'reports/module2-data-audit.json']
manifest={'date':date.today().isoformat(),'archives':[],'module1_service_sha256':{str(p.relative_to(ROOT)).replace('\\','/'):hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted((ROOT/'gpu_service').glob('*.py'))}}
for name,files in [('AI-FIT-source.zip',source),('AI-FIT-datasets.zip',dataset)]:
    path=OUT/name
    with zipfile.ZipFile(path,'w',compression=zipfile.ZIP_DEFLATED,compresslevel=9) as z:
        for p in sorted(set(files)):z.write(p,'ai-fit/'+p.relative_to(ROOT).as_posix())
    with zipfile.ZipFile(path) as z:
        assert z.testzip() is None
        for p in set(files):assert hashlib.sha256(z.read('ai-fit/'+p.relative_to(ROOT).as_posix())).digest()==hashlib.sha256(p.read_bytes()).digest()
    row={'file':name,'bytes':path.stat().st_size,'sha256':hashlib.sha256(path.read_bytes()).hexdigest(),'files':len(set(files)),'integrity':'CRC and all archived file SHA-256 comparisons passed'}
    manifest['archives'].append(row);print(json.dumps(row))
(OUT/'manifest.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')
