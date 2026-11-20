"""Resumable, private full photo x inventory upper-garment audit.

Use --prepare before --run. Outputs stay under ignored instance/. A successful
HTTP response is a technical pass, not a human judgement of garment fidelity.
"""
import argparse
from contextlib import contextmanager
import csv
import hashlib
import html
import io
import json
import os
from pathlib import Path
import sqlite3
import sys
import time
import urllib.error
import urllib.request

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from gpu_service.runtime_fingerprint import runtime_fingerprint
FOLDER = ROOT / 'instance' / 'all-photo-audit'
EXTENSIONS = {'.jpg', '.jpeg', '.png', '.webp'}


@contextmanager
def keep_awake():
    """Prevent idle system sleep only while this local audit process runs."""
    previous=0
    if os.name=='nt':
        import ctypes
        previous=ctypes.windll.kernel32.SetThreadExecutionState(0x80000001)
    try:
        yield bool(previous)
    finally:
        if previous:
            ctypes.windll.kernel32.SetThreadExecutionState(previous)


def save_json(path, value):
    temporary = path.with_suffix(path.suffix + '.tmp')
    with temporary.open('w',encoding='utf-8') as stream:
        stream.write(json.dumps(value, indent=2, ensure_ascii=False))
        stream.flush();os.fsync(stream.fileno())
    temporary.replace(path)


def load_results(journal):
    """Recover an incomplete final journal line; never ignore interior damage."""
    if not journal.exists():
        return {}
    data=journal.read_bytes()
    lines=data.splitlines(keepends=True)
    results={};valid_bytes=0
    for index,line in enumerate(lines):
        try:
            item=json.loads(line)
        except (ValueError,UnicodeDecodeError):
            if index != len(lines)-1:
                raise RuntimeError('Audit journal contains interior damage; preserve it for recovery')
            with journal.open('r+b') as stream:
                stream.truncate(valid_bytes);stream.flush();os.fsync(stream.fileno())
            return results
        results[item['case']]=item
        valid_bytes+=len(line)
    if data and not data.endswith(b'\n'):
        with journal.open('ab') as stream:
            stream.write(b'\n');stream.flush();os.fsync(stream.fileno())
    return results


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def source_version():
    return runtime_fingerprint(ROOT)


def ensure_service(opener, expected):
    with opener.open('http://127.0.0.1:8001/health',timeout=10) as response:
        health=json.load(response)
    if not health.get('ready') or health.get('source_fingerprint')!=expected:
        raise RuntimeError('Running service fingerprint differs from this audit; restart the matching service before resuming')


def ordered_pairs(photos, products):
    # Cover all people and garment styles early, then complete every pair.
    for offset in range(len(products)):
        for index,photo in enumerate(photos):
            yield photo,products[(index+offset)%len(products)]


def prepare():
    from PIL import Image, ImageOps
    import torch
    torch.set_num_threads(4)
    os.environ.update(HF_HUB_CACHE=str(ROOT / '.models/hf-cache'),
                      HF_HOME=str(ROOT / '.models/hf-home'), HF_HUB_OFFLINE='1')
    for path in (ROOT / '.external/CatVTON', ROOT / '.preprocess-deps', ROOT / 'gpu_service'):
        sys.path.insert(0, str(path))
    from huggingface_hub import snapshot_download
    from preprocessing import PhotoPreprocessor
    from clothing_mask import ClothingMask, long_sleeve_target
    checkpoint = Path(snapshot_download('zhengchong/CatVTON', local_files_only=True))
    prep = PhotoPreprocessor(ROOT / '.models/preprocessing')
    masker = ClothingMask(checkpoint / 'SCHP/exp-schp-201908261155-lip.pth')
    folder = ROOT.parent / 'MODEL PICTURES'
    photos = sorted(p for p in folder.rglob('*') if p.is_file()
                    and p.suffix.lower() in EXTENSIONS
                    and 'outfits' not in [part.lower() for part in p.relative_to(folder).parts])
    with sqlite3.connect(ROOT / 'instance/aifit.sqlite3') as connection:
        products = [json.loads(row[0]) for row in connection.execute(
            'SELECT body FROM documents WHERE collection=?', ('products',))]
    products = sorted((p for p in products if p.get('category') == 'upper'), key=lambda p: p['id'])
    manifest = {'source_version': source_version(), 'prepared_at': time.time(),
                'photos': [], 'products': [], 'excluded_outfit_sources': [str(p.relative_to(folder))
                for p in (folder / 'outfits').rglob('*') if p.suffix.lower() in EXTENSIONS]}
    for path in photos:
        row = {'name': str(path.relative_to(folder)), 'path': str(path), 'sha256': digest(path)}
        try:
            im = ImageOps.exif_transpose(Image.open(path)).convert('RGB')
            points, foreground = prep.validate_person(im, 'upper', include_foreground=True)
            masker.prepare(im, 'upper', (512, 768), landmarks=points, foreground=foreground)
            row.update(status='ready', width=im.width, height=im.height)
        except ValueError as error:
            row.update(status='input_rejected', error=str(error))
        except (OSError, Image.DecompressionBombError) as error:
            row.update(status='unreadable_input', error=str(error))
        manifest['photos'].append(row)
        save_json(FOLDER / 'manifest.json', manifest)
        print(json.dumps({'photo': row['name'], 'status': row['status'], 'error': row.get('error')}, ensure_ascii=False), flush=True)
    for product in products:
        path = ROOT / 'instance/images' / product.get('image', '')
        row = {k: product.get(k) for k in ('id', 'name', 'image', 'store')}
        row['upper_length']=product.get('upper_length') or 'standard'
        row['path'] = str(path)
        try:
            if row['upper_length'] not in ('standard','long'):
                raise ValueError('Invalid inventory upper garment length')
            if not path.is_file():
                raise ValueError('Inventory product has no available image')
            row['sha256'] = digest(path)
            image = ImageOps.exif_transpose(Image.open(path)).convert('RGBA')
            clean = prep.clean_garment(image)
            row['donor_person'] = prep.garment_has_person(clean)
            if row['donor_person']:
                clean = masker.garment_only(clean, 'upper')
            row.update(status='ready', long_sleeves=long_sleeve_target(clean))
        except (ValueError, OSError) as error:
            row.update(status='garment_rejected', error=str(error))
        manifest['products'].append(row)
        save_json(FOLDER / 'manifest.json', manifest)
        print(json.dumps({'product': row['name'], 'status': row['status'], 'error': row.get('error')}, ensure_ascii=False), flush=True)
    manifest['complete'] = True
    manifest['total_pairs'] = len(photos) * len(products)
    save_json(FOLDER / 'manifest.json', manifest)
    prep.close()
    return manifest


def pair_id(photo, product, version):
    value = version + photo['name'] + photo['sha256'] + product['id'] + product.get('sha256', '') + product.get('upper_length','standard')
    return hashlib.sha256(value.encode()).hexdigest()[:24]


def write_report(manifest, results, active=None):
    from collections import Counter
    rows = list(results.values())
    counts = Counter(r['status'] for r in rows)
    times = [r['seconds'] for r in rows if r.get('seconds', 0) > 1]
    mean = sum(times)/len(times) if times else 60
    eligible = sum(p['status']=='ready' for p in manifest['photos']) * sum(p['status']=='ready' for p in manifest['products'])
    generated = counts['technical_pass_needs_visual_review'] + counts['generation_rejected']
    summary = {'total_pairs': manifest['total_pairs'], 'finished_pairs': len(rows),
               'counts': dict(counts), 'remaining_pairs': manifest['total_pairs']-len(rows),
               'eligible_gpu_pairs': eligible,
               'photos_sent_to_gpu':len({r['photo'] for r in rows if r['status'] in ('technical_pass_needs_visual_review','generation_rejected')}),
               'garments_sent_to_gpu':len({r['product_id'] for r in rows if r['status'] in ('technical_pass_needs_visual_review','generation_rejected')}),
               'estimated_remaining_hours': round(max(0,eligible-generated)*mean/3600, 2),
               'input_issues': [{'photo':p['name'],'error':p.get('error')} for p in manifest['photos'] if p['status']!='ready'],
               'inventory_issues': [{'product':p['name'],'error':p.get('error')} for p in manifest['products'] if p['status']!='ready'],
               'active': active, 'source_version': manifest['source_version'], 'updated_at': time.time(),
               'note': 'Technical pass requires visual review; no arbitrary-photo perfection guarantee.'}
    reviews_path=FOLDER/'visual-issues.json'
    if reviews_path.exists():
        summary['visual_issues']=json.loads(reviews_path.read_text(encoding='utf-8'))
    save_json(FOLDER / 'progress.json', summary)
    fields = ['photo', 'product', 'product_id', 'status', 'error', 'seconds', 'attempts', 'backend', 'refinement', 'output', 'diagnostic', 'case']
    with (FOLDER / 'results.csv').open('w', newline='', encoding='utf-8-sig') as stream:
        writer = csv.DictWriter(stream, fields, extrasaction='ignore'); writer.writeheader(); writer.writerows(rows)
    body = []
    photo_paths={p['name']:p['path'] for p in manifest['photos']}
    product_paths={p['id']:p['path'] for p in manifest['products']}
    def thumb(path):
        if not path:return ''
        url=html.escape(str(path).replace('\\','/'),quote=True)
        return '<a href="'+url+'"><img loading="lazy" width="90" src="'+url+'"></a>'
    for row in rows:
        output = row.get('output') or row.get('diagnostic')
        preview=thumb(output)
        source=thumb(os.path.relpath(photo_paths[row['photo']],FOLDER))
        product=thumb(os.path.relpath(product_paths[row['product_id']],FOLDER))
        body.append('<tr>'+''.join('<td>'+html.escape(str(row.get(k, '')) or '')+'</td>'
                                  for k in ('photo', 'product', 'status', 'error', 'seconds'))+'<td>'+source+'</td><td>'+product+'</td><td>'+preview+'</td></tr>')
    page = '''<!doctype html><meta charset="utf-8"><title>AI-FIT upper outfit audit</title>
<style>body{font:15px system-ui;margin:25px;background:#f5f8f6}table{border-collapse:collapse;width:100%}td,th{padding:8px;border:1px solid #ccc;text-align:left}input{padding:10px;width:60%}img{max-height:120px;object-fit:contain}</style>
<h1>AI-FIT upper outfit audit</h1><p>Refresh to see progress. Technical passes still need visual review.</p>'''
    page += '<pre>'+html.escape(json.dumps(summary, indent=2))+'</pre>'
    page += '<input id="search" placeholder="Filter by photo, outfit, or error" oninput="document.querySelectorAll(\'tbody tr\').forEach(r=>r.hidden=!r.innerText.toLowerCase().includes(this.value.toLowerCase()))">'
    page += '<table><thead><tr><th>Photo</th><th>Outfit</th><th>Status</th><th>Issue</th><th>Seconds</th><th>Source</th><th>Reference</th><th>Result</th></tr></thead><tbody>'+''.join(body)+'</tbody></table>'
    (FOLDER / 'report.html').write_text(page, encoding='utf-8')


def encode_request(photo, product):
    boundary = 'aifit-' + os.urandom(12).hex()
    chunks = []
    for name, path in (('person', photo['path']), ('garment', product['path'])):
        chunks.append(('--'+boundary+'\r\nContent-Disposition: form-data; name="'+name+'"; filename="image.png"\r\nContent-Type: application/octet-stream\r\n\r\n').encode())
        chunks.append(Path(path).read_bytes()); chunks.append(b'\r\n')
    chunks.append(('--'+boundary+'\r\nContent-Disposition: form-data; name="category"\r\n\r\nupper\r\n--'+boundary+'\r\nContent-Disposition: form-data; name="upper_length"\r\n\r\n'+product.get('upper_length','standard')+'\r\n--'+boundary+'--\r\n').encode())
    return b''.join(chunks), boundary


def run(limit=None):
    manifest = json.loads((FOLDER / 'manifest.json').read_text(encoding='utf-8'))
    if not manifest.get('complete'):
        raise RuntimeError('Preparation is incomplete; rerun --prepare')
    if manifest['source_version'] != source_version():
        raise RuntimeError('Service source changed; rerun --prepare in a new audit folder before resuming')
    import msvcrt
    with (FOLDER / 'run.lock').open('a+b') as lock:
        if lock.seek(0, 2) == 0:
            lock.write(b'0'); lock.flush()
        lock.seek(0)
        msvcrt.locking(lock.fileno(), msvcrt.LK_NBLCK, 1)
        _run_cases(manifest, limit)


def _run_cases(manifest, limit):
    journal = FOLDER / 'results.jsonl'
    results = load_results(journal)
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    key = (ROOT / 'instance/gpu.key').read_text().strip()
    completed = 0
    # Diagonal rounds expose different people and garment styles early.
    priority = {'Forest Green Twill Shirt': 0, 'Shadow Black Hooded Puffer Jacket': 1}
    products = sorted(manifest['products'], key=lambda p: priority.get(p['name'], 2))
    photos = sorted(manifest['photos'], key=lambda p: (
        0 if p['name'].replace('\\','/') == 'male/male_person_19.png' else 1))
    for photo,product in ordered_pairs(photos,products):
        case = pair_id(photo, product, manifest['source_version'])
        if case in results:
            continue
        if (FOLDER / 'STOP').exists():
            write_report(manifest, results); return
        if manifest['source_version'] != source_version():
            raise RuntimeError('Source changed during audit. Checkpoint saved; prepare a new version.')
        row = {'case': case, 'photo': photo['name'], 'product': product['name'],
               'product_id': product['id'], 'seconds': 0}
        if photo['status'] != 'ready':
            row.update(status='input_rejected', error=photo.get('error', photo['status']))
        elif product['status'] != 'ready':
            row.update(status='garment_rejected', error=product.get('error', product['status']))
        else:
            # Verify the manifest still names exactly the files being sent.
            if digest(Path(photo['path'])) != photo['sha256'] or digest(Path(product['path'])) != product['sha256']:
                raise RuntimeError('An input file changed after preparation')
            payload, boundary = encode_request(photo, product)
            write_report(manifest, results, {'photo': photo['name'], 'product': product['name']})
            start = time.monotonic()
            request = urllib.request.Request('http://127.0.0.1:8001/infer', data=payload,
                headers={'Authorization': 'Bearer '+key, 'Content-Type': 'multipart/form-data; boundary='+boundary,
                         'X-AIFIT-Audit-Case':case})
            # Busy/service errors are not quality failures and are never
            # checkpointed as completed cases. A restart resumes them.
            while True:
                if (FOLDER/'STOP').exists():
                    write_report(manifest,results);return
                try:
                    ensure_service(opener,manifest['source_version'])
                    with opener.open(request, timeout=240) as response:
                        data = response.read()
                        if response.headers.get_content_type() != 'image/jpeg':
                            raise RuntimeError('Inference returned unexpected content')
                        from PIL import Image
                        Image.open(io.BytesIO(data)).verify()
                        relative = 'outputs/'+case+'.jpg'
                        image_path=FOLDER / relative
                        temporary=image_path.with_suffix('.tmp')
                        with temporary.open('wb') as stream:
                            stream.write(data);stream.flush();os.fsync(stream.fileno())
                        temporary.replace(image_path)
                        row.update(status='technical_pass_needs_visual_review', output=relative,
                                   attempts=response.headers.get('X-AIFIT-Attempts'),
                                   backend=response.headers.get('X-AIFIT-Backend'),
                                   refinement=response.headers.get('X-AIFIT-Sleeve-Refinement'))
                        break
                except urllib.error.HTTPError as error:
                    raw = error.read()
                    if error.code in (429, 503):
                        time.sleep(5); continue
                    try:
                        detail = json.loads(raw).get('error', 'Unknown inference error')
                    except (ValueError, AttributeError):
                        detail = 'HTTP '+str(error.code)
                    if error.code >= 500:
                        raise RuntimeError(detail) from error
                    row.update(status='generation_rejected', error=detail, http_status=error.code)
                    for attempt in (2,1):
                        diagnostic=ROOT/'instance/upper-audit-diagnostics'/f'{case}-{attempt}.jpg'
                        if diagnostic.is_file():
                            row['diagnostic']=os.path.relpath(diagnostic,FOLDER).replace('\\','/')
                            break
                    break
            row['seconds'] = round(time.monotonic()-start, 2)
        results[case] = row
        with journal.open('a', encoding='utf-8') as stream:
            stream.write(json.dumps(row, ensure_ascii=False)+'\n'); stream.flush(); os.fsync(stream.fileno())
        write_report(manifest, results)
        print(json.dumps(row, ensure_ascii=False), flush=True)
        completed += 1
        if limit and completed >= limit:
            return
    write_report(manifest, results)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--prepare', action='store_true')
    parser.add_argument('--run', action='store_true')
    parser.add_argument('--limit', type=int)
    parser.add_argument('--output', type=Path, default=FOLDER)
    args = parser.parse_args()
    FOLDER = args.output.resolve(); FOLDER.mkdir(parents=True, exist_ok=True)
    (FOLDER / 'outputs').mkdir(exist_ok=True)
    if args.prepare:
        prepare()
    if args.run:
        try:
            with keep_awake() as awake:
                save_json(FOLDER/'run-state.json', {'status':'running','pid':os.getpid(),'keep_awake':awake,'updated_at':time.time()})
                run(args.limit)
        except Exception as error:
            save_json(FOLDER/'run-state.json', {'status':'error','error':str(error),'updated_at':time.time()})
            raise
        progress=json.loads((FOLDER/'progress.json').read_text(encoding='utf-8'))
        save_json(FOLDER/'run-state.json', {'status':'complete' if progress['remaining_pairs']==0 else 'checkpointed',
                                         'updated_at':time.time()})
    if not args.prepare and not args.run:
        parser.error('Choose --prepare and/or --run')
