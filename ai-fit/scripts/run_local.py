"""Start the local GPU service and web app, preserving logs for troubleshooting."""
import os
from pathlib import Path
import subprocess
import sys
import time
import urllib.request

ROOT = Path(__file__).resolve().parents[1]
os.chdir(ROOT)
env = os.environ.copy()
env.pop('PYTHONPATH', None)
env.setdefault('AIFIT_SPEED_PROFILE','balanced')
env.update(AIFIT_EXISTING_PROJECT=str(ROOT / '.external'),
           HF_HUB_CACHE=str(ROOT / '.models' / 'hf-cache'),
           HF_HOME=str(ROOT / '.models' / 'hf-home'),
           HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1')
port = os.environ.get('PORT', '5050')
web_url = 'http://127.0.0.1:' + port
env['PORT'] = port
env.setdefault('PUBLIC_URL', web_url)
# local_windows.py expects PROJECT/CatVTON.
logs = ROOT / 'instance'
logs.mkdir(exist_ok=True)
if os.name == 'nt':
    import msvcrt
    startup_lock = (logs / 'launcher.lock').open('a+b')
    if startup_lock.seek(0,2) == 0:
        startup_lock.write(b'0')
        startup_lock.flush()
    startup_lock.seek(0)
    try:
        msvcrt.locking(startup_lock.fileno(),msvcrt.LK_NBLCK,1)
    except OSError:
        raise SystemExit('AI-FIT startup is already in progress. Wait for its ready message.')
opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))

def ready(url):
    try:
        with opener.open(url, timeout=3) as response:
            return response.status == 200
    except Exception:
        return False

def start(script, name, variables):
    out = (logs / (name + '-output.log')).open('a')
    err = (logs / (name + '-error.log')).open('a')
    return subprocess.Popen([sys.executable, '-u', script], cwd=ROOT, env=variables,
                            stdout=out, stderr=err,
                            creationflags=subprocess.CREATE_NO_WINDOW)

if not ready('http://127.0.0.1:8001/health'):
    gpu = start('gpu_service/local_windows.py', 'gpu', env)
    print('Loading GPU model. Logs: instance/gpu-error.log', flush=True)
    for _ in range(180):
        if ready('http://127.0.0.1:8001/health'):
            break
        if gpu.poll() is not None:
            raise SystemExit('GPU startup failed. See instance/gpu-error.log')
        time.sleep(2)
    else:
        raise SystemExit('GPU startup timed out. See instance/gpu-error.log')
env['CATVTON_URL'] = 'http://127.0.0.1:8001'
env['CATVTON_API_KEY'] = (logs / 'gpu.key').read_text().strip()
if not ready(web_url + '/'):
    web = start('app.py', 'app', env)
    for _ in range(30):
        if ready(web_url + '/'):
            break
        if web.poll() is not None:
            raise SystemExit('App startup failed. See instance/app-error.log')
        time.sleep(1)
    else:
        raise SystemExit('App startup timed out. See instance/app-error.log')
print('GPU ready. Open ' + web_url, flush=True)
