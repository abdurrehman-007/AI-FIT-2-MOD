"""Disposable progress UI fixture, with optional real local GPU. Never deploy."""
import io,json,os,sys,tempfile,time,threading
from contextlib import ExitStack
from pathlib import Path
from unittest.mock import Mock,patch
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from app import create_app
from flask import jsonify,request,send_file
from PIL import Image
from storage import Store
real=os.environ.get('AIFIT_TEST_REAL_GPU')=='1'
stream=io.BytesIO();Image.new('RGB',(256,384),'#7a9b8d').save(stream,format='JPEG');fixture=stream.getvalue()
rows={};guard=threading.Lock();mode={'fail':False}
def fake_post(url,**kwargs):
    key=kwargs['headers']['X-AIFIT-Request-ID'];started=time.monotonic()
    with guard:rows[key]=started
    time.sleep(12)
    response=Mock(status_code=422 if mode['fail'] else 200,content=fixture,headers={})
    response.json.return_value={'error':'This preview could not be completed. Please try again.'}
    response.raise_for_status.return_value=None
    return response
def fake_get(url,**kwargs):
    key=url.rsplit('/',1)[-1]
    with guard:started=rows.get(key)
    response=Mock(status_code=200 if started is not None else 404)
    if started is not None:
        elapsed=time.monotonic()-started
        stage='validating' if elapsed<2 else 'preparing' if elapsed<3 else 'generating' if elapsed<9 else 'checking' if elapsed<11 else 'saving'
        response.json.return_value=dict(status='running',stage=stage,percent=min(99,round(elapsed/12*99)),remaining_seconds=max(0,12-elapsed),estimated_total_seconds=12,estimated=True)
    return response
with tempfile.TemporaryDirectory(prefix='aifit-progress-ui-') as folder,ExitStack() as stack:
    config=dict(DATA_DIR=folder,TESTING=True,AUTH_BACKEND='local',STORAGE_BACKEND='sqlite',CATVTON_URL='http://127.0.0.1:8001' if real else 'http://mock-gpu.test',CATVTON_API_KEY=(ROOT/'instance/gpu.key').read_text().strip() if real else '')
    app=create_app(config)
    @app.get('/test-fixture')
    def photo():
        if real:
            name='male_person_19.png' if request.args.get('bare')=='1' else 'male_person_22.png'
            return send_file(ROOT.parent/'MODEL PICTURES/male'/name,mimetype='image/png')
        return send_file(io.BytesIO(fixture),mimetype='image/jpeg')
    @app.get('/test-garment')
    def garment():
        if real:
            product=next(p for p in Store(ROOT/'instance').all('products') if p['name']=='Forest Green Twill Shirt')
            return send_file(ROOT/'instance/images'/product['image'])
        return send_file(io.BytesIO(fixture),mimetype='image/jpeg')
    @app.get('/test-failure')
    def failure():mode['fail']=True;return jsonify(ok=True)
    if not real:
        stack.enter_context(patch('requests.post',side_effect=fake_post))
        stack.enter_context(patch('requests.get',side_effect=fake_get))
    app.run(host='127.0.0.1',port=5017,use_reloader=False,threaded=True)
