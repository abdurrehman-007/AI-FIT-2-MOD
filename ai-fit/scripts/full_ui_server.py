"""Disposable UI fixture server. GPU output is mocked; never deploy this script."""
import sys,tempfile,io
from pathlib import Path
from unittest.mock import patch,Mock
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from app import create_app
from flask import send_file
from PIL import Image
out=io.BytesIO();Image.new('RGB',(256,384),'#336699').save(out,format='JPEG');fixture=out.getvalue()
with tempfile.TemporaryDirectory(prefix='aifit-full-ui-') as folder:
    app=create_app({'DATA_DIR':folder,'TESTING':True,'AUTH_BACKEND':'local','STORAGE_BACKEND':'sqlite','CATVTON_URL':'http://mock-gpu.test'})
    @app.get('/test-fixture')
    def image_fixture():return send_file(io.BytesIO(fixture),mimetype='image/jpeg')
    response=Mock(content=fixture,status_code=200,headers={'Content-Type':'image/jpeg'})
    response.raise_for_status.return_value=None
    with patch('requests.post',return_value=response):
        app.run(host='127.0.0.1',port=5013,use_reloader=False)
