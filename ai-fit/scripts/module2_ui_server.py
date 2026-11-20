import sys,tempfile
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from app import create_app
with tempfile.TemporaryDirectory(prefix='aifit-module2-ui-') as folder:
    app=create_app({'DATA_DIR':folder,'TESTING':True,'AUTH_BACKEND':'local','STORAGE_BACKEND':'sqlite','CATVTON_URL':''})
    app.run(host='127.0.0.1',port=5012,use_reloader=False)
