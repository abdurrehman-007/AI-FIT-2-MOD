import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parent

def _vendor_usable(root):
    if sys.version_info[:2] != (3,12):
        return False
    marker = root / '.vendor' / 'flask' / '__init__.py'
    try:
        return marker.is_file() and len(marker.read_bytes()) > 0
    except OSError:
        return False

if _vendor_usable(ROOT):
    sys.path.insert(0, str(ROOT / '.vendor'))
if (ROOT / '.localdeps').is_dir():
    sys.path.insert(0, str(ROOT / '.localdeps'))
import csv
import hashlib
import hmac
import io
import json
import math
import os
import secrets
import smtplib
import ssl
import time
import uuid
from collections import Counter
from datetime import timedelta
from functools import wraps
from urllib.parse import quote, urlparse
from email.message import EmailMessage
from itsdangerous import BadSignature, URLSafeTimedSerializer
from flask import Flask, jsonify, request, session, render_template, send_file, abort
from PIL import Image, ImageOps, UnidentifiedImageError
from werkzeug.security import generate_password_hash, check_password_hash
from analytics import extra_analytics
from storage import Store
from sizing import DEFAULT_CHART, DEFAULT_CHARTS, SIZES, measurements, recommend, store_charts, validate_charts, chart_fields


def create_app(test_config=None):
    app = Flask(__name__)
    app.config.update(DATA_DIR=os.getenv('DATA_DIR',str(ROOT/'instance')), STORAGE_BACKEND=os.getenv('STORAGE_BACKEND','sqlite'), AUTH_BACKEND=os.getenv('AUTH_BACKEND','local'), MAX_CONTENT_LENGTH=12*1024*1024, SESSION_COOKIE_HTTPONLY=True, SESSION_COOKIE_SAMESITE='Lax', SESSION_COOKIE_SECURE=os.getenv('COOKIE_SECURE','0')=='1', PERMANENT_SESSION_LIFETIME=timedelta(hours=12), CATVTON_URL=os.getenv('CATVTON_URL',''), CATVTON_API_KEY=os.getenv('CATVTON_API_KEY',''), MODEL_PATH=os.getenv('SIZE_MODEL_PATH',str(ROOT/'models'/'size.joblib')), PUBLIC_URL=os.getenv('PUBLIC_URL','http://127.0.0.1:5000'))
    if test_config:
        app.config.update(test_config)
    data_dir = Path(app.config['DATA_DIR'])
    data_dir.mkdir(parents=True,exist_ok=True)
    (data_dir/'images').mkdir(exist_ok=True)
    keyfile = data_dir/'session.key'
    if not keyfile.exists():
        try:
            with keyfile.open('x') as f:
                f.write(secrets.token_hex(32))
        except FileExistsError:
            pass
    app.secret_key = os.getenv('SECRET_KEY') or keyfile.read_text()
    db = Store(data_dir,app.config['STORAGE_BACKEND'])
    app.store = db
    demo = {'id':'demo','name':'AI-FIT Studio','owner':None,'chart':DEFAULT_CHART,'charts':DEFAULT_CHARTS,'chart_status':'illustrative','baseline_orders':0,'baseline_returns':0}
    db.create('stores','demo',demo)
    for i,(name,category,color,price) in enumerate([('Everyday cotton shirt','upper','#d9ddd3',3290),('Midnight tailored shirt','upper','#292f3b',4490),('Sand relaxed kurta','overall','#b7a58d',5290),('Indigo straight trousers','lower','#455c77',3990)]):
        pid = 'demo-'+str(i)
        db.create('products',pid,{'id':pid,'store':'demo','sku':pid,'name':name,'category':category,'color':color,'price':price,'sizes':SIZES,'stock':25,'image':'','sample':True,'active':True})

    def user():
        account = db.get('users',session.get('uid','_'))
        if account and session.get('password_version') == account.get('password_version'):
            return account

    def require_user(fn):
        @wraps(fn)
        def wrapper(*a,**kw):
            if not user():
                return jsonify(error='Sign in to continue.'),401
            return fn(*a,**kw)
        return wrapper

    def owned_store():
        u = user()
        st = db.get('stores',u.get('store','_')) if u else None
        if not st or st['owner'] != u['id']:
            abort(403,description='Retailer access required.')
        return st

    def event(kind,store,**extra):
        if kind == 'tryon' and user():
            extra['customer_key'] = hmac.new(app.secret_key.encode(), (store+':'+user()['id']).encode(), hashlib.sha256).hexdigest()
        row = {'id':uuid.uuid4().hex,'kind':kind,'store':store,'time':time.time(), **extra}
        db.put('events',row['id'],row)
        return row

    def image_bytes(stream):
        try:
            img = Image.open(stream)
            if img.format not in ('JPEG','PNG') or img.width*img.height>20_000_000:
                raise ValueError('Use a JPEG or PNG under 20 megapixels.')
            img.load()
            img = ImageOps.exif_transpose(img)
            transparent = 'A' in img.getbands() or 'transparency' in img.info
            img = img.convert('RGBA' if transparent else 'RGB')
            if min(img.size)<128:
                raise ValueError('Use an image at least 128 pixels on each side.')
            img.thumbnail((1536,2048))
            output = io.BytesIO()
            img.save(output,format='PNG' if transparent else 'JPEG',quality=92)
            return output.getvalue()
        except (UnidentifiedImageError,OSError,Image.DecompressionBombError):
            raise ValueError('The uploaded file is not a valid JPEG or PNG.')

    def save_image(content,kind,owner):
        key = uuid.uuid4().hex
        (data_dir/'images'/key).write_bytes(content)
        db.put('images',key,{'id':key,'kind':kind,'owner':owner,'created':time.time()})
        return key

    @app.before_request
    def csrf():
        if 'csrf' not in session:
            session['csrf']=secrets.token_urlsafe(32)
        if request.method in ('POST','PUT','DELETE','PATCH') and not secrets.compare_digest(request.headers.get('X-CSRF-Token',''),session['csrf']):
            abort(403,description='Session expired. Refresh the page and try again.')

    @app.after_request
    def headers(response):
        response.headers['X-Content-Type-Options']='nosniff'
        response.headers['Referrer-Policy']='same-origin'
        response.headers['X-Frame-Options']='SAMEORIGIN'
        response.headers['Content-Security-Policy']="default-src 'self'; img-src 'self' blob: data:; style-src 'self' 'unsafe-inline'; script-src 'self'; connect-src 'self'; frame-ancestors 'self'; base-uri 'self'; form-action 'self'"
        if request.path.startswith('/api') or request.path.startswith('/media'):
            response.headers['Cache-Control']='no-store'
        return response

    @app.errorhandler(400)
    @app.errorhandler(401)
    @app.errorhandler(403)
    @app.errorhandler(404)
    @app.errorhandler(413)
    @app.errorhandler(429)
    def error(e):
        return jsonify(error='File too large. Maximum upload is 12 MB.' if e.code==413 else e.description),e.code

    @app.errorhandler(ValueError)
    def invalid(e):
        return jsonify(error=str(e)),400

    @app.route('/')
    @app.route('/kiosk')
    def index():
        return render_template('index.html')

    @app.get('/api/bootstrap')
    def bootstrap():
        u = user()
        return jsonify(csrf=session['csrf'],user={k:v for k,v in u.items() if k not in ('password','password_version')} if u else None,stores=[{'id':s['id'],'name':s['name']} for s in db.all('stores')],tryon_ready=bool(app.config['CATVTON_URL']),model_ready=False,auth_backend=app.config['AUTH_BACKEND'])

    def establish(u):
        session.clear()
        session.update(uid=u['id'],csrf=secrets.token_urlsafe(32),password_version=u.get('password_version'))
        session.permanent=True
        return jsonify(ok=True)

    reset_tokens = URLSafeTimedSerializer(app.secret_key, salt='aifit-password-reset')

    def password_fingerprint(account):
        return hashlib.sha256(account['password'].encode()).hexdigest()

    def request_password_reset(body):
        email = str(body.get('email','')).strip().lower()
        if '@' not in email or len(email)>254:
            raise ValueError('Enter a valid email address.')
        if app.config['AUTH_BACKEND']=='firebase':
            import requests
            key = os.getenv('FIREBASE_WEB_API_KEY')
            if not key:
                return jsonify(error='Password reset email is not configured.'),503
            try:
                result = requests.post(
                    'https://identitytoolkit.googleapis.com/v1/accounts:sendOobCode',
                    params={'key':key}, json={'requestType':'PASSWORD_RESET','email':email}, timeout=10)
                if not result.ok and result.json().get('error', {}).get('message') != 'EMAIL_NOT_FOUND':
                    return jsonify(error='Password reset email is temporarily unavailable.'),503
            except (requests.RequestException, ValueError):
                return jsonify(error='Password reset email is temporarily unavailable.'),503
            return jsonify(ok=True)

        host, sender = os.getenv('SMTP_HOST'), os.getenv('SMTP_FROM')
        if not host or not sender:
            return jsonify(error='Password reset email is not configured. Contact your AI-FIT administrator.'),503
        uid = hashlib.sha256(email.encode()).hexdigest()
        account = db.get('users',uid)
        if account and account.get('password'):
            minute = int(time.time()//60)
            if db.create('reset_limits', uid, {'minute':minute}) or db.get('reset_limits',uid).get('minute') != minute:
                db.put('reset_limits',uid,{'minute':minute})
                token = reset_tokens.dumps([uid,password_fingerprint(account)])
                link = app.config['PUBLIC_URL'].rstrip('/') + '/#reset=' + quote(token,safe='')
                message = EmailMessage()
                message['From'], message['To'] = sender, account['email']
                message['Subject'] = 'Reset your AI-FIT password'
                message.set_content(f'Use this link within 30 minutes to reset your password:\n\n{link}\n\nIf you did not request a reset, you can ignore this email.')
                try:
                    with smtplib.SMTP_SSL(host,int(os.getenv('SMTP_PORT','465')),timeout=10,context=ssl.create_default_context()) as smtp:
                        if os.getenv('SMTP_USER'):
                            smtp.login(os.environ['SMTP_USER'],os.environ.get('SMTP_PASSWORD',''))
                        smtp.send_message(message)
                except (OSError, ValueError):
                    app.logger.warning('Could not send password reset email')
        return jsonify(ok=True)

    def reset_password(body):
        password = str(body.get('password',''))
        if not 10<=len(password)<=128:
            raise ValueError('Use a password with 10–128 characters.')
        error = 'This reset link is invalid or expired. Request a new link.'
        try:
            uid, fingerprint = reset_tokens.loads(str(body.get('token','')),max_age=1800)
        except BadSignature:
            raise ValueError(error)
        account = db.get('users',uid)
        if not account or not account.get('password') or not secrets.compare_digest(fingerprint,password_fingerprint(account)):
            raise ValueError(error)
        if not db.replace_password(uid,account['password'],generate_password_hash(password),secrets.token_hex(16)):
            raise ValueError(error)
        db.delete('limits',uid)
        session.pop('uid',None)
        return jsonify(ok=True)

    @app.post('/api/auth/<action>')
    def auth(action):
        if action=='logout':
            session.clear()
            return jsonify(ok=True)
        body = request.get_json() or {}
        if action=='forgot-password':
            return request_password_reset(body)
        if action=='reset-password' and app.config['AUTH_BACKEND']=='local':
            return reset_password(body)
        if app.config['AUTH_BACKEND']=='firebase':
            if action not in ('firebase','register','login'):
                raise ValueError('Unknown authentication action.')
            import firebase_admin
            from firebase_admin import auth as fauth
            if not firebase_admin._apps:
                firebase_admin.initialize_app()
            token=body.get('idToken','')
            if action in ('register','login'):
                import requests
                api_key=os.getenv('FIREBASE_WEB_API_KEY','')
                if not api_key:
                    return jsonify(error='Firebase email sign-in needs FIREBASE_WEB_API_KEY on the server.'),503
                endpoint='signUp' if action=='register' else 'signInWithPassword'
                try:
                    result=requests.post('https://identitytoolkit.googleapis.com/v1/accounts:'+endpoint,params={'key':api_key},json={'email':body.get('email',''),'password':body.get('password',''),'returnSecureToken':True},timeout=20)
                    result.raise_for_status()
                    token=result.json()['idToken']
                except (requests.RequestException,KeyError):
                    abort(401,description='Firebase could not sign you in. Check your credentials or account status.')
            try:
                claims=fauth.verify_id_token(token,check_revoked=True)
            except Exception:
                abort(401,description='Invalid Firebase token.')
            uid=claims['uid']
            u=db.get('users',uid)
            if not u:
                u={'id':uid,'email':claims.get('email',''),'name':str(body.get('name') or claims.get('name') or 'Shopper')[:80],'store':None}
                db.create('users',uid,u)
            return establish(u)
        email=str(body.get('email','')).strip().lower()
        password=str(body.get('password',''))
        if '@' not in email or len(email)>254:
            raise ValueError('Enter a valid email address.')
        uid=hashlib.sha256(email.encode()).hexdigest()
        # Persistent per-account throttling; no client-controlled forwarding headers.
        limit=db.get('limits',uid) or {'id':uid,'attempts':0,'until':0}
        if limit['until']>time.time() and limit['attempts']>=10:
            abort(429,description='Too many attempts. Try again in 15 minutes.')
        if limit['until']<time.time():
            limit.update(attempts=0,until=time.time()+900)
        limit['attempts']+=1
        db.put('limits',uid,limit)
        u=db.get('users',uid)
        if action=='register':
            if len(password)<10 or len(password)>128:
                raise ValueError('Use a password with 10–128 characters.')
            name=str(body.get('name','')).strip()[:80]
            if not name:
                raise ValueError('Enter your name.')
            u={'id':uid,'email':email,'name':name,'password':generate_password_hash(password),'store':None}
            if not db.create('users',uid,u):
                raise ValueError('An account already exists for this email.')
        elif action!='login' or not u or not check_password_hash(u.get('password',''),password):
            abort(401,description='Email or password is incorrect.')
        db.delete('limits',uid)
        return establish(u)

    @app.post('/api/store')
    @require_user
    def new_store():
        u=user()
        if u.get('store'):
            raise ValueError('You already have a store.')
        name=str((request.get_json() or {}).get('name','')).strip()[:80]
        if not name:
            raise ValueError('Enter a store name.')
        sid=u['id']
        st={'id':sid,'name':name,'owner':u['id'],'chart':DEFAULT_CHART,'charts':DEFAULT_CHARTS,'chart_status':'illustrative','baseline_orders':0,'baseline_returns':0}
        if not db.create('stores',sid,st):
            raise ValueError('A store already exists for this account.')
        u['store']=sid
        db.put('users',u['id'],u)
        return jsonify(store=st)

    @app.get('/api/products')
    def products():
        sid=request.args.get('store','demo')
        st=db.get('stores',sid)
        public={k:st[k] for k in ('id','name','chart')} if st else None
        if public is not None:
            public.update(charts=store_charts(st), chart_units='in', chart_status=st.get('chart_status','legacy chart converted from cm; verify with retailer'))
        return jsonify(products=[p for p in db.all('products') if p['store']==sid and p['active']],store=public)

    def product_data(body,sid,pid=None):
        name=str(body.get('name','')).strip()[:120]
        sku=str(body.get('sku','')).strip()[:80]
        category=body.get('category','upper')
        previous=db.get('products',pid or sid+'-'+hashlib.sha256(sku.encode()).hexdigest()[:24])
        upper_length=(body.get('upper_length') or (previous or {}).get('upper_length','standard')) if category=='upper' else 'standard'
        if upper_length not in ('standard','long'):
            raise ValueError('Choose standard or long upper garment length.')
        sizes=body.get('sizes',[])
        if isinstance(sizes,str): sizes=[s.strip() for s in sizes.split('|') if s.strip()]
        if not name or not sku or category not in ('upper','lower','overall') or not sizes or any(s not in SIZES for s in sizes):
            raise ValueError('Provide name, SKU, category (upper/lower/overall), and valid sizes (XS–XXL).')
        try:
            price=float(body.get('price',0)); stock=int(body.get('stock',0))
        except (ValueError,TypeError):
            raise ValueError('Price and stock must be numbers.')
        if not math.isfinite(price) or price<0 or stock<0:
            raise ValueError('Price and stock cannot be negative.')
        image=str(body.get('image',''))
        if image:
            asset=db.get('images',image)
            if not asset or asset['owner']!=user()['id'] or asset['kind']!='garment':
                raise ValueError('Upload a garment image owned by your account.')
        color=str(body.get('color','#c9d1ce'))
        if len(color)!=7 or color[0]!='#' or any(c not in '0123456789abcdefABCDEF' for c in color[1:]):
            raise ValueError('Color must be a hex value such as #334455.')
        return {'id':pid or sid+'-'+hashlib.sha256(sku.encode()).hexdigest()[:24],'store':sid,'sku':sku,'name':name,'category':category,'upper_length':upper_length,'sizes':sizes,'price':round(price,2),'stock':stock,'image':image,'color':color,'sample':False,'active':True}

    @app.post('/api/products')
    @require_user
    def add_product():
        st=owned_store(); p=product_data(request.get_json() or {},st['id'])
        db.put('products',p['id'],p)
        return jsonify(product=p)

    @app.delete('/api/products/<pid>')
    @require_user
    def delete_product(pid):
        st=owned_store(); p=db.get('products',pid)
        if not p or p['store']!=st['id']: abort(404)
        p['active']=False; db.put('products',pid,p)
        return jsonify(ok=True)

    @app.post('/api/inventory/import')
    @require_user
    def import_csv():
        st=owned_store(); file=request.files.get('file')
        if not file: raise ValueError('Choose a CSV file.')
        try: text=file.read(1_000_001).decode('utf-8-sig')
        except UnicodeDecodeError: raise ValueError('Save the CSV as UTF-8.')
        if len(text)>1_000_000: raise ValueError('CSV must be under 1 MB.')
        reader=csv.DictReader(io.StringIO(text))
        if not {'sku','name','category','price','stock','sizes'}.issubset(reader.fieldnames or []):
            raise ValueError('Missing required CSV columns. Download the template.')
        pending=[]; seen=set()
        for line,row in enumerate(reader,2):
            if line>201: raise ValueError('Import up to 200 products at a time.')
            try: p=product_data(row,st['id'])
            except ValueError as e: raise ValueError(f'Row {line}: {e}')
            previous=db.get('products',p['id'])
            if previous and not row.get('image'):
                p['image']=previous.get('image','')
            if p['sku'] in seen: raise ValueError(f'Row {line}: duplicate SKU.')
            seen.add(p['sku']); pending.append((p['id'],p))
        if not pending: raise ValueError('The CSV contains no products.')
        db.put_many('products',pending)
        return jsonify(imported=len(pending))

    @app.get('/api/inventory/template')
    def template():
        return send_file(io.BytesIO(b'sku,name,category,price,stock,sizes,color,image,upper_length\nK001,Cotton Kurta,overall,4990,20,XS|S|M|L|XL|XXL,#b7a58d,,standard\n'),mimetype='text/csv',as_attachment=True,download_name='aifit-inventory.csv')

    @app.post('/api/uploads')
    @require_user
    def upload():
        kind=request.form.get('kind','person')
        if kind not in ('person','garment'): raise ValueError('Invalid image type.')
        if kind=='garment': owned_store()
        elif request.form.get('consent')!='true': raise ValueError('Confirm you have permission to use this photo.')
        file=request.files.get('file')
        if not file: raise ValueError('Choose a JPEG or PNG.')
        key=save_image(image_bytes(file.stream),kind,user()['id'])
        return jsonify(id=key,url='/media/'+key)

    @app.get('/media/<key>')
    def media(key):
        record=db.get('images',key)
        if not record: abort(404)
        if record['kind']!='garment' and (not user() or record['owner']!=user()['id']): abort(403)
        if record['kind']!='garment' and record['created']<time.time()-86400:
            abort(404,description='Photo expired after 24 hours.')
        path=data_dir/'images'/key
        if not path.is_file(): abort(404)
        content = path.read_bytes()
        is_png = content.startswith(b'\x89PNG\r\n\x1a\n')
        return send_file(io.BytesIO(content),mimetype='image/png' if is_png else 'image/jpeg',as_attachment=request.args.get('download')=='1',download_name='ai-fit.png' if is_png else 'ai-fit.jpg')

    @app.delete('/api/uploads/<key>')
    @require_user
    def delete_upload(key):
        row=db.get('images',key)
        if not row or row['owner']!=user()['id']: abort(404)
        if row['kind']=='garment' and any(p.get('image')==key and p['active'] for p in db.all('products')):
            raise ValueError('Remove this image from active products first.')
        (data_dir/'images'/key).unlink(missing_ok=True); db.delete('images',key)
        return jsonify(ok=True)

    @app.post('/api/size')
    def size():
        body=request.get_json() or {}; p=db.get('products',str(body.get('product','')))
        if not p or not p['active']: abort(404)
        if body.get('consent') is not True: raise ValueError('Consent is required to process body measurements.')
        st=db.get('stores',p['store'])
        category=p.get('category','overall')
        if body.get('category',category) != category:
            raise ValueError('Select a garment from the chosen body category.')
        chart=store_charts(st)[category]
        values=measurements(body,category,chart_fields(chart,category))
        model_path=app.config['MODEL_PATH'] if category in ('upper','lower') else None
        result=recommend(values,chart,p['sizes'],model_path=model_path,category=category,retailer=st['id'])
        result['chart_status']=st.get('chart_status','legacy chart converted from cm; verify with retailer')
        # Keep only coarse aggregate segments, never raw body measurements.
        segment='Not collected'
        if 'chest' in values and 'hips' in values:
            segment='Balanced' if abs(values['chest']-values['hips'])<6/2.54 else 'Chest-dominant' if values['chest']>values['hips'] else 'Hip-dominant'
        event('size',p['store'],product=p['id'],size=result['size'],segment=segment,category=p['category'])
        return jsonify(result)

    @app.post('/api/tryon')
    @require_user
    def tryon():
        if not app.config['CATVTON_URL']: return jsonify(error='Photo generation needs a CatVTON GPU service. Follow the setup guide to connect it.'),503
        body=request.get_json() or {}; p=db.get('products',str(body.get('product',''))); asset=db.get('images',str(body.get('photo','')))
        if not p or not p['active'] or not p.get('image'): raise ValueError('Select a product with a real garment image.')
        if not asset or asset['owner']!=user()['id'] or asset['kind']!='person' or asset['created']<time.time()-86400: raise ValueError('Upload a new photo first.')
        request_id=body.get('request_id') or secrets.token_hex(16)
        if not isinstance(request_id,str) or len(request_id)!=32 or any(c not in '0123456789abcdef' for c in request_id):
            raise ValueError('Invalid generation identifier.')
        for old in db.all('tryon_jobs'):
            if old.get('created',0)<time.time()-3600:db.delete('tryon_jobs',old['id'])
        job=dict(id=request_id,owner=user()['id'],created=time.time(),status='running')
        if not db.create('tryon_jobs',request_id,job):return jsonify(error='This generation has already been submitted.'),409
        def finish_job(status,result=None):
            job.update(status=status,finished=time.time())
            if result is not None:job['result']=result
            db.put('tryon_jobs',request_id,job)
        start=time.time()
        import requests
        try:
            response=requests.post(app.config['CATVTON_URL'].rstrip('/')+'/infer',headers={'Authorization':'Bearer '+app.config['CATVTON_API_KEY'],'X-AIFIT-Request-ID':request_id},files={'person':('person.jpg',(data_dir/'images'/asset['id']).read_bytes(),'image/jpeg'),'garment':('garment.jpg',(data_dir/'images'/p['image']).read_bytes(),'image/jpeg')},data={'category':p['category'],'upper_length':p.get('upper_length','standard')},timeout=(10,360))
            status=getattr(response,'status_code',200)
            if isinstance(status,int) and status!=200:
                message='The try-on service could not generate this image. Check the photo and try again.'
                try:
                    payload=response.json()
                    if isinstance(payload,dict) and payload.get('error'):
                        message=str(payload['error'])
                except ValueError:
                    pass
                event('tryon_failed',p['store'],product=p['id'],seconds=round(time.time()-start,2))
                finish_job('failed')
                return jsonify(error=message),502 if status>=500 else status
            response.raise_for_status()
            content=image_bytes(io.BytesIO(response.content))
        except (requests.RequestException,ValueError,OSError):
            finish_job('failed')
            event('tryon_failed',p['store'],product=p['id'],seconds=round(time.time()-start,2))
            return jsonify(error='The try-on service could not generate this image. Check the photo and try again.'),502
        try:
            key=save_image(content,'result',user()['id'])
            result=event('tryon',p['store'],product=p['id'],category=p['category'],seconds=round(time.time()-start,2),result=key)
            payload=dict(url='/media/'+key,id=key,tryon=result['id'])
            finish_job('complete',payload)
            return jsonify(**payload,request_id=request_id)
        except Exception:
            finish_job('failed')
            raise

    @app.get('/api/tryon-progress/<request_id>')
    @require_user
    def tryon_progress(request_id):
        job=db.get('tryon_jobs',request_id)
        if not job or job.get('owner')!=user()['id'] or time.time()-job['created']>3600:
            return jsonify(error='Generation progress unavailable.'),404
        elapsed=max(0,(job.get('finished') or time.time())-job['created'])
        data=dict(status=job['status'],stage=job['status'],percent=100 if job['status']=='complete' else 0,
            elapsed_seconds=round(elapsed,1),remaining_seconds=None,estimated_total_seconds=None,estimated=True)
        if job['status']=='running':
            import requests
            try:
                response=requests.get(app.config['CATVTON_URL'].rstrip('/')+'/progress/'+request_id,
                    headers={'Authorization':'Bearer '+app.config['CATVTON_API_KEY']},timeout=(2,2))
                if response.status_code==200:
                    snapshot=response.json()
                    if isinstance(snapshot,dict):
                        for field in ('stage','percent','remaining_seconds','estimated_total_seconds','attempt'):
                            if field in snapshot:data[field]=snapshot[field]
                        # The image still has to be verified and saved by this app.
                        data['percent']=min(99,max(0,int(data['percent'])))
                        if snapshot.get('status')=='complete':data.update(stage='saving',percent=99)
                else:data.update(stage='starting',tracking_available=False)
            except (requests.RequestException,ValueError,TypeError):
                data.update(stage='starting',tracking_available=False)
        if job['status']=='complete':data['result']=job.get('result')
        response=jsonify(data);response.headers['Cache-Control']='no-store'
        return response

    @app.get('/api/chart-presets')
    def chart_presets():
        return jsonify(presets=json.loads((ROOT/'datasets/pakistan_charts/verified_presets.json').read_text(encoding='utf-8')))

    @app.get('/api/analytics')
    @require_user
    def analytics():
        st=owned_store(); days=int(request.args.get('days',30))
        if days not in (7,30,90): raise ValueError('Choose 7, 30 or 90 days.')
        now=time.time()
        rows=[e for e in db.all('events') if e['store']==st['id'] and now-days*86400<=e['time']<=now]
        tries=[e for e in rows if e['kind']=='tryon']; sales=[e for e in rows if e['kind']=='sale']; returns=[e for e in rows if e['kind']=='return']
        linked={e['tryon'] for e in sales if e.get('tryon') in {t['id'] for t in tries}}
        buy_orders=[e for e in sales if e.get('source')=='tryon_buy']
        product_performance=[]
        for pid in sorted({e.get('product') for e in tries if e.get('product')}):
            pt=[e for e in tries if e.get('product')==pid]; ids={e['id'] for e in pt}
            ps=[e for e in sales if e.get('tryon') in ids]
            product=db.get('products',pid) or {}
            product_performance.append({'product':pid,'name':product.get('name','Unavailable garment'),
                'tryons':len(pt),'orders':len(ps),'converted_tryons':len({e['tryon'] for e in ps}),
                'conversion':round(len({e['tryon'] for e in ps})/len(pt)*100,1)})
        commerce={'buy_orders':len(buy_orders),'manual_sales':len(sales)-len(buy_orders),
            'recorded_value':round(sum(e.get('amount',0) for e in sales),2),'priced_orders':sum('amount' in e for e in sales),
            'linked_tryons':len(linked),'unconverted_tryons':len(tries)-len(linked),
            'products':sorted(product_performance,key=lambda p:(-p['orders'],-p['tryons'])),
            'daily_orders':dict(Counter(time.strftime('%Y-%m-%d',time.gmtime(e['time'])) for e in sales))}
        baseline=st['baseline_returns']/st['baseline_orders']*100 if st['baseline_orders'] else None
        extra=extra_analytics(rows,db.all('products'),st['id'],store_charts(st),app.config['MODEL_PATH'])
        # Display-only sales samples: never enter event storage or real conversion metrics.
        sample_counts=[2,4,3,5,2,6,4]
        sales_preview={'demo':True,'orders':sum(sample_counts),'value':93600,
            'daily_orders':{time.strftime('%Y-%m-%d',time.gmtime(now-(6-i)*86400)):v for i,v in enumerate(sample_counts)}}
        return jsonify(data_mode="live",sales_preview=sales_preview,commerce=commerce,**extra,tryons=len(tries),sales=len(sales),returns=len(returns),conversion=round(len(linked)/len(tries)*100,1) if tries else None,return_rate=round(len(returns)/len(sales)*100,1) if sales else None,baseline=baseline,inference_seconds=round(sum(e['seconds'] for e in tries)/len(tries),1) if tries else None,failed=sum(e['kind']=='tryon_failed' for e in rows),sizes=dict(Counter(e['size'] for e in rows if e['kind']=='size')),segments=dict(Counter(e['segment'] for e in rows if e['kind']=='size' and e.get('segment') not in (None,'Not collected'))),popular=dict(Counter(e['product'] for e in tries)),trend=dict(Counter(time.strftime('%Y-%m-%d',time.gmtime(e['time'])) for e in tries)),recent=sorted([{k:e[k] for k in ('id','kind','time','reference') if k in e} for e in rows if e['kind'] in ('tryon','sale','return')],key=lambda e:e['time'],reverse=True)[:30],store=st)

    @app.post('/api/retail/events')
    @require_user
    def retail_event():
        st=owned_store(); body=request.get_json() or {}; kind=body.get('kind'); reference=str(body.get('reference','')).strip()[:100]
        if kind not in ('sale','return') or not reference: raise ValueError('Provide a sale/return and a unique order reference.')
        rows=[e for e in db.all('events') if e['store']==st['id']]
        key=hashlib.sha256((st['id']+kind+reference).encode()).hexdigest()
        if kind=='return' and not any(e['kind']=='sale' and e.get('reference')==reference for e in rows): raise ValueError('Record the original sale first.')
        tid=str(body.get('tryon','')).strip()
        if tid and not any(e['id']==tid and e['kind']=='tryon' for e in rows): raise ValueError('Try-on ID does not belong to this store.')
        row={'id':key,'kind':kind,'store':st['id'],'time':time.time(),'reference':reference,'tryon':tid}
        if not db.create('events',key,row): raise ValueError('This order event was already recorded.')
        return jsonify(ok=True)

    @app.put('/api/store/settings')
    @require_user
    def settings():
        st=owned_store(); body=request.get_json() or {}
        if 'chart' in body: raise ValueError('Use separate upper/lower charts and chart_units=in.')
        charts=store_charts(st)
        if 'charts' in body:
            if body.get('chart_units') != 'in': raise ValueError('Chart units must be in (inches).')
            incoming=body['charts']
            if isinstance(incoming,dict) and set(incoming)=={'upper','lower'}:
                incoming={**incoming,'overall':charts['overall']}
            charts=validate_charts(incoming)
        orders=int(body.get('baseline_orders',st['baseline_orders'])); returns=int(body.get('baseline_returns',st['baseline_returns']))
        if orders<0 or returns<0 or returns>orders: raise ValueError('Baseline returns cannot exceed orders.')
        st.update(charts=charts,baseline_orders=orders,baseline_returns=returns)
        if 'charts' in body: st['chart_status']='retailer supplied'
        db.put('stores',st['id'],st); return jsonify(ok=True)

    @app.get('/api/qr/<sid>')
    def qr(sid):
        if not db.get('stores',sid): abort(404)
        import qrcode
        from qrcode.image.pil import PilImage
        payload=app.config['PUBLIC_URL'].rstrip('/')+'/?store='+sid
        qr=qrcode.QRCode(error_correction=qrcode.constants.ERROR_CORRECT_M, box_size=8, border=2)
        qr.add_data(payload)
        qr.make(fit=True)
        out=io.BytesIO()
        qr.make_image(image_factory=PilImage, fill_color='black', back_color='white').save(out, format='PNG')
        out.seek(0)
        return send_file(out,mimetype='image/png',download_name='ai-fit-store-qr.png',as_attachment=request.args.get('download')=='1')

    @app.cli.command('cleanup-photos')
    def cleanup():
        for row in db.all('images'):
            if row['kind']!='garment' and row['created']<time.time()-86400:
                (data_dir/'images'/row['id']).unlink(missing_ok=True); db.delete('images',row['id'])
        print('Expired personal photos removed.')

    return app


if __name__=='__main__':
    create_app().run(host='127.0.0.1',port=int(os.getenv('PORT','5000')),debug=False)
