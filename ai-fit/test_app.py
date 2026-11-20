import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).parent))
from app import create_app
import io
import tempfile
import time
import unittest
from unittest.mock import patch,Mock
from PIL import Image
from sizing import measurements,recommend,DEFAULT_CHART,SIZES


def jpeg():
    stream=io.BytesIO();Image.new('RGB',(256,384),'#778899').save(stream,format='JPEG');stream.seek(0);return stream


class ApplicationTests(unittest.TestCase):
    def test_progress_ownership_completion_and_duplicate_submission(self):
        st,p=self.product();g=self.upload('garment').json;person=self.upload().json
        p['image']=g['id'];self.app.store.put('products',p['id'],p)
        self.app.config['CATVTON_URL']='http://gpu.test'
        rid='a'*32
        response=Mock(content=jpeg().getvalue(),status_code=200,headers={});response.raise_for_status.return_value=None
        with patch('requests.post',return_value=response) as gpu:
            result=self.post('/tryon',dict(photo=person['id'],product=p['id'],request_id=rid))
            self.assertEqual(result.status_code,200)
            self.assertEqual(gpu.call_args.kwargs['headers']['X-AIFIT-Request-ID'],rid)
            self.assertEqual(self.post('/tryon',dict(photo=person['id'],product=p['id'],request_id=rid)).status_code,409)
            self.assertEqual(gpu.call_count,1)
        progress=self.client.get('/api/tryon-progress/'+rid)
        self.assertEqual(progress.json['percent'],100)
        self.assertEqual(progress.json['status'],'complete')
        self.assertEqual(progress.json['result']['id'],result.json['id'])
        self.assertEqual(progress.headers['Cache-Control'],'no-store')
        other=self.app.test_client();self.register('progress-other@example.test',other)
        self.assertEqual(other.get('/api/tryon-progress/'+rid).status_code,404)

    def test_gpu_completion_is_not_100_until_app_has_saved_result(self):
        self.register();account=self.client.get('/api/bootstrap').json['user']
        rid='b'*32
        self.app.store.put('tryon_jobs',rid,dict(id=rid,owner=account['id'],created=time.time(),status='running'))
        self.app.config['CATVTON_URL']='http://gpu.test'
        response=Mock(status_code=200);response.json.return_value=dict(status='complete',stage='complete',percent=100,remaining_seconds=0)
        with patch('requests.get',return_value=response):
            snapshot=self.client.get('/api/tryon-progress/'+rid).json
        self.assertEqual(snapshot['status'],'running')
        self.assertEqual(snapshot['stage'],'saving')
        self.assertEqual(snapshot['percent'],99)

    def test_failed_and_expired_jobs_never_report_success(self):
        self.register();account=self.client.get('/api/bootstrap').json['user'];rid='c'*32
        self.app.store.put('tryon_jobs',rid,dict(id=rid,owner=account['id'],created=time.time(),finished=time.time(),status='failed'))
        snapshot=self.client.get('/api/tryon-progress/'+rid).json
        self.assertEqual(snapshot['status'],'failed');self.assertEqual(snapshot['percent'],0)
        self.app.store.put('tryon_jobs',rid,dict(id=rid,owner=account['id'],created=time.time()-3601,status='running'))
        self.assertEqual(self.client.get('/api/tryon-progress/'+rid).status_code,404)

    def test_long_upper_length_is_saved_preserved_and_validated(self):
        st,p=self.product()
        body={**p,'upper_length':'long'}
        saved=self.post('/products',body)
        self.assertEqual(saved.status_code,200)
        self.assertEqual(saved.json['product']['upper_length'],'long')
        body.pop('upper_length')
        self.assertEqual(self.post('/products',body).json['product']['upper_length'],'long')
        body['upper_length']='unknown'
        self.assertEqual(self.post('/products',body).status_code,400)

    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory()
        self.app=create_app({'TESTING':True,'DATA_DIR':self.tmp.name,'MODEL_PATH':self.tmp.name+'/none','CATVTON_URL':'','AUTH_BACKEND':'local','STORAGE_BACKEND':'sqlite'})
        self.client=self.app.test_client()
        self.token=self.client.get('/api/bootstrap').json['csrf']

    def tearDown(self): self.tmp.cleanup()
    def post(self,path,body=None,client=None):
        c=client or self.client
        token=c.get('/api/bootstrap').json['csrf']
        return c.post('/api'+path,json=body or {},headers={'X-CSRF-Token':token})
    def register(self,email='retailer@example.test',client=None):
        return self.post('/auth/register',{'email':email,'name':'Test Retailer','password':'A-long-password-123'},client)
    def store(self):
        self.register();return self.post('/store',{'name':'Test Boutique'}).json['store']
    def product(self):
        st=self.store()
        p=self.post('/products',{'name':'Cotton Shirt','sku':'SH01','category':'upper','price':3200,'stock':12,'sizes':['S','M','L']}).json['product']
        return st,p
    def upload(self,kind='person'):
        token=self.client.get('/api/bootstrap').json['csrf']
        return self.client.post('/api/uploads',data={'file':(jpeg(),'photo.jpg'),'kind':kind,'consent':'true'},headers={'X-CSRF-Token':token})

    def test_demo_analytics_never_changes_live_activity(self):
        self.store()
        before=self.client.get('/api/analytics').json
        demo=self.client.get('/api/analytics?days=7&demo=1').json
        self.assertEqual(demo['data_mode'],'live')
        self.assertEqual((demo['tryons'],demo['sales'],demo['returns']),(0,0,0))
        self.assertEqual(demo['sales_preview']['orders'],26)
        self.assertIsNone(demo['conversion'])
        after=self.client.get('/api/analytics').json
        self.assertEqual(after['data_mode'],'live')
        for key in ('tryons','sales','returns','recent','popular','sizes','category_interest','customer_preferences','conversion','return_rate'):
            self.assertEqual(before[key],after[key])
        other=self.app.test_client();self.register('outsider@example.test',other)
        self.assertEqual(other.get('/api/analytics?demo=1').status_code,403)

    def test_pages_and_assets(self):
        for path in ['/','/kiosk','/static/app.js','/static/style.css','/api/products','/api/inventory/template','/api/qr/demo']:
            with self.client.get(path) as response:
                self.assertEqual(response.status_code,200,path)

    def test_csrf_and_access(self):
        self.assertEqual(self.client.post('/api/store',json={'name':'X'}).status_code,403)
        self.assertEqual(self.client.get('/api/analytics').status_code,401)

    def test_signup_login_logout(self):
        self.assertEqual(self.register().status_code,200)
        self.assertEqual(self.register().status_code,400)
        self.post('/auth/logout')
        self.assertEqual(self.post('/auth/login',{'email':'retailer@example.test','password':'bad'}).status_code,401)
        self.assertEqual(self.post('/auth/login',{'email':'retailer@example.test','password':'A-long-password-123'}).status_code,200)
        self.assertNotIn('password',self.client.get('/api/bootstrap').json['user'])

    def test_inventory_isolation(self):
        st,p=self.product(); other=self.app.test_client();self.register('other@example.test',other)
        self.assertEqual(other.get('/api/analytics').status_code,403)
        self.post('/store',{'name':'Other store'},other)
        token=other.get('/api/bootstrap').json['csrf']
        self.assertEqual(other.delete('/api/products/'+p['id'],headers={'X-CSRF-Token':token}).status_code,404)
        self.assertEqual(self.client.get('/api/products?store='+st['id']).json['products'][0]['name'],'Cotton Shirt')

    def test_csv_atomic_validation_and_import(self):
        st=self.store(); token=self.client.get('/api/bootstrap').json['csrf']
        header='sku,name,category,price,stock,sizes\n'
        for text,status in [(header+'A,Shirt,upper,20,3,S|M\nB,Bad,wrong,1,1,S\n',400),(header+'A,Shirt,upper,20,3,S|M\n',200)]:
            r=self.client.post('/api/inventory/import',data={'file':(io.BytesIO(text.encode()),'items.csv')},headers={'X-CSRF-Token':token})
            self.assertEqual(r.status_code,status,r.json)
            count=len(self.client.get('/api/products?store='+st['id']).json['products'])
            self.assertEqual(count,0 if status==400 else 1)

    def test_photo_privacy_delete_expiry(self):
        self.register();r=self.upload();self.assertEqual(r.status_code,200)
        key=r.json['id'];self.assertEqual(self.client.get('/media/'+key).status_code,200)
        stranger=self.app.test_client();self.assertEqual(stranger.get('/media/'+key).status_code,403)
        token=self.client.get('/api/bootstrap').json['csrf']
        self.assertEqual(self.client.delete('/api/uploads/'+key,headers={'X-CSRF-Token':token}).status_code,200)
        self.assertEqual(self.client.get('/media/'+key).status_code,404)
        key=self.upload().json['id'];row=self.app.store.get('images',key);row['created']=time.time()-90000;self.app.store.put('images',key,row)
        self.assertEqual(self.client.get('/media/'+key).status_code,404)
        self.app.test_cli_runner().invoke(args=['cleanup-photos'])
        self.assertFalse((Path(self.tmp.name)/'images'/key).exists())

    def test_transparent_upload_roundtrip(self):
        self.store()
        source=Image.new('RGBA',(256,256),(0,0,0,0))
        source.paste((20,40,200,255),(64,64,192,192))
        stream=io.BytesIO();source.save(stream,format='PNG');stream.seek(0)
        token=self.client.get('/api/bootstrap').json['csrf']
        response=self.client.post('/api/uploads',data={'file':(stream,'cutout.png'),'kind':'garment','consent':'true'},headers={'X-CSRF-Token':token})
        self.assertEqual(response.status_code,200)
        media=self.client.get('/media/'+response.json['id'])
        self.assertEqual(media.mimetype,'image/png')
        im=Image.open(io.BytesIO(media.data))
        self.assertEqual(im.getpixel((0,0))[3],0)
        self.assertEqual(im.getpixel((128,128)),(20,40,200,255))

    def test_invalid_image(self):
        self.register();token=self.client.get('/api/bootstrap').json['csrf']
        r=self.client.post('/api/uploads',data={'file':(io.BytesIO(b'<script>alert(1)</script>'),'fake.png'),'consent':'true'},headers={'X-CSRF-Token':token})
        self.assertEqual(r.status_code,400)

    def test_sizing_and_no_measurement_retention(self):
        data={'product':'demo-0','units':'in','height_feet':5,'height_inches':7,'weight':65,'chest':96/2.54,'waist':80/2.54,'hips':100/2.54,'consent':True}
        r=self.post('/size',data);self.assertEqual(r.status_code,200);self.assertEqual(r.json['size'],'M');self.assertIsNone(r.json['confidence'])
        rows=self.app.store.all('events');self.assertEqual(rows[0]['kind'],'size');self.assertNotIn('chest',rows[0])
        data['chest']='NaN';self.assertEqual(self.post('/size',data).status_code,400)
        data['chest']=96/2.54;data['consent']=False;self.assertEqual(self.post('/size',data).status_code,400)

    def test_disabled_inference_is_explicit(self):
        self.register();self.assertEqual(self.post('/tryon').status_code,503)
        self.assertFalse(self.app.store.all('events'))

    def test_inference_contract_and_analytics(self):
        st,p=self.product();garment=self.upload('garment').json;person=self.upload().json;p['image']=garment['id'];self.app.store.put('products',p['id'],p)
        p['upper_length']='long';self.app.store.put('products',p['id'],p)
        self.app.config['CATVTON_URL']='http://gpu.test'
        response=Mock(content=jpeg().getvalue(),status_code=200,headers={'Content-Type':'image/jpeg'});response.raise_for_status.return_value=None
        with patch('requests.post',return_value=response) as called:
            result=self.post('/tryon',{'photo':person['id'],'product':p['id']})
            self.assertEqual(result.status_code,200,result.json);self.assertEqual(called.call_args.kwargs['data']['category'],'upper')
            self.assertEqual(called.call_args.kwargs['data']['upper_length'],'long')
        self.assertEqual(self.client.get(result.json['url']).status_code,200)
        self.assertEqual(self.post('/retail/events',{'kind':'sale','reference':'order-1','tryon':result.json['tryon']}).status_code,200)
        self.assertEqual(self.post('/retail/events',{'kind':'sale','reference':'order-1'}).status_code,400)
        self.assertEqual(self.post('/retail/events',{'kind':'return','reference':'order-1'}).status_code,200)
        a=self.client.get('/api/analytics').json;self.assertEqual(a['conversion'],100);self.assertEqual(a['return_rate'],100);self.assertEqual(a['tryons'],1)

    def test_inference_failure_not_counted_as_success(self):
        import requests
        st,p=self.product();g=self.upload('garment').json;p['image']=g['id'];self.app.store.put('products',p['id'],p);person=self.upload().json
        self.app.config['CATVTON_URL']='http://gpu.test'
        with patch('requests.post',side_effect=requests.Timeout):
            self.assertEqual(self.post('/tryon',{'photo':person['id'],'product':p['id']}).status_code,502)
        a=self.client.get('/api/analytics').json;self.assertEqual(a['failed'],1);self.assertEqual(a['tryons'],0)

    def test_store_chart_validation(self):
        self.store();token=self.client.get('/api/bootstrap').json['csrf']
        r=self.client.put('/api/store/settings',json={'baseline_orders':2,'baseline_returns':3},headers={'X-CSRF-Token':token});self.assertEqual(r.status_code,400)
        r=self.client.put('/api/store/settings',json={'chart':{'S':{'chest':2}}},headers={'X-CSRF-Token':token});self.assertEqual(r.status_code,400)

    def test_restart_persistence(self):
        st,p=self.product();again=create_app({'TESTING':True,'DATA_DIR':self.tmp.name,'STORAGE_BACKEND':'sqlite'})
        self.assertEqual(again.store.get('products',p['id'])['sku'],'SH01')

    def test_training_requires_independent_label_provenance(self):
        from train_size import train
        with self.assertRaisesRegex(ValueError,'separate calibration/test'):
            train('unlabelled.csv',Path(self.tmp.name)/'model.joblib')

    def test_csv_upsert_preserves_product_photo(self):
        st,p=self.product();p['image']=self.upload('garment').json['id'];self.app.store.put('products',p['id'],p)
        token=self.client.get('/api/bootstrap').json['csrf']
        data='sku,name,category,price,stock,sizes\nSH01,Updated,upper,4000,10,S|M\n'
        r=self.client.post('/api/inventory/import',data={'file':(io.BytesIO(data.encode()),'items.csv')},headers={'X-CSRF-Token':token})
        self.assertEqual(r.status_code,200)
        self.assertEqual(self.app.store.get('products',p['id'])['image'],p['image'])


    def test_analytics_new_fields_and_scoping(self):
        st,p=self.product()
        now=time.time()
        for key,store,when,category in [('one',st['id'],now,'upper'),('foreign','foreign',now,'lower'),('future',st['id'],now+1000,'lower'),('expired',st['id'],now-100*86400,'lower')]:
            self.app.store.put('events',key,{'id':key,'store':store,'kind':'tryon','time':when,'product':p['id'],'category':category,'seconds':2,'customer_key':'private-key'})
        data=self.client.get('/api/analytics').json
        self.assertEqual(data['tryons'],1)
        self.assertEqual(data['category_popularity']['upper'],{p['id']:1})
        self.assertEqual(data['customer_preferences'],{'upper':1})
        self.assertNotIn('private-key',str(data))
        self.assertFalse(data['evaluations']['upper']['available'])
        self.assertEqual(self.client.get('/api/analytics?days=bad').status_code,400)
        presets=self.client.get('/api/chart-presets').json['presets']
        self.assertEqual(set(presets[0]['charts']),{'upper','lower'})
        self.assertEqual(self.app.store.get('stores',st['id'])['charts'],st['charts'])


    def test_retail_outcome_isolation_and_duplicate_conversion(self):
        st,p=self.product();now=time.time()
        self.app.store.put('events','own',{'id':'own','store':st['id'],'kind':'tryon','time':now,'product':p['id'],'category':'upper','seconds':1})
        self.app.store.put('events','foreign',{'id':'foreign','store':'other','kind':'tryon','time':now,'product':'secret','seconds':1})
        self.app.store.put('events','foreign-sale',{'id':'foreign-sale','store':'other','kind':'sale','time':now,'reference':'foreign-order'})
        self.assertEqual(self.post('/retail/events',{'kind':'sale','reference':'wrong','tryon':'foreign'}).status_code,400)
        self.assertEqual(self.post('/retail/events',{'kind':'return','reference':'foreign-order'}).status_code,400)
        for ref in ('one','two'):
            self.assertEqual(self.post('/retail/events',{'kind':'sale','reference':ref,'tryon':'own'}).status_code,200)
        data=self.client.get('/api/analytics').json
        self.assertEqual(data['sales'],2);self.assertEqual(data['conversion'],100)
        self.assertEqual(data['returns'],0);self.assertEqual(data['return_rate'],0)


    def test_pose_rejection_is_not_saved_or_counted_successful(self):
        st,p=self.product();p['image']=self.upload('garment').json['id'];self.app.store.put('products',p['id'],p)
        person=self.upload().json;self.app.config['CATVTON_URL']='http://gpu.test'
        before=len(self.app.store.all('images'))
        response=Mock(status_code=422);response.json.return_value={'error':'The outfit changed your pose too much.'}
        with patch('requests.post',return_value=response):
            result=self.post('/tryon',{'photo':person['id'],'product':p['id']})
        self.assertEqual(result.status_code,422);self.assertIn('pose',result.json['error'])
        self.assertEqual(len(self.app.store.all('images')),before)
        data=self.client.get('/api/analytics').json
        self.assertEqual(data['tryons'],0);self.assertEqual(data['failed'],1)


if __name__=='__main__': unittest.main(verbosity=2)
