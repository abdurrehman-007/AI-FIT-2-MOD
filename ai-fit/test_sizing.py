import copy,json,tempfile,unittest
from pathlib import Path
import numpy as np
from app import create_app
from sizing import DEFAULT_CHART,DEFAULT_CHARTS,SIZES,measurements,recommend,store_charts,validate_charts
from train_size import probability_metrics,temperature_scale,train

class SizingTests(unittest.TestCase):
    def body(self,**extra):return {'units':'in','height_feet':5,'height_inches':8,'weight':65,'chest':38,'waist':32,'hips':40,**extra}
    def test_height_conversion_and_weight(self):
        v=measurements(self.body());self.assertEqual(v['height'],68);self.assertEqual(v['weight'],65);self.assertEqual(v['chest'],38)
    def test_invalid_units_feet_inches_and_values(self):
        for change in ({'units':'cm'},{'height_inches':12},{'height_feet':5.5},{'height_feet':True},{'weight':'NaN'},{'chest':0},{'height_inches':-1},{'height_inches':None}):
            with self.subTest(change=change),self.assertRaises(ValueError):measurements(self.body(**change))
    def test_category_required_inputs(self):
        body=self.body();del body['chest'];self.assertNotIn('chest',measurements(body,'lower'))
        with self.assertRaises(ValueError):measurements(body,'upper')
        body=self.body();del body['hips'];measurements(body,'upper')
        with self.assertRaises(ValueError):measurements(body,'overall')
    def test_upper_ignores_hips_lower_ignores_chest(self):
        for c,irrelevant in [('upper','hips'),('lower','chest')]:
            first=recommend(measurements(self.body(),c),DEFAULT_CHARTS[c],SIZES,category=c)
            second=recommend(measurements(self.body(**{irrelevant:999}),c),DEFAULT_CHARTS[c],SIZES,category=c)
            self.assertEqual(first,second);self.assertNotIn(irrelevant,first['fields'])
    def test_only_available_sizes_and_no_model_confidence(self):
        r=recommend(measurements(self.body(),'upper'),DEFAULT_CHARTS['upper'],['L'],__file__,'upper')
        self.assertEqual(r['size'],'L');self.assertIsNone(r['confidence'])
        with self.assertRaises(ValueError):recommend(measurements(self.body()),DEFAULT_CHARTS['overall'],[])
    def test_outside_chart_does_not_claim_perfect_fit(self):
        r=recommend(measurements(self.body(chest=70,waist=70),'upper'),DEFAULT_CHARTS['upper'],SIZES,category='upper')
        self.assertTrue(r['out_of_range']);self.assertEqual(r['fit'],'Tight');self.assertEqual(r['fit_basis'],'Chart-estimated')
    def test_legacy_conversion_preserves_source(self):
        st={'chart':copy.deepcopy(DEFAULT_CHART)};before=copy.deepcopy(st)
        charts=store_charts(st);self.assertEqual(st,before);self.assertAlmostEqual(charts['upper']['M']['chest']*2.54,96)
    def test_chart_validation_atomic_and_monotonic(self):
        for mutate in [lambda c:c.pop('lower'),lambda c:c['upper']['S'].update(chest=1),lambda c:c['lower']['S'].update(waist=65),lambda c:c['upper']['M'].update(hips=40),lambda c:c['overall']['M'].update(chest=float('nan')),lambda c:c['lower']['M'].update(inseam=30)]:
            charts=copy.deepcopy(DEFAULT_CHARTS);mutate(charts)
            with self.assertRaises(ValueError):validate_charts(charts)
    def test_optional_length_required_when_chart_includes_it(self):
        charts=copy.deepcopy(DEFAULT_CHARTS)
        for row in charts['lower'].values():row['inseam']=32
        validate_charts(charts)
        with self.assertRaises(ValueError):measurements(self.body(),'lower',['waist','hips','inseam'])
        v=measurements(self.body(inseam=28),'lower',['waist','hips','inseam']);r=recommend(v,charts['lower'],SIZES,category='lower');self.assertIn('alterations',r['notes'])
    def test_probability_metrics_known_answers(self):
        p=np.array([[.8,.2],[.2,.8]]);r=probability_metrics(['a','b'],p,['a','b'])
        self.assertEqual(r['accuracy'],1);self.assertAlmostEqual(r['ece_10_equal_width_bins'],.2);self.assertAlmostEqual(r['multiclass_brier_sum'],.08)
        np.testing.assert_allclose(temperature_scale(p,1),p)
        self.assertTrue(np.all(temperature_scale(p,2).max(axis=1)<.8))
    def test_synthetic_provenance_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'manifest.json';p.write_text(json.dumps({'label_origin':'synthetic'}))
            with self.assertRaisesRegex(ValueError,'physical fittings'):train('a',Path(d)/'model',calibration_source='b',test_source='c',manifest=p)

class CategoryApiTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.app=create_app({'TESTING':True,'DATA_DIR':self.tmp.name,'AUTH_BACKEND':'local','STORAGE_BACKEND':'sqlite'});self.c=self.app.test_client();self.token=self.c.get('/api/bootstrap').json['csrf'];self.headers={'X-CSRF-Token':self.token}
        self.c.post('/api/auth/register',json={'email':'module2@example.test','name':'Tester','password':'LongPassword-123!'},headers=self.headers)
        self.token=self.c.get('/api/bootstrap').json['csrf'];self.headers={'X-CSRF-Token':self.token}
        self.st=self.c.post('/api/store',json={'name':'Module 2 store'},headers=self.headers).json['store']
    def tearDown(self):self.tmp.cleanup()
    def product(self,category):return self.c.post('/api/products',json={'name':category,'sku':category,'category':category,'price':100,'stock':2,'sizes':SIZES},headers=self.headers).json['product']
    def test_all_categories_and_private_measurements(self):
        for category in ('upper','lower','overall'):
            p=self.product(category);body={'units':'in','height_feet':5,'height_inches':8,'weight':65,'consent':True,'product':p['id'],'waist':32}
            if category!='lower':body['chest']=38
            if category!='upper':body['hips']=40
            r=self.c.post('/api/size',json=body,headers=self.headers);self.assertEqual(r.status_code,200,r.json);self.assertEqual(r.json['category'],category);self.assertEqual(r.json['units'],'in')
        for row in self.app.store.all('events'):
            self.assertFalse(set(row)&{'height','height_feet','weight','chest','waist','hips'})
    def test_overall_requires_all_three_circumferences(self):
        p=self.product('overall')
        self.assertEqual(p['category'],'overall')
        body={'product':p['id'],'consent':True,'units':'in','height_feet':5,
              'height_inches':8,'weight':65,'waist':32,'chest':38}
        r=self.c.post('/api/size',json=body,headers=self.headers)
        self.assertEqual(r.status_code,400)
        self.assertIn('hips',r.json['error'].lower())
        r=self.c.post('/api/size',json={**body,'hips':40},headers=self.headers)
        self.assertEqual(r.status_code,200,r.json)
        self.assertIsNone(r.json['confidence'])
    def test_category_mismatch_rejected(self):
        p=self.product('upper')
        r=self.c.post('/api/size',json={'product':p['id'],'consent':True,'category':'lower'},headers=self.headers)
        self.assertEqual(r.status_code,400)
    def test_api_ignores_irrelevant_measurements(self):
        for category,irrelevant in [('upper','hips'),('lower','chest')]:
            p=self.product(category)
            body={'product':p['id'],'consent':True,'units':'in','height_feet':5,'height_inches':8,'weight':65,'chest':38,'waist':32,'hips':40}
            first=self.c.post('/api/size',json=body,headers=self.headers)
            body[irrelevant]='invalid but irrelevant'
            second=self.c.post('/api/size',json=body,headers=self.headers)
            self.assertEqual(first.status_code,200)
            self.assertEqual(first.json,second.json)
    def test_two_chart_update_preserves_legacy_overall(self):
        charts=copy.deepcopy(DEFAULT_CHARTS);old=copy.deepcopy(charts['overall'])
        charts.pop('overall');charts['upper']['XS']['chest']+=.1
        r=self.c.put('/api/store/settings',json={'charts':charts,'chart_units':'in'},headers=self.headers)
        self.assertEqual(r.status_code,200,r.json)
        saved=self.app.store.get('stores',self.st['id'])['charts']
        self.assertEqual(saved['overall'],old)
        self.assertEqual(saved['upper'],charts['upper'])
    def test_chart_update_persistence_and_isolation(self):
        charts=copy.deepcopy(DEFAULT_CHARTS)
        for row in charts['upper'].values():row['chest']+=2
        r=self.c.put('/api/store/settings',json={'chart_units':'in','charts':charts},headers=self.headers);self.assertEqual(r.status_code,200,r.json)
        public=self.c.get('/api/products?store='+self.st['id']).json['store'];self.assertEqual(public['charts'],charts);self.assertEqual(public['chart_status'],'retailer supplied')
        again=create_app({'TESTING':True,'DATA_DIR':self.tmp.name,'STORAGE_BACKEND':'sqlite'});self.assertEqual(again.store.get('stores',self.st['id'])['charts'],charts)
        invalid=copy.deepcopy(charts);invalid['lower']['M']['waist']=float('nan')
        self.assertEqual(self.c.put('/api/store/settings',json={'chart_units':'in','charts':invalid},headers=self.headers).status_code,400)
        self.assertEqual(self.app.store.get('stores',self.st['id'])['charts'],charts)
    def test_legacy_store_migrates_on_read_only(self):
        st=self.app.store.get('stores',self.st['id']);st.pop('charts');self.app.store.put('stores',st['id'],st)
        self.assertEqual(self.c.get('/api/products?store='+st['id']).status_code,200)
        self.assertNotIn('charts',self.app.store.get('stores',st['id']))
    def test_invalid_chart_units(self):
        self.assertEqual(self.c.put('/api/store/settings',json={'charts':DEFAULT_CHARTS,'chart_units':'cm'},headers=self.headers).status_code,400)

class ModelIntegrationTests(unittest.TestCase):
    """Artificial fixtures test software behavior only, never fitting accuracy."""
    def setUp(self):
        import joblib
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.neighbors import KNeighborsClassifier
        from sizing import chart_fingerprint
        self.tmp=tempfile.TemporaryDirectory();self.path=Path(self.tmp.name)/'upper.joblib'
        self.chart=copy.deepcopy(DEFAULT_CHARTS['upper'])
        self.values=measurements({'units':'in','height_feet':5,'height_inches':8,'weight':65,'chest':self.chart['M']['chest'],'waist':self.chart['M']['waist']},'upper')
        x=[[68,65,self.chart[z]['chest'],self.chart[z]['waist']] for z in SIZES]
        rf=RandomForestClassifier(n_estimators=10,random_state=1).fit(x,SIZES)
        knn=KNeighborsClassifier(n_neighbors=1).fit(x,SIZES)
        self.artifact={'schema_version':2,'units':'in_kg','production_enabled':True,'category':'upper','features':['height','weight','chest','waist'],'rf':rf,'knn':knn,'temperature':1.,'provenance':{'retailer':'test-store','label_origin':'independent_physical_fitting','chart_sha256':chart_fingerprint(self.chart)},'evaluation':{'independent_test':True,'test_rows':6,'ensemble_calibrated':{'accuracy':.5,'log_loss':1.,'ece_10_equal_width_bins':.2}}}
        # Metadata is deliberately a unit-test fixture, not a real evaluation.
        self.save()
    def tearDown(self):self.tmp.cleanup()
    def save(self):
        import joblib
        joblib.dump(self.artifact,self.path)
    def result(self,**kw):
        return recommend(self.values,self.chart,kw.pop('available',SIZES),model_path=self.tmp.name,category='upper',retailer=kw.pop('retailer','test-store'),**kw)
    def test_enabled_model_inference(self):
        r=self.result();self.assertEqual(r['size'],'M');self.assertEqual(r['method'],'RF + KNN size prediction');self.assertIsNotNone(r['confidence'])
    def test_wrong_retailer_falls_back(self):
        r=self.result(retailer='other');self.assertEqual(r['method'],'Category body-chart estimate');self.assertIsNone(r['confidence'])
    def test_changed_chart_falls_back(self):
        self.chart['XS']['chest']+=.1
        self.assertEqual(self.result()['method'],'Category body-chart estimate')
    def test_disabled_and_legacy_model_fall_back(self):
        for key,value in [('production_enabled',False),('schema_version',1),('category','lower'),('units','cm'),('temperature',float('nan'))]:
            original=self.artifact[key];self.artifact[key]=value;self.save()
            self.assertEqual(self.result()['method'],'Category body-chart estimate')
            self.artifact[key]=original
    def test_no_calibration_evidence_no_confidence(self):
        self.artifact['evaluation']={};self.save()
        r=self.result();self.assertEqual(r['method'],'RF + KNN size prediction');self.assertIsNone(r['confidence'])
    def test_unavailable_prediction_has_no_reassigned_confidence(self):
        r=self.result(available=['XL']);self.assertEqual(r['size'],'XL');self.assertIsNone(r['confidence']);self.assertEqual(r['method'],'Category body-chart estimate')
    def test_corrupt_file_falls_back(self):
        self.path.write_bytes(b'not a model')
        self.assertEqual(self.result()['method'],'Category body-chart estimate')
    def test_missing_file_falls_back(self):
        self.path.unlink();self.assertEqual(self.result()['method'],'Category body-chart estimate')

if __name__=='__main__':unittest.main(verbosity=2)
