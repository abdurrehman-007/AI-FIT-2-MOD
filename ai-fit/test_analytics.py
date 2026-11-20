import copy
import csv
import json
import tempfile
import unittest
from pathlib import Path
from analytics import extra_analytics, evaluation_reports
from sizing import DEFAULT_CHARTS, SIZES, chart_fingerprint, validate_charts


class AnalyticsTests(unittest.TestCase):
    def test_categories_preferences_and_isolation(self):
        rows = [{'kind':'tryon','store':'one','product':'p','category':c,'customer_key':u} for c,u in [('upper','a'),('upper','a'),('lower','a'),('upper','b'),('lower','b')]]
        rows += [{'kind':'tryon','store':'one','product':'old','category':'overall'}, {'kind':'tryon','store':'two','product':'secret','category':'lower','customer_key':'secret'}]
        result=extra_analytics(rows, [{'id':'p','store':'one','name':'Shirt'},{'id':'secret','store':'two','name':'Secret'}], 'one', DEFAULT_CHARTS, '')
        self.assertEqual(result['customer_preferences'], {'upper':1,'mixed':1})
        self.assertEqual(result['category_popularity']['upper'], {'p':3})
        self.assertEqual(result['untracked_tryons'],1)
        self.assertNotIn('secret',str(result))
        self.assertFalse(result['evaluations']['upper']['available'])

    def test_empty_data(self):
        result=extra_analytics([], [], 'one', DEFAULT_CHARTS, '')
        self.assertEqual(result['customer_preferences'],{})
        self.assertEqual(result['category_interest'],{})
        self.assertEqual(result['preference_customers'],0)

    def report(self):
        return {'category':'upper','label_origin':'independent_physical_fitting','provenance':{'label_origin':'independent_physical_fitting','retailer':'one','chart_sha256':chart_fingerprint(DEFAULT_CHARTS['upper'])},'features':['height','weight','chest','waist'],'units':'height and circumferences inches; weight kg','classes':SIZES,'split_sha256':{'train':'a'*64,'calibration':'b'*64,'test':'c'*64},'split_counts':{'test':{'rows':30}},**{m:{'accuracy':.6,'log_loss':1.1,'ece_10_equal_width_bins':.1} for m in ('rf','knn','ensemble_calibrated')}}

    def test_evaluation_scope_and_invalid_reports(self):
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'upper.metrics.json'
            path.write_text(json.dumps(self.report()))
            self.assertTrue(evaluation_reports(tmp,'one',DEFAULT_CHARTS)['upper']['available'])
            self.assertFalse(evaluation_reports(tmp,'two',DEFAULT_CHARTS)['upper']['available'])
            self.assertFalse(evaluation_reports(tmp,'one',DEFAULT_CHARTS)['lower']['available'])
            for field,value in [('accuracy',float('nan')),('accuracy',2),('ece_10_equal_width_bins',-1),('log_loss',True)]:
                report=self.report();report['rf'][field]=value;path.write_text(json.dumps(report))
                self.assertFalse(evaluation_reports(tmp,'one',DEFAULT_CHARTS)['upper']['available'])
            report=self.report();report['provenance']['chart_sha256']='f'*64;path.write_text(json.dumps(report))
            self.assertFalse(evaluation_reports(tmp,'one',DEFAULT_CHARTS)['upper']['available'])
            path.write_text('{broken')
            self.assertFalse(evaluation_reports(tmp,'one',DEFAULT_CHARTS)['upper']['available'])

    def test_reference_csvs_match_presets(self):
        root=Path(__file__).parent/'datasets/pakistan_charts'
        preset=json.loads((root/'verified_presets.json').read_text())[0]
        validate_charts({**preset['charts'],'overall':DEFAULT_CHARTS['overall']})
        for category in ('upper','lower'):
            with (root/('verified_'+category+'_charts.csv')).open() as f: rows=list(csv.DictReader(f))
            self.assertEqual(len(rows),12)
            for row in rows:
                self.assertEqual(row['basis'],'body circumference')
                self.assertEqual(float(row['value_in']),preset['charts'][category][row['size']][row['measurement']])
                self.assertLessEqual(abs(float(row['source_value_cm'])-float(row['value_in'])*2.54),.051)

if __name__=='__main__':unittest.main()
