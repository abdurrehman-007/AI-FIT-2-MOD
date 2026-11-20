"""Read-only audit of raw research data. No size labels or imputed measurements."""
import csv,hashlib,json,sys,importlib.metadata
from collections import Counter
from pathlib import Path
import openpyxl
ROOT=Path(__file__).resolve().parents[1];data=ROOT/'datasets/module2'
workbook=openpyxl.load_workbook(data/'pone.0317534.s001.xlsx',read_only=True,data_only=True)
sheets=[]
for sheet in workbook:
    values=list(sheet.values);headers=list(values[0]);rows=[r for r in values[1:] if any(v is not None for v in r)]
    ids=[r[headers.index('ID')] for r in rows]
    summary={'sheet':sheet.title,'rows':len(rows),'columns':headers,'unique_IDs':len(set(ids)),'duplicate_ID_rows':len(ids)-len(set(ids)),'exact_duplicate_rows':len(rows)-len(set(rows)),'missing_counts':{k:sum(r[i] is None for r in rows) for i,k in enumerate(headers)},'numeric_ranges':{}}
    for key in ('Age','agemean','Weight_Kg','Height_M','postweight','posthight'):
        nums=[r[headers.index(key)] for r in rows if isinstance(r[headers.index(key)],(int,float))]
        summary['numeric_ranges'][key]={'min':min(nums),'max':max(nums),'count':len(nums)}
    sheets.append(summary)
with (data/'national-anthropometry.csv').open(newline='',encoding='utf-8-sig') as f:
    rows=[r for r in csv.DictReader(f) if r.get('ISO3')=='PAK']
report={'python':sys.version,'packages':{p:importlib.metadata.version(p) for p in ('numpy','scikit-learn','joblib','openpyxl')},'school_workbook':sheets,'national_anthropometry':{'indicator_rows':len(rows),'surveys':sorted({r['SurveyId'] for r in rows}),'indicators':sorted({r['Indicator'] for r in rows}),'unit_of_observation':'aggregate survey indicator, not individual fitting'},'fitting_labels':0,'training_performed':False,'external_test_accuracy':None,'confidence_calibration':None,'reason':'Neither delivered source contains category-specific independently confirmed garment size/fit labels or the required adult shopper measurements. No chart-derived or synthetic labels generated.','files':{p.name:{'bytes':p.stat().st_size,'sha256':hashlib.sha256(p.read_bytes()).hexdigest()} for p in data.iterdir() if p.suffix in ('.xlsx','.csv')}}
(ROOT/'reports/module2-data-audit.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
print(json.dumps({**report,'school_workbook':[{k:v for k,v in s.items() if k not in ('missing_counts','columns')} for s in sheets]},indent=2))
