"""Offline RF/KNN research evaluation on independently confirmed fitting labels.
Requires disjoint training, calibration and external test CSVs plus provenance.
Artifacts are not auto-enabled by the production chart recommendation endpoint.
"""
import argparse,csv,hashlib,json
from collections import Counter
from pathlib import Path
import numpy as np
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score,balanced_accuracy_score,classification_report,confusion_matrix,log_loss
from sizing import SIZES,REQUIRED,measurements


def temperature_scale(probabilities,temperature):
    logits=np.log(np.clip(probabilities,1e-9,1))/temperature
    values=np.exp(logits-logits.max(axis=1,keepdims=True))
    return values/values.sum(axis=1,keepdims=True)


def probability_metrics(y,probabilities,classes):
    y=np.asarray(y);classes=np.asarray(classes);pred=classes[probabilities.argmax(axis=1)]
    confidence=probabilities.max(axis=1);correct=pred==y;bins=[];ece=0.
    for i in range(10):
        mask=(confidence>=i/10)&((confidence<(i+1)/10) if i<9 else (confidence<=1))
        count=int(mask.sum())
        observed=float(correct[mask].mean()) if count else None
        mean=float(confidence[mask].mean()) if count else None
        if count:ece+=count/len(y)*abs(observed-mean)
        bins.append({'lower':i/10,'upper':(i+1)/10,'count':count,'accuracy':observed,'mean_confidence':mean})
    truth=(y[:,None]==classes[None,:]).astype(float)
    return {'accuracy':float(accuracy_score(y,pred)),'balanced_accuracy':float(balanced_accuracy_score(y,pred)),
            'log_loss':float(log_loss(y,probabilities,labels=list(classes))),
            'multiclass_brier_sum':float(np.mean(np.sum((probabilities-truth)**2,axis=1))),
            'ece_10_equal_width_bins':ece,'reliability_bins':bins,
            'confusion_matrix':confusion_matrix(y,pred,labels=list(classes)).tolist(),
            'classification_report':classification_report(y,pred,labels=list(classes),output_dict=True,zero_division=0)}


def train(source,output,*,calibration_source=None,test_source=None,manifest=None):
    if not calibration_source or not test_source or not manifest:
        raise ValueError('Provide separate calibration/test files and a provenance manifest; chart-derived or synthetic labels cannot establish real fitting accuracy.')
    provenance=json.loads(Path(manifest).read_text(encoding='utf-8-sig'))
    if provenance.get('label_origin')!='independent_physical_fitting':
        raise ValueError('Labels must come from independent physical fittings, not charts or synthetic data.')
    for key in ('source','license','population','collection_protocol','retailer','chart_version','test_collection'):
        if not isinstance(provenance.get(key),str) or not provenance[key].strip():raise ValueError('Manifest requires '+key)
    category=provenance.get('category')
    if category not in ('upper','lower'):raise ValueError('Manifest category must be upper/lower.')
    chart_hash=provenance.get('chart_sha256','')
    if len(chart_hash)!=64 or any(c not in '0123456789abcdef' for c in chart_hash):raise ValueError('Manifest requires chart_sha256 for the exact retailer category chart.')
    features=['height','weight',*REQUIRED[category]]
    datasets=[];seen_people=set();seen_fittings=set();hashes={}
    for split,path in [('train',source),('calibration',calibration_source),('test',test_source)]:
        with open(path,encoding='utf-8-sig',newline='') as f:rows=list(csv.DictReader(f))
        x=[];y=[];people=set();fittings=set()
        for row in rows:
            if row.get('category')!=category:raise ValueError('Each fitting must match the manifest category.')
            if not row.get('person_id') or not row.get('fitting_id'):raise ValueError('Every fitting requires person_id and fitting_id.')
            if row.get('label_origin')!='independent_physical_fitting':raise ValueError('Unverified fitting label origin.')
            if row['fitting_id'] in fittings:raise ValueError('Duplicate fitting_id within split.')
            if row.get('size') not in SIZES:raise ValueError('Every row needs a confirmed XS-XXL size label.')
            v=measurements(row,category);x.append([v[k] for k in features]);y.append(row['size']);people.add(row['person_id']);fittings.add(row['fitting_id'])
        if not rows or set(y)!=set(SIZES):raise ValueError('Every split must cover all six sizes.')
        if min(Counter(y).values())<5:raise ValueError('Each split needs at least five examples per size; this is an execution minimum, not evidence of adequacy.')
        if seen_people&people or seen_fittings&fittings:raise ValueError('Participant/fitting leakage between train, calibration and test.')
        seen_people|=people;seen_fittings|=fittings
        datasets.append((np.asarray(x),np.asarray(y),people));hashes[split]=hashlib.sha256(Path(path).read_bytes()).hexdigest()
    (xt,yt,pt),(xc,yc,pc),(xe,ye,pe)=datasets
    rf=RandomForestClassifier(n_estimators=200,min_samples_leaf=2,class_weight='balanced',random_state=42,n_jobs=1)
    knn=make_pipeline(StandardScaler(),KNeighborsClassifier(n_neighbors=min(7,len(xt)),weights='distance'))
    rf.fit(xt,yt);knn.fit(xt,yt)
    classes=rf.classes_;assert np.array_equal(classes,knn.classes_)
    cal=(rf.predict_proba(xc)+knn.predict_proba(xc))/2
    temperatures=np.geomspace(.25,4,81)
    temperature=float(min(temperatures,key=lambda t:log_loss(yc,temperature_scale(cal,t),labels=list(classes))))
    rp=rf.predict_proba(xe);kp=knn.predict_proba(xe);raw=(rp+kp)/2;scaled=temperature_scale(raw,temperature)
    report={'status':'offline evaluation; not production enabled','label_origin':provenance['label_origin'],'provenance':provenance,'category':category,'features':features,'units':'height and circumferences inches; weight kg','classes':classes.tolist(),'split':'separate supplied files; participants and fitting IDs disjoint','split_sha256':hashes,'rows':sum(len(d[0]) for d in datasets),'split_counts':{s:{'rows':len(d[0]),'people':len(d[2]),'sizes':dict(Counter(d[1]))} for s,d in zip(('train','calibration','test'),datasets)},'temperature':temperature,'majority_baseline_accuracy':float(np.mean(ye==Counter(yt).most_common(1)[0][0])),'rf':probability_metrics(ye,rp,classes),'knn':probability_metrics(ye,kp,classes),'ensemble_uncalibrated':probability_metrics(ye,raw,classes),'ensemble_calibrated':probability_metrics(ye,scaled,classes),'limitations':'Manifest provenance requires human audit. Disjoint IDs alone do not prove independent collection or representativeness. Metrics concern confirmed size labels for this retailer/category only, not garment appearance, fit satisfaction, or Pakistani population accuracy.'}
    output=Path(output);output.parent.mkdir(parents=True,exist_ok=True)
    joblib.dump({'schema_version':2,'units':'in_kg','evaluation':{'independent_test':True,'test_rows':len(ye),'ensemble_calibrated':report['ensemble_calibrated']},'rf':rf,'knn':knn,'temperature':temperature,'features':features,'category':category,'provenance':provenance,'production_enabled':False},output)
    output.with_suffix('.metrics.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    return report

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('csv');parser.add_argument('--calibration',required=True);parser.add_argument('--test',required=True);parser.add_argument('--manifest',required=True);parser.add_argument('--output',default='models/size-research.joblib');args=parser.parse_args()
    print(json.dumps(train(args.csv,args.output,calibration_source=args.calibration,test_source=args.test,manifest=args.manifest),indent=2))
