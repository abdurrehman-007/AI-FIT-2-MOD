"""Category-specific body-chart estimates. Public units: inches and kg."""
import math
import hashlib
import json
import logging
from pathlib import Path
from copy import deepcopy

SIZES = ['XS', 'S', 'M', 'L', 'XL', 'XXL']
FEATURES = ['height', 'weight', 'chest', 'waist', 'hips']  # legacy training schema only
DEFAULT_CHART = {s: dict(zip(('chest','waist','hips'), (84+i*6,68+i*6,88+i*6))) for i,s in enumerate(SIZES)}
REQUIRED = {'upper': ('chest','waist'), 'lower': ('waist','hips'), 'overall': ('chest','waist','hips')}
OPTIONAL = {'upper': ('shoulder','sleeve'), 'lower': ('inseam','thigh'), 'overall': ('shoulder','sleeve','inseam','thigh')}
LIMITS = {'chest':(20,75), 'waist':(16,75), 'hips':(20,80), 'shoulder':(8,30), 'sleeve':(10,40), 'inseam':(15,45), 'thigh':(10,45)}
DEFAULT_CHARTS = {c: {s: {k: round(DEFAULT_CHART[s][k]/2.54,4) for k in keys} for s in SIZES} for c,keys in REQUIRED.items()}


def number(value, name, low, high):
    try:
        if isinstance(value, bool): raise ValueError()
        value = float(value)
    except (TypeError, ValueError):
        raise ValueError(f'Enter a valid {name}.') from None
    if not math.isfinite(value) or not low <= value <= high:
        raise ValueError(f'{name} must be between {low} and {high}.')
    return value


def chart_fields(chart, category):
    if category not in REQUIRED: raise ValueError('Invalid garment category.')
    return list(REQUIRED[category]) + [k for k in OPTIONAL[category] if any(k in row for row in chart.values())]


def validate_charts(charts):
    if not isinstance(charts, dict) or set(charts) != set(REQUIRED):
        raise ValueError('Provide separate upper, lower and overall charts.')
    result = {}
    for category, chart in charts.items():
        if not isinstance(chart, dict) or set(chart) != set(SIZES):
            raise ValueError(f'{category}: include XS through XXL.')
        if any(not isinstance(row, dict) for row in chart.values()): raise ValueError('Invalid chart row.')
        fields = chart_fields(chart, category)
        result[category] = {}
        for size in SIZES:
            if set(chart[size]) != set(fields):
                raise ValueError(f'{category}: every size needs the same relevant fields: {", ".join(fields)}.')
            result[category][size] = {k: number(chart[size].get(k), f'{category} {size} {k} (inches)', *LIMITS[k]) for k in fields}
        for k in fields:
            values = [result[category][s][k] for s in SIZES]
            if any(a > b for a,b in zip(values,values[1:])):
                raise ValueError(f'{category} {k}: measurements cannot decrease as sizes increase.')
    return result


def store_charts(store):
    """Read old centimetre charts without modifying stored retailer data."""
    if store.get('charts') is not None:
        return validate_charts(store['charts'])
    old = store.get('chart', DEFAULT_CHART)
    return validate_charts({c: {s: {k: float(old[s][k])/2.54 for k in fields} for s in SIZES} for c,fields in REQUIRED.items()})


def measurements(data, category='overall', fields=None):
    if category not in REQUIRED: raise ValueError('Invalid garment category.')
    # Explicit units prevent silent reinterpretation of old centimetre clients.
    if data.get('units') != 'in': raise ValueError('Use units="in", height_feet and height_inches; weight stays in kg.')
    feet = number(data.get('height_feet'), 'Height feet', 3, 7)
    inches = number(data.get('height_inches'), 'Height inches', 0, 11.99)
    if not feet.is_integer(): raise ValueError('Height feet must be a whole number.')
    height = feet*12 + inches
    if not 39.37 <= height <= 90.55: raise ValueError('Combined height must be between 3 ft 3.37 in and 7 ft 6.55 in.')
    result = {'height':height, 'weight':number(data.get('weight'), 'Weight (kg)', 25, 250)}
    for key in fields or REQUIRED[category]:
        result[key] = number(data.get(key), key.title()+' (inches)', *LIMITS[key])
    return result


def chart_fingerprint(chart):
    return hashlib.sha256(json.dumps(chart, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def model_prediction(path, values, category, retailer, chart):
    """Load only administrator-installed joblib artifacts; never accept upload paths."""
    if not path or not retailer or category not in ('upper', 'lower'):
        return None
    path = Path(path)
    if path.is_dir(): path = path / (category + '.joblib')
    if not path.is_file() or path.suffix != '.joblib': return None
    try:
        import joblib
        import numpy as np
        artifact = joblib.load(path)
        features = ['height', 'weight', *REQUIRED[category]]
        provenance = artifact.get('provenance', {})
        if (artifact.get('schema_version') != 2 or artifact.get('production_enabled') is not True
                or artifact.get('category') != category or artifact.get('features') != features
                or artifact.get('units') != 'in_kg'
                or provenance.get('label_origin') != 'independent_physical_fitting'
                or provenance.get('retailer') != retailer
                or provenance.get('chart_sha256') != chart_fingerprint(chart)):
            return None
        rf, knn = artifact['rf'], artifact['knn']
        classes = list(rf.classes_)
        if classes != list(knn.classes_) or set(classes) != set(SIZES): return None
        x = [[values[k] for k in features]]
        arrays = [np.asarray(est.predict_proba(x), dtype=float) for est in (rf, knn)]
        if any(a.shape != (1,len(classes)) or not np.isfinite(a).all() or (a<0).any()
               or not np.allclose(a.sum(axis=1),1) for a in arrays): return None
        temperature = float(artifact['temperature'])
        if not math.isfinite(temperature) or temperature <= 0: return None
        raw = (arrays[0][0]+arrays[1][0])/2
        logits = np.log(np.clip(raw,1e-9,1))/temperature
        probabilities = np.exp(logits-logits.max()); probabilities /= probabilities.sum()
        index = int(probabilities.argmax())
        evaluation = artifact.get('evaluation', {})
        metrics = evaluation.get('ensemble_calibrated', {})
        calibrated = (evaluation.get('independent_test') is True and
                      isinstance(evaluation.get('test_rows'), int) and evaluation['test_rows'] > 0 and
                      all(isinstance(metrics.get(k), (int,float)) and math.isfinite(metrics[k])
                          for k in ('accuracy','log_loss','ece_10_equal_width_bins')))
        return {'size':str(classes[index]), 'confidence':round(float(probabilities[index])*100,1) if calibrated else None}
    except Exception:
        logging.getLogger(__name__).warning('Size model unavailable or incompatible; using chart fallback.', exc_info=False)
        return None


def recommend(values, chart, available, model_path=None, category='overall', retailer=None):
    fields = chart_fields(chart, category)
    candidates = [s for s in SIZES if s in available and s in chart]
    if not candidates: raise ValueError('This product has no available sizes with a size chart.')
    # Circumferences constrain fit. Lengths help choose between accommodating sizes.
    girths = [k for k in fields if k not in ('sleeve','inseam')]
    fitting = [s for s in candidates if all(values[k] <= chart[s][k]+.2 for k in girths)]
    distance = lambda s: sum(((values[k]-chart[s][k])/(1 if k in girths else 2))**2 for k in fields)
    size = min(fitting or candidates, key=lambda s:(distance(s), SIZES.index(s)))
    prediction = model_prediction(model_path, values, category, retailer, chart)
    confidence = None
    method = 'Category body-chart estimate'
    # Preserve the calibrated model's original top-label decision. Do not renormalize
    # over available sizes or present confidence for a chart-adjusted decision.
    if prediction and prediction['size'] in (fitting or candidates):
        size = prediction['size']
        confidence = prediction['confidence']
        method = 'RF + KNN size prediction'
    deltas = {k:round(values[k]-chart[size][k],2) for k in fields}
    outside = not fitting or all(all(values[k]<chart[s][k]-1.2 for k in girths) for s in candidates)
    fit = 'Tight' if any(deltas[k]>.2 for k in girths) else 'Loose' if any(deltas[k]<-1.2 for k in girths) else 'Perfect'
    notes = 'Fit tag is chart-estimated, not a physical fitting guarantee. Fabric, cut and garment ease can change the fit. Model confidence, when available, concerns the predicted size label, not satisfaction.'
    if outside: notes += ' No close available chart match; ask the retailer before ordering.'
    if any(abs(deltas[k])>1 for k in fields if k in ('inseam','sleeve')): notes += ' Check sleeve/inseam length or alterations separately.'
    # Old uncalibrated .joblib files must not silently produce fit confidence.
    return {'size':size,'fit':fit,'confidence':confidence,'confidence_label':'Calibrated size-label model support' if confidence is not None else 'Fitting confidence unavailable','fit_basis':'Chart-estimated','method':method,'category':category,'units':'in','fields':fields,'out_of_range':outside,'notes':notes,'comparisons':[{'size':s,**{k:round(chart[s][k],2) for k in fields},'available':s in available} for s in SIZES if s in chart],'deltas':deltas}
