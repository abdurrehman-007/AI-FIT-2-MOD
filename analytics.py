"""Retailer-scoped aggregates and validated, offline evaluation reporting."""
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from sizing import chart_fingerprint, SIZES, REQUIRED


def evaluation_reports(model_path, retailer, charts):
    results = {}
    for category in ('upper', 'lower'):
        unavailable = {'available': False, 'reason': 'No compatible independent fitting evaluation report installed.'}
        results[category] = unavailable
        if not model_path:
            continue
        path = Path(model_path)
        if path.is_dir():
            path = path / (category + '.joblib')
        path = path.with_suffix('.metrics.json')
        try:
            if path.stat().st_size > 1_000_000:
                continue
            report = json.loads(path.read_text(encoding='utf-8'))
            provenance = report['provenance']
            if (report['category'] != category or report['label_origin'] != 'independent_physical_fitting'
                    or provenance['label_origin'] != 'independent_physical_fitting'
                    or provenance['retailer'] != retailer
                    or provenance['chart_sha256'] != chart_fingerprint(charts[category])
                    or report['features'] != ['height', 'weight', *REQUIRED[category]]
                    or report['units'] != 'height and circumferences inches; weight kg'
                    or set(report['classes']) != set(SIZES)):
                continue
            splits = report['split_sha256']
            hashes = [splits[k] for k in ('train', 'calibration', 'test')]
            if len(set(hashes)) != 3 or any(len(h) != 64 or any(c not in '0123456789abcdef' for c in h) for h in hashes):
                continue
            count = report['split_counts']['test']['rows']
            if type(count) is not int or count <= 0:
                continue
            metrics = {}
            for model in ('rf', 'knn', 'ensemble_calibrated'):
                metrics[model] = {}
                for name in ('accuracy', 'log_loss', 'ece'):
                    value = report[model]['ece_10_equal_width_bins' if name == 'ece' else name]
                    if type(value) not in (int, float) or not math.isfinite(value) or value < 0 or (name != 'log_loss' and value > 1):
                        raise ValueError('Invalid metric')
                    metrics[model][name] = value
            results[category] = {'available': True, 'test_rows': count, 'metrics': metrics,
                'scope': 'Offline evaluation for this retailer and exact category chart. Not deployment status or population accuracy; provenance still requires human audit.'}
        except (OSError, ValueError, KeyError, TypeError, AttributeError):
            continue
    return results


def extra_analytics(rows, products, retailer, charts, model_path):
    # Defensive retailer filtering even when the caller has already scoped its query.
    rows = [e for e in rows if e.get('store') == retailer]
    catalog = {p['id']: p for p in products if p.get('store') == retailer}
    popular = {k: Counter() for k in ('upper', 'lower', 'overall', 'unknown')}
    categories = Counter()
    customers = defaultdict(Counter)
    untracked = 0
    for e in rows:
        if e.get('kind') != 'tryon':
            continue
        category = e.get('category', catalog.get(e.get('product'), {}).get('category', 'unknown'))
        if category not in popular:
            category = 'unknown'
        popular[category][e.get('product', 'unknown')] += 1
        categories[category] += 1
        if e.get('customer_key'):
            customers[e['customer_key']][category] += 1
        else:
            untracked += 1
    preferences = Counter()
    for counts in customers.values():
        top = max(counts.values())
        winners = [k for k, v in counts.items() if v == top]
        preferences[winners[0] if len(winners) == 1 else 'mixed'] += 1
    return {'category_popularity': {k: dict(v) for k, v in popular.items()},
        'product_names': {k: p['name'] for k, p in catalog.items()},
        'category_interest': dict(categories), 'customer_preferences': dict(preferences),
        'preference_customers': len(customers), 'untracked_tryons': untracked,
        'evaluations': evaluation_reports(model_path, retailer, charts)}
