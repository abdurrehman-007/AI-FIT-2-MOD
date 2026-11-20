"""Extract only the audited local Sapphire table; never import it into the app."""
import csv
import hashlib
import re
from decimal import Decimal
from html import unescape
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / 'datasets' / 'pakistan_charts'
FIELDS = ['brand', 'product', 'sku', 'category', 'size', 'measurement',
          'original_value', 'original_unit', 'measurement_basis', 'verification_status',
          'unit_status', 'basis_status', 'value_inches', 'conversion', 'waist_state',
          'body_import_eligible', 'source_url', 'source_file', 'source_line', 'evidence']


def extract():
    sources = list(csv.DictReader((DATA / 'sources.csv').open(encoding='utf-8-sig')))
    output = {'upper': [], 'lower': []}
    hashes = []
    for source in sources:
        path = DATA / (source['name'] + '.html')
        raw = path.read_bytes()
        html = raw.decode('utf-8')
        hashes.append((path.name, hashlib.sha256(raw).hexdigest()))
        tables = re.findall(r'<table\b[^>]*>.*?</table>', html, re.S | re.I)
        if source['category'] == 'lower':
            assert 'data-action="Search-Show"' in html
            assert 'We are sorry, but no results were found for:' in html
            assert not tables, 'New lower table requires manual audit before extraction'
            continue
        assert source['name'] == 'sapphire_upper_waistcoat'
        assert len(tables) == 1 and 'active-inches' in tables[0]
        assert '>INCHES</a>' in html
        table = tables[0]
        rows = []
        for tr in re.findall(r'<tr\b[^>]*>(.*?)</tr>', table, re.S | re.I):
            cells = re.findall(r'<t[hd]\b[^>]*>(.*?)</t[hd]>', tr, re.S | re.I)
            rows.append([unescape(re.sub('<[^>]+>', '', c)).strip() for c in cells])
        assert rows[0] == ['Size', 'XS', 'S', 'M', 'L', 'XL']
        assert [r[0] for r in rows[1:]] == ['Length', 'Chest Half', 'Hem', 'Shoulders']
        for cells in rows[1:]:
            label, *values = cells
            assert len(values) == len(rows[0]) - 1
            # Half is explicit; no full girth or body circumference is derived.
            basis = 'half_chest_width' if label == 'Chest Half' else 'unverified'
            verified = label == 'Chest Half'
            line = html[:html.index('<th>' + label + '</th>')].count('\n') + 1
            for size, value in zip(rows[0][1:], values):
                assert Decimal(value).is_finite() and Decimal(value) > 0
                output['upper'].append(dict(zip(FIELDS, [
                    'SAPPHIRE', 'Waist Coat (generic category chart)', '', 'upper', size,
                    label, value, 'INCHES', basis,
                    'verified' if verified else 'unverified', 'verified',
                    'verified_half_width; flat_lay_method_unspecified' if verified else 'unverified',
                    value, 'none; source already inches', '', 'false', source['url'],
                    path.name, line,
                    'INCHES active tab + active-inches table; ' +
                    ('Chest Half explicitly denotes half width; no body or full circumference claim'
                     if verified else label + ': measurement endpoints/width/circumference basis unspecified')
                ])))
    for category, rows in output.items():
        with (DATA / (category + '_charts.csv')).open('w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=FIELDS)
            writer.writeheader()
            writer.writerows(rows)
    assert len(output['upper']) == 20 and not output['lower']
    for category in output:
        with (DATA / (category + '_charts.csv')).open(newline='', encoding='utf-8') as f:
            assert list(csv.DictReader(f)) == [
                {k: str(v) for k, v in row.items()} for row in output[category]]
    print('Verified CSV round-trip: upper=20 rows; lower=0 rows; eligible body imports=0')
    for name, digest in hashes:
        print(name, digest)


if __name__ == '__main__':
    extract()
