"""Local Vendor Invoice Intelligence: SQL, reproducible demo data, training and inference."""
from contextlib import contextmanager
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
import argparse
import csv
import io
import json
import os
import sqlite3
import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score, mean_absolute_error,
                             mean_squared_error, precision_score, r2_score, recall_score, roc_auc_score)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

ROOT = Path(__file__).resolve().parent
DB_PATH = Path(os.environ.get('INVOICE_DATA_DIR', str(ROOT / 'data'))) / 'invoices.db'
MODEL_DIR = Path(os.environ.get('INVOICE_MODEL_DIR', str(ROOT / 'models')))
MODES = ('Standard', 'Express', 'Economy')
REGIONS = ('North', 'South', 'East', 'West', 'Central')
FREIGHT_FEATURES = ['quantity', 'invoice_amount', 'weight_kg', 'distance_km', 'line_count', 'shipping_mode', 'region']
RISK_FEATURES = FREIGHT_FEATURES + ['freight_cost', 'line_total', 'payment_terms_days',
                                  'amount_mismatch_ratio', 'freight_ratio', 'cost_per_unit']
REQUIRED = ['quantity', 'invoice_amount', 'weight_kg', 'distance_km', 'freight_cost', 'line_total']
DEFAULTS = dict(line_count=3, shipping_mode='Standard', region='North', payment_terms_days=30)
LIMITS = {'quantity': (1, 1000000), 'invoice_amount': (.01, 100000000),
          'weight_kg': (.01, 1000000), 'distance_km': (.1, 50000),
          'freight_cost': (0, 10000000), 'line_total': (.01, 100000000),
          'line_count': (1, 10000), 'payment_terms_days': (0, 365)}
SCHEMA = """
CREATE TABLE IF NOT EXISTS metadata (key TEXT PRIMARY KEY, value TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS vendors (
 vendor_id INTEGER PRIMARY KEY, vendor_name TEXT NOT NULL, region TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS invoices (
 invoice_id TEXT PRIMARY KEY, vendor_id INTEGER NOT NULL REFERENCES vendors(vendor_id),
 invoice_date TEXT NOT NULL, invoice_amount REAL NOT NULL CHECK(invoice_amount > 0),
 freight_cost REAL NOT NULL CHECK(freight_cost >= 0), weight_kg REAL NOT NULL CHECK(weight_kg > 0),
 distance_km REAL NOT NULL CHECK(distance_km > 0), shipping_mode TEXT NOT NULL,
 payment_terms_days INTEGER NOT NULL, needs_review INTEGER NOT NULL CHECK(needs_review IN (0,1)));
CREATE TABLE IF NOT EXISTS invoice_lines (
 line_id INTEGER PRIMARY KEY, invoice_id TEXT NOT NULL REFERENCES invoices(invoice_id),
 quantity INTEGER NOT NULL CHECK(quantity > 0), unit_price REAL NOT NULL CHECK(unit_price > 0));
CREATE INDEX IF NOT EXISTS lines_invoice ON invoice_lines(invoice_id);
CREATE TABLE IF NOT EXISTS audit_log (
 id INTEGER PRIMARY KEY AUTOINCREMENT,
 created_at TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ','now')),
 invoice_reference TEXT NOT NULL, source TEXT NOT NULL, input_json TEXT NOT NULL,
 expected_freight REAL NOT NULL, risk_score REAL NOT NULL, decision TEXT NOT NULL, reasons_json TEXT NOT NULL);
"""

@contextmanager
def connect(db_path=None):
    path = Path(db_path or DB_PATH)
    path.parent.mkdir(parents=True, exist_ok=True)
    con = sqlite3.connect(str(path), timeout=20)
    con.execute('PRAGMA foreign_keys = ON')
    try:
        yield con
        con.commit()
    except Exception:
        con.rollback()
        raise
    finally:
        con.close()

def initialize_database(db_path=None, n_invoices=3000, seed=42):
    """Initialize once. Never overwrite existing invoices or saved reviews."""
    with connect(db_path) as con:
        con.executescript(SCHEMA)
        if con.execute('SELECT COUNT(*) FROM invoices').fetchone()[0]:
            return False
        if n_invoices < 100:
            raise ValueError('At least 100 demo invoices are needed for training.')
        rng = np.random.default_rng(seed)
        names = ['Atlas', 'Summit', 'Meridian', 'Cedar', 'Harbor', 'Orion', 'Nova', 'Evergreen']
        vendors = [(i, f'{names[(i-1)%8]} Supply {i:02d}', REGIONS[(i-1)%5]) for i in range(1, 41)]
        con.executemany('INSERT OR IGNORE INTO vendors VALUES (?,?,?)', vendors)
        invoices, lines = [], []
        for i in range(n_invoices):
            vendor_id = int(rng.integers(1, 41))
            reference = f'DEMO-{i+1:05d}'
            quantity = int(rng.integers(10, 850))
            count = int(rng.integers(1, 8))
            portions = rng.multinomial(quantity-count, np.ones(count)/count) + 1
            prices = np.round(rng.uniform(15, 190)*rng.uniform(.75, 1.25, count), 2)
            total = round(float(np.dot(portions, prices)), 2)
            weight = round(float(quantity*rng.uniform(.3, 3.2)), 2)
            distance = round(float(rng.uniform(35, 2400)), 1)
            mode = str(rng.choice(MODES, p=[.6, .2, .2]))
            multiplier = {'Standard': 1., 'Express': 1.55, 'Economy': .78}[mode]
            region_factor = 1 + ((vendor_id-1)%5)*.035
            freight = max(10, (80 + weight*.65 + distance*.32 + quantity*.25 + total*.004)
                          *multiplier*region_factor*float(rng.normal(1, .055)))
            amount, terms = total, int(rng.choice([15, 30, 45, 60]))
            # Latent scenario generates demo review labels; never included among model inputs.
            risky = int(rng.random() < .25)
            if risky:
                scenario = int(rng.integers(0, 4))
                if scenario == 0:
                    freight *= float(rng.uniform(1.7, 3.))
                elif scenario == 1:
                    amount *= float(rng.uniform(1.12, 1.55))
                elif scenario == 2:
                    terms = int(rng.choice([90, 120, 150]))
                else:
                    freight *= float(rng.uniform(1.3, 1.8))
                    amount *= float(rng.uniform(1.04, 1.12))
            day = date(2025, 1, 1) + timedelta(days=int(i*540/n_invoices))
            invoices.append((reference, vendor_id, day.isoformat(), round(amount, 2), round(freight, 2),
                             weight, distance, mode, terms, risky))
            lines.extend((None, reference, int(q), float(p)) for q,p in zip(portions, prices))
        con.executemany('INSERT INTO invoices VALUES (?,?,?,?,?,?,?,?,?,?)', invoices)
        con.executemany('INSERT INTO invoice_lines VALUES (?,?,?,?)', lines)
        con.executemany('INSERT OR REPLACE INTO metadata VALUES (?,?)',
                        [('data_source', 'synthetic_demo'), ('seed', str(seed)), ('schema_version', '1')])
    return True

def load_invoices(db_path=None):
    # Aggregate line items before joining, avoiding inflated invoice totals.
    query = """
    WITH lines AS (SELECT invoice_id, SUM(quantity) AS quantity,
                  ROUND(SUM(quantity*unit_price),2) AS line_total, COUNT(*) AS line_count
                  FROM invoice_lines GROUP BY invoice_id)
    SELECT i.*, v.vendor_name, v.region, l.quantity, l.line_total, l.line_count
    FROM invoices i JOIN vendors v ON v.vendor_id=i.vendor_id
    JOIN lines l ON l.invoice_id=i.invoice_id ORDER BY i.invoice_date,i.invoice_id
    """
    with connect(db_path) as con:
        return pd.read_sql_query(query, con)

def validate_record(record):
    clean = {**DEFAULTS, **record}
    errors = [f'{field} is required' for field in REQUIRED if field not in clean]
    for field, (low, high) in LIMITS.items():
        if field not in clean:
            continue
        try:
            value = float(clean[field])
            if not np.isfinite(value) or not low <= value <= high:
                raise ValueError()
            if field in {'quantity', 'line_count', 'payment_terms_days'} and not value.is_integer():
                errors.append(f'{field} must be a whole number')
            clean[field] = value
        except (ValueError, TypeError, OverflowError):
            errors.append(f'{field} must be a finite number between {low:g} and {high:g}')
    for field, choices in [('shipping_mode', MODES), ('region', REGIONS)]:
        value = clean.get(field)
        if not isinstance(value, str) or value.strip() not in choices:
            errors.append(f'{field} must be one of: {", ".join(choices)}')
        else:
            clean[field] = value.strip()
    if not errors and clean['line_count'] > clean['quantity']:
        errors.append('line_count cannot exceed quantity')
    return clean, errors

def engineer_features(frame):
    result = frame.copy()
    for field, value in DEFAULTS.items():
        if field not in result:
            result[field] = value
    result['amount_mismatch_ratio'] = (result.invoice_amount-result.line_total).abs()/result.line_total.clip(lower=.01)
    result['freight_ratio'] = result.freight_cost/result.invoice_amount.clip(lower=.01)
    result['cost_per_unit'] = result.invoice_amount/result.quantity.clip(lower=1)
    return result

def make_pipeline(features, estimator):
    categorical = ['shipping_mode', 'region']
    return Pipeline([('prepare', ColumnTransformer([
        ('numeric', 'passthrough', [c for c in features if c not in categorical]),
        ('category', OneHotEncoder(handle_unknown='ignore', sparse_output=False), categorical)])),
        ('model', estimator)])

def train_models(db_path=None, model_dir=None):
    initialize_database(db_path)
    frame = engineer_features(load_invoices(db_path))
    if len(frame) < 100:
        raise ValueError('At least 100 complete invoices are required.')
    train, test = frame.iloc[:int(.8*len(frame))], frame.iloc[int(.8*len(frame)):]
    if train.needs_review.nunique() != 2 or test.needs_review.nunique() != 2:
        raise ValueError('Both chronological partitions must contain both review labels.')
    normal_train, normal_test = train[train.needs_review == 0], test[test.needs_review == 0]
    if len(normal_train) < 30 or len(normal_test) < 10:
        raise ValueError('Too few normal invoices to evaluate freight prediction.')
    freight = make_pipeline(FREIGHT_FEATURES, RandomForestRegressor(
        n_estimators=140, max_depth=18, min_samples_leaf=2, random_state=42, n_jobs=-1))
    risk = make_pipeline(RISK_FEATURES, RandomForestClassifier(
        n_estimators=160, max_depth=16, min_samples_leaf=2, class_weight='balanced', random_state=42, n_jobs=-1))
    freight.fit(normal_train[FREIGHT_FEATURES], normal_train.freight_cost)
    risk.fit(train[RISK_FEATURES], train.needs_review)
    predicted = freight.predict(normal_test[FREIGHT_FEATURES])
    labels = risk.predict(test[RISK_FEATURES])
    scores = risk.predict_proba(test[RISK_FEATURES])[:, 1]
    def importance(model):
        names = model.named_steps['prepare'].get_feature_names_out()
        values = model.named_steps['model'].feature_importances_
        return sorted([dict(feature=str(n).split('__',1)[-1], importance=float(v))
                       for n,v in zip(names, values)], key=lambda x:x['importance'], reverse=True)[:12]
    metrics = dict(data_source='synthetic_demo', artifact_version=1, sklearn_version=sklearn.__version__,
        trained_at_utc=datetime.now(timezone.utc).isoformat(), rows=len(frame), train_rows=len(train), holdout_rows=len(test),
        split='First 80% by invoice date for training; latest 20% untouched for evaluation.',
        train_end=str(train.invoice_date.max()), holdout_start=str(test.invoice_date.min()),
        freight_train_rows=len(normal_train), freight_holdout_rows=len(normal_test),
        freight=dict(r2=float(r2_score(normal_test.freight_cost, predicted)),
            mae=float(mean_absolute_error(normal_test.freight_cost, predicted)),
            rmse=float(np.sqrt(mean_squared_error(normal_test.freight_cost, predicted))),
            baseline_mae=float(mean_absolute_error(normal_test.freight_cost,
                               np.repeat(normal_train.freight_cost.median(), len(normal_test)))),
            feature_importance=importance(freight)),
        risk=dict(accuracy=float(accuracy_score(test.needs_review, labels)),
            precision=float(precision_score(test.needs_review, labels, zero_division=0)),
            recall=float(recall_score(test.needs_review, labels, zero_division=0)),
            f1=float(f1_score(test.needs_review, labels, zero_division=0)),
            roc_auc=float(roc_auc_score(test.needs_review, scores)), threshold=.5,
            confusion_matrix=confusion_matrix(test.needs_review, labels, labels=[0,1]).tolist(),
            feature_importance=importance(risk)),
        regression_examples=[dict(actual=float(a), predicted=float(p))
                             for a,p in zip(normal_test.freight_cost.iloc[:150], predicted[:150])],
        training_ranges={c:[float(train[c].min()),float(train[c].max())]
                         for c in ['quantity','invoice_amount','weight_kg','distance_km']})
    folder = Path(model_dir or MODEL_DIR)
    folder.mkdir(parents=True, exist_ok=True)
    bundle = dict(freight=freight, risk=risk, metrics=metrics)
    temporary = folder / 'invoice_models.tmp'
    joblib.dump(bundle, temporary, compress=3)
    temporary.replace(folder / 'invoice_models.joblib')
    (folder / 'metrics.json').write_text(json.dumps(metrics, indent=2), encoding='utf-8')
    return bundle

def ensure_models(db_path=None, model_dir=None):
    initialize_database(db_path)
    path = Path(model_dir or MODEL_DIR) / 'invoice_models.joblib'
    if path.exists():
        # Artifacts are generated locally; the app never accepts uploaded pickle/joblib files.
        bundle = joblib.load(path)
        if (bundle['metrics'].get('artifact_version') == 1
                and bundle['metrics'].get('sklearn_version') == sklearn.__version__):
            return bundle
    return train_models(db_path, model_dir)

def score_frame(frame, bundle):
    if frame.empty:
        return []
    data = engineer_features(frame)
    freight = bundle['freight'].predict(data[FREIGHT_FEATURES])
    scores = bundle['risk'].predict_proba(data[RISK_FEATURES])[:,1]
    output = []
    for (_,row), estimate, score in zip(data.iterrows(), freight, scores):
        reasons = []
        if row.amount_mismatch_ratio > .03:
            reasons.append('Invoice and itemized line total differ by more than 3%.')
        if row.freight_cost > estimate*1.5:
            reasons.append('Quoted freight exceeds the model estimate by more than 50%.')
        if row.payment_terms_days > 75:
            reasons.append('Payment terms exceed the 75-day review policy.')
        if score >= .5:
            reasons.append('ML review score meets the 50% screening threshold.')
        warnings = [f'{c} is outside the training data range.'
                    for c,(low,high) in bundle['metrics']['training_ranges'].items()
                    if not low <= float(row[c]) <= high]
        output.append(dict(expected_freight=round(max(0,float(estimate)),2), risk_score=float(score),
                           decision='Review required' if reasons else 'No flags detected',
                           reasons=reasons,warnings=warnings, amount_mismatch_pct=round(row.amount_mismatch_ratio*100,2)))
    return output

def analyze_invoice(record, bundle):
    clean, errors = validate_record(record)
    if errors:
        raise ValueError('; '.join(errors))
    return score_frame(pd.DataFrame([clean]), bundle)[0]

def read_batch_csv(content):
    try:
        text = content.decode('utf-8-sig') if isinstance(content, bytes) else content
        reader = csv.reader(io.StringIO(text))
        columns = [c.strip() for c in next(reader, [])]
        if not columns or any(not c for c in columns):
            raise ValueError('CSV must have non-empty column names.')
        if len(columns) != len(set(columns)):
            raise ValueError('CSV contains duplicate column names.')
        rows = []
        for row in reader:
            if len(row) != len(columns):
                raise ValueError('Every CSV row must have the same number of fields as the header.')
            rows.append(row)
            if len(rows) > 5000:
                raise ValueError('Maximum 5,000 invoices per file.')
        if not rows:
            raise ValueError('CSV contains no invoice rows.')
        missing = [c for c in REQUIRED if c not in columns]
        if missing:
            raise ValueError('Missing required columns: '+', '.join(missing))
        return pd.DataFrame(rows, columns=columns)
    except (UnicodeError, csv.Error) as exc:
        raise ValueError('Upload a valid UTF-8 CSV file.') from exc

def analyze_batch(frame, bundle):
    missing = [c for c in REQUIRED if c not in frame]
    if missing:
        raise ValueError('Missing required columns: '+', '.join(missing))
    if not 1 <= len(frame) <= 5000:
        raise ValueError('Provide between 1 and 5,000 rows.')
    valid, positions, rows = [], [], []
    for position,(_,record) in enumerate(frame.iterrows()):
        clean, errors = validate_record(record.to_dict())
        rows.append(dict(row_number=position+2,status='Invalid' if errors else 'Scored',error='; '.join(errors),
                         expected_freight=None,risk_score=None,decision='Invalid input' if errors else '',reasons='',warnings=''))
        if not errors:
            valid.append(clean)
            positions.append(position)
    for pos,result in zip(positions, score_frame(pd.DataFrame(valid), bundle)):
        rows[pos].update(expected_freight=result['expected_freight'],risk_score=round(result['risk_score'],6),
                         decision=result['decision'],reasons=' | '.join(result['reasons']),warnings=' | '.join(result['warnings']))
    original, generated = frame.reset_index(drop=True).copy(), pd.DataFrame(rows)
    # Preserve arbitrary input columns, even if their names overlap generated results.
    used = set(original.columns) | set(generated.columns)
    rename = {}
    for column in original:
        if column in generated:
            replacement = 'input_'+column
            while replacement in used:
                replacement = 'input_'+replacement
            rename[column] = replacement
            used.add(replacement)
    return pd.concat([original.rename(columns=rename), generated],axis=1)

def safe_csv(frame):
    output = frame.copy()
    for column in output.select_dtypes(include=['object','string']).columns:
        output[column] = output[column].map(lambda v: "'"+v if isinstance(v,str)
                                            and v.lstrip().startswith(('=','+','-','@')) else v)
    return output.to_csv(index=False).encode('utf-8-sig')

def save_review(record, result, source='Manual', db_path=None):
    reference = str(record.get('invoice_id','MANUAL')).strip()[:120] or 'MANUAL'
    with connect(db_path) as con:
        cur = con.execute('''INSERT INTO audit_log
            (invoice_reference,source,input_json,expected_freight,risk_score,decision,reasons_json)
            VALUES (?,?,?,?,?,?,?)''', (reference,source,json.dumps(record,default=str),result['expected_freight'],
            result['risk_score'],result['decision'],json.dumps(result['reasons'])))
        return cur.lastrowid

def load_history(db_path=None):
    with connect(db_path) as con:
        return pd.read_sql_query('''SELECT id,created_at,invoice_reference,source,expected_freight,
            risk_score,decision,reasons_json FROM audit_log ORDER BY id DESC LIMIT 1000''', con)

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--train', action='store_true',help='Retrain both models on the local database')
    parser.add_argument('--setup', action='store_true',help='Initialize missing data and models')
    args = parser.parse_args()
    bundle = train_models() if args.train else ensure_models()
    sample = load_invoices().drop(columns=['needs_review','vendor_id']).head(25)
    (DB_PATH.parent/'sample_invoices.csv').write_bytes(safe_csv(sample))
    print(json.dumps(dict(database=str(DB_PATH),model_folder=str(MODEL_DIR),
        demo_invoices=bundle['metrics']['rows'],freight_r2=bundle['metrics']['freight']['r2'],
        risk_f1=bundle['metrics']['risk']['f1']),indent=2))

if __name__ == '__main__':
    main()
