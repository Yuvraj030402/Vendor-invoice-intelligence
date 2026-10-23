from pathlib import Path
import tempfile
import unittest
import joblib
import numpy as np
import pandas as pd
import engine


class PipelineTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp = tempfile.TemporaryDirectory(prefix='invoice-test-')
        cls.db = Path(cls.temp.name)/'test.db'
        cls.models = Path(cls.temp.name)/'models'
        engine.initialize_database(cls.db, n_invoices=1200)
        cls.bundle = engine.train_models(cls.db, cls.models)
        cls.data = engine.load_invoices(cls.db)

    @classmethod
    def tearDownClass(cls):
        cls.temp.cleanup()

    def test_database_initialization_is_idempotent(self):
        self.assertFalse(engine.initialize_database(self.db))
        self.assertEqual(len(engine.load_invoices(self.db)), 1200)
        with engine.connect(self.db) as con:
            self.assertEqual(con.execute('PRAGMA foreign_key_check').fetchall(), [])
            self.assertEqual(con.execute('SELECT COUNT(*) FROM vendors').fetchone()[0],40)

    def test_sql_aggregation_independent_fixture(self):
        db = Path(self.temp.name)/'fixture.db'
        with engine.connect(db) as con:
            con.executescript(engine.SCHEMA)
            con.execute("INSERT INTO vendors VALUES (1,'Fixture vendor','North')")
            con.execute("INSERT INTO invoices VALUES ('FIX-1',1,'2025-01-01',80,10,4,50,'Standard',30,0)")
            con.executemany('INSERT INTO invoice_lines VALUES (?,?,?,?)',[(1,'FIX-1',2,10),(2,'FIX-1',3,20)])
        row = engine.load_invoices(db).iloc[0]
        self.assertEqual(row.quantity,5)
        self.assertEqual(row.line_total,80)
        self.assertEqual(row.line_count,2)
        self.assertEqual(row.invoice_amount,80)

    def test_models_are_fitted_evaluated_and_reloadable(self):
        loaded = engine.ensure_models(self.db,self.models)
        record = self.data.iloc[0].to_dict()
        before,after = engine.analyze_invoice(record,self.bundle),engine.analyze_invoice(record,loaded)
        self.assertAlmostEqual(before['expected_freight'],after['expected_freight'])
        self.assertAlmostEqual(before['risk_score'],after['risk_score'])
        self.assertTrue(np.isfinite(before['expected_freight']))
        self.assertTrue(0 <= before['risk_score'] <= 1)
        metrics = loaded['metrics']
        self.assertEqual(metrics['train_rows']+metrics['holdout_rows'],1200)
        self.assertLess(metrics['train_end'],metrics['holdout_start'])
        self.assertLess(metrics['freight']['mae'],metrics['freight']['baseline_mae'])
        self.assertGreater(metrics['risk']['f1'],.65)

    def test_input_validation_rejects_nonfinite_and_invalid_values(self):
        record = self.data.iloc[0].to_dict()
        for field,value in [('quantity',0),('invoice_amount',-1),('freight_cost',float('inf')),
                            ('line_total',float('nan')),('quantity',1.5),('weight_kg','text'),
                            ('shipping_mode','Rocket'),('region',None),('payment_terms_days',366)]:
            with self.subTest(field=field,value=value):
                with self.assertRaises(ValueError):
                    engine.analyze_invoice({**record,field:value},self.bundle)

    def test_batch_preserves_invalid_rows_order_and_identifiers(self):
        frame = self.data.head(3).drop(columns=['needs_review']).copy()
        frame.loc[frame.index[1],'invoice_amount'] = -5
        result = engine.analyze_batch(frame,self.bundle)
        self.assertEqual(result.invoice_id.tolist(),frame.invoice_id.tolist())
        self.assertEqual(result.status.tolist(),['Scored','Invalid','Scored'])
        self.assertTrue(pd.isna(result.loc[1,'expected_freight']))
        self.assertIn('invoice_amount',result.loc[1,'error'])
        parsed = engine.read_batch_csv(engine.safe_csv(frame))
        self.assertEqual(len(parsed),3)

    def test_csv_errors_are_readable(self):
        for content in [b'',b'x,x\n1,2',b'quantity\n2',b'x,y\n1',b'\xff\xfe']:
            with self.subTest(content=content),self.assertRaises(ValueError):
                engine.read_batch_csv(content)

    def test_policy_flags_are_independent_of_model_score(self):
        class LowScore:
            def predict_proba(self, frame):
                return np.tile([1.,0.],(len(frame),1))
        record = self.data.iloc[0].to_dict()
        record.update(invoice_amount=record['line_total']*1.3,payment_terms_days=120)
        result = engine.analyze_invoice(record,{**self.bundle,'risk':LowScore()})
        self.assertEqual(result['risk_score'],0)
        self.assertEqual(result['decision'],'Review required')
        self.assertGreaterEqual(len(result['reasons']),2)

    def test_history_persists_and_initialization_preserves_it(self):
        record = self.data.iloc[0].to_dict()
        result = engine.analyze_invoice(record,self.bundle)
        saved_id = engine.save_review(record,result,db_path=self.db)
        engine.initialize_database(self.db)
        history = engine.load_history(self.db)
        self.assertIn(saved_id,history.id.tolist())
        self.assertEqual(history.iloc[0].invoice_reference,record['invoice_id'])

    def test_export_handles_formula_text_and_column_collisions(self):
        frame = self.data.head(1).copy()
        frame['invoice_id'] = '=1+1'
        frame['status'] = 'original'
        frame['input_status'] = 'preserved'
        result = engine.analyze_batch(frame,self.bundle)
        self.assertTrue(result.columns.is_unique)
        text = engine.safe_csv(result).decode('utf-8-sig')
        self.assertIn("'=1+1",text)

if __name__=='__main__':
    unittest.main()
