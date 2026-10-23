from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import engine
from streamlit.testing.v1 import AppTest


class AppWorkflowTests(unittest.TestCase):
    def test_all_pages_and_prediction_workflows(self):
        import streamlit as st
        with tempfile.TemporaryDirectory(prefix='invoice-ui-test-') as folder:
            db,models = Path(folder)/'test.db',Path(folder)/'models'
            with patch.object(engine,'DB_PATH',db),patch.object(engine,'MODEL_DIR',models):
                engine.initialize_database(n_invoices=400)
                engine.ensure_models()
                st.cache_resource.clear()
                st.cache_data.clear()
                app = AppTest.from_file(str(engine.ROOT/'app.py'),default_timeout=90).run()
                self.assertEqual(len(app.exception),0)
                def navigate(name):
                    app.sidebar.radio[0].set_value(name).run()
                    self.assertEqual(len(app.exception),0,f'{name}: {app.exception}')
                def click(label):
                    next(button for button in app.button if button.label==label).click().run()
                    self.assertEqual(len(app.exception),0,f'{label}: {app.exception}')
                navigate('Freight estimator')
                click('Estimate freight')
                self.assertGreater(len(app.metric),0)
                navigate('Invoice review')
                app.number_input(key='i_freight').set_value(100000.)
                click('Analyze invoice')
                self.assertTrue(any('Review required' in item.value for item in app.warning))
                click('Save to review history')
                self.assertEqual(len(engine.load_history()),1)
                navigate('Review history')
                self.assertGreater(len(app.dataframe),0)
                navigate('Batch analysis')
                click('Run example batch')
                self.assertGreater(len(app.dataframe),0)
                navigate('Model report')
                self.assertGreater(len(app.metric),0)
                navigate('Overview')
                st.cache_resource.clear()
                st.cache_data.clear()

if __name__=='__main__':
    unittest.main()
