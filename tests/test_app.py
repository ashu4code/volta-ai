"""Smoke tests for the Streamlit workspace.

    python -m unittest tests.test_app -v

Uses Streamlit's own ``AppTest`` harness, so no browser or server is needed.
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from streamlit.testing.v1 import AppTest  # noqa: E402

from src.ledger import load_ledgers  # noqa: E402

APP_FILE = ROOT / "main.py"
DATASET_FILE = ROOT / "consumer_data" / "sample_consumer_120_rows.xlsx"
LEDGER_DIR = ROOT / "consumer_data" / "ledgers"


def load_dataset() -> pd.DataFrame:
    return pd.read_excel(DATASET_FILE)


def build_app() -> AppTest:
    app = AppTest.from_file(str(APP_FILE), default_timeout=60)
    df = load_dataset()
    app.session_state["current_data"] = df
    app.session_state["edited_data"] = df
    app.session_state["active_file"] = DATASET_FILE.name
    return app


def build_ledger_app() -> AppTest:
    app = AppTest.from_file(str(APP_FILE), default_timeout=120)
    df = load_ledgers(LEDGER_DIR)
    app.session_state["current_data"] = df
    app.session_state["edited_data"] = df
    app.session_state["active_file"] = "Consumer_Ledger_Report_test.xls"
    return app


def assistant_replies(app: AppTest) -> list[str]:
    """Every markdown body written inside an assistant chat message."""
    return [
        child.value
        for message in app.chat_message
        if message.name == "assistant"
        for child in message.markdown
    ]


class AppTests(unittest.TestCase):
    def test_app_starts_without_exceptions(self):
        app = build_app().run()
        self.assertEqual(len(app.exception), 0)

    def test_engine_is_reported_online(self):
        app = build_app().run()
        self.assertEqual(len(app.exception), 0)
        banners = [m.value for m in app.markdown]
        self.assertTrue(any("AI ENGINE ONLINE" in b for b in banners))

    def test_scan_command_reports_the_faulty_reading(self):
        app = build_app().run()
        app.chat_input[0].set_value("scan anomalies").run()
        self.assertEqual(len(app.exception), 0)
        replies = assistant_replies(app)
        self.assertTrue(any("4,850" in r for r in replies))
        self.assertTrue(any("anomalous meter reading" in r for r in replies))
        self.assertTrue(any("51,701.50" in r for r in replies))
        self.assertTrue(any("50,420.49" in r for r in replies))

    def test_reconcile_command_reports_slab_mismatches(self):
        app = build_app().run()
        app.chat_input[0].set_value("reconcile bills").run()
        self.assertEqual(len(app.exception), 0)
        replies = assistant_replies(app)
        self.assertTrue(any("rows do not match the engine" in r for r in replies))

    def test_apply_command_rewrites_the_buffer(self):
        app = build_app().run()
        before = len(app.chat_message)
        app.chat_input[0].set_value("apply corrections").run()
        self.assertEqual(len(app.exception), 0)
        self.assertGreater(len(app.chat_message), before)

        updated = app.session_state["current_data"]
        self.assertIn("Units_Consumed", updated.columns)
        self.assertNotIn("Is_Anomaly", updated.columns)
        self.assertLessEqual(float(updated["Units_Consumed"].max()), 1000.0)

    def test_scan_after_correction_finds_nothing(self):
        app = build_app().run()
        app.chat_input[0].set_value("apply corrections").run()
        app.chat_input[0].set_value("scan anomalies").run()
        self.assertEqual(len(app.exception), 0)
        replies = assistant_replies(app)
        self.assertTrue(any("No billing exceptions discovered" in r for r in replies))

    def test_unknown_command_falls_back_to_help(self):
        app = build_app().run()
        app.chat_input[0].set_value("what can you do").run()
        self.assertEqual(len(app.exception), 0)
        replies = assistant_replies(app)
        self.assertTrue(any("scan anomalies" in r for r in replies))

    def test_missing_units_column_is_reported_not_crashed(self):
        app = AppTest.from_file(str(APP_FILE), default_timeout=60)
        app.session_state["current_data"] = pd.DataFrame({"Consumer_ID": ["A"], "Bill_Amount": [100.0]})
        app.session_state["edited_data"] = pd.DataFrame({"Consumer_ID": ["A"], "Bill_Amount": [100.0]})
        app.run()
        app.chat_input[0].set_value("scan anomalies").run()
        self.assertEqual(len(app.exception), 0)
        replies = assistant_replies(app)
        self.assertTrue(any("Column missing error" in r for r in replies))



class LedgerAppTests(unittest.TestCase):
    def test_ledger_scan_and_reconcile(self):
        app = build_ledger_app().run()
        self.assertEqual(len(app.exception), 0)

        app.chat_input[0].set_value("scan anomalies").run()
        self.assertEqual(len(app.exception), 0)
        replies = assistant_replies(app)
        self.assertTrue(any("meter defect" in r for r in replies))

        app.chat_input[0].set_value("reconcile bills").run()
        self.assertEqual(len(app.exception), 0)
        replies = assistant_replies(app)
        self.assertTrue(any("Monthly identity:" in r for r in replies))
        self.assertTrue(any("Strict tariff:" in r for r in replies))

    def test_consumer_lookup_and_bill_anomaly(self):
        app = build_ledger_app().run()
        app.chat_input[0].set_value("DUDOB14460").run()
        self.assertEqual(len(app.exception), 0)
        replies = assistant_replies(app)
        self.assertTrue(any("23030008718 (DUDOB14460)" in r for r in replies))
        self.assertTrue(any("CHANDAN KUMAR" in r for r in replies))
        self.assertEqual(app.session_state["selected_consumer"], "23030008718")

        app.chat_input[0].set_value("bill anomaly").run()
        self.assertEqual(len(app.exception), 0)
        replies = assistant_replies(app)
        self.assertTrue(any("month-wise billing anomalies" in r for r in replies))
        self.assertTrue(any("2022-07" in r and "DROP" in r for r in replies))

    def test_bill_anomaly_accepts_direct_consumer_id(self):
        app = build_ledger_app().run()
        app.chat_input[0].set_value("bill anomaly 23030008718").run()
        self.assertEqual(len(app.exception), 0)
        replies = assistant_replies(app)
        self.assertTrue(any("month-wise billing anomalies" in r for r in replies))
        self.assertTrue(any("METER_DEFECT" in r for r in replies))

    def test_unknown_consumer_id_returns_suggestions(self):
        app = build_ledger_app().run()
        app.chat_input[0].set_value("consumer 999").run()
        self.assertEqual(len(app.exception), 0)
        replies = assistant_replies(app)
        self.assertTrue(any("was not found" in r for r in replies))
        self.assertTrue(any("Did you mean" in r for r in replies))

    def test_ledger_apply_corrects_only_model_anomalies(self):
        app = build_ledger_app().run()
        before = float(app.session_state["current_data"]["CURRENT_TOTAL"].sum())
        app.chat_input[0].set_value("apply corrections").run()
        self.assertEqual(len(app.exception), 0)
        updated = app.session_state["current_data"]
        self.assertNotIn("Is_Model_Anomaly", updated.columns)
        self.assertNotEqual(float(updated["CURRENT_TOTAL"].sum()), before)
        self.assertTrue(
            any("meter-defect row(s) remain unchanged" in r for r in assistant_replies(app))
        )


class LedgerAutoLoadTests(unittest.TestCase):
    def test_app_auto_loads_the_dataset_corpus(self):
        app = AppTest.from_file(str(APP_FILE), default_timeout=180).run()
        self.assertEqual(len(app.exception), 0)
        frame = app.session_state["current_data"]
        self.assertEqual(len(frame), 3943)
        self.assertEqual(frame["Consumer_ID"].nunique(), 83)
        self.assertIn("consumer_data ledger corpus", app.session_state["active_file"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
