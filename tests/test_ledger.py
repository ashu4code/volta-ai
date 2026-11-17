"""Regression tests for consumer-ledger ingestion."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.ledger import (  # noqa: E402
    BILLING_DAYS_NORMALIZED_COLUMN,
    BILLING_MONTH_COLUMN,
    CONSUMER_COLUMN,
    CURRENT_TOTAL_COLUMN,
    DATE_COLUMN,
    FISCAL_YEAR_COLUMN,
    METER_DEFECT_COLUMN,
    MONTHLY_UNITS_COLUMN,
    SOURCE_FILE_COLUMN,
    consumer_catalog,
    consumer_label,
    duplicate_period_count,
    find_consumers,
    is_ledger_frame,
    ledger_summary,
    ledger_warnings,
    load_ledger,
    load_ledger_bytes,
    load_ledgers,
    suggest_consumers,
)

DATASET_DIR = ROOT / "consumer_data" / "ledgers"
DUDOB_FILE = DATASET_DIR / "Consumer_Ledger_Report_2421_DUDOB14460.xls"
ACCOUNT_FILE = DATASET_DIR / "Consumer_Ledger_Report_2421_52355973850.xls"
SAMPLE_FILE = ROOT / "consumer_data" / "sample_consumer_120_rows.xlsx"


class LedgerNormalizationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.df = load_ledgers(DATASET_DIR)

    def test_combined_ledger_shape(self):
        self.assertEqual(len(self.df), 3943)
        self.assertEqual(self.df[CONSUMER_COLUMN].nunique(), 83)
        self.assertIn(SOURCE_FILE_COLUMN, self.df.columns)
        self.assertEqual(self.df[SOURCE_FILE_COLUMN].nunique(), 84)
        self.assertEqual(duplicate_period_count(self.df), 0)

    def test_required_normalized_columns_exist(self):
        expected = {
            CONSUMER_COLUMN,
            "Account_No",
            BILLING_MONTH_COLUMN,
            DATE_COLUMN,
            FISCAL_YEAR_COLUMN,
            BILLING_DAYS_NORMALIZED_COLUMN,
            MONTHLY_UNITS_COLUMN,
            "Billed_Units",
            METER_DEFECT_COLUMN,
            "Category",
            "Bill_Amount",
            "Tax_Amount",
            "Total_Amount",
            "Fixed_Charges",
            SOURCE_FILE_COLUMN,
        }
        self.assertTrue(expected.issubset(self.df.columns))

    def test_periods_are_zero_padded(self):
        self.assertTrue(self.df[BILLING_MONTH_COLUMN].str.len().eq(6).all())
        self.assertIn("012023", set(self.df[BILLING_MONTH_COLUMN]))
        self.assertTrue(self.df[DATE_COLUMN].notna().all())

    def test_long_billing_period_is_normalized_to_monthly_units(self):
        rows = self.df[
            (self.df["Account_No"] == "DUDOB14460")
            & (self.df[BILLING_MONTH_COLUMN] == "012023")
        ]
        self.assertEqual(len(rows), 1)
        row = rows.iloc[0]
        self.assertEqual(float(row[BILLING_DAYS_NORMALIZED_COLUMN]), 99.0)
        self.assertEqual(float(row["Billed_Units"]), 199.0)
        self.assertAlmostEqual(float(row[MONTHLY_UNITS_COLUMN]), 199 * 30 / 99, places=3)
        self.assertFalse(bool(row[METER_DEFECT_COLUMN]))

    def test_meter_defects_come_from_read_status(self):
        self.assertEqual(int(self.df[METER_DEFECT_COLUMN].sum()), 3155)
        statuses = self.df.loc[self.df[METER_DEFECT_COLUMN], "READ_STAT"]
        self.assertTrue(statuses.isin(["MD", "LK", "UM"]).all())
        zero_recorded = self.df[
            (pd.to_numeric(self.df["UNITS_CONS"], errors="coerce").fillna(0.0) <= 0)
            & (pd.to_numeric(self.df["UNITS_BILLED"], errors="coerce").fillna(0.0) > 0)
        ]
        self.assertTrue((self.df.loc[zero_recorded.index, METER_DEFECT_COLUMN]).all())

    def test_account_number_can_differ_from_consumer_id(self):
        ledger = load_ledger(ACCOUNT_FILE)
        pairs = ledger[["Consumer_ID", "Account_No"]].drop_duplicates()
        self.assertEqual(len(pairs), 1)
        self.assertEqual(pairs.iloc[0]["Consumer_ID"], "242107645338")
        self.assertEqual(pairs.iloc[0]["Account_No"], "52355973850")

    def test_bytes_loader_matches_path_loader(self):
        from_path = load_ledger(DUDOB_FILE)
        from_bytes = load_ledger_bytes(DUDOB_FILE.read_bytes(), DUDOB_FILE.name)
        pd.testing.assert_frame_equal(from_path, from_bytes)

    def test_summary_and_warnings(self):
        summary = ledger_summary(self.df)
        self.assertEqual(summary["rows"], 3943)
        self.assertEqual(summary["consumers"], 83)
        self.assertEqual(summary["files"], 84)
        self.assertEqual(summary["meter_defects"], 3155)
        self.assertEqual(len(summary["fiscal_years"]), 10)
        self.assertEqual(ledger_warnings(self.df), [])
        self.assertAlmostEqual(float(self.df[CURRENT_TOTAL_COLUMN].sum()), 10133856.60, places=2)

    def test_lookup_accepts_consumer_id_and_account_number(self):
        by_consumer = find_consumers(self.df, "23030008718")
        by_account = find_consumers(self.df, "dudob14460")
        self.assertEqual(len(by_consumer), 43)
        self.assertEqual(len(by_account), 43)
        self.assertEqual(
            set(by_consumer[CONSUMER_COLUMN]), set(by_account[CONSUMER_COLUMN])
        )
        self.assertEqual(
            consumer_label("23030008718", "DUDOB14460"),
            "23030008718 (DUDOB14460)",
        )

    def test_consumer_catalog_and_suggestions(self):
        catalog = consumer_catalog(self.df)
        self.assertEqual(len(catalog), 83)
        self.assertEqual(find_consumers(self.df, "999999").shape[0], 0)
        suggestions = suggest_consumers(self.df, "DUDOB144", limit=3)
        self.assertIn("23030008718 (DUDOB14460)", suggestions)

    def test_ledger_and_sample_frames_are_distinguished(self):
        self.assertTrue(is_ledger_frame(self.df))
        sample = pd.read_excel(SAMPLE_FILE)
        self.assertFalse(is_ledger_frame(sample))


if __name__ == "__main__":
    unittest.main(verbosity=2)
