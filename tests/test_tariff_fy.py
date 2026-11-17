"""Regression tests for the ledger-native FY tariff engine."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.ledger import CURRENT_TOTAL_COLUMN, load_ledgers  # noqa: E402
from src.tariff_engine import (  # noqa: E402
    TARIFF_SCHEDULE,
    fiscal_year_for,
    reconcile_ledger,
    reprice_ledger,
    tariff_for,
)

DATASET_DIR = ROOT / "consumer_data" / "ledgers"


class FiscalYearTests(unittest.TestCase):
    def test_april_starts_a_new_financial_year(self):
        self.assertEqual(fiscal_year_for("032023"), "2022-23")
        self.assertEqual(fiscal_year_for("042023"), "2023-24")
        self.assertEqual(fiscal_year_for("012026"), "2025-26")
        self.assertEqual(fiscal_year_for(pd.Timestamp("2024-03-31")), "2023-24")
        self.assertEqual(fiscal_year_for(pd.Timestamp("2024-04-01")), "2024-25")

    def test_schedule_has_every_ledger_year(self):
        years = {year for year, _ in TARIFF_SCHEDULE}
        self.assertEqual(
            years,
            {
                "2016-17",
                "2017-18",
                "2018-19",
                "2019-20",
                "2020-21",
                "2021-22",
                "2022-23",
                "2023-24",
                "2024-25",
                "2025-26",
            },
        )


class FYTariffTests(unittest.TestCase):
    def test_ds1_2016_slabs_mmc_and_meter_rent(self):
        policy = tariff_for("DS1", "2016-17")
        self.assertIsNotNone(policy)
        self.assertEqual(policy.energy_charge(100), 225.0)
        self.assertEqual(policy.energy_charge(101), 227.8)
        self.assertEqual(policy.billable_units(10, 1), 40.0)
        self.assertEqual(policy.fixed_charge(1, 30), 0.0)
        self.assertEqual(policy.meter_rent(30), 20.0)

    def test_kj_2016_leading_units_and_minimum(self):
        policy = tariff_for("KJ", "2016-17")
        self.assertIsNotNone(policy)
        self.assertEqual(policy.energy_charge(50), 93.0)
        self.assertEqual(policy.energy_charge(10), 40.0)

    def test_demand_tariff_prorates_by_days(self):
        policy = tariff_for("DS1D", "2022-23")
        self.assertIsNotNone(policy)
        self.assertEqual(policy.demand_charge(5, 30), 100.0)
        self.assertEqual(policy.demand_charge(5, 15), 50.0)

    def test_subsidy_rates_are_applied_per_unit(self):
        policy = tariff_for("DS1", "2023-24")
        self.assertAlmostEqual(policy.subsidy.amount(100), 502.0)
        self.assertAlmostEqual(tariff_for("KJ", "2017-18").subsidy.amount(10), 35.8)

    def test_unknown_category_or_year_is_unmapped(self):
        self.assertIsNone(tariff_for("DS9", "2022-23"))
        self.assertIsNone(tariff_for("DS1", "1999-00"))


class LedgerReconciliationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.raw = load_ledgers(DATASET_DIR)
        cls.df = reconcile_ledger(cls.raw)

    def test_known_ledger_counts(self):
        self.assertEqual(len(self.df), 3943)
        self.assertEqual(int(self.df["Current_Identity_Matches"].sum()), 3488)
        self.assertEqual(int(self.df["Reconcilable"].sum()), 3281)
        self.assertEqual(int(self.df["Strict_Tariff_Matches"].sum()), 2757)
        self.assertEqual(
            int(self.df["Current_Total_With_Ledger_Subsidy_Matches"].sum()), 3097
        )
        self.assertEqual(int(self.df["Strict_Current_Total_Matches"].sum()), 2807)

    def test_tariff_statuses(self):
        counts = self.df["Tariff_Status"].value_counts().to_dict()
        self.assertEqual(counts, {"OK": 3082, "UNVERIFIED": 498, "UNMAPPED": 363})

    def test_identity_residuals_are_reported(self):
        residuals = self.df.loc[~self.df["Current_Identity_Matches"]]
        self.assertEqual(len(residuals), 455)
        rounding = self.df[
            (self.df["Consumer_ID"] == "242203044701")
            & (self.df["Billing_Month"] == "112023")
        ].iloc[0]
        self.assertAlmostEqual(float(rounding["Current_Identity_Residual"]), 0.02, places=2)

    def test_reprice_selected_row_matches_the_tariff(self):
        rows = self.df[
            (self.df["Consumer_ID"] == "23030008718")
            & (self.df["Billing_Month"] == "082022")
        ]
        self.assertEqual(len(rows), 1)
        row = rows.copy()
        out = reprice_ledger(row, pd.Series(True, index=row.index))
        self.assertEqual(float(out[CURRENT_TOTAL_COLUMN].iloc[0]), 200.55)
        self.assertEqual(float(out["EC_CHRG"].iloc[0]), 152.50)
        self.assertEqual(float(out["ED_CURRENT"].iloc[0]), 9.15)
        self.assertEqual(float(out["GOV_SUBSIDY"].iloc[0]), -87.50)

    def test_reprice_only_touches_masked_rows(self):
        rows = self.df.head(2).copy()
        rows.loc[rows.index[1], CURRENT_TOTAL_COLUMN] = 999.0
        mask = pd.Series([True, False], index=rows.index)
        out = reprice_ledger(rows, mask)
        self.assertEqual(float(out[CURRENT_TOTAL_COLUMN].iloc[1]), 999.0)
        self.assertNotEqual(float(out[CURRENT_TOTAL_COLUMN].iloc[0]), 999.0)

    def test_unmapped_rows_are_left_untouched(self):
        row = self.df.head(1).copy()
        row["Fiscal_Year"] = "1999-00"
        row[CURRENT_TOTAL_COLUMN] = 1234.0
        out = reprice_ledger(row, pd.Series(True, index=row.index))
        self.assertEqual(float(out[CURRENT_TOTAL_COLUMN].iloc[0]), 1234.0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
