"""Regression tests for the billing workspace.

    python -m unittest discover -s tests -v

Stdlib only, so the suite runs without installing pytest.
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.anomaly import (  # noqa: E402
    ANOMALY_COLUMN,
    BASELINE_COLUMN,
    CONSUMER_COLUMN,
    TYPE_COLUMN,
    UNITS_COLUMN,
    DetectorConfig,
    MeterAnomalyDetector,
)
from src.tariff_engine import (  # noqa: E402
    BILL_COLUMN,
    DEFAULT_POLICY,
    SLAB_DOMESTIC,
    TAX_COLUMN,
    apply_billing,
    bill_breakdown,
    calculate_bill,
    reconcile_bills,
)

DATASET_FILE = ROOT / "consumer_data" / "sample_consumer_120_rows.xlsx"
CSV_FILE = ROOT / "consumer_data" / "con.csv"


def load_dataset() -> pd.DataFrame:
    return pd.read_excel(DATASET_FILE)


class TariffTests(unittest.TestCase):
    def test_default_policy_is_the_telescopic_slab(self):
        self.assertIs(DEFAULT_POLICY, SLAB_DOMESTIC)

    def test_sample_bills_are_priced_with_telescopic_slabs(self):
        df = load_dataset()
        reconciled = reconcile_bills(df, DEFAULT_POLICY)
        row = reconciled[reconciled[UNITS_COLUMN] == 282].iloc[0]
        self.assertAlmostEqual(float(row["Expected_Bill"]), 1401.00, places=2)
        self.assertAlmostEqual(float(row["Expected_Tax"]), 84.06, places=2)

        spike = reconciled[reconciled[UNITS_COLUMN] == 4850].iloc[0]
        self.assertAlmostEqual(float(spike["Expected_Bill"]), 48775.00, places=2)
        self.assertAlmostEqual(float(spike["Expected_Tax"]), 2926.50, places=2)

    def test_sample_tax_is_six_percent_of_the_slab_subtotal(self):
        df = load_dataset()
        reconciled = reconcile_bills(df, DEFAULT_POLICY)
        self.assertTrue(
            np.allclose(
                reconciled["Expected_Tax"],
                reconciled["Expected_Bill"] * 0.06,
                atol=0.01,
            )
        )

    def test_csv_sample_is_priced_with_the_slab_tariff(self):
        df = pd.read_csv(CSV_FILE)
        reconciled = reconcile_bills(df, DEFAULT_POLICY)
        row = reconciled.iloc[0]
        self.assertAlmostEqual(float(row["Expected_Bill"]), 1005.00, places=2)
        self.assertAlmostEqual(float(row["Expected_Tax"]), 60.30, places=2)
        self.assertAlmostEqual(
            bill_breakdown(210, DEFAULT_POLICY, row["Fixed_Charges"])["total"],
            1065.30,
            places=2,
        )

    def test_breakdown_is_internally_consistent(self):
        breakdown = bill_breakdown(282, DEFAULT_POLICY)
        self.assertAlmostEqual(breakdown["energy_charge"], 1301.00, places=2)
        self.assertAlmostEqual(breakdown["subtotal"], 1401.00, places=2)
        self.assertAlmostEqual(breakdown["tax"], 84.06, places=2)
        self.assertAlmostEqual(breakdown["total"], 1485.06, places=2)
        self.assertEqual(calculate_bill(282, DEFAULT_POLICY), breakdown["total"])

    def test_zero_units_bills_fixed_charge_only(self):
        breakdown = bill_breakdown(0, DEFAULT_POLICY)
        self.assertEqual(breakdown["energy_charge"], 0.0)
        self.assertEqual(breakdown["subtotal"], 100.0)
        self.assertEqual(breakdown["tax"], 6.0)

    def test_slab_policy_prices_increasingly(self):
        low = calculate_bill(100, DEFAULT_POLICY)
        high = calculate_bill(1000, DEFAULT_POLICY)
        self.assertGreater(high, low)
        self.assertGreater(calculate_bill(1, DEFAULT_POLICY), calculate_bill(0, DEFAULT_POLICY))

    def test_apply_billing_only_touches_masked_rows(self):
        df = load_dataset().head(10)
        masked = pd.Series([True] + [False] * 9, index=df.index)
        updated = apply_billing(df, DEFAULT_POLICY, mask=masked)
        expected = bill_breakdown(float(df[UNITS_COLUMN].iloc[0]), DEFAULT_POLICY)["subtotal"]
        self.assertAlmostEqual(float(updated[BILL_COLUMN].iloc[0]), expected, places=2)
        for i in range(1, 10):
            self.assertEqual(updated[BILL_COLUMN].iloc[i], df[BILL_COLUMN].iloc[i])

    def test_apply_billing_preserves_all_rows_when_mask_is_empty(self):
        df = load_dataset().head(5)
        empty = pd.Series([False] * 5, index=df.index)
        updated = apply_billing(df, DEFAULT_POLICY, mask=empty)
        self.assertTrue(np.allclose(updated[BILL_COLUMN], df[BILL_COLUMN]))
        self.assertNotIn("Bill_Amount_Expected", updated.columns)


class DetectorTests(unittest.TestCase):
    def setUp(self):
        self.df = load_dataset()
        self.detector = MeterAnomalyDetector(DetectorConfig()).fit(self.df)

    def test_detects_the_faulty_meter_with_no_false_positives(self):
        metrics = self.detector.evaluate(self.df)
        self.assertTrue(metrics["available"])
        self.assertEqual(metrics["actual_faults"], 1)
        self.assertEqual(metrics["true_positives"], 1)
        self.assertEqual(metrics["false_positives"], 0)
        self.assertEqual(metrics["false_negatives"], 0)

    def test_flagged_row_is_the_faulty_one(self):
        report = self.detector.detect(self.df)
        flagged = report[report[ANOMALY_COLUMN]]
        self.assertEqual(len(flagged), 1)
        self.assertEqual(str(flagged["Meter_Status"].iloc[0]), "FAULTY")
        self.assertEqual(flagged[TYPE_COLUMN].iloc[0], "SPIKE")

    def test_normal_readings_are_not_flagged(self):
        report = self.detector.detect(self.df)
        normal = report[report["Meter_Status"] == "OK"]
        self.assertEqual(int(normal[ANOMALY_COLUMN].sum()), 0)

    def test_baseline_is_the_consumer_median(self):
        report = self.detector.detect(self.df)
        self.assertAlmostEqual(
            float(report[BASELINE_COLUMN].iloc[0]), float(self.df[UNITS_COLUMN].median()), places=2
        )

    def test_baselines_are_computed_per_consumer(self):
        frame = pd.DataFrame(
            {
                CONSUMER_COLUMN: ["A"] * 20 + ["B"] * 20,
                UNITS_COLUMN: [100.0] * 20 + [1000.0] * 20,
            }
        )
        detector = MeterAnomalyDetector().fit(frame)
        report = detector.detect(frame)
        self.assertAlmostEqual(float(report[BASELINE_COLUMN].iloc[0]), 100.0, places=2)
        self.assertAlmostEqual(float(report[BASELINE_COLUMN].iloc[-1]), 1000.0, places=2)

    def test_a_spike_in_a_second_consumer_is_detected(self):
        frame = pd.DataFrame(
            {
                CONSUMER_COLUMN: ["A"] * 20 + ["B"] * 20,
                UNITS_COLUMN: [100.0] * 20 + [1000.0] * 19 + [20000.0],
            }
        )
        detector = MeterAnomalyDetector().fit(frame)
        report = detector.detect(frame)
        flagged = report[report[ANOMALY_COLUMN]]
        self.assertEqual(len(flagged), 1)
        self.assertAlmostEqual(float(flagged[UNITS_COLUMN].iloc[0]), 20000.0, places=2)

    def test_dead_meter_is_detected(self):
        frame = pd.DataFrame(
            {
                CONSUMER_COLUMN: ["A"] * 20,
                UNITS_COLUMN: [100.0] * 19 + [0.0],
            }
        )
        detector = MeterAnomalyDetector().fit(frame)
        report = detector.detect(frame)
        self.assertEqual(int(report[ANOMALY_COLUMN].sum()), 1)
        self.assertEqual(report[TYPE_COLUMN].iloc[-1], "DEAD")

    def test_detector_round_trips_through_disk(self, tmp=None):
        import tempfile

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "detector.joblib"
            self.detector.save(path)
            reloaded = MeterAnomalyDetector.load(path)
            self.assertEqual(
                int(reloaded.detect(self.df)[ANOMALY_COLUMN].sum()),
                int(self.detector.detect(self.df)[ANOMALY_COLUMN].sum()),
            )

    def test_missing_units_column_raises(self):
        with self.assertRaises(KeyError):
            self.detector.detect(pd.DataFrame({"Consumer_ID": ["A"], "Bill_Amount": [1.0]}))


class CorrectionTests(unittest.TestCase):
    def test_correction_resets_units_and_reprices_the_row(self):
        df = load_dataset()
        detector = MeterAnomalyDetector().fit(df)
        report = detector.detect(df)
        mask = report[ANOMALY_COLUMN]
        corrected = df.copy()
        corrected.loc[mask, UNITS_COLUMN] = report.loc[mask, BASELINE_COLUMN].to_numpy(dtype=float)
        corrected = apply_billing(corrected, DEFAULT_POLICY, mask=mask)

        row = corrected[mask].iloc[0]
        self.assertAlmostEqual(float(row[UNITS_COLUMN]), float(report.loc[mask, BASELINE_COLUMN].iloc[0]), places=2)
        self.assertAlmostEqual(
            float(row[BILL_COLUMN]),
            bill_breakdown(row[UNITS_COLUMN], DEFAULT_POLICY)["subtotal"],
            places=2,
        )
        self.assertAlmostEqual(
            float(row[TAX_COLUMN]),
            bill_breakdown(row[UNITS_COLUMN], DEFAULT_POLICY)["tax"],
            places=2,
        )

    def test_correction_removes_the_anomaly(self):
        df = load_dataset()
        detector = MeterAnomalyDetector().fit(df)
        report = detector.detect(df)
        mask = report[ANOMALY_COLUMN]
        corrected = df.copy()
        corrected.loc[mask, UNITS_COLUMN] = report.loc[mask, BASELINE_COLUMN].to_numpy(dtype=float)
        corrected = apply_billing(corrected, DEFAULT_POLICY, mask=mask)
        self.assertEqual(int(detector.detect(corrected)[ANOMALY_COLUMN].sum()), 0)

    def test_correction_does_not_disturb_other_rows(self):
        df = load_dataset()
        detector = MeterAnomalyDetector().fit(df)
        report = detector.detect(df)
        mask = report[ANOMALY_COLUMN]
        corrected = df.copy()
        corrected.loc[mask, UNITS_COLUMN] = report.loc[mask, BASELINE_COLUMN].to_numpy(dtype=float)
        corrected = apply_billing(corrected, DEFAULT_POLICY, mask=mask)
        untouched = ~mask
        self.assertTrue(np.allclose(corrected.loc[untouched, BILL_COLUMN], df.loc[untouched, BILL_COLUMN]))


if __name__ == "__main__":
    unittest.main(verbosity=2)
