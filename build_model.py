"""Train the meter anomaly detector and export it to disk.

    python build_model.py

Training data is discovered from
``consumer_data/ledgers/Consumer_Ledger_Report_*.xls``. The sample workbook
remains a regression fixture and is used only when no ledger exports exist or
for the printed sample check.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

APP_DIR = Path(__file__).resolve().parent
if str(APP_DIR) not in sys.path:
    sys.path.insert(0, str(APP_DIR))

from src.anomaly import (
    CONSUMER_COLUMN,
    METER_DEFECT_COLUMN,
    UNITS_COLUMN,
    DetectorConfig,
    MeterAnomalyDetector,
)
from src.ledger import is_ledger_frame, ledger_summary, load_ledgers
from src.tariff_engine import DEFAULT_POLICY, reconcile_bills, reconcile_ledger

DATA_DIR = APP_DIR / "consumer_data" / "ledgers"
SAMPLE_FILE = APP_DIR / "consumer_data" / "sample_consumer_120_rows.xlsx"
MODEL_FILE = APP_DIR / "models" / "meter_anomaly_model.joblib"
LEDGER_PATTERNS = ("Consumer_Ledger_Report_*.xls", "Consumer_Ledger_Report_*.xlsx")


def ledger_files(directory: Path = DATA_DIR) -> list[Path]:
    if not directory.exists():
        return []
    return sorted(
        path
        for pattern in LEDGER_PATTERNS
        for path in directory.glob(pattern)
    )


def load_sample(path: Path = SAMPLE_FILE) -> pd.DataFrame | None:
    if not path.exists():
        legacy = APP_DIR / "consumer_data" / "sample_consumer_120_rows.xlsx"
        path = legacy if legacy.exists() else path
    if not path.exists():
        return None
    df = pd.read_excel(path)
    if UNITS_COLUMN not in df.columns:
        raise KeyError(f"sample is missing the '{UNITS_COLUMN}' column")
    df[UNITS_COLUMN] = pd.to_numeric(df[UNITS_COLUMN], errors="coerce").fillna(0.0)
    return df


def load_training_data() -> tuple[pd.DataFrame, str]:
    files = ledger_files()
    if files:
        df = load_ledgers(DATA_DIR)
        return df, f"{len(files)} ledger file(s)"
    sample = load_sample()
    if sample is not None:
        return sample, SAMPLE_FILE.name
    raise FileNotFoundError(
        f"no training data found. Expected '{DATA_DIR / LEDGER_PATTERNS[0]}' "
        f"or '{SAMPLE_FILE.name}'."
    )


def _print_sample_check(detector: MeterAnomalyDetector) -> None:
    sample = load_sample()
    if sample is None:
        return
    print("Checking the sample regression fixture...")
    reconciled = reconcile_bills(sample, DEFAULT_POLICY)
    matches = int(reconciled["Bill_Matches"].sum())
    print(
        f"  sample bills priced under '{DEFAULT_POLICY.name}'; "
        f"{matches}/{len(reconciled)} match the historical flat dataset bills"
    )
    metrics = detector.evaluate(sample)
    if metrics.get("available"):
        print(
            "  sample detector: "
            f"TP={metrics['true_positives']} FP={metrics['false_positives']} "
            f"FN={metrics['false_negatives']} (actual faults={metrics['actual_faults']})"
        )


def main() -> int:
    print("Loading training data...")
    df, source = load_training_data()
    consumers = df[CONSUMER_COLUMN].nunique() if CONSUMER_COLUMN in df.columns else 1
    print(f"  {len(df)} rows / {consumers} consumer(s) from {source}")

    if is_ledger_frame(df):
        summary = ledger_summary(df)
        print(
            f"  {summary['files']} file(s), {summary['meter_defects']} meter-defect row(s), "
            f"FYs {', '.join(summary['fiscal_years'])}"
        )

    print("Training the meter anomaly detector...")
    detector = MeterAnomalyDetector(DetectorConfig()).fit(df)
    print(
        f"  trained on {detector.training_summary.get('training_rows', len(df))} rows; "
        f"{int(df[METER_DEFECT_COLUMN].sum()) if METER_DEFECT_COLUMN in df.columns else 0} "
        "rule-based meter defects will be reported separately"
    )

    metrics = detector.evaluate(df)
    if metrics.get("available"):
        print(
            "  evaluation vs Meter_Status: "
            f"TP={metrics['true_positives']} FP={metrics['false_positives']} "
            f"FN={metrics['false_negatives']} (actual faults={metrics['actual_faults']})"
        )
        if metrics["false_positives"] or metrics["false_negatives"]:
            print("  WARNING: detector is not perfectly aligned with the ground truth")

    if is_ledger_frame(df):
        print("Reconciling ledger rows against the FY tariff schedule...")
        reconciled = reconcile_ledger(df)
        identity_ok = int(reconciled["Current_Identity_Matches"].sum())
        strict_ok = int(reconciled["Strict_Tariff_Matches"].sum())
        reconcilable = int(reconciled["Reconcilable"].sum())
        print(
            f"  current-total identity: {identity_ok}/{len(reconciled)} rows; "
            f"strict tariff: {strict_ok}/{reconcilable} reconcilable rows"
        )
        unresolved = reconciled[~reconciled["Current_Identity_Matches"]]
        if len(unresolved):
            print(
                f"  {len(unresolved)} row(s) have current-total residuals "
                "(see Current_Identity_Residual)"
            )

    _print_sample_check(detector)

    detector.save(MODEL_FILE)
    print(f"Success! Compiled detector written to '{MODEL_FILE}'")
    return 0


if __name__ == "__main__":
    sys.exit(main())
