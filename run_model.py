"""Command line entry point for the meter anomaly detector.

    python run_model.py                  # scan the shipped ledgers
    python run_model.py --readings 240 4950

The detector's baselines are derived from the training data, never hard coded.
Ledger meter defects (``MD``/``LK``) are reported separately from statistical
consumption anomalies.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

APP_DIR = Path(__file__).resolve().parent
if str(APP_DIR) not in sys.path:
    sys.path.insert(0, str(APP_DIR))

from src.anomaly import (
    ANOMALY_COLUMN,
    BASELINE_COLUMN,
    CONSUMER_COLUMN,
    DETECTION_SOURCE_COLUMN,
    METER_DEFECT_ANOMALY_COLUMN,
    MODEL_ANOMALY_COLUMN,
    RATIO_COLUMN,
    REASON_COLUMN,
    TYPE_COLUMN,
    UNITS_COLUMN,
    Z_COLUMN,
    MeterAnomalyDetector,
)
from src.ledger import is_ledger_frame, load_ledgers
from src.tariff_engine import DEFAULT_POLICY, bill_breakdown

DATA_DIR = APP_DIR / "consumer_data" / "ledgers"
SAMPLE_FILE = APP_DIR / "consumer_data" / "sample_consumer_120_rows.xlsx"
MODEL_FILE = APP_DIR / "models" / "meter_anomaly_model.joblib"
LEDGER_PATTERNS = ("Consumer_Ledger_Report_*.xls", "Consumer_Ledger_Report_*.xlsx")


def _period(value) -> str:
    """Render a date, ``MMYYYY`` or the sample's ``MYY`` code."""
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return "-"
    if isinstance(value, pd.Timestamp):
        return value.strftime("%Y-%m")
    if isinstance(value, str) and len(value.strip()) == 6 and value.strip().isdigit():
        return f"{value.strip()[2:]}-{value.strip()[:2]}"
    try:
        code = int(value)
    except (TypeError, ValueError):
        return str(value)
    if code >= 100000:  # YYYYMM
        return f"{code // 100:04d}-{code % 100:02d}"
    month, year = code // 100, code % 100
    if 1 <= month <= 12 and year < 100:
        return f"{2000 + year:04d}-{month:02d}"
    return str(value)


def load_scan_data() -> tuple[pd.DataFrame, bool]:
    paths = sorted(
        path
        for pattern in LEDGER_PATTERNS
        for path in DATA_DIR.glob(pattern)
    )
    if paths:
        return load_ledgers(DATA_DIR), True
    if not SAMPLE_FILE.exists():
        raise FileNotFoundError(
            f"no ledger files or sample '{SAMPLE_FILE.name}' found in '{DATA_DIR}'"
        )
    return pd.read_excel(SAMPLE_FILE), False


def scan_dataset(detector: MeterAnomalyDetector, df: pd.DataFrame, policy) -> int:
    ledger_mode = is_ledger_frame(df)
    report = detector.detect(df)
    flagged = report[report[ANOMALY_COLUMN]]
    consumers = df[CONSUMER_COLUMN].nunique() if CONSUMER_COLUMN in df.columns else 1

    print(f"Scanned {len(report)} rows across {consumers} consumer(s).")
    if ledger_mode:
        defects = int(report[METER_DEFECT_ANOMALY_COLUMN].sum())
        model = int(report[MODEL_ANOMALY_COLUMN].sum())
        both = int(
            (report[METER_DEFECT_ANOMALY_COLUMN] & report[MODEL_ANOMALY_COLUMN]).sum()
        )
        print(
            f"Flagged {len(flagged)} row(s): {defects} meter defect(s), "
            f"{model} model anomaly/anomalies, {both} overlapping."
        )
    else:
        print(f"Flagged {len(flagged)} anomalous reading(s).")

    if flagged.empty:
        print("No billing exceptions discovered.")
        return 0

    for _, row in flagged.iterrows():
        period = _period(row.get("Date", row.get("Billing_Month")))
        baseline = float(row[BASELINE_COLUMN])
        print(f"\n  [{row[TYPE_COLUMN]}] {period} / {row.get(CONSUMER_COLUMN, '-')}")
        print(
            f"    reading   : {row[UNITS_COLUMN]:.1f} units "
            f"({row[RATIO_COLUMN]:.2f}x baseline, z={row[Z_COLUMN]:.1f})"
        )
        print(f"    reason    : {row[REASON_COLUMN]}")
        if ledger_mode:
            billed = row.get("CURRENT_TOTAL")
            expected = row.get("Expected_Current_Total_With_Ledger_Subsidy")
            source = row.get(DETECTION_SOURCE_COLUMN, "")
            print(f"    source    : {source}")
            if pd.notna(billed):
                print(f"    current   : Rs {float(billed):,.2f}")
            if pd.notna(expected):
                print(f"    tariff exp: Rs {float(expected):,.2f}")
        else:
            billed = bill_breakdown(row[UNITS_COLUMN], policy)
            corrected = bill_breakdown(baseline, policy)
            refund = round(billed["total"] - corrected["total"], 2)
            print(f"    billed    : Rs {billed['total']:.2f}")
            print(f"    corrected : Rs {corrected['total']:.2f} (baseline {baseline:.0f} units)")
            print(f"    refund due: Rs {refund:.2f}")
    return 0


def scan_readings(
    detector: MeterAnomalyDetector,
    readings: list[float],
    df: pd.DataFrame,
    policy,
) -> int:
    """Evaluate ad-hoc readings against the dataset's consumer baseline."""
    ledger_mode = is_ledger_frame(df)
    baseline = float(detector.baseline(df).median())
    consumer = str(df[CONSUMER_COLUMN].iloc[0]) if CONSUMER_COLUMN in df.columns else "<adhoc>"
    for reading in readings:
        frame = pd.DataFrame({UNITS_COLUMN: [reading], CONSUMER_COLUMN: [consumer]})
        row = detector.detect(frame, baseline_override=baseline).iloc[0]
        verdict = row[TYPE_COLUMN] if row[ANOMALY_COLUMN] else "OK"
        note = f" - {row[REASON_COLUMN]}" if row[ANOMALY_COLUMN] else ""
        if ledger_mode:
            print(
                f"  {reading:>10.1f} units -> {verdict:<12} "
                f"(baseline {baseline:.0f}){note}"
            )
        else:
            billed = bill_breakdown(reading, policy)
            print(
                f"  {reading:>10.1f} units -> {verdict:<6} "
                f"(baseline {baseline:.0f}, {billed['total']:.2f} Rs){note}"
            )
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Meter anomaly detection CLI")
    parser.add_argument("--readings", type=float, nargs="*", help="ad-hoc unit readings to evaluate")
    parser.add_argument(
        "--policy",
        default=DEFAULT_POLICY.name,
        help=f"tariff policy name (only '{DEFAULT_POLICY.name}' is available)",
    )
    args = parser.parse_args(argv)

    from src.tariff_engine import POLICIES

    policy = POLICIES.get(args.policy, DEFAULT_POLICY)

    if not MODEL_FILE.exists():
        print(f"Model artifact '{MODEL_FILE.name}' not found. Run build_model.py first.", file=sys.stderr)
        return 1
    detector = MeterAnomalyDetector.load(MODEL_FILE)
    print(f"Loaded detector from {MODEL_FILE.name} (policy: {policy.name})")

    try:
        df, ledger_mode = load_scan_data()
    except FileNotFoundError as exc:
        print(str(exc), file=sys.stderr)
        return 1

    print("\n--- Dataset scan ---")
    scan_dataset(detector, df, policy)

    if args.readings:
        print("\n--- Ad-hoc readings ---")
        scan_readings(detector, list(args.readings), df, policy)

    return 0


if __name__ == "__main__":
    sys.exit(main())
