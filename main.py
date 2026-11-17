"""Utility Billing AI Workspace - Streamlit application.

    streamlit run main.py

Layout
------
* Left  - file loader, editable consumer grid, export.
* Right - model status indicator and the Volta AI command console.

Every rupee figure comes from ``src/tariff_engine.py`` and every detection
comes from ``src/anomaly.py`` so the dashboard, the CLI and the tests can never
disagree.
"""

from __future__ import annotations

import html
import re
import sys
from pathlib import Path

APP_DIR = Path(__file__).resolve().parent
if str(APP_DIR) not in sys.path:
    sys.path.insert(0, str(APP_DIR))

import joblib
import numpy as np
import pandas as pd
import streamlit as st

from src.anomaly import (
    ANOMALY_COLUMN,
    BASELINE_COLUMN,
    CONSUMER_COLUMN,
    DETECTION_SOURCE_COLUMN,
    METER_DEFECT_ANOMALY_COLUMN,
    METER_DEFECT_COLUMN,
    MODEL_ANOMALY_COLUMN,
    RATIO_COLUMN,
    REASON_COLUMN,
    TAG_COLUMN,
    TYPE_COLUMN,
    UNITS_COLUMN,
    Z_COLUMN,
    MeterAnomalyDetector,
)
from src.ledger import (
    ACCOUNT_NUMBER_COLUMN,
    CATEGORY_COLUMN,
    CURRENT_TOTAL_COLUMN,
    DATE_COLUMN,
    FISCAL_YEAR_COLUMN,
    SOURCE_FILE_COLUMN,
    SUBSIDY_COLUMN,
    consumer_label,
    find_consumers,
    is_ledger_frame,
    ledger_summary,
    ledger_warnings,
    load_ledger_bytes,
    load_ledgers,
    normalize_ledger,
    suggest_consumers,
)
from src.tariff_engine import (
    BILL_COLUMN,
    DEFAULT_POLICY,
    FIXED_COLUMN,
    TAX_COLUMN,
    TariffPolicy,
    apply_billing,
    bill_breakdown,
    reconcile_bills,
    reconcile_ledger,
    reprice_ledger,
)

MODEL_FILE = APP_DIR / "models" / "meter_anomaly_model.joblib"
DATA_DIR = APP_DIR / "consumer_data" / "ledgers"

GREETING = (
    "I am Volta AI, your Utility Billing assistant. Enter a consumer ID (`CON_ID` or `ACT_NO`) "
    "to inspect a profile, then ask `bill anomaly` for month-wise exceptions. Other "
    "commands: `scan anomalies`, `reconcile bills`, `apply corrections`, `show summary`."
)


# --------------------------------------------------------------------- engine
@st.cache_resource(show_spinner=False)
def _load_detector(model_mtime: float) -> MeterAnomalyDetector | None:
    """Load the compiled detector. Cached until the artifact changes on disk."""
    try:
        detector = joblib.load(MODEL_FILE)
    except Exception:
        return None
    return detector if isinstance(detector, MeterAnomalyDetector) else None


def load_detector() -> tuple[MeterAnomalyDetector | None, str]:
    """Return ``(detector, status)`` where status is one of ok/missing/invalid."""
    if not MODEL_FILE.exists():
        return None, "missing"
    try:
        detector = _load_detector(MODEL_FILE.stat().st_mtime)
    except Exception:
        return None, "invalid"
    if detector is None:
        return None, "invalid"
    return detector, "ok"


def status_banner(status: str) -> str:
    if status == "ok":
        return (
            '<div style="background-color:#d4edda;color:#155724;padding:12px;border-radius:5px;'
            'border-left:5px solid #28a745;margin-bottom:15px;font-weight:bold;display:flex;'
            'align-items:center;font-family:sans-serif;">'
            '<span style="color:#28a745;font-size:22px;margin-right:10px;line-height:1;">●</span>'
            "AI ENGINE ONLINE: detector loaded ('meter_anomaly_model.joblib')"
            "</div>"
        )
    if status == "missing":
        message = "AI ENGINE OFFLINE: 'meter_anomaly_model.joblib' is missing. Run build_model.py!"
    else:
        message = (
            "AI ENGINE ERROR: 'meter_anomaly_model.joblib' could not be initialised. "
            "Re-run build_model.py to regenerate the artifact."
        )
    return (
        '<div style="background-color:#f8d7da;color:#721c24;padding:12px;border-radius:5px;'
        'border-left:5px solid #dc3545;margin-bottom:15px;font-weight:bold;display:flex;'
        'align-items:center;font-family:sans-serif;">'
        '<span style="color:#dc3545;font-size:22px;margin-right:10px;line-height:1;">●</span>'
        f"{html.escape(message)}"
        "</div>"
    )


# ------------------------------------------------------------------ data prep
def prepare_frame(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    """Normalise an uploaded frame and report any coercion warnings."""
    warnings: list[str] = []
    out = df.copy()
    out.columns = [str(c).strip() for c in out.columns]

    if UNITS_COLUMN in out.columns:
        numeric = pd.to_numeric(out[UNITS_COLUMN], errors="coerce")
        broken = int(numeric.isna().sum())
        if broken:
            warnings.append(f"{broken} row(s) had a non-numeric {UNITS_COLUMN} and were treated as 0")
        out[UNITS_COLUMN] = numeric.fillna(0.0).astype(float)
    for column in (BILL_COLUMN, TAX_COLUMN, FIXED_COLUMN):
        if column in out.columns:
            out[column] = pd.to_numeric(out[column], errors="coerce")
    if CONSUMER_COLUMN in out.columns:
        out[CONSUMER_COLUMN] = out[CONSUMER_COLUMN].astype(str)
    return out, warnings


def format_period(value) -> str:
    """Render the dataset's ``MYY`` period code as ``YYYY-MM``."""
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return "-"
    if isinstance(value, pd.Timestamp):
        return value.strftime("%Y-%m")
    if isinstance(value, str):
        text = value.strip()
        if len(text) == 6 and text.isdigit():
            return f"{text[2:]}-{text[:2]}"
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


def markdown_table(frame: pd.DataFrame, columns: list[tuple[str, str]]) -> str:
    """Render a small dataframe as a markdown table (no tabulate dependency)."""
    headers = [label for _, label in columns]
    lines = ["| " + " | ".join(headers) + " |", "| " + " | ".join("---" for _ in headers) + " |"]
    for _, row in frame.iterrows():
        cells = []
        for key, _ in columns:
            value = row.get(key, "")
            if isinstance(value, float):
                cells.append(f"{value:,.2f}")
            else:
                cells.append(str(value))
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def _text(value, default: str = "-") -> str:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return default
    if isinstance(value, float) and value.is_integer():
        return str(int(value))
    text = str(value).strip()
    return text or default


def _money(value) -> str:
    number = pd.to_numeric(pd.Series([value]), errors="coerce").iloc[0]
    return "-" if pd.isna(number) else f"{float(number):,.2f}"


def _unique_text(frame: pd.DataFrame, column: str, limit: int = 3) -> str:
    if column not in frame.columns:
        return "-"
    values = [v for v in (_text(value, "") for value in frame[column]) if v]
    unique = list(dict.fromkeys(values))
    if not unique:
        return "-"
    shown = ", ".join(unique[:limit])
    return shown if len(unique) <= limit else f"{shown} (+{len(unique) - limit} more)"


def _consumer_error(df: pd.DataFrame, identifier: str) -> str:
    query = _text(identifier, "")
    suggestions = suggest_consumers(df, query)
    message = (
        f"Consumer **{query or '<blank>'}** was not found. Use the numeric "
        "`Consumer_ID` or the `Account_No` shown in the workspace grid."
    )
    if suggestions:
        message += "\n\nDid you mean: " + ", ".join(f"`{item}`" for item in suggestions) + "?"
    return message


def _resolve_consumer(
    df: pd.DataFrame, identifier: str
) -> tuple[pd.DataFrame | None, str, str | None, str | None]:
    matches = find_consumers(df, identifier)
    if matches.empty:
        return None, "", None, _consumer_error(df, identifier)

    consumer_id = (
        _text(matches[CONSUMER_COLUMN].iloc[0]) if CONSUMER_COLUMN in matches.columns else "-"
    )
    account_no = (
        _text(matches[ACCOUNT_NUMBER_COLUMN].iloc[0])
        if ACCOUNT_NUMBER_COLUMN in matches.columns
        else "-"
    )
    label = consumer_label(consumer_id, "" if account_no == "-" else account_no)
    if consumer_id == "-":
        return None, "", None, _consumer_error(df, identifier)
    return matches, label, consumer_id, None


def command_consumer(
    df: pd.DataFrame,
    identifier: str,
    detector: MeterAnomalyDetector | None,
    policy: TariffPolicy,
    z_threshold: float,
) -> tuple[str, str | None]:
    """Render a consumer profile and remember the selected consumer."""
    rows, label, consumer_id, error = _resolve_consumer(df, identifier)
    if error or rows is None:
        return error or "Consumer lookup failed.", None

    ledger_mode = is_ledger_frame(df)
    months = rows["Billing_Month"].nunique() if "Billing_Month" in rows.columns else len(rows)
    fiscal_years = (
        sorted(str(year) for year in rows[FISCAL_YEAR_COLUMN].dropna().unique())
        if FISCAL_YEAR_COLUMN in rows.columns
        else []
    )

    lines = [
        f"**Volta AI:** consumer profile for **{label}**.",
        "",
        (
            f"- Consumer ID: **{_text(rows[CONSUMER_COLUMN].iloc[0])}** | "
            f"Account No: **{_text(rows[ACCOUNT_NUMBER_COLUMN].iloc[0]) if ACCOUNT_NUMBER_COLUMN in rows else '-'}**"
        ),
        f"- Rows: **{len(rows):,}** | Months: **{months:,}** | Files: **{_unique_text(rows, SOURCE_FILE_COLUMN)}**",
    ]

    profile_fields = (
        ("CNAME", "Name"),
        ("CADRESS", "Address"),
        ("TELE_NO1", "Phone"),
        ("TARIFF_ID", "Tariff"),
        (CATEGORY_COLUMN, "Category"),
        ("SANC_LOAD", "Sanctioned load"),
        ("PHASE", "Phase"),
        ("CONNECTION_TYPE", "Connection"),
        ("METER_NO", "Meter no"),
    )
    for column, title in profile_fields:
        if column in rows.columns:
            value = _unique_text(rows, column, limit=1)
            if value != "-":
                lines.append(f"- {title}: **{value}**")

    if DATE_COLUMN in rows.columns:
        dates = pd.to_datetime(rows[DATE_COLUMN], errors="coerce").dropna()
        if len(dates):
            lines.append(
                f"- Billing period: **{dates.min():%Y-%m}** to **{dates.max():%Y-%m}**"
            )
    if fiscal_years:
        lines.append(f"- Fiscal years: **{', '.join(fiscal_years)}**")

    units = pd.to_numeric(
        rows[UNITS_COLUMN] if UNITS_COLUMN in rows.columns else pd.Series(dtype=float),
        errors="coerce",
    )
    billed_units = pd.to_numeric(
        rows["UNITS_BILLED"] if "UNITS_BILLED" in rows.columns else pd.Series(dtype=float),
        errors="coerce",
    )
    lines.append(
        f"- Monthly units: average **{units.mean():,.1f}**, median **{units.median():,.1f}** | "
        f"billed total **{billed_units.sum():,.1f}**"
    )

    if ledger_mode:
        current_total = pd.to_numeric(
            rows[CURRENT_TOTAL_COLUMN] if CURRENT_TOTAL_COLUMN in rows.columns else pd.Series(dtype=float),
            errors="coerce",
        ).sum()
        subsidy = pd.to_numeric(
            rows[SUBSIDY_COLUMN] if SUBSIDY_COLUMN in rows.columns else pd.Series(dtype=float),
            errors="coerce",
        ).sum()
        reconciled = reconcile_ledger(rows)
        report = detector.detect(reconciled, z_threshold=z_threshold) if detector else None
        statuses = reconciled["Tariff_Status"].value_counts().to_dict()
        lines.extend(
            [
                f"- Current billed: **Rs {current_total:,.2f}** | subsidy **Rs {abs(subsidy):,.2f}**",
                (
                    f"- Tariff status: **OK {int(statuses.get('OK', 0))}** | "
                    f"**UNVERIFIED {int(statuses.get('UNVERIFIED', 0))}** | "
                    f"identity **{int(reconciled['Current_Identity_Matches'].sum())}/{len(reconciled)}**"
                ),
            ]
        )
        if report is not None:
            lines.append(
                f"- Exceptions: **{int(report[ANOMALY_COLUMN].sum())}** total | "
                f"meter defects **{int(report[METER_DEFECT_ANOMALY_COLUMN].sum())}** | "
                f"model anomalies **{int(report[MODEL_ANOMALY_COLUMN].sum())}**"
            )
        else:
            lines.append("- Exceptions: detector offline; run `build_model.py`")
    else:
        total_billed = pd.to_numeric(
            rows[BILL_COLUMN] if BILL_COLUMN in rows.columns else pd.Series(dtype=float),
            errors="coerce",
        ).sum()
        total_tax = pd.to_numeric(
            rows[TAX_COLUMN] if TAX_COLUMN in rows.columns else pd.Series(dtype=float),
            errors="coerce",
        ).sum()
        report = detector.detect(rows, z_threshold=z_threshold) if detector else None
        lines.append(
            f"- Billed amount: **Rs {total_billed:,.2f}** (tax Rs {total_tax:,.2f})"
        )
        if report is not None:
            lines.append(f"- Anomalous readings: **{int(report[ANOMALY_COLUMN].sum())}**")
        else:
            lines.append("- Anomalous readings: detector offline; run `build_model.py`")

    lines.extend(
        [
            "",
            f"Run **bill anomaly** to list every flagged month for **{label}**.",
        ]
    )
    return "\n".join(lines), consumer_id


def command_bill_anomaly(
    df: pd.DataFrame,
    identifier: str,
    detector: MeterAnomalyDetector | None,
    policy: TariffPolicy,
    z_threshold: float,
) -> tuple[str, str | None]:
    """Show month-wise anomaly details for one consumer."""
    rows, label, consumer_id, error = _resolve_consumer(df, identifier)
    if error or rows is None:
        return error or "Consumer lookup failed.", None
    if detector is None:
        return "AI Engine Error: the detector artifact is offline. Run build_model.py.", consumer_id

    ledger_mode = is_ledger_frame(df)
    enriched = reconcile_ledger(rows) if ledger_mode else rows.copy()
    report = detector.detect(enriched, z_threshold=z_threshold)
    flagged = report[report[ANOMALY_COLUMN]].copy()
    if DATE_COLUMN in flagged.columns:
        flagged = flagged.sort_values(DATE_COLUMN, kind="stable")

    header = (
        f"**Volta AI:** month-wise billing anomalies for **{label}** "
        f"({len(rows)} month(s)).\n\n"
    )
    if flagged.empty:
        return header + "No billing anomalies found for this consumer.", consumer_id

    defects = int(flagged[METER_DEFECT_ANOMALY_COLUMN].sum())
    model_flags = int(flagged[MODEL_ANOMALY_COLUMN].sum())
    both = int((flagged[METER_DEFECT_ANOMALY_COLUMN] & flagged[MODEL_ANOMALY_COLUMN]).sum())
    summary = (
        f"Flagged **{len(flagged)}** month(s): **{defects}** meter defect(s), "
        f"**{model_flags}** model anomaly/anomalies, **{both}** overlapping.\n\n"
    )

    table_rows = []
    for _, row in flagged.iterrows():
        period = format_period(row.get(DATE_COLUMN, row.get("Billing_Month")))
        if ledger_mode:
            current = row.get(CURRENT_TOTAL_COLUMN)
            expected = row.get("Expected_Current_Total_With_Ledger_Subsidy")
            delta = (
                float(current) - float(expected)
                if pd.notna(current) and pd.notna(expected)
                else None
            )
            table_rows.append(
                {
                    "Period": period,
                    "Units": f"{float(row[UNITS_COLUMN]):,.1f}",
                    "Baseline": f"{float(row[BASELINE_COLUMN]):,.1f}",
                    "Type": row[TYPE_COLUMN],
                    "Source": row.get(DETECTION_SOURCE_COLUMN, ""),
                    "Current": _money(current),
                    "Expected": _money(expected),
                    "Delta": _money(delta) if delta is not None else "-",
                    "Reason": row[REASON_COLUMN],
                }
            )
        else:
            billed = bill_breakdown(row[UNITS_COLUMN], policy)
            corrected = bill_breakdown(row[BASELINE_COLUMN], policy)
            table_rows.append(
                {
                    "Period": period,
                    "Units": f"{float(row[UNITS_COLUMN]):,.0f}",
                    "Baseline": f"{float(row[BASELINE_COLUMN]):,.0f}",
                    "Type": row[TYPE_COLUMN],
                    "Source": row.get(DETECTION_SOURCE_COLUMN, ""),
                    "Current": f"{billed['total']:,.2f}",
                    "Expected": f"{corrected['total']:,.2f}",
                    "Delta": f"{billed['total'] - corrected['total']:,.2f}",
                    "Reason": row[REASON_COLUMN],
                }
            )

    detail = markdown_table(
        pd.DataFrame(table_rows),
        [
            ("Period", "Period"),
            ("Units", "Monthly units"),
            ("Baseline", "Baseline"),
            ("Type", "Type"),
            ("Source", "Source"),
            ("Current", "Current bill (Rs)"),
            ("Expected", "Tariff expected (Rs)"),
            ("Delta", "Delta (Rs)"),
            ("Reason", "Diagnostic"),
        ],
    )
    note = (
        "\n\nMeter defects need field verification; model anomalies are consumption "
        "outliers. Delta is the current bill minus the tariff-expected bill."
        if ledger_mode
        else "\n\nDelta is the current bill minus the corrected baseline bill."
    )
    return header + summary + detail + note, consumer_id


# ------------------------------------------------------------------- commands
def command_scan(df: pd.DataFrame, detector: MeterAnomalyDetector, policy: TariffPolicy, z_threshold: float) -> str:
    report = detector.detect(df, z_threshold=z_threshold)
    flagged = report[report[ANOMALY_COLUMN]]
    consumers = df[CONSUMER_COLUMN].nunique() if CONSUMER_COLUMN in df.columns else 1
    ledger_mode = is_ledger_frame(df)

    if ledger_mode:
        basis = "with meter-defect rules, the consumption model and the FY tariff schedule"
    else:
        basis = f"using the '{policy.name}' tariff"
    header = (
        f"**Volta AI:** scanned **{len(report)}** rows across **{consumers}** consumer(s) "
        f"{basis}.\n\n"
    )
    if flagged.empty:
        return header + "No billing exceptions discovered - every reading sits within its consumer's expected band."

    if ledger_mode:
        defects = int(report[METER_DEFECT_ANOMALY_COLUMN].sum())
        model_flags = int(report[MODEL_ANOMALY_COLUMN].sum())
        both = int(
            (report[METER_DEFECT_ANOMALY_COLUMN] & report[MODEL_ANOMALY_COLUMN]).sum()
        )
        rows = []
        for _, r in flagged.head(25).iterrows():
            rows.append(
                {
                    "Period": format_period(r.get("Date", r.get("Billing_Month"))),
                    "Consumer": r.get(CONSUMER_COLUMN, "-"),
                    "Category": r.get("Category", r.get("TARIFF_ID", "-")),
                    "Units": f"{r[UNITS_COLUMN]:,.1f}",
                    "Baseline": f"{r[BASELINE_COLUMN]:,.1f}",
                    "Type": r[TYPE_COLUMN],
                    "Source": r.get(DETECTION_SOURCE_COLUMN, ""),
                    "Current": f"{float(r.get(CURRENT_TOTAL_COLUMN, 0.0)):,.2f}",
                }
            )
        detail = markdown_table(
            pd.DataFrame(rows),
            [
                ("Period", "Period"),
                ("Consumer", "Consumer"),
                ("Category", "Category"),
                ("Units", "Monthly units"),
                ("Baseline", "Baseline"),
                ("Type", "Type"),
                ("Source", "Source"),
                ("Current", "Current bill (Rs)"),
            ],
        )
        reasons = "\n".join(
            f"- **{format_period(r.get('Date', r.get('Billing_Month')))}**: {r[REASON_COLUMN]}"
            for _, r in flagged.head(10).iterrows()
        )
        tail = "" if len(flagged) <= 25 else f"\n\n...and {len(flagged) - 25} more row(s)."
        return (
            header
            + f"Flagged **{len(flagged)}** billing exception(s): **{defects}** meter defect(s), "
            f"**{model_flags}** model anomaly/anomalies, **{both}** overlapping.\n\n"
            + detail
            + tail
            + "\n\nMeter defects need field verification; model anomalies are consumption outliers.\n\n"
            + "Diagnostics:\n"
            + reasons
        )

    rows = []
    for _, r in flagged.iterrows():
        billed = bill_breakdown(r[UNITS_COLUMN], policy)
        corrected = bill_breakdown(r[BASELINE_COLUMN], policy)
        rows.append(
            {
                "Period": format_period(r.get("Date")),
                "Consumer": r.get(CONSUMER_COLUMN, "-"),
                "Units": f"{r[UNITS_COLUMN]:,.0f}",
                "Baseline": f"{r[BASELINE_COLUMN]:,.0f}",
                "Type": r[TYPE_COLUMN],
                "Billed": f"{billed['total']:,.2f}",
                "Corrected": f"{corrected['total']:,.2f}",
                "Refund": f"{billed['total'] - corrected['total']:,.2f}",
            }
        )
    detail = markdown_table(
        pd.DataFrame(rows),
        [
            ("Period", "Period"),
            ("Consumer", "Consumer"),
            ("Units", "Units"),
            ("Baseline", "Baseline"),
            ("Type", "Type"),
            ("Billed", "Billed (Rs)"),
            ("Corrected", "Corrected (Rs)"),
            ("Refund", "Refund (Rs)"),
        ],
    )
    reasons = "\n".join(f"- **{format_period(r.get('Date'))}**: {r[REASON_COLUMN]}" for _, r in flagged.iterrows())
    return (
        header
        + f"Found **{len(flagged)}** anomalous meter reading(s):\n\n"
        + detail
        + "\n\nDiagnostics:\n"
        + reasons
    )


def command_reconcile(df: pd.DataFrame, policy: TariffPolicy) -> str:
    if is_ledger_frame(df):
        reconciled = reconcile_ledger(df)
        total = len(reconciled)
        identity_ok = int(reconciled["Current_Identity_Matches"].sum())
        strict_ok = int(reconciled["Strict_Tariff_Matches"].sum())
        reconcilable = int(reconciled["Reconcilable"].sum())
        header = (
            "**Volta AI:** ledger reconciliation against the FY SBPDCL/BERC "
            "schedule and the monthly current-total identity.\n\n"
        )
        if identity_ok == total and strict_ok == reconcilable:
            return (
                header
                + f"All **{total}** monthly current totals and **{reconcilable}** "
                "reconcilable tariff rows match."
            )

        bad = reconciled[
            (~reconciled["Current_Identity_Matches"])
            | (
                reconciled["Reconcilable"]
                & ~reconciled["Strict_Tariff_Matches"]
            )
        ].copy()
        rows = []
        for _, r in bad.head(15).iterrows():
            rows.append(
                {
                    "Period": format_period(r.get("Date", r.get("Billing_Month"))),
                    "Consumer": r.get(CONSUMER_COLUMN, "-"),
                    "Units": f"{r.get(UNITS_COLUMN, 0.0):,.1f}",
                    "Current": f"{r.get(CURRENT_TOTAL_COLUMN, 0.0):,.2f}",
                    "Identity": f"{r.get('Current_Identity_Residual', 0.0):,.2f}",
                    "EC delta": f"{r.get('EC_Delta', 0.0):,.2f}",
                    "Subsidy delta": f"{r.get('Subsidy_Delta', 0.0):,.2f}",
                }
            )
        table = markdown_table(
            pd.DataFrame(rows),
            [
                ("Period", "Period"),
                ("Consumer", "Consumer"),
                ("Units", "Units"),
                ("Current", "Current (Rs)"),
                ("Identity", "Identity residual"),
                ("EC delta", "EC delta"),
                ("Subsidy delta", "Subsidy delta"),
            ],
        )
        tail = "" if len(bad) <= 15 else f"\n\n...and {len(bad) - 15} more row(s)."
        return (
            header
            + f"Monthly identity: **{identity_ok}/{total}** rows exact. "
            f"Strict tariff: **{strict_ok}/{reconcilable}** reconcilable rows. "
            f"Ledger-subsidy current total: "
            f"**{int(reconciled['Current_Total_With_Ledger_Subsidy_Matches'].sum())}/{total}** rows.\n\n"
            + table
            + tail
        )

    reconciled = reconcile_bills(df, policy)
    matches = int(reconciled["Bill_Matches"].sum())
    total = len(reconciled)
    header = (
        f"**Volta AI:** tariff reconciliation against '{policy.name}' "
        f"({policy.description}).\n\n"
    )
    if matches == total:
        return header + f"All **{total}** rows reconcile with the tariff engine (max delta Rs 0.00)."

    mismatched = reconciled[~reconciled["Bill_Matches"]].copy()
    rows = []
    for _, r in mismatched.head(15).iterrows():
        rows.append(
            {
                "Period": format_period(r.get("Date")),
                "Consumer": r.get(CONSUMER_COLUMN, "-"),
                "Units": f"{r[UNITS_COLUMN]:,.0f}",
                "Billed": f"{r.get(BILL_COLUMN, float('nan')):,.2f}",
                "Expected": f"{r['Expected_Bill']:,.2f}",
                "Delta": f"{r['Bill_Delta']:,.2f}",
            }
        )
    table = markdown_table(
        pd.DataFrame(rows),
        [
            ("Period", "Period"),
            ("Consumer", "Consumer"),
            ("Units", "Units"),
            ("Billed", "Billed (Rs)"),
            ("Expected", "Expected (Rs)"),
            ("Delta", "Delta (Rs)"),
        ],
    )
    tail = "" if len(mismatched) <= 15 else f"\n\n...and {len(mismatched) - 15} more row(s)."
    return (
        header
        + f"**{total - matches}** of **{total}** rows do not match the engine:\n\n"
        + table
        + tail
    )


def command_summary(df: pd.DataFrame, detector: MeterAnomalyDetector, policy: TariffPolicy, z_threshold: float) -> str:
    units = pd.to_numeric(df[UNITS_COLUMN], errors="coerce").fillna(0.0)
    report = detector.detect(df, z_threshold=z_threshold)
    flagged = int(report[ANOMALY_COLUMN].sum())
    consumers = df[CONSUMER_COLUMN].nunique() if CONSUMER_COLUMN in df.columns else 1

    if is_ledger_frame(df):
        summary = ledger_summary(df)
        defects = int(report[METER_DEFECT_ANOMALY_COLUMN].sum())
        model_flags = int(report[MODEL_ANOMALY_COLUMN].sum())
        current_total = pd.to_numeric(
            df.get(CURRENT_TOTAL_COLUMN, pd.Series(dtype=float)), errors="coerce"
        ).sum()
        subsidy = pd.to_numeric(
            df.get("GOV_SUBSIDY", pd.Series(dtype=float)), errors="coerce"
        ).sum()
        identity_ok = 0
        reconciled = reconcile_ledger(df)
        if "Current_Identity_Matches" in reconciled.columns:
            identity_ok = int(reconciled["Current_Identity_Matches"].sum())
        return "\n".join(
            [
                "**Volta AI:** ledger workspace summary",
                "",
                f"- Rows: **{len(df)}** | Consumers: **{consumers}** | Files: **{summary['files']}**",
                f"- FYs: **{', '.join(summary['fiscal_years']) or '-'}**",
                f"- Monthly units: min **{units.min():,.1f}**, median **{units.median():,.1f}**, max **{units.max():,.1f}**",
                f"- Current billed: **Rs {current_total:,.2f}** (subsidy Rs {abs(subsidy):,.2f})",
                f"- Meter defects: **{defects}** | Model anomalies: **{model_flags}** | Total flags: **{flagged}**",
                f"- Current-total identity: **{identity_ok}/{len(df)}** rows exact",
            ]
        )

    total_billed = pd.to_numeric(df.get(BILL_COLUMN, pd.Series(dtype=float)), errors="coerce").sum()
    total_tax = pd.to_numeric(df.get(TAX_COLUMN, pd.Series(dtype=float)), errors="coerce").sum()

    lines = [
        "**Volta AI:** workspace summary",
        "",
        f"- Rows: **{len(df)}** | Consumers: **{consumers}** | Tariff: **{policy.name}**",
        f"- Units consumed: min **{units.min():,.0f}**, median **{units.median():,.0f}**, max **{units.max():,.0f}**, total **{units.sum():,.0f}**",
        f"- Billed amount: **Rs {total_billed:,.2f}** (tax Rs {total_tax:,.2f})",
        f"- Anomalous readings: **{flagged}**",
    ]
    return "\n".join(lines)


def command_apply(
    df: pd.DataFrame,
    detector: MeterAnomalyDetector,
    policy: TariffPolicy,
    z_threshold: float,
) -> tuple[str, pd.DataFrame | None]:
    """Rewrite anomalous rows to their consumer baseline and reprice them."""
    report = detector.detect(df, z_threshold=z_threshold)
    ledger_mode = is_ledger_frame(df)

    if ledger_mode:
        mask = report[MODEL_ANOMALY_COLUMN]
        defects = int(report[METER_DEFECT_ANOMALY_COLUMN].sum())
        if not mask.any():
            return (
                "**Volta AI:** no model consumption anomalies found. "
                f"The **{defects}** meter-defect row(s) were left unchanged for field verification.",
                None,
            )

        corrected = df.copy()
        for column in (UNITS_COLUMN, "Monthly_Units", "Billed_Units", "UNITS_BILLED", "UNITS_CONS"):
            if column in corrected.columns:
                corrected[column] = pd.to_numeric(
                    corrected[column], errors="coerce"
                ).astype(float)
        baseline = report.loc[mask, BASELINE_COLUMN].astype(float)
        days = pd.to_numeric(
            corrected.get("Billing_Days", pd.Series(30.0, index=corrected.index)),
            errors="coerce",
        ).fillna(30.0)
        period_units = (baseline * days / 30.0).round(2)
        corrected.loc[mask, UNITS_COLUMN] = baseline
        corrected.loc[mask, "Monthly_Units"] = baseline
        corrected.loc[mask, "Billed_Units"] = period_units
        corrected.loc[mask, "UNITS_BILLED"] = period_units
        corrected.loc[mask, "UNITS_CONS"] = baseline
        corrected = reprice_ledger(corrected, mask)

        before = pd.to_numeric(df.loc[mask, CURRENT_TOTAL_COLUMN], errors="coerce").sum()
        after = pd.to_numeric(corrected.loc[mask, CURRENT_TOTAL_COLUMN], errors="coerce").sum()
        change = after - before
        detail = "\n".join(
            f"- **{format_period(r.get('Date', r.get('Billing_Month')))}**: "
            f"{r[UNITS_COLUMN]:,.1f} -> {r[BASELINE_COLUMN]:,.1f} monthly units "
            f"({r[TYPE_COLUMN]}, {r[REASON_COLUMN]})"
            for _, r in report.loc[mask].iterrows()
        )
        message = (
            f"**Volta AI:** rewrote **{int(mask.sum())}** model anomaly/anomalies "
            "to their consumer baseline and repriced only those rows with the FY tariff engine. "
            f"**{defects}** meter-defect row(s) remain unchanged.\n\n"
            + detail
            + f"\n\nCurrent-bill adjustment on corrected rows: **Rs {change:,.2f}**."
        )
        return message, corrected

    flagged = report[report[ANOMALY_COLUMN]]
    if flagged.empty:
        return (
            "**Volta AI:** no anomalous readings found, so the buffer was left untouched.",
            None,
        )

    corrected = df.copy()
    mask = report[ANOMALY_COLUMN]
    corrected.loc[mask, UNITS_COLUMN] = report.loc[mask, BASELINE_COLUMN].to_numpy(dtype=float)

    # Only reprice the rows we actually touched; every other row keeps its bill.
    corrected = apply_billing(corrected, policy, mask=mask)
    if "Meter_Status" in corrected.columns:
        corrected.loc[mask, "Meter_Status"] = "CORRECTED"

    savings = 0.0
    for _, r in flagged.iterrows():
        before = bill_breakdown(r[UNITS_COLUMN], policy)
        after = bill_breakdown(r[BASELINE_COLUMN], policy)
        savings += before["total"] - after["total"]

    detail = "\n".join(
        f"- **{format_period(r.get('Date'))}**: {r[UNITS_COLUMN]:,.0f} -> {r[BASELINE_COLUMN]:,.0f} units "
        f"({r[TYPE_COLUMN]}, {r[REASON_COLUMN]})"
        for _, r in flagged.iterrows()
    )
    message = (
        f"**Volta AI:** rewrote **{len(flagged)}** anomalous row(s) to their "
        f"consumer baseline and repriced them under '{policy.name}'.\n\n"
        + detail
        + f"\n\nTotal over-billing recovered: **Rs {savings:,.2f}**."
    )
    return message, corrected


def handle_command(
    prompt: str,
    df: pd.DataFrame | None,
    detector: MeterAnomalyDetector | None,
    policy: TariffPolicy,
    z_threshold: float,
    selected_consumer: str | None = None,
) -> tuple[str, pd.DataFrame | None, str | None]:
    """Route a chat command. Returns ``(response, updated_frame, selected_consumer)``."""
    text = prompt.strip()
    lowered = text.lower()
    selected = selected_consumer

    if df is None or df.empty:
        return (
            "Target context frame empty: upload a consumer profile on the left canvas first.",
            None,
            selected,
        )

    if UNITS_COLUMN not in df.columns:
        return (
            f"Column missing error: the dataset must expose a `{UNITS_COLUMN}` feature field "
            f"(found: {', '.join(map(str, df.columns)) or 'no columns'}).",
            None,
            selected,
        )

    anomaly_match = re.match(
        r"^(?:bill(?:ing)?\s+anomal(?:y|ies)|anomal(?:y|ies)\s+bill)"
        r"(?:\s+(?:for|of|consumer))?\s*[:#-]?\s*(.*)$",
        text,
        re.IGNORECASE,
    )
    if anomaly_match:
        target = anomaly_match.group(1).strip() or selected
        if not target:
            return (
                "Select a consumer first: enter a `Consumer_ID` or `Account_No`, "
                "or run `bill anomaly <id>`.",
                None,
                selected,
            )
        message, chosen = command_bill_anomaly(
            df, target, detector, policy, z_threshold
        )
        return message, None, chosen or selected

    consumer_match = re.match(
        r"^(?:consumer(?:\s+id)?|details?|show\s+consumer|account(?:\s+no)?)"
        r"\s*[:#-]?\s*(.+)$",
        text,
        re.IGNORECASE,
    )
    query = consumer_match.group(1).strip() if consumer_match else None
    if query is None and len(text) >= 3 and not find_consumers(df, text).empty:
        query = text
    if query:
        message, chosen = command_consumer(
            df, query, detector, policy, z_threshold
        )
        return message, None, chosen or selected

    if any(word in lowered for word in ("reconcile", "verify", "bill check", "tariff check")):
        return command_reconcile(df, policy), None, selected

    if any(word in lowered for word in ("scan", "anomaly", "anomalies", "detect", "spike")):
        if detector is None:
            return (
                "AI Engine Error: the detector artifact is offline (red status flag). Run build_model.py.",
                None,
                selected,
            )
        return command_scan(df, detector, policy, z_threshold), None, selected

    if any(word in lowered for word in ("correct", "apply", "fix", "repair")):
        if detector is None:
            return (
                "Workspace operational failure: the green engine indicator must be active to run corrections.",
                None,
                selected,
            )
        message, updated = command_apply(df, detector, policy, z_threshold)
        return message, updated, selected

    if any(word in lowered for word in ("summary", "overview", "report", "stats")):
        if detector is None:
            return "AI Engine Error: the detector artifact is offline (red status flag).", None, selected
        return command_summary(df, detector, policy, z_threshold), None, selected

    if any(word in lowered for word in ("bill", "tariff", "price", "pricing")):
        return command_reconcile(df, policy), None, selected

    return (
        "I can read your configuration file variables. Enter a consumer ID, or try "
        "`bill anomaly`, `scan anomalies`, `reconcile bills`, `apply corrections` or `show summary`.",
        None,
        selected,
    )


# ----------------------------------------------------------------- app shell
def load_dataset_corpus() -> pd.DataFrame | None:
    """Load every dataset ledger, or return ``None`` when the corpus is absent."""
    try:
        return load_ledgers(DATA_DIR)
    except FileNotFoundError:
        return None


def init_state() -> None:
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = [{"role": "assistant", "content": GREETING}]
    st.session_state.setdefault("current_data", None)
    st.session_state.setdefault("edited_data", None)
    st.session_state.setdefault("active_file", None)
    st.session_state.setdefault("grid_version", 0)
    st.session_state.setdefault("selected_consumer", None)


def main() -> None:
    st.set_page_config(page_title="Utility Billing AI", page_icon="💻", layout="wide")
    st.title("💻 Utility Billing AI Workspace")
    st.caption("An interactive IDE and Volta AI for processing billing anomalies and enforcing tariff configurations.")

    init_state()
    detector, engine_status = load_detector()

    if st.session_state.current_data is None:
        with st.spinner("Loading the consumer_data ledger corpus..."):
            corpus = load_dataset_corpus()
        if corpus is not None and not corpus.empty:
            st.session_state.current_data = corpus
            st.session_state.edited_data = corpus
            st.session_state.active_file = (
                f"consumer_data ledger corpus ({corpus[SOURCE_FILE_COLUMN].nunique()} files)"
            )
            st.session_state.grid_version += 1
            for warning in ledger_warnings(corpus):
                st.warning(warning)

    # ---------------------------------------------------------------- sidebar
    with st.sidebar:
        st.subheader("⚙️ Configuration")
        policy = DEFAULT_POLICY
        st.caption(f"Tariff policy: **{policy.name}** — {policy.description}")

        z_threshold = st.slider(
            "Anomaly sensitivity (robust z)",
            min_value=2.0,
            max_value=6.0,
            value=float(detector.config.z_threshold) if detector else 3.5,
            step=0.5,
            help="Lower values catch milder deviations; higher values only flag extreme readings.",
        )
        if detector is not None:
            st.caption(
                f"Detector trained on {detector.training_summary.get('rows', 0)} rows / "
                f"{detector.training_summary.get('consumers', 0)} consumer(s)"
            )

    left_col = st.container()

    # ------------------------------------------------------------- left canvas
    with left_col:
        st.subheader("📂 Project Workspace & File Editor")

        uploaded_file = st.file_uploader(
            "Open Active Consumer File (.csv / .xlsx / .xls)",
            type=["csv", "xlsx", "xls"],
        )

        if uploaded_file is not None and uploaded_file.name != st.session_state.active_file:
            try:
                name = uploaded_file.name
                warnings: list[str] = []
                if name.lower().endswith(".xls"):
                    frame = load_ledger_bytes(uploaded_file.getvalue(), name)
                elif name.lower().endswith(".xlsx"):
                    raw = pd.read_excel(uploaded_file, engine="openpyxl")
                    if is_ledger_frame(raw):
                        frame = normalize_ledger(raw, source_file=name)
                    else:
                        frame, warnings = prepare_frame(raw)
                else:
                    raw = pd.read_csv(uploaded_file)
                    frame, warnings = prepare_frame(raw)
                st.session_state.current_data = frame
                st.session_state.edited_data = frame
                st.session_state.active_file = uploaded_file.name
                st.session_state.selected_consumer = None
                st.session_state.grid_version += 1
                st.success(f"Loaded active file buffer: {uploaded_file.name} ({len(frame)} rows)")
                for warning in warnings:
                    st.warning(warning)
            except Exception as exc:
                st.error(f"Error loading file matrix: {exc}")

        frame = st.session_state.current_data
        if frame is not None:
            st.write(f"📁 **`{st.session_state.active_file}` (Interactive Canvas Area)**")

            # The editor gets a versioned key so that a fresh buffer (new upload or
            # applied corrections) replaces the edit state instead of fighting it.
            number_formats = {
                UNITS_COLUMN: st.column_config.NumberColumn("Monthly units", format="%.1f"),
                BILL_COLUMN: st.column_config.NumberColumn("Bill (Rs)", format="%.2f"),
                TAX_COLUMN: st.column_config.NumberColumn("Tax (Rs)", format="%.2f"),
                CURRENT_TOTAL_COLUMN: st.column_config.NumberColumn("Current total (Rs)", format="%.2f"),
            }
            edited = st.data_editor(
                frame,
                key=f"grid_{st.session_state.grid_version}",
                height=450,
                width="stretch",
                column_config={
                    column: config for column, config in number_formats.items() if column in frame.columns
                },
            )
            st.session_state.edited_data = edited

            csv_buffer = frame.to_csv(index=False).encode("utf-8")
            st.download_button(
                "📥 Export Current Buffers to CSV",
                data=csv_buffer,
                file_name="corrected_profile.csv",
                mime="text/csv",
                width="stretch",
            )

            active = st.session_state.edited_data
            if active is not None and UNITS_COLUMN in active.columns:
                if detector is not None:
                    report = detector.detect(active, z_threshold=z_threshold)
                    flagged = int(report[ANOMALY_COLUMN].sum())
                    if is_ledger_frame(active):
                        defects = int(report[METER_DEFECT_ANOMALY_COLUMN].sum())
                        model_flags = int(report[MODEL_ANOMALY_COLUMN].sum())
                        c1, c2, c3, c4 = st.columns(4)
                        c1.metric("Rows", f"{len(active):,}")
                        c2.metric(
                            "Consumers",
                            f"{active[CONSUMER_COLUMN].nunique() if CONSUMER_COLUMN in active.columns else 1:,}",
                        )
                        c3.metric("Meter defects", f"{defects:,}")
                        c4.metric("Model flags", f"{model_flags:,}")
                    else:
                        c1, c2, c3 = st.columns(3)
                        c1.metric("Rows", f"{len(active):,}")
                        c2.metric(
                            "Consumers",
                            f"{active[CONSUMER_COLUMN].nunique() if CONSUMER_COLUMN in active.columns else 1:,}",
                        )
                        c3.metric(
                            "Anomalies",
                            f"{flagged:,}",
                            delta=None if flagged == 0 else "review",
                        )
        else:
            st.info("Your file canvas is empty. Drag a consumer profile here to load the dataset matrix into memory.")

    # ---------------------------------------------------------- chat canvas
    st.divider()
    with st.container():
        st.subheader("Volta AI")
        st.markdown(status_banner(engine_status), unsafe_allow_html=True)
        if st.session_state.selected_consumer:
            st.caption(f"Selected consumer: **{st.session_state.selected_consumer}**")

        chat_container = st.container(height=450)
        for message in st.session_state.chat_history:
            with chat_container.chat_message(message["role"]):
                st.write(message["content"])

        prompt = st.chat_input(
            "Enter a consumer ID or command (e.g. 'DUDOB14460', 'bill anomaly')"
        )
        if prompt:
            st.session_state.chat_history.append({"role": "user", "content": prompt})
            with chat_container.chat_message("user"):
                st.write(prompt)

            active = st.session_state.edited_data
            if active is None:
                active = st.session_state.current_data

            previous_selected = st.session_state.selected_consumer
            response, updated, selected = handle_command(
                prompt,
                active,
                detector,
                policy,
                z_threshold,
                previous_selected,
            )
            st.session_state.selected_consumer = selected
            with chat_container.chat_message("assistant"):
                st.write(response)
            st.session_state.chat_history.append({"role": "assistant", "content": response})

            if updated is not None:
                st.session_state.current_data = updated
                st.session_state.edited_data = updated
                st.session_state.grid_version += 1
                st.rerun()
            if selected != previous_selected:
                st.rerun()


if __name__ == "__main__":
    main()
