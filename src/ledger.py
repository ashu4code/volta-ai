"""Consumer-ledger ingestion and normalization.

The SBPDCL ledger exports share a 79-column schema but use raw utility names
(``CON_ID``, ``UNITS_BILLED``, ``CURRENT_TOTAL``) and a zero-padded ``MMYYYY``
period. This module is the single place that turns those exports into the
columns the detector and tariff engine consume.
"""

from __future__ import annotations

from io import BytesIO
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

CON_ID_COLUMN = "CON_ID"
ACCOUNT_COLUMN = "ACT_NO"
PERIOD_COLUMN = "MMYYYY"
TARIFF_COLUMN = "TARIFF_ID"
READ_STATUS_COLUMN = "READ_STAT"
RECORDED_UNITS_COLUMN = "UNITS_CONS"
BILLED_UNITS_COLUMN = "UNITS_BILLED"
BILLING_DAYS_COLUMN = "NO_MONTHS"

CONSUMER_COLUMN = "Consumer_ID"
DATE_COLUMN = "Date"
FISCAL_YEAR_COLUMN = "Fiscal_Year"
BILLING_MONTH_COLUMN = "Billing_Month"
BILLING_DAYS_NORMALIZED_COLUMN = "Billing_Days"
MONTHLY_UNITS_COLUMN = "Monthly_Units"
UNITS_COLUMN = "Units_Consumed"
METER_DEFECT_COLUMN = "Meter_Defect"
CATEGORY_COLUMN = "Category"
SOURCE_FILE_COLUMN = "Source_File"
ACCOUNT_NUMBER_COLUMN = "Account_No"

CONSUMER_LOOKUP_COLUMNS = (CONSUMER_COLUMN, ACCOUNT_NUMBER_COLUMN)

BILL_COLUMN = "Bill_Amount"
TAX_COLUMN = "Tax_Amount"
TOTAL_COLUMN = "Total_Amount"
FIXED_COLUMN = "Fixed_Charges"

ENERGY_COLUMN = "EC_CHRG"
ED_COLUMN = "ED_CURRENT"
FIXED_RAW_COLUMN = "FIXED_CHRG"
DEMAND_COLUMN = "DEMAND_CHRG"
DEMAND_EXCESS_COLUMN = "DEMAND_CHRG_EXC"
METER_RENT_COLUMN = "MRENT"
SHUNT_COLUMN = "SHUNT_CHRG"
OTHER_COLUMN = "OTH_CHRG"
SUBSIDY_COLUMN = "GOV_SUBSIDY"
DPS_CURRENT_COLUMN = "DPS_CUR"
CURRENT_TOTAL_COLUMN = "CURRENT_TOTAL"

REQUIRED_COLUMNS = (
    CON_ID_COLUMN,
    ACCOUNT_COLUMN,
    PERIOD_COLUMN,
    TARIFF_COLUMN,
    READ_STATUS_COLUMN,
    RECORDED_UNITS_COLUMN,
    BILLED_UNITS_COLUMN,
    BILLING_DAYS_COLUMN,
    ENERGY_COLUMN,
    ED_COLUMN,
    FIXED_RAW_COLUMN,
    DEMAND_COLUMN,
    METER_RENT_COLUMN,
    SUBSIDY_COLUMN,
    CURRENT_TOTAL_COLUMN,
)

_CHARGE_COLUMNS = (
    ENERGY_COLUMN,
    FIXED_RAW_COLUMN,
    DEMAND_COLUMN,
    DEMAND_EXCESS_COLUMN,
    METER_RENT_COLUMN,
    SHUNT_COLUMN,
    OTHER_COLUMN,
    DPS_CURRENT_COLUMN,
)


def _as_string(series: pd.Series) -> pd.Series:
    """Return trimmed strings, removing the ``.0`` pandas adds to numeric IDs."""
    text = series.astype("string").str.strip()
    return text.str.replace(r"\.0$", "", regex=True).fillna("")


def _as_number(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce").astype(float)


def _format_period(value) -> str:
    if pd.isna(value):
        return ""
    text = str(value).strip()
    if text.endswith(".0"):
        text = text[:-2]
    if text.isdigit() and len(text) <= 6:
        return text.zfill(6)
    return text


def _fiscal_year(dates: pd.Series) -> pd.Series:
    start = np.where(dates.dt.month >= 4, dates.dt.year, dates.dt.year - 1)
    labels = [
        f"{int(year)}-{(int(year) + 1) % 100:02d}" if not pd.isna(year) else ""
        for year in start
    ]
    return pd.Series(labels, index=dates.index, dtype="string")


def _read_excel(source, suffix: str) -> pd.DataFrame:
    engine = "xlrd" if suffix.lower() == ".xls" else "openpyxl"
    return pd.read_excel(source, sheet_name=0, engine=engine)


def normalize_ledger(raw: pd.DataFrame, source_file: str = "") -> pd.DataFrame:
    """Return a normalized copy of one ledger export."""
    out = raw.copy()
    out.columns = [str(column).strip() for column in out.columns]

    missing = [column for column in REQUIRED_COLUMNS if column not in out.columns]
    if missing:
        raise KeyError(
            "ledger is missing required column(s): " + ", ".join(missing)
        )

    out[CONSUMER_COLUMN] = _as_string(out[CON_ID_COLUMN])
    out["Account_No"] = _as_string(out[ACCOUNT_COLUMN])
    out[SOURCE_FILE_COLUMN] = source_file

    periods = out[PERIOD_COLUMN].map(_format_period)
    out[BILLING_MONTH_COLUMN] = periods
    dates = pd.to_datetime(periods, format="%m%Y", errors="coerce")
    out[DATE_COLUMN] = dates
    out[FISCAL_YEAR_COLUMN] = _fiscal_year(dates)
    out[CATEGORY_COLUMN] = _as_string(out[TARIFF_COLUMN]).str.upper()

    for column in (
        RECORDED_UNITS_COLUMN,
        BILLED_UNITS_COLUMN,
        BILLING_DAYS_COLUMN,
        ENERGY_COLUMN,
        ED_COLUMN,
        FIXED_RAW_COLUMN,
        DEMAND_COLUMN,
        DEMAND_EXCESS_COLUMN,
        METER_RENT_COLUMN,
        SUBSIDY_COLUMN,
        CURRENT_TOTAL_COLUMN,
        SHUNT_COLUMN,
        OTHER_COLUMN,
        DPS_CURRENT_COLUMN,
    ):
        if column in out.columns:
            out[column] = _as_number(out[column])

    days = out[BILLING_DAYS_COLUMN].fillna(30.0)
    days = days.where(days > 1, 30.0)
    out[BILLING_DAYS_NORMALIZED_COLUMN] = days.astype(float)

    billed = out[BILLED_UNITS_COLUMN].fillna(0.0)
    out["Billed_Units"] = billed.astype(float)
    out[MONTHLY_UNITS_COLUMN] = (billed * 30.0 / out[BILLING_DAYS_NORMALIZED_COLUMN]).round(4)
    out[UNITS_COLUMN] = out[MONTHLY_UNITS_COLUMN]

    read_status = _as_string(out[READ_STATUS_COLUMN]).str.upper()
    recorded = out[RECORDED_UNITS_COLUMN].fillna(0.0)
    out[METER_DEFECT_COLUMN] = (
        read_status.isin(["MD", "LK"]) | ((recorded <= 0) & (billed > 0))
    )

    fixed_components = [
        column
        for column in (FIXED_RAW_COLUMN, DEMAND_COLUMN, DEMAND_EXCESS_COLUMN)
        if column in out.columns
    ]
    out[FIXED_COLUMN] = (
        out[fixed_components].fillna(0.0).sum(axis=1) if fixed_components else 0.0
    )

    charge_columns = [column for column in _CHARGE_COLUMNS if column in out.columns]
    out[BILL_COLUMN] = out[charge_columns].fillna(0.0).sum(axis=1)
    out[TAX_COLUMN] = out[ED_COLUMN].fillna(0.0)
    out[TOTAL_COLUMN] = out[CURRENT_TOTAL_COLUMN].fillna(
        out[BILL_COLUMN] + out[TAX_COLUMN] + out[SUBSIDY_COLUMN]
    )

    return out


def load_ledger(source, name: str | None = None) -> pd.DataFrame:
    """Read and normalize a ledger from a path or an uploaded file object."""
    source_name = Path(name or getattr(source, "name", "ledger.xls")).name
    raw = _read_excel(source, Path(source_name).suffix)
    return normalize_ledger(raw, source_file=source_name)


def load_ledger_bytes(data: bytes, name: str) -> pd.DataFrame:
    return load_ledger(BytesIO(data), name=name)


def load_ledgers(dataset_dir: str | Path = "consumer_data/ledgers") -> pd.DataFrame:
    """Load every ``Consumer_Ledger_Report_*`` workbook in a directory."""
    directory = Path(dataset_dir)
    if not directory.exists():
        raise FileNotFoundError(f"ledger directory not found: '{directory}'")

    paths = sorted(
        path
        for pattern in ("Consumer_Ledger_Report_*.xls", "Consumer_Ledger_Report_*.xlsx")
        for path in directory.glob(pattern)
    )
    if not paths:
        raise FileNotFoundError(
            f"no 'Consumer_Ledger_Report_*.xls' files found in '{directory}'"
        )

    frames = [load_ledger(path) for path in paths]
    combined = pd.concat(frames, ignore_index=True)
    return combined.sort_values(
        [CONSUMER_COLUMN, DATE_COLUMN, SOURCE_FILE_COLUMN], na_position="last"
    ).reset_index(drop=True)


def _normalize_identifier(value) -> str:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return ""
    text = str(value).strip()
    if text.endswith(".0") and text[:-2].isdigit():
        text = text[:-2]
    return text.upper()


def find_consumers(df: pd.DataFrame, identifier: str) -> pd.DataFrame:
    """Return rows whose ``Consumer_ID`` or ``Account_No`` matches exactly."""
    query = _normalize_identifier(identifier)
    if not query or CONSUMER_COLUMN not in df.columns:
        return df.iloc[0:0].copy()

    mask = pd.Series(False, index=df.index)
    for column in CONSUMER_LOOKUP_COLUMNS:
        if column in df.columns:
            mask |= _as_string(df[column]).str.upper().eq(query)
    return df.loc[mask].copy()


def consumer_catalog(df: pd.DataFrame) -> pd.DataFrame:
    """Unique consumer identifier pairs, sorted for display and suggestions."""
    columns = [c for c in CONSUMER_LOOKUP_COLUMNS if c in df.columns]
    if not columns:
        return pd.DataFrame(columns=list(CONSUMER_LOOKUP_COLUMNS))
    catalog = df[columns].copy()
    for column in columns:
        catalog[column] = _as_string(catalog[column])
    return catalog.drop_duplicates().sort_values(columns).reset_index(drop=True)


def suggest_consumers(df: pd.DataFrame, identifier: str, limit: int = 5) -> list[str]:
    """Return up to ``limit`` human-readable IDs containing the query."""
    query = _normalize_identifier(identifier)
    if not query:
        return []
    catalog = consumer_catalog(df)
    matches: list[str] = []
    for _, row in catalog.iterrows():
        consumer_id = str(row.get(CONSUMER_COLUMN, ""))
        account = str(row.get(ACCOUNT_NUMBER_COLUMN, ""))
        haystack = f"{consumer_id} {account}".upper()
        if query in haystack:
            matches.append(consumer_label(consumer_id, account))
        if len(matches) >= limit:
            break
    return matches


def consumer_label(consumer_id, account_no) -> str:
    consumer_id = _normalize_identifier(consumer_id)
    account_no = _normalize_identifier(account_no)
    if consumer_id and account_no and consumer_id != account_no:
        return f"{consumer_id} ({account_no})"
    return consumer_id or account_no or "<unknown>"


def duplicate_period_count(df: pd.DataFrame) -> int:
    """Number of rows duplicated on consumer and billing month."""
    keys = [CONSUMER_COLUMN, BILLING_MONTH_COLUMN]
    if not set(keys).issubset(df.columns):
        return 0
    return int(df.duplicated(subset=keys).sum())


def is_ledger_frame(df: pd.DataFrame) -> bool:
    required = {CON_ID_COLUMN, PERIOD_COLUMN, ENERGY_COLUMN, CURRENT_TOTAL_COLUMN}
    return required.issubset({str(column) for column in df.columns})


def ledger_warnings(df: pd.DataFrame) -> list[str]:
    warnings: list[str] = []
    if CON_ID_COLUMN not in df.columns:
        return warnings
    blank_ids = int((_as_string(df[CON_ID_COLUMN]) == "").sum())
    if blank_ids:
        warnings.append(f"{blank_ids} row(s) have a blank CON_ID")
    if DATE_COLUMN in df.columns:
        invalid_dates = int(df[DATE_COLUMN].isna().sum())
        if invalid_dates:
            warnings.append(f"{invalid_dates} row(s) have an invalid MMYYYY period")
    if BILLED_UNITS_COLUMN in df.columns:
        negative = int((_as_number(df[BILLED_UNITS_COLUMN]) < 0).sum())
        if negative:
            warnings.append(f"{negative} row(s) have negative billed units")
    duplicates = duplicate_period_count(df)
    if duplicates:
        warnings.append(
            f"{duplicates} duplicate consumer-month row(s); remove copied ledger files"
        )
    return warnings


def ledger_summary(df: pd.DataFrame) -> dict:
    return {
        "rows": int(len(df)),
        "consumers": int(df[CONSUMER_COLUMN].nunique()) if CONSUMER_COLUMN in df.columns else 0,
        "files": int(df[SOURCE_FILE_COLUMN].nunique()) if SOURCE_FILE_COLUMN in df.columns else 0,
        "meter_defects": int(df[METER_DEFECT_COLUMN].sum()) if METER_DEFECT_COLUMN in df.columns else 0,
        "duplicate_periods": duplicate_period_count(df),
        "fiscal_years": sorted(
            str(value) for value in df[FISCAL_YEAR_COLUMN].dropna().unique()
        ) if FISCAL_YEAR_COLUMN in df.columns else [],
    }
