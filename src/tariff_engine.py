"""Deterministic tariff / billing rules engine.

Single source of truth for every rupee amount produced by this workspace.

Billing semantics follow the shipped consumer dataset exactly:

    Bill_Amount = energy_charge + fixed_charge      (pre-tax subtotal)
    Tax_Amount  = Bill_Amount * tax_rate
    Total       = Bill_Amount + Tax_Amount           (what the consumer pays)

Verified against all 120 rows of ``sample_consumer_120_rows.xlsx`` and every
row of ``con.csv`` (see ``tests/test_pipeline.py``).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Mapping

import numpy as np
import pandas as pd

UNITS_COLUMN = "Units_Consumed"
BILL_COLUMN = "Bill_Amount"
TAX_COLUMN = "Tax_Amount"
FIXED_COLUMN = "Fixed_Charges"

#: Absolute tolerance (rupees) used when comparing a billed amount to the engine.
TOLERANCE = 0.01


@dataclass(frozen=True)
class TariffPolicy:
    """A named electricity tariff.

    Either ``energy_rate`` (flat per-unit rate) or ``slabs`` (telescopic slab
    matrix) defines how energy is charged; ``fixed_charge`` is billed whether
    or not any unit is consumed, and ``tax_rate`` is applied to the subtotal.
    """

    name: str
    fixed_charge: float
    tax_rate: float
    energy_rate: float | None = None
    slabs: tuple[tuple[float, float], ...] = ()

    @property
    def is_flat(self) -> bool:
        return not self.slabs

    @property
    def description(self) -> str:
        if self.is_flat:
            return f"₹{self.energy_rate:.2f}/unit + ₹{self.fixed_charge:.2f} fixed, {self.tax_rate:.0%} tax"
        return f"telescopic slabs + ₹{self.fixed_charge:.2f} fixed, {self.tax_rate:.0%} tax"

    def energy_charge(self, units: float) -> float:
        """Energy component of the bill for ``units`` consumed."""
        units = max(0.0, float(units))
        if not self.is_flat:
            charge = 0.0
            remaining = units
            for limit, rate in self.slabs:
                if remaining > limit:
                    charge += limit * rate
                    remaining -= limit
                else:
                    charge += remaining * rate
                    break
            return charge
        return units * float(self.energy_rate or 0.0)

    def breakdown(self, units: float, fixed_charge: float | None = None) -> Mapping[str, float]:
        """Full rupee breakdown for ``units`` consumed.

        ``fixed_charge`` overrides the policy default (used to honour a
        per-consumer ``Fixed_Charges`` column when the dataset provides one).
        """
        fixed = self.fixed_charge if fixed_charge is None else float(fixed_charge)
        energy = self.energy_charge(units)
        subtotal = energy + fixed
        tax = subtotal * self.tax_rate
        return {
            "units": round(float(units), 4),
            "energy_charge": round(energy, 2),
            "fixed_charge": round(fixed, 2),
            "subtotal": round(subtotal, 2),
            "tax": round(tax, 2),
            "total": round(subtotal + tax, 2),
        }


#: The only workspace tariff for sample/CSV/XLSX billing.
SLAB_DOMESTIC = TariffPolicy(
    name="Domestic Telescopic Slab",
    energy_rate=None,
    fixed_charge=100.00,
    tax_rate=0.06,
    slabs=((100.0, 3.00), (200.0, 5.50), (200.0, 8.00), (float("inf"), 10.50)),
)

POLICIES: dict[str, TariffPolicy] = {SLAB_DOMESTIC.name: SLAB_DOMESTIC}
DEFAULT_POLICY: TariffPolicy = SLAB_DOMESTIC


def bill_breakdown(
    units: float,
    policy: TariffPolicy = DEFAULT_POLICY,
    fixed_charge: float | None = None,
) -> Mapping[str, float]:
    """Rupee breakdown for ``units`` under ``policy``."""
    return policy.breakdown(units, fixed_charge)


def calculate_bill(
    units: float,
    policy: TariffPolicy = DEFAULT_POLICY,
    fixed_charge: float | None = None,
) -> float:
    """Tax-inclusive total payable for ``units``."""
    return float(bill_breakdown(units, policy, fixed_charge)["total"])


def _fixed_charges_for(df: pd.DataFrame, policy: TariffPolicy) -> pd.Series:
    """Per-row fixed charge, preferring the dataset's own ``Fixed_Charges``."""
    if FIXED_COLUMN in df.columns:
        fixed = pd.to_numeric(df[FIXED_COLUMN], errors="coerce")
        return fixed.fillna(policy.fixed_charge).astype(float)
    return pd.Series(float(policy.fixed_charge), index=df.index)


def apply_billing(
    df: pd.DataFrame,
    policy: TariffPolicy = DEFAULT_POLICY,
    mask: pd.Series | None = None,
) -> pd.DataFrame:
    """Return a copy of ``df`` with ``Bill_Amount``/``Tax_Amount`` recomputed.

    Only rows selected by ``mask`` are rewritten; when ``mask`` is ``None`` every
    row is recomputed. ``Bill_Amount`` holds the pre-tax subtotal and
    ``Tax_Amount`` the tax on that subtotal, matching the shipped dataset.
    """
    if UNITS_COLUMN not in df.columns:
        raise KeyError(f"column '{UNITS_COLUMN}' is required to price a bill")

    out = df.copy()
    units = pd.to_numeric(out[UNITS_COLUMN], errors="coerce").fillna(0.0)
    fixed = _fixed_charges_for(out, policy)

    energy = pd.Series(
        [policy.energy_charge(u) for u in units.to_numpy(dtype=float)],
        index=out.index,
        dtype=float,
    )
    subtotal = energy + fixed
    tax = subtotal * policy.tax_rate

    target = out.index if mask is None else out.index[mask]
    if len(target) == 0:
        return out

    for column, values in ((BILL_COLUMN, subtotal), (TAX_COLUMN, tax)):
        if column not in out.columns:
            out[column] = pd.Series(float("nan"), index=out.index, dtype=float)
        out.loc[target, column] = values.loc[target].round(2)
        out[column] = pd.to_numeric(out[column], errors="coerce")
    return out


def reconcile_bills(
    df: pd.DataFrame,
    policy: TariffPolicy = DEFAULT_POLICY,
) -> pd.DataFrame:
    """Add engine expectations and a match flag for every billed row.

    Adds ``Expected_Bill``, ``Expected_Tax``, ``Bill_Delta`` and ``Bill_Matches``.
    """
    if UNITS_COLUMN not in df.columns:
        raise KeyError(f"column '{UNITS_COLUMN}' is required to price a bill")

    out = df.copy()
    units = pd.to_numeric(out[UNITS_COLUMN], errors="coerce").fillna(0.0)
    fixed = _fixed_charges_for(out, policy)

    energy = pd.Series(
        [policy.energy_charge(u) for u in units.to_numpy(dtype=float)],
        index=out.index,
        dtype=float,
    )
    subtotal = (energy + fixed).round(2)
    tax = (subtotal * policy.tax_rate).round(2)

    out["Expected_Bill"] = subtotal
    out["Expected_Tax"] = tax

    billed = pd.to_numeric(out.get(BILL_COLUMN, pd.Series(np.nan, index=out.index)), errors="coerce")
    out["Bill_Delta"] = (billed - subtotal).round(2)
    out["Bill_Matches"] = billed.sub(subtotal).abs().le(TOLERANCE).fillna(False)
    return out

# ---------------------------------------------------------------------------
# Ledger-native FY tariff rules
# ---------------------------------------------------------------------------
# The sample-policy classes above intentionally keep their original semantics:
# tax is charged on the pre-tax subtotal. Ledger bills are different: ED is 6%
# of the energy charge only, and fixed/demand charges are prorated by billing
# days. The classes below are the ledger path and do not affect the sample
# regression.

LEDGER_TOLERANCE = 0.50
LEDGER_IDENTITY_TOLERANCE = 0.01
TARIFF_EC_TOLERANCE = 10.00
TARIFF_ED_TOLERANCE = 0.60
TARIFF_SUBSIDY_TOLERANCE = 5.00
TARIFF_TOTAL_TOLERANCE = 12.00


@dataclass(frozen=True)
class SubsidyPolicy:
    """Government subsidy for a category/FY.

    ``flat_per_unit`` wins when set. Otherwise ``slabs`` is a telescopic
    width/rate table exactly like :class:`TariffPolicy`.
    """

    flat_per_unit: float | None = None
    slabs: tuple[tuple[float, float], ...] = ()
    source: str = ""
    verified: bool = True

    def amount(self, units: float) -> float:
        units = max(0.0, float(units))
        if self.flat_per_unit is not None:
            return units * self.flat_per_unit
        charge = 0.0
        remaining = units
        for limit, rate in self.slabs:
            if remaining > limit:
                charge += limit * rate
                remaining -= limit
            else:
                charge += remaining * rate
                break
        return charge


@dataclass(frozen=True)
class FYTariff:
    """A category tariff for one Indian financial year (April-March)."""

    fiscal_year: str
    category: str
    slabs: tuple[tuple[float, float], ...]
    fixed_per_kw: float = 0.0
    fixed_per_connection: float = 0.0
    demand_per_kw: float = 0.0
    meter_rent_per_month: float = 0.0
    tax_rate: float = 0.06
    subsidy: SubsidyPolicy = field(default_factory=SubsidyPolicy)
    mmc_first_kw: float = 0.0
    mmc_additional_kw: float = 0.0
    leading_units: float = 0.0
    leading_rate: float = 0.0
    minimum_charge: float = 0.0
    source: str = ""
    verified: bool = True

    def billable_units(self, units: float, load: float) -> float:
        units = max(0.0, float(units))
        if self.mmc_first_kw <= 0 and self.mmc_additional_kw <= 0:
            return units
        load = max(0.0, float(load))
        minimum = self.mmc_first_kw * min(load, 1.0)
        minimum += self.mmc_additional_kw * max(load - 1.0, 0.0)
        return max(units, minimum)

    def energy_charge(self, units: float) -> float:
        units = max(0.0, float(units))
        charge = 0.0
        if self.leading_units > 0:
            leading = min(units, self.leading_units)
            charge += leading * self.leading_rate
            units -= leading
        remaining = units
        for limit, rate in self.slabs:
            if remaining > limit:
                charge += limit * rate
                remaining -= limit
            else:
                charge += remaining * rate
                break
        return max(charge, self.minimum_charge)

    def fixed_charge(self, load: float, days: float) -> float:
        months = max(0.0, float(days)) / 30.0
        load = max(0.0, float(load))
        return (
            self.fixed_per_kw * load * months
            + self.fixed_per_connection * months
        )

    def demand_charge(self, load: float, days: float) -> float:
        months = max(0.0, float(days)) / 30.0
        return self.demand_per_kw * max(1.0, float(load)) * months

    def meter_rent(self, days: float) -> float:
        return self.meter_rent_per_month * max(0.0, float(days)) / 30.0


def _subsidy(flat: float | None = None, slabs=(), source="", verified=True) -> SubsidyPolicy:
    return SubsidyPolicy(
        flat_per_unit=flat,
        slabs=tuple(slabs),
        source=source,
        verified=verified,
    )


def _t(
    fy: str,
    category: str,
    slabs,
    *,
    fixed_kw: float = 0.0,
    fixed_conn: float = 0.0,
    demand_kw: float = 0.0,
    meter_rent: float = 0.0,
    subsidy: SubsidyPolicy | None = None,
    mmc_first: float = 0.0,
    mmc_additional: float = 0.0,
    leading_units: float = 0.0,
    leading_rate: float = 0.0,
    minimum_charge: float = 0.0,
    source: str = "",
    verified: bool = True,
) -> FYTariff:
    return FYTariff(
        fiscal_year=fy,
        category=category,
        slabs=tuple(slabs),
        fixed_per_kw=fixed_kw,
        fixed_per_connection=fixed_conn,
        demand_per_kw=demand_kw,
        meter_rent_per_month=meter_rent,
        subsidy=subsidy or _subsidy(0.0, source="no subsidy"),
        mmc_first_kw=mmc_first,
        mmc_additional_kw=mmc_additional,
        leading_units=leading_units,
        leading_rate=leading_rate,
        minimum_charge=minimum_charge,
        source=source,
        verified=verified,
    )


_INF = float("inf")

# Subsidy rates are the notified/billing flat rates observed in the ledgers and
# cross-checked against the FY2017-18 order. ``verified=False`` entries are
# deliberately visible in reconciliation instead of being silently guessed.
_SUB_0 = _subsidy(0.0, source="no subsidy", verified=True)
_SUB_DS1_1718 = _subsidy(
    3.10,
    source="BERC FY2017-18; ledger applies flat 3.10",
    verified=False,
)
_SUB_DS1_1819 = _subsidy(3.50, source="ledger FY2018-19", verified=True)
_SUB_DS1_1921 = _subsidy(3.50, source="inferred from FY2018-19", verified=False)
_SUB_DS1_2123 = _subsidy(3.50, source="ledger FY2021-22/FY2022-23", verified=True)
_SUB_DS1_2324 = _subsidy(5.02, source="ledger FY2023-24", verified=True)
_SUB_DS1_2426 = _subsidy(4.98, source="ledger FY2024-25/FY2025-26", verified=True)
_SUB_KJ_1718 = _subsidy(3.58, source="BERC FY2017-18", verified=True)
_SUB_KJ_1822 = _subsidy(3.98, source="ledger FY2018-19 through FY2021-22", verified=True)

TARIFF_SCHEDULE: dict[tuple[str, str], FYTariff] = {}


def _add(tariff: FYTariff) -> None:
    TARIFF_SCHEDULE[(tariff.fiscal_year, tariff.category)] = tariff


# FY 2016-17: BERC Final SBPDCL Order 21.03.2016, Appendix I.
_add(
    _t(
        "2016-17",
        "DS1",
        ((50, 2.10), (50, 2.40), (_INF, 2.80)),
        meter_rent=20.0,
        subsidy=_SUB_0,
        mmc_first=40.0,
        mmc_additional=20.0,
        source="BERC FY2016-17 Appendix I",
    )
)
_add(
    _t(
        "2016-17",
        "KJ",
        ((50, 2.10), (50, 2.40), (_INF, 2.80)),
        meter_rent=20.0,
        subsidy=_SUB_0,
        leading_units=30.0,
        leading_rate=1.70,
        minimum_charge=40.0,
        source="BERC FY2016-17 Appendix I",
    )
)

# FY 2017-18.
_add(
    _t(
        "2017-18",
        "DS1",
        ((50, 5.75), (50, 6.00), (_INF, 6.25)),
        fixed_kw=20.0,
        meter_rent=20.0,
        subsidy=_SUB_DS1_1718,
        source="BERC FY2017-18",
    )
)
_add(
    _t(
        "2017-18",
        "KJ",
        ((_INF, 5.75),),
        fixed_conn=10.0,
        meter_rent=20.0,
        subsidy=_SUB_KJ_1718,
        source="BERC FY2017-18",
    )
)

# FY 2018-19 was kept at the FY 2019-20 approved rates.
for _fy in ("2018-19", "2019-20"):
    _add(
        _t(
            _fy,
            "DS1",
            ((50, 6.15), (50, 6.40), (100, 6.70), (_INF, 7.05)),
            fixed_kw=20.0,
            meter_rent=20.0,
            subsidy=_SUB_DS1_1819 if _fy == "2018-19" else _SUB_DS1_1921,
            source=f"BERC FY{_fy}",
        )
    )
    _add(
        _t(
            _fy,
            "KJ",
            ((_INF, 6.15),),
            fixed_conn=10.0,
            meter_rent=20.0,
            subsidy=_SUB_KJ_1822,
            source=f"BERC FY{_fy}",
        )
    )
    _add(
        _t(
            _fy,
            "DS1D",
            ((50, 6.15), (50, 6.40), (100, 6.70), (_INF, 7.05)),
            demand_kw=20.0,
            meter_rent=20.0,
            subsidy=_SUB_DS1_1819 if _fy == "2018-19" else _SUB_DS1_1921,
            source=f"BERC FY{_fy}; demand-based option",
            verified=False,
        )
    )

# FY 2020-21.
_add(
    _t(
        "2020-21",
        "DS1",
        ((50, 6.05), (50, 6.30), (100, 6.60), (_INF, 6.95)),
        fixed_kw=20.0,
        meter_rent=0.0,
        subsidy=_SUB_DS1_1921,
        source="BERC FY2020-21",
    )
)
_add(
    _t(
        "2020-21",
        "KJ",
        ((_INF, 6.05),),
        fixed_conn=10.0,
        meter_rent=0.0,
        subsidy=_SUB_KJ_1822,
        source="BERC FY2020-21",
    )
)
_add(
    _t(
        "2020-21",
        "DS1D",
        ((50, 6.05), (50, 6.30), (100, 6.60), (_INF, 6.95)),
        demand_kw=20.0,
        meter_rent=0.0,
        subsidy=_SUB_DS1_1921,
        source="BERC FY2020-21; demand-based option",
        verified=False,
    )
)

# FY 2021-22 and FY 2022-23.
for _fy in ("2021-22", "2022-23"):
    _add(
        _t(
            _fy,
            "DS1",
            ((50, 6.10), (50, 6.40), (_INF, 6.70)),
            fixed_kw=20.0,
            meter_rent=0.0,
            subsidy=_SUB_DS1_2123,
            source=f"BERC FY{_fy}",
        )
    )
    _add(
        _t(
            _fy,
            "KJ",
            ((_INF, 6.10),),
            fixed_conn=10.0,
            meter_rent=0.0,
            subsidy=_SUB_KJ_1822,
            source=f"BERC FY{_fy}",
        )
    )
    _add(
        _t(
            _fy,
            "DS1D",
            ((50, 6.10), (50, 6.40), (_INF, 6.70)),
            demand_kw=20.0,
            meter_rent=0.0,
            subsidy=_SUB_DS1_2123,
            source=f"BERC FY{_fy}; demand-based option",
            verified=True,
        )
    )

# FY 2023-24.
_add(
    _t(
        "2023-24",
        "DS1",
        ((50, 7.57), (_INF, 8.11)),
        fixed_kw=40.0,
        meter_rent=0.0,
        subsidy=_SUB_DS1_2324,
        source="BERC FY2023-24",
    )
)
_add(
    _t(
        "2023-24",
        "KJ",
        ((_INF, 7.57),),
        fixed_conn=20.0,
        meter_rent=0.0,
        subsidy=_SUB_0,
        source="FY2023-24 chart; KJ fixed charge not exercised by current ledger",
        verified=False,
    )
)
_add(
    _t(
        "2023-24",
        "DS1D",
        ((50, 7.57), (_INF, 8.11)),
        demand_kw=40.0,
        meter_rent=0.0,
        subsidy=_SUB_DS1_2324,
        source="BERC FY2023-24",
    )
)

# FY 2024-25.
_add(
    _t(
        "2024-25",
        "DS1",
        ((50, 7.42), (_INF, 7.96)),
        fixed_kw=40.0,
        meter_rent=0.0,
        subsidy=_SUB_DS1_2426,
        source="BERC FY2024-25",
    )
)
_add(
    _t(
        "2024-25",
        "KJ",
        ((_INF, 7.42),),
        fixed_conn=20.0,
        meter_rent=0.0,
        subsidy=_SUB_0,
        source="FY2024-25 chart; KJ fixed charge not exercised by current ledger",
        verified=False,
    )
)
_add(
    _t(
        "2024-25",
        "DS1D",
        ((50, 7.42), (_INF, 7.96)),
        demand_kw=40.0,
        meter_rent=0.0,
        subsidy=_SUB_DS1_2426,
        source="BERC FY2024-25",
    )
)

# FY 2025-26: DS-I slabs merged at the lower rate.
_add(
    _t(
        "2025-26",
        "DS1",
        ((_INF, 7.42),),
        fixed_kw=40.0,
        meter_rent=0.0,
        subsidy=_SUB_DS1_2426,
        source="BERC FY2025-26",
    )
)
_add(
    _t(
        "2025-26",
        "KJ",
        ((_INF, 7.42),),
        fixed_conn=20.0,
        meter_rent=0.0,
        subsidy=_SUB_0,
        source="FY2025-26 chart; KJ fixed charge not exercised by current ledger",
        verified=False,
    )
)
_add(
    _t(
        "2025-26",
        "DS1D",
        ((_INF, 7.42),),
        demand_kw=40.0,
        meter_rent=0.0,
        subsidy=_SUB_DS1_2426,
        source="BERC FY2025-26",
    )
)


def fiscal_year_for(period) -> str:
    """Return the April-March financial year for a date or ``MMYYYY`` value."""
    if period is None or (isinstance(period, float) and np.isnan(period)):
        return ""
    if isinstance(period, str):
        text = period.strip()
        if len(text) == 6 and text.isdigit():
            dates = pd.to_datetime(text, format="%m%Y", errors="coerce")
        else:
            dates = pd.to_datetime(text, errors="coerce")
    else:
        dates = pd.to_datetime(period, errors="coerce")
    if pd.isna(dates):
        return ""
    month = int(dates.month)
    year = int(dates.year)
    start = year if month >= 4 else year - 1
    return f"{start}-{(start + 1) % 100:02d}"


def tariff_for(category: str, fiscal_year: str | None = None, period=None) -> FYTariff | None:
    """Look up a ledger tariff, returning ``None`` when the FY/category is unmapped."""
    fy = fiscal_year or fiscal_year_for(period)
    normalized = str(category or "").strip().upper()
    return TARIFF_SCHEDULE.get((fy, normalized))


def _number(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce").fillna(0.0)


def _text(series: pd.Series) -> pd.Series:
    return series.astype("string").fillna("").str.strip().str.upper()


def reconcile_ledger(df: pd.DataFrame) -> pd.DataFrame:
    """Reconcile ledger rows against the FY schedule and the monthly identity.

    ``Expected_Current_Total`` prices the tariff components (energy, ED, fixed,
    demand, subsidy) and keeps ledger-only charges (meter rent, shunt, other,
    current DPS) as recorded. ``Current_Identity_*`` checks the ledger's own
    monthly current-total arithmetic. Arrears are deliberately excluded.
    """
    required = {"CURRENT_TOTAL", "EC_CHRG", "ED_CURRENT"}
    missing = [column for column in required if column not in df.columns]
    if missing:
        raise KeyError("ledger reconciliation requires: " + ", ".join(sorted(missing)))

    out = df.copy()
    if "Billed_Units" not in out.columns:
        out["Billed_Units"] = _number(out.get("UNITS_BILLED", pd.Series(0.0, index=out.index)))
    if "Monthly_Units" not in out.columns:
        out["Monthly_Units"] = _number(out["Billed_Units"])
    if "Billing_Days" not in out.columns:
        days = _number(out.get("NO_MONTHS", pd.Series(30.0, index=out.index)))
        out["Billing_Days"] = days.where(days > 1, 30.0)
    if "Fiscal_Year" not in out.columns:
        out["Fiscal_Year"] = out.get("Date", pd.Series(pd.NaT, index=out.index)).map(fiscal_year_for)
    if "Category" not in out.columns:
        out["Category"] = _text(out.get("TARIFF_ID", pd.Series("", index=out.index)))

    for column in (
        "SANC_LOAD",
        "FIXED_CHRG",
        "DEMAND_CHRG",
        "DEMAND_CHRG_EXC",
        "MRENT",
        "SHUNT_CHRG",
        "OTH_CHRG",
        "GOV_SUBSIDY",
        "DPS_CUR",
        "CURRENT_TOTAL",
        "EC_CHRG",
        "ED_CURRENT",
        "Billed_Units",
        "Billing_Days",
    ):
        if column in out.columns:
            out[column] = _number(out[column])

    ec_values: list[float] = []
    ed_values: list[float] = []
    fixed_values: list[float] = []
    demand_values: list[float] = []
    subsidy_values: list[float] = []
    expected_totals: list[float] = []
    expected_ex_subsidy_totals: list[float] = []
    statuses: list[str] = []
    sources: list[str] = []
    subsidy_sources: list[str] = []

    for _, row in out.iterrows():
        policy = tariff_for(
            str(row.get("Category", row.get("TARIFF_ID", ""))),
            str(row.get("Fiscal_Year", "")),
        )
        if policy is None:
            ec = ed = fixed = demand = subsidy = np.nan
            expected = np.nan
            expected_ex_subsidy = np.nan
            statuses.append("UNMAPPED")
            sources.append("")
            subsidy_sources.append("")
        else:
            load = float(row.get("SANC_LOAD", 1.0) or 1.0)
            days = float(row.get("Billing_Days", 30.0) or 30.0)
            billed = float(row.get("Billed_Units", 0.0) or 0.0)
            units = policy.billable_units(billed, load)
            ec = policy.energy_charge(units)
            ed = round(ec * policy.tax_rate, 2)
            fixed = policy.fixed_charge(load, days)
            demand = policy.demand_charge(load, days)
            subsidy = -policy.subsidy.amount(units)
            ledger_only = (
                float(row.get("DEMAND_CHRG_EXC", 0.0) or 0.0)
                + float(row.get("MRENT", 0.0) or 0.0)
                + float(row.get("SHUNT_CHRG", 0.0) or 0.0)
                + float(row.get("OTH_CHRG", 0.0) or 0.0)
                + float(row.get("DPS_CUR", 0.0) or 0.0)
            )
            expected = ec + ed + fixed + demand + ledger_only + subsidy
            expected_ex_subsidy = (
                ec
                + ed
                + fixed
                + demand
                + ledger_only
                + float(row.get("GOV_SUBSIDY", 0.0) or 0.0)
            )
            statuses.append("OK" if policy.verified else "UNVERIFIED")
            sources.append(policy.source)
            subsidy_sources.append(policy.subsidy.source)

        ec_values.append(ec)
        ed_values.append(ed)
        fixed_values.append(fixed)
        demand_values.append(demand)
        subsidy_values.append(subsidy)
        expected_totals.append(expected)
        expected_ex_subsidy_totals.append(expected_ex_subsidy)

    out["Expected_EC"] = ec_values
    out["Expected_ED"] = ed_values
    out["Expected_Fixed"] = fixed_values
    out["Expected_Demand"] = demand_values
    out["Expected_Subsidy"] = subsidy_values
    out["Expected_Current_Total"] = expected_totals
    out["Expected_Current_Total_With_Ledger_Subsidy"] = expected_ex_subsidy_totals
    out["Tariff_Status"] = statuses
    out["Rate_Source"] = sources
    out["Subsidy_Source"] = subsidy_sources

    out["EC_Delta"] = (out["EC_CHRG"] - out["Expected_EC"]).round(2)
    out["ED_Delta"] = (out["ED_CURRENT"] - out["Expected_ED"]).round(2)
    out["Fixed_Delta"] = (out["FIXED_CHRG"] - out["Expected_Fixed"]).round(2)
    out["Demand_Delta"] = (out["DEMAND_CHRG"] - out["Expected_Demand"]).round(2)
    out["Subsidy_Delta"] = (out["GOV_SUBSIDY"] - out["Expected_Subsidy"]).round(2)
    out["Current_Total_Residual"] = (
        out["CURRENT_TOTAL"] - out["Expected_Current_Total"]
    ).round(2)
    out["Current_Total_With_Ledger_Subsidy_Residual"] = (
        out["CURRENT_TOTAL"] - out["Expected_Current_Total_With_Ledger_Subsidy"]
    ).round(2)

    out["Tariff_Matches"] = (
        out["EC_Delta"].abs().le(TARIFF_EC_TOLERANCE)
        & out["ED_Delta"].abs().le(TARIFF_ED_TOLERANCE)
        & out["Fixed_Delta"].abs().le(LEDGER_TOLERANCE)
        & out["Demand_Delta"].abs().le(LEDGER_TOLERANCE)
    )
    out["Subsidy_Matches"] = out["Subsidy_Delta"].abs().le(TARIFF_SUBSIDY_TOLERANCE)
    out["Current_Total_Matches"] = out["Current_Total_Residual"].abs().le(
        TARIFF_TOTAL_TOLERANCE
    )
    out["Current_Total_With_Ledger_Subsidy_Matches"] = (
        out["Current_Total_With_Ledger_Subsidy_Residual"].abs().le(
            TARIFF_TOTAL_TOLERANCE
        )
    )
    out["Multi_Period"] = out["Billing_Days"] >= 45
    if "Date" in out.columns:
        out["Transition_Row"] = out["Date"].dt.month.eq(4) & out["Multi_Period"].eq(False)
    else:
        out["Transition_Row"] = False
    out["Reconcilable"] = ~out["Multi_Period"] & ~out["Transition_Row"]
    out["Strict_Tariff_Matches"] = out["Tariff_Matches"] & out["Reconcilable"]
    out["Strict_Current_Total_Matches"] = (
        out["Current_Total_With_Ledger_Subsidy_Matches"] & out["Reconcilable"]
    )

    identity = (
        out["EC_CHRG"]
        + out["ED_CURRENT"]
        + out["FIXED_CHRG"]
        + out["DEMAND_CHRG"]
        + out.get("DEMAND_CHRG_EXC", 0.0)
        + out["MRENT"]
        + out.get("SHUNT_CHRG", 0.0)
        + out.get("OTH_CHRG", 0.0)
        + out["GOV_SUBSIDY"]
        + out.get("DPS_CUR", 0.0)
    )
    out["Current_Identity"] = identity.round(2)
    out["Current_Identity_Residual"] = (out["CURRENT_TOTAL"] - identity).round(2)
    out["Current_Identity_Matches"] = out["Current_Identity_Residual"].abs().le(
        LEDGER_IDENTITY_TOLERANCE
    )
    return out

def reprice_ledger(df: pd.DataFrame, mask: pd.Series | None = None) -> pd.DataFrame:
    """Recompute ledger charges for selected rows from the FY schedule.

    The caller is responsible for setting ``Billed_Units``/``UNITS_BILLED``
    (period units) and ``Monthly_Units``/``Units_Consumed`` (normalized units)
    before calling. Rows whose FY/category is unmapped are left untouched.
    """
    out = df.copy()
    if mask is None:
        selected = pd.Series(True, index=out.index, dtype=bool)
    else:
        selected = mask.reindex(out.index, fill_value=False).astype(bool)

    for index in out.index[selected]:
        row = out.loc[index]
        category = str(row.get("Category", row.get("TARIFF_ID", "")))
        fiscal_year = str(row.get("Fiscal_Year", "")) or fiscal_year_for(row.get("Date"))
        policy = tariff_for(category, fiscal_year)
        if policy is None:
            continue

        load = float(row.get("SANC_LOAD", 1.0) or 1.0)
        days = float(row.get("Billing_Days", 30.0) or 30.0)
        if days <= 1:
            days = 30.0
        units = float(row.get("Billed_Units", row.get("UNITS_BILLED", 0.0)) or 0.0)
        units = policy.billable_units(units, load)

        ec = round(policy.energy_charge(units), 2)
        ed = round(ec * policy.tax_rate, 2)
        fixed = round(policy.fixed_charge(load, days), 2)
        demand = round(policy.demand_charge(load, days), 2)
        subsidy = round(-policy.subsidy.amount(units), 2)
        other = (
            float(row.get("DEMAND_CHRG_EXC", 0.0) or 0.0)
            + float(row.get("MRENT", 0.0) or 0.0)
            + float(row.get("SHUNT_CHRG", 0.0) or 0.0)
            + float(row.get("OTH_CHRG", 0.0) or 0.0)
            + float(row.get("DPS_CUR", 0.0) or 0.0)
        )
        current = round(ec + ed + fixed + demand + other + subsidy, 2)

        out.at[index, "EC_CHRG"] = ec
        out.at[index, "ED_CURRENT"] = ed
        if policy.demand_per_kw > 0:
            out.at[index, "DEMAND_CHRG"] = demand
            out.at[index, "FIXED_CHRG"] = 0.0
        else:
            out.at[index, "FIXED_CHRG"] = fixed
            out.at[index, "DEMAND_CHRG"] = 0.0
        out.at[index, "GOV_SUBSIDY"] = subsidy
        out.at[index, "CURRENT_TOTAL"] = current

        charges = (
            ec
            + fixed
            + demand
            + float(row.get("DEMAND_CHRG_EXC", 0.0) or 0.0)
            + float(row.get("MRENT", 0.0) or 0.0)
            + float(row.get("SHUNT_CHRG", 0.0) or 0.0)
            + float(row.get("OTH_CHRG", 0.0) or 0.0)
            + float(row.get("DPS_CUR", 0.0) or 0.0)
        )
        out.at[index, "Bill_Amount"] = round(charges, 2)
        out.at[index, "Tax_Amount"] = ed
        out.at[index, "Fixed_Charges"] = round(fixed + demand, 2)
        out.at[index, "Total_Amount"] = current

    return out
