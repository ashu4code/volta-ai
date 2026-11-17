# Utility Billing AI Agent Specification

This document defines the architecture and operating rules for the
tariff-based electricity bill anomaly engine in this Streamlit workspace.

## 1. Core model

- **Detection:** deterministic meter-defect rules plus unsupervised
  `IsolationForest` scoring.
- **Normalized target:** `Monthly_Units = UNITS_BILLED * 30 / Billing_Days`.
- **Grouping:** per `Consumer_ID` (ledger `CON_ID`); the sample path keeps its
  existing `Consumer_ID` behavior.
- **Artifacts:** `models/meter_anomaly_model.joblib`, rebuilt by `build_model.py`.
- **Dependencies:** `scikit-learn`, `joblib`, `pandas`, `numpy`, `openpyxl`,
  `xlrd`, `streamlit`.

The detector is scale invariant because it scores relative log deviation from
each consumer's median rather than raw units. A robust modified z-score and
the model must agree before a statistical flag is raised; hard spike, hard drop
and dead-meter rules are applied independently.

## 2. Data sources and normalization

- Training data is discovered from
  `consumer_data/ledgers/Consumer_Ledger_Report_*.xls`.
- `consumer_data/sample_consumer_120_rows.xlsx` is a regression fixture and
  fallback only.
- `src/ledger.py` is the single ingestion point:
  - `CON_ID` -> `Consumer_ID`
  - `ACT_NO` -> `Account_No`
  - `MMYYYY` -> zero-padded `Billing_Month`, `Date`, `Fiscal_Year`
  - `NO_MONTHS` -> `Billing_Days` (`30` when `<= 1`)
  - `UNITS_BILLED` -> `Billed_Units` and `Monthly_Units`
  - `READ_STAT`/recorded units -> `Meter_Defect`
- Ledger frames are identified by `CON_ID`, `MMYYYY`, `EC_CHRG` and
  `CURRENT_TOTAL`. The sample CSV/XLSX format must not be routed through the
  ledger normalizer.

The shipped corpus contains 3,943 rows, 83 consumers, 84 ledger files, 10
fiscal years and 3,155 meter-defect rows. Duplicate copied ledger files are
removed so each consumer-month appears once.

The Streamlit app auto-loads this corpus at startup. The chat accepts either
`Consumer_ID` (`CON_ID`) or `Account_No` (`ACT_NO`) for consumer lookup.

## 3. Financial rules

### Workspace sample tariff (Domestic Telescopic Slab)

The only policy for sample/CSV/XLSX billing is the telescopic slab schedule:

```
First 100 units      x Rs 3.00
Next 200 units       x Rs 5.50
Next 200 units       x Rs 8.00
Remaining units      x Rs 10.50
Bill_Amount = slab energy + Fixed_Charges
Tax_Amount  = Bill_Amount * 6%
Total       = Bill_Amount + Tax_Amount
```

The shipped sample's flat Rs 6.50/unit bills are historical and are not
expected to match the workspace policy.

### Ledger tariff

`src/tariff_engine.py` holds FY schedules for DS1, DS1D and KJ. Ledger semantics are:

```
ED_CURRENT = 6% * EC_CHRG
CURRENT_TOTAL = EC_CHRG + ED_CURRENT + FIXED_CHRG + DEMAND_CHRG
              + DEMAND_CHRG_EXC + MRENT + SHUNT_CHRG + OTH_CHRG
              + GOV_SUBSIDY + DPS_CUR
```

Arrears are excluded. Fixed/demand charges are prorated by billing days.
`Tariff_Status=UNVERIFIED` marks inferred fixed charges or subsidy rates; these
must be visible rather than silently forced to match.

Ten-year regression results:

- Current-total identity exact: 3,488/3,943.
- Strict tariff components: 2,757/3,281 reconcilable rows.
- Current total with ledger subsidy: 3,097/3,943; strict: 2,807/3,281.
- Status counts: `OK=3,082`, `UNVERIFIED=498`, `UNMAPPED=363`.

## 4. Workspace commands

| Prompt | Backend behavior |
| --- | --- |
| `<consumer id>` | Finds `Consumer_ID` or `Account_No` and stores the selection |
| `bill anomaly` | Lists month-wise anomalies for the selected consumer |
| `bill anomaly <id>` | Resolves a consumer and lists its month-wise anomalies |
| `scan anomalies` | Runs defect rules and the model; reports counts, sources and bills |
| `reconcile bills` | Runs `reconcile_ledger` or the sample `reconcile_bills` path |
| `apply corrections` | Rewrites model anomalies to baseline and reprices those rows only |
| `show summary` | Reports rows, consumers, FYs, billing, defects and model counts |

Consumer details include identifiers, name/address, tariff/load, period,
consumption, billing, tariff status and exception counts. `bill anomaly`
renders every flagged month with current bill, tariff-expected bill, delta and
diagnostic. Meter-defect rows are never auto-corrected; `apply corrections`
leaves them unchanged for field verification.

## 5. Validation and deployment

```bash
python build_model.py
python run_model.py
python -m unittest discover -s tests -v
streamlit run main.py
```

Required checks:

- `tests/test_pipeline.py` sample regression must remain green.
- `tests/test_app.py` Streamlit smoke tests must remain green.
- `tests/test_ledger.py` and `tests/test_tariff_fy.py` cover normalization,
  FY lookup, reconciliation counts and targeted repricing.
- The engine banner must report `AI ENGINE ONLINE` before commands are used.

Known gaps: KJ fixed charges for FY2023-26, some inferred DS1 subsidies after
FY2019-20, and the full arrears/`NET_AMT` formula.
