# Utility Billing AI Workspace

A Streamlit workspace for detecting faulty electricity meter readings and
reconciling consumer bills against SBPDCL/BERC tariffs.

## Layout

| File | Role |
| --- | --- |
| `main.py` | Streamlit app: file loader, editable grid, copilot console |
| `src/ledger.py` | Consumer-ledger ingestion, period normalization, defect flags |
| `src/tariff_engine.py` | Rupee engine: telescopic sample policy plus FY ledger schedules |
| `src/anomaly.py` | Meter-defect rules plus per-consumer IsolationForest detector |
| `src/__init__.py` | Makes `src` an importable package |
| `build_model.py` | Trains the detector and writes `models/meter_anomaly_model.joblib` |
| `run_model.py` | Command line scan for ledgers or the sample fixture |
| `consumer_data/ledgers/Consumer_Ledger_Report_*.xls` | 84 SBPDCL consumer-ledger exports |
| `consumer_data/sample_consumer_120_rows.xlsx` | 120-row regression fixture (`CON-9841`) |
| `consumer_data/con.csv` | 4-row sample profile (`CON-101`) containing a faulty meter |
| `models/meter_anomaly_model.joblib` | Compiled detector artifact |
| `tests/` | Stdlib `unittest` suite (no pytest required) |

> **Local data note:** `consumer_data/ledgers/` contains real customer PII and
> is intentionally excluded from the repository. Ledger-dependent tests and the
> app's startup auto-load expect those `.xls` files to be supplied locally;
> otherwise upload a ledger through the app. `consumer_data/con.csv` is also
> excluded because Windows reserves the `CON` filename, so
> `tests/test_pipeline.py` needs that fixture supplied locally.

## Running

```bash
pip install -r requirements.txt
python build_model.py            # compile the detector artifact
streamlit run main.py            # launch the workspace
python run_model.py --readings 240 4950   # command line scan
python -m unittest discover -s tests -v   # run the test suite
```

The app accepts `.csv`, `.xlsx` and `.xls`. It auto-loads the ledger corpus
from `consumer_data/ledgers/` at startup, and an upload overrides that buffer.
Ledger exports are detected by their raw columns (`CON_ID`, `MMYYYY`,
`EC_CHRG`, `CURRENT_TOTAL`) and routed through `src/ledger.py`; the sample
CSV/XLSX path is unchanged.

Consumer lookup accepts either the numeric `Consumer_ID` (`CON_ID`) or the
account code in `Account_No` (`ACT_NO`). Enter one in the chat, for example
`23030008718` or `DUDOB14460`, to inspect the profile.

## Ledger normalization

- `Consumer_ID` is `CON_ID`; `ACT_NO` is retained as `Account_No`.
- `MMYYYY` is zero-padded to six digits and converted to `Date` and
  `Fiscal_Year` (April-March).
- `Billing_Days = 30` when `NO_MONTHS <= 1`, otherwise `NO_MONTHS`.
- `Monthly_Units = UNITS_BILLED * 30 / Billing_Days`; the detector uses this
  normalized value while bill repricing uses `UNITS_BILLED`.
- A row is a meter defect when `READ_STAT` is `MD`/`LK`, or when recorded units
  are zero while billed units are positive.

The corpus contains 3,943 rows across 83 consumers, 84 ledger files and 10
financial years; 3,155 rows are rule-based meter defects. Duplicate copied
ledger files have been removed, so each consumer-month appears once.

## Detection

Two signals are reported separately:

1. **Meter-defect rules** from the ledger's own `READ_STAT` / recorded units.
2. **IsolationForest** on relative log deviation from each consumer's median,
   combined with a robust modified z-score. A reading is also flagged when it
   exceeds 5x its baseline, falls below one-fifth of baseline, or reports zero
   units.

A model flag never overrides a meter defect. `apply corrections` rewrites only
model anomalies to the consumer baseline and reprices those rows; defect rows
are left untouched for field verification.

On the ledger corpus this yields 3,155 meter defects and 171 model anomalies. On
the shipped 120-row sample the detector yields **1 true positive, 0 false
positives** against `Meter_Status`.

## Tariff engine

The workspace uses only the **Domestic Telescopic Slab** policy for
sample/CSV/XLSX billing:

```
First 100 units      x Rs 3.00
Next 200 units       x Rs 5.50
Next 200 units       x Rs 8.00
Remaining units      x Rs 10.50
Bill_Amount = slab energy + Fixed_Charges
Tax_Amount  = Bill_Amount * 6%
Total       = Bill_Amount + Tax_Amount
```

Each block is billed at its own rate. The shipped sample's historical bills were
computed with a flat Rs 6.50/unit tariff, so they are not expected to match the
workspace policy exactly.

Ledger bills use the FY schedule in `TARIFF_SCHEDULE` for DS1, DS1D and KJ from
FY2016-17 through FY2025-26. Ledger semantics differ from the sample: `ED` is 6%
of the energy charge and fixed/demand charges are prorated by billing days.
`reconcile bills` checks both the tariff components and the monthly
`CURRENT_TOTAL` identity (arrears excluded).

Current regression results on the ledger corpus:

- Current-total identity: **3,488/3,943** rows exact.
- Strict tariff components: **2,757/3,281** reconcilable rows.
- Current total using the ledger's own subsidy: **3,097/3,943** rows.
- Strict current total: **2,807/3,281** reconcilable rows.
- Status counts: `OK=3,082`, `UNVERIFIED=498`, `UNMAPPED=363`.

`Tariff_Status` marks FY/category combinations whose fixed charge or subsidy is
still inferred (`UNVERIFIED`) or has no mapped FY schedule (`UNMAPPED`) rather
than silently forcing a match.

## Volta AI commands

| Command | Effect |
| --- | --- |
| `<consumer id>` | Shows the profile for a `Consumer_ID` or `Account_No` |
| `bill anomaly` | Lists every flagged month for the selected consumer |
| `bill anomaly <id>` | Selects a consumer and lists that consumer's flagged months |
| `scan anomalies` | Separates meter defects from model anomalies and shows current bills |
| `reconcile bills` | Compares rows against the FY tariff and monthly current-total identity |
| `apply corrections` | Rewrites model anomalies to baseline and reprices only those rows |
| `show summary` | Row, consumer, consumption, billing and detection totals |

## Known caveats

- KJ fixed charges for FY2023-26 are not exercised by the current ledgers.
- Some DS1 subsidy rates after FY2019-20 are inferred; they are marked
  `UNVERIFIED`.
- Arrears and the full `NET_AMT` formula are intentionally out of scope.
