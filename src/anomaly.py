"""Meter anomaly detection for utility consumer profiles.

Design notes
------------
The previous detector flagged any row the IsolationForest called "anomalous",
which on the shipped dataset produced 5 false positives out of 6 flags (normal
months at 180-318 units). Two changes fix that:

1. **Per-consumer baselines.** A global median is meaningless when a sheet
   mixes consumers, so every statistic is computed inside a consumer group and
   falls back to the global median for unknown consumers.

2. **Two independent signals must agree.** A row is an anomaly only when the
   robust per-consumer modified z-score breaches the threshold *and* the
   IsolationForest agrees, or when the reading exceeds a hard ratio ceiling
   (a physically impossible spike that no model should miss).

The model itself is fitted on *relative* log deviation from the consumer
baseline rather than on raw units, which makes it scale invariant: a 20x spike
looks the same for a consumer using 200 units/month as for one using 2000.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest

UNITS_COLUMN = "Units_Consumed"
CONSUMER_COLUMN = "Consumer_ID"

BASELINE_COLUMN = "Baseline_Units"
RATIO_COLUMN = "Deviation_Ratio"
TAG_COLUMN = "Model_Tag"
Z_COLUMN = "Robust_Z"
ANOMALY_COLUMN = "Is_Anomaly"
MODEL_ANOMALY_COLUMN = "Is_Model_Anomaly"
METER_DEFECT_ANOMALY_COLUMN = "Is_Meter_Defect"
DETECTION_SOURCE_COLUMN = "Detection_Source"
TYPE_COLUMN = "Anomaly_Type"
REASON_COLUMN = "Anomaly_Reason"
METER_DEFECT_COLUMN = "Meter_Defect"
READ_STATUS_COLUMN = "READ_STAT"
RECORDED_UNITS_COLUMN = "UNITS_CONS"
BILLED_UNITS_COLUMN = "UNITS_BILLED"

#: Modified z-score constant that makes a MAD score comparable to a normal score.
MAD_CONSTANT = 0.6745

#: Below this many rows a consumer group has no usable spread of its own.
MIN_GROUP_SIZE = 3


def _number(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce").fillna(0.0)


@dataclass
class DetectorConfig:
    """Tunables for :class:`MeterAnomalyDetector`."""

    contamination: float = 0.05
    random_state: int = 42
    #: Robust modified z-score beyond which a reading is statistically unusual.
    z_threshold: float = 3.5
    #: Reading this many times the consumer baseline is always a fault.
    hard_ratio: float = 5.0
    #: Readings at or below this are treated as a dead meter.
    min_units: float = 0.0

    def to_dict(self) -> dict:
        return asdict(self)


class MeterAnomalyDetector:
    """Finds faulty meter readings in a consumer billing dataframe."""

    def __init__(self, config: DetectorConfig | None = None):
        self.config = config or DetectorConfig()
        self.model: IsolationForest | None = None
        self.training_baselines: dict[str, float] = {}
        self.training_mad: dict[str, float] = {}
        self.training_summary: dict = {}

    # ------------------------------------------------------------------ utils
    @staticmethod
    def _require_columns(df: pd.DataFrame) -> None:
        if UNITS_COLUMN not in df.columns:
            raise KeyError(
                f"column '{UNITS_COLUMN}' is required; found {list(df.columns)}"
            )

    def baseline(self, df: pd.DataFrame) -> pd.Series:
        """Public per-row baseline (median units) for the owning consumer."""
        return self._baseline_series(df)

    @staticmethod
    def _units(df: pd.DataFrame) -> pd.Series:
        return pd.to_numeric(df[UNITS_COLUMN], errors="coerce").fillna(0.0).astype(float)

    @staticmethod
    def _valid_mask(df: pd.DataFrame) -> pd.Series:
        """Rows with a usable units value for fitting a baseline."""
        if UNITS_COLUMN not in df.columns:
            return pd.Series(False, index=df.index, dtype=bool)
        return df[UNITS_COLUMN].notna()

    def _baseline_series(self, df: pd.DataFrame) -> pd.Series:
        """Per-row baseline (median units) for the owning consumer."""
        valid = self._valid_mask(df)
        source = df.loc[valid] if valid.any() else df
        units = self._units(source)
        global_median = float(units.median()) if len(units) else 0.0
        if CONSUMER_COLUMN not in source.columns:
            return pd.Series(global_median, index=df.index, dtype=float)
        medians = source.groupby(CONSUMER_COLUMN, observed=True)[UNITS_COLUMN].median()
        mapped = df[CONSUMER_COLUMN].map(medians).astype(float)
        return mapped.fillna(global_median)

    def _live_mad(self, df: pd.DataFrame, baseline: pd.Series) -> pd.Series:
        """Median absolute deviation around the per-row baseline, per consumer."""
        valid = self._valid_mask(df)
        source = df.loc[valid] if valid.any() else df
        units = self._units(source)
        deviation = (units - baseline.loc[source.index]).abs()
        if CONSUMER_COLUMN not in source.columns:
            value = float(deviation.median()) if len(deviation) else 0.0
            return pd.Series(value, index=df.index, dtype=float)
        grouped = (
            pd.DataFrame({"_dev": deviation, "_c": source[CONSUMER_COLUMN]})
            .groupby("_c", observed=True)["_dev"]
            .transform("median")
            .astype(float)
        )
        return grouped.reindex(df.index)

    def _mad_series(self, df: pd.DataFrame, baseline: pd.Series) -> pd.Series:
        """MAD per consumer, falling back to the learned spread for tiny groups.

        A consumer with fewer than ``MIN_GROUP_SIZE`` rows has no measurable
        spread of its own, so the MAD captured during training is reused instead
        of collapsing to zero (which would silently disable the z-score gate).
        """
        mad = self._live_mad(df, baseline)
        if CONSUMER_COLUMN not in df.columns:
            return mad.fillna(0.0)

        sizes = df.groupby(CONSUMER_COLUMN, observed=True)[UNITS_COLUMN].transform("size")
        learned = df[CONSUMER_COLUMN].map(self.training_mad).astype(float)
        mad = mad.where(sizes >= MIN_GROUP_SIZE, learned)
        return mad.fillna(mad.median() if mad.notna().any() else 0.0).fillna(0.0)

    @staticmethod
    def _features(units: np.ndarray, baseline: np.ndarray) -> np.ndarray:
        """Relative log deviation from the baseline (scale invariant)."""
        safe_units = np.log1p(np.maximum(units, 0.0))
        safe_baseline = np.log1p(np.maximum(baseline, 0.0))
        return (safe_units - safe_baseline).reshape(-1, 1)

    # ------------------------------------------------------------------- train
    def fit(self, df: pd.DataFrame) -> "MeterAnomalyDetector":
        """Fit the IsolationForest on relative deviations from the baseline."""
        self._require_columns(df)
        valid = self._valid_mask(df)
        train = df.loc[valid] if valid.any() else df
        units = self._units(train).to_numpy(dtype=float)
        baseline = self._baseline_series(train).to_numpy(dtype=float)

        self.model = IsolationForest(
            contamination=self.config.contamination,
            random_state=self.config.random_state,
        ).fit(self._features(units, baseline))

        if CONSUMER_COLUMN in train.columns:
            medians = train.groupby(CONSUMER_COLUMN, observed=True)[UNITS_COLUMN].median()
            self.training_baselines = {str(k): float(v) for k, v in medians.items()}
            learned_mad = self._live_mad(train, self._baseline_series(train))
            self.training_mad = {
                str(k): float(v)
                for k, v in learned_mad.groupby(
                    train[CONSUMER_COLUMN], observed=True
                ).median().items()
            }
        else:
            self.training_baselines = {"__global__": float(np.median(units))}
            self.training_mad = {
                "__global__": float(np.median(np.abs(units - np.median(units))))
            }

        self.training_summary = {
            "rows": int(len(df)),
            "training_rows": int(len(train)),
            "excluded_missing_units": int((~valid).sum()),
            "consumers": int(df[CONSUMER_COLUMN].nunique()) if CONSUMER_COLUMN in df.columns else 1,
            "config": self.config.to_dict(),
        }
        return self

    # ----------------------------------------------------------------- detect
    def detect(
        self,
        df: pd.DataFrame,
        z_threshold: float | None = None,
        baseline_override: float | None = None,
    ) -> pd.DataFrame:
        """Return a copy of ``df`` annotated with anomaly diagnostics.

        ``baseline_override`` forces a single baseline for every row, which is
        how a single ad-hoc reading is scored against a known consumer.
        """
        self._require_columns(df)
        threshold = self.config.z_threshold if z_threshold is None else float(z_threshold)

        out = df.copy()
        units = self._units(out)
        if baseline_override is None:
            baseline = self._baseline_series(out)
        else:
            baseline = pd.Series(float(baseline_override), index=out.index, dtype=float)
        mad = self._mad_series(out, baseline)

        unit_values = units.to_numpy(dtype=float)
        baseline_values = baseline.to_numpy(dtype=float)

        ratio = np.divide(
            unit_values,
            baseline_values,
            out=np.zeros_like(unit_values),
            where=baseline_values > 0,
        )
        safe_mad = np.where(mad.to_numpy(dtype=float) > 0, mad.to_numpy(dtype=float), np.nan)
        z = np.divide(
            MAD_CONSTANT * (unit_values - baseline_values),
            safe_mad,
            out=np.zeros_like(unit_values),
            where=~np.isnan(safe_mad),
        )

        if self.model is not None:
            tags = self.model.predict(self._features(unit_values, baseline_values))
        else:
            tags = np.ones(len(out), dtype=int)
        model_flag = tags == -1

        stat_flag = np.abs(z) >= threshold
        hard_spike = ratio >= self.config.hard_ratio
        hard_drop = (ratio > 0) & (ratio <= 1.0 / self.config.hard_ratio)
        dead = unit_values <= self.config.min_units

        meter_defect = np.zeros(len(out), dtype=bool)
        if METER_DEFECT_COLUMN in out.columns:
            meter_defect |= out[METER_DEFECT_COLUMN].fillna(False).astype(bool).to_numpy()
        if READ_STATUS_COLUMN in out.columns:
            status = out[READ_STATUS_COLUMN].astype(str).str.upper().to_numpy()
            meter_defect |= np.isin(status, ["MD", "LK"])

        model_anomaly = (
            (model_flag & stat_flag) | hard_spike | hard_drop | dead
        ) & ~meter_defect
        is_anomaly = model_anomaly | meter_defect

        out[BASELINE_COLUMN] = np.round(baseline_values, 2)
        out[RATIO_COLUMN] = np.round(ratio, 3)
        out[TAG_COLUMN] = tags
        out[Z_COLUMN] = np.round(z, 3)
        out[MODEL_ANOMALY_COLUMN] = model_anomaly
        out[METER_DEFECT_ANOMALY_COLUMN] = meter_defect
        out[ANOMALY_COLUMN] = is_anomaly
        out[DETECTION_SOURCE_COLUMN] = np.where(
            meter_defect & model_anomaly,
            "both",
            np.where(meter_defect, "meter_defect", np.where(model_anomaly, "model", "")),
        )
        out[TYPE_COLUMN] = ""
        out[REASON_COLUMN] = ""

        rule_type = np.where(dead, "DEAD", np.where(ratio >= 1, "SPIKE", "DROP"))
        anomaly_types = np.where(
            meter_defect,
            "METER_DEFECT",
            np.where(model_anomaly, rule_type, ""),
        )
        out[TYPE_COLUMN] = anomaly_types

        if READ_STATUS_COLUMN in out.columns:
            statuses = out[READ_STATUS_COLUMN].astype(str).str.upper().fillna("").tolist()
        else:
            statuses = [""] * len(out)
        if RECORDED_UNITS_COLUMN in out.columns:
            recorded = _number(out[RECORDED_UNITS_COLUMN]).to_numpy(dtype=float)
        else:
            recorded = np.zeros(len(out), dtype=float)
        if BILLED_UNITS_COLUMN in out.columns:
            billed = _number(out[BILLED_UNITS_COLUMN]).to_numpy(dtype=float)
        else:
            billed = unit_values

        reasons: list[str] = []
        for flagged, defect, model_hit, unit, base, r, zz, kind, agreed, status, rec, bill in zip(
            is_anomaly,
            meter_defect,
            model_anomaly,
            unit_values,
            baseline_values,
            ratio,
            z,
            anomaly_types,
            model_flag,
            statuses,
            recorded,
            billed,
        ):
            if not flagged:
                reasons.append("")
            elif defect:
                label = status if status in ("MD", "LK") else "defect"
                text = (
                    f"meter reading {label}; recorded {rec:.0f} units, "
                    f"billed {bill:.0f} units"
                )
                if model_hit:
                    text += "; model also flagged the billed units"
                reasons.append(text)
            elif kind == "DEAD":
                reasons.append(f"meter reported {unit:.0f} units (dead/zero reading)")
            elif kind == "SPIKE":
                reasons.append(
                    f"{r:.1f}x the baseline of {base:.0f} units "
                    f"(robust z={zz:.1f}, model={'agree' if agreed else 'ratio rule'})"
                )
            else:
                reasons.append(
                    f"only {r:.2f}x the baseline of {base:.0f} units "
                    f"(robust z={zz:.1f}, suspected under-recording)"
                )
        out[REASON_COLUMN] = reasons
        return out

    def anomalies(self, df: pd.DataFrame, z_threshold: float | None = None) -> pd.DataFrame:
        """Only the rows flagged as anomalies."""
        report = self.detect(df, z_threshold=z_threshold)
        return report[report[ANOMALY_COLUMN]].reset_index(drop=True)

    # --------------------------------------------------------------- evaluate
    def evaluate(self, df: pd.DataFrame, truth_column: str = "Meter_Status") -> dict:
        """Score the detector against a ground-truth status column.

        Returns true positives, false positives and false negatives. A row is
        considered faulty when ``truth_column`` is ``FAULTY`` (case-insensitive).
        """
        report = self.detect(df)
        if truth_column not in df.columns:
            return {"available": False}

        truth = df[truth_column].astype(str).str.upper().eq("FAULTY").to_numpy()
        flagged = report[ANOMALY_COLUMN].to_numpy()
        return {
            "available": True,
            "rows": int(len(df)),
            "actual_faults": int(truth.sum()),
            "true_positives": int((flagged & truth).sum()),
            "false_positives": int((flagged & ~truth).sum()),
            "false_negatives": int((~flagged & truth).sum()),
            "flagged": int(flagged.sum()),
        }

    # ---------------------------------------------------------------- persist
    def __setstate__(self, state: dict) -> None:
        """Tolerate artifacts pickled before a field existed."""
        self.__dict__.update(state)
        for attribute, default in (
            ("training_baselines", {}),
            ("training_mad", {}),
            ("training_summary", {}),
        ):
            self.__dict__.setdefault(attribute, default)

    def save(self, path: str | Path) -> Path:
        path = Path(path)
        joblib.dump(self, path)
        return path

    @classmethod
    def load(cls, path: str | Path) -> "MeterAnomalyDetector":
        obj = joblib.load(Path(path))
        if not isinstance(obj, MeterAnomalyDetector):
            raise TypeError(
                f"'{path}' does not contain a MeterAnomalyDetector; "
                "re-run build_model.py to regenerate the artifact"
            )
        return obj
