"""
Data Explorer Agent — automatic data profiling and task identification.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd

from endforecast.engine.fingerprint import FeatureFingerprint, FingerprintCalculator
from endforecast.engine.label_noise import LabelNoiseDetector, LabelNoiseReport

logger = logging.getLogger(__name__)


@dataclass
class ExplorationReport:
    """Structured output from the DataExplorer agent."""
    task_type: str
    fingerprint: FeatureFingerprint
    n_rows: int
    n_cols: int
    time_column: Optional[str] = None
    target_column: str = "y"
    id_column: Optional[str] = None
    missing_ratio: float = 0.0
    duplicate_ratio: float = 0.0
    quality_flags: list[str] = field(default_factory=list)
    suggested_models: list[str] = field(default_factory=list)
    suggested_metrics: list[str] = field(default_factory=list)
    suggested_features: list[str] = field(default_factory=list)
    # Risk detection reports (P1)
    label_noise_report: Optional[LabelNoiseReport] = None
    hierarchy_detected: bool = False

    def summary(self) -> str:
        flags = ", ".join(self.quality_flags) if self.quality_flags else "clean"
        extra = []
        if self.label_noise_report and self.label_noise_report.noise_detected:
            extra.append("label_noise")
        if self.hierarchy_detected:
            extra.append("hierarchy")
        if self.fingerprint.tags.get("is_cold_start"):
            extra.append("cold_start")
        extras = f" [{', '.join(extra)}]" if extra else ""
        return f"Task={self.task_type} | {self.n_rows}r × {self.n_cols}c | miss={self.missing_ratio:.1%} | {flags}{extras}"


class DataExplorer:
    """First-pass data profiling and task identification.

    Usage:
        >>> explorer = DataExplorer()
        >>> report = explorer.explore(df, time_col="ds", target_col="y")
    """

    def __init__(self) -> None:
        self.calculator = FingerprintCalculator()

    def explore(
        self, df: pd.DataFrame, time_col: Optional[str] = "ds",
        target_col: str = "y", id_col: Optional[str] = "unique_id",
    ) -> ExplorationReport:
        if target_col not in df.columns:
            raise ValueError(f"Target column '{target_col}' not found.")

        n_rows, n_cols = len(df), len(df.columns)
        target = df[target_col].dropna()
        has_time = bool(time_col and time_col in df.columns)

        task_type = self._detect_task_type(target, has_time)
        missing_ratio = float(df[target_col].isna().mean())
        duplicate_ratio = float(df.duplicated().mean())

        quality_flags = []
        if missing_ratio > 0.1:
            quality_flags.append(f"high_missing({missing_ratio:.0%})")
        if duplicate_ratio > 0.05:
            quality_flags.append(f"duplicates({duplicate_ratio:.0%})")

        fingerprint = self.calculator.calculate(df, time_col=time_col, target_col=target_col, id_col=id_col)
        suggestions = self._make_suggestions(task_type, fingerprint)

        # ── P1: Label noise detection (classification only) ──
        label_noise_report = None
        if task_type == "classification":
            detector = LabelNoiseDetector()
            label_noise_report = detector.detect(df, target_col=target_col)
            if label_noise_report.noise_detected:
                quality_flags.append(f"label_noise(~{label_noise_report.noise_ratio_estimate:.0%})")

        # ── P2: Hierarchy detection ────────────────────────
        hierarchy_detected = False
        if id_col and id_col in df.columns:
            # Check for potential hierarchical columns (categorical with < 50 unique values)
            cat_cols = df.select_dtypes(include=["object", "category"]).columns
            cat_cols = [c for c in cat_cols if c not in {target_col, id_col, time_col}]
            # Simple heuristic: if multiple categorical columns with different cardinalities exist
            if len(cat_cols) >= 2:
                hierarchy_detected = True
                quality_flags.append("hierarchy_detected")

        return ExplorationReport(
            task_type=task_type, fingerprint=fingerprint,
            n_rows=n_rows, n_cols=n_cols,
            time_column=time_col, target_column=target_col, id_column=id_col,
            missing_ratio=missing_ratio, duplicate_ratio=duplicate_ratio,
            quality_flags=quality_flags, **suggestions,
            label_noise_report=label_noise_report,
            hierarchy_detected=hierarchy_detected,
        )

    def _detect_task_type(self, target: pd.Series, has_time: bool) -> str:
        unique = target.nunique()
        if unique <= 10 or pd.api.types.is_bool_dtype(target) or pd.api.types.is_categorical_dtype(target):
            return "classification"
        if has_time:
            return "timeseries"
        return "regression"

    def _make_suggestions(self, task_type: str, fp: FeatureFingerprint) -> dict:
        s = {"suggested_models": [], "suggested_metrics": [], "suggested_features": []}
        if task_type == "classification":
            s["suggested_models"] = ["logistic", "lightgbm", "xgboost", "random_forest"]
            s["suggested_metrics"] = ["accuracy", "f1", "auc"]
            # Calendar features when time column exists
            s["suggested_features"].extend(["hour", "day_of_week"])
            if fp.seasonal_strength > 0.2:
                s["suggested_features"].extend(["month"])
            if abs(fp.acf1) > 0.3:
                s["suggested_features"].extend(["lag_24"])
        elif task_type == "regression":
            s["suggested_models"] = ["ridge", "lightgbm", "xgboost", "random_forest"]
            s["suggested_metrics"] = ["mae", "mape", "rmse"]
            s["suggested_features"].extend(["hour", "day_of_week"])
            if fp.seasonal_strength > 0.2:
                s["suggested_features"].extend(["month"])
            if abs(fp.acf1) > 0.3:
                s["suggested_features"].extend(["lag_24", "lag_168"])
        else:
            s["suggested_models"] = ["ridge", "lightgbm", "xgboost", "random_forest"]
            s["suggested_metrics"] = ["mase", "smape"]
            s["suggested_features"].extend(["lag_24", "lag_168", "hour", "day_of_week", "month"])
        return s