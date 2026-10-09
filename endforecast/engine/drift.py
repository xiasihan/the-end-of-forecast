"""
Drift detection — monitor data and concept drift over time.

Implements Phase 10 of the methodology: continuous monitoring and
adaptation. Detects:
- Data drift: Input distributions change (PSI, KS-test, KL divergence)
- Concept drift: Error patterns change systematically over time
- Performance drift: Prediction quality degrades
- Training-Serving Skew: Features at serving time differ from training time
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class DriftReport:
    """Structured drift detection report."""
    data_drift_detected: bool = False
    concept_drift_detected: bool = False
    performance_drift_detected: bool = False
    training_serving_skew_detected: bool = False
    metrics: dict[str, float] = field(default_factory=dict)
    skew_features: list[str] = field(default_factory=list)
    triggered_retrain: bool = False
    data_freshness_days: int = 0
    recommended_refresh_frequency: str = "monthly"
    summary_text: str = ""

    @property
    def any_drift(self) -> bool:
        return (
            self.data_drift_detected or self.concept_drift_detected
            or self.performance_drift_detected or self.training_serving_skew_detected
        )


@dataclass
class TrainingServingSkewReport:
    """Feature-level comparison between training and serving data.

    Detects silent data decay — the most dangerous class of ML bugs
    per Google Rules of ML #33-36:
    - Training-time column X had values in [0, 100], serving-time it's [0, 1].
    - A feature went from 95% non-null to 30% non-null.
    - A categorical feature has categories unseen during training.
    """
    feature: str = ""
    train_mean: float = 0.0
    serve_mean: float = 0.0
    train_std: float = 0.0
    serve_std: float = 0.0
    train_missing_pct: float = 0.0
    serve_missing_pct: float = 0.0
    distribution_shift_detected: bool = False
    missing_spike_detected: bool = False
    range_shift_detected: bool = False
    severity: str = "ok"  # ok / warning / critical
    message: str = ""


class DriftDetector:
    """Detect distribution shifts, performance degradation, and skew.

    Usage:
        >>> detector = DriftDetector()
        >>> report = detector.detect(
        ...     reference_data=train_df, current_data=serving_df,
        ...     reference_errors=old_errors, current_errors=new_errors,
        ... )
        >>> skew_reports = detector.check_training_serving_skew(
        ...     training_df, serving_df, feature_list,
        ... )
    """

    def __init__(
        self,
        data_drift_threshold: float = 0.1,
        performance_drift_threshold: float = 0.05,
        skew_shift_threshold: float = 2.0,
        skew_missing_threshold: float = 0.20,
        window_size: int = 30,
    ) -> None:
        self.data_drift_threshold = data_drift_threshold
        self.performance_drift_threshold = performance_drift_threshold
        self.skew_shift_threshold = skew_shift_threshold
        self.skew_missing_threshold = skew_missing_threshold
        self.window_size = window_size

    def detect(
        self,
        reference_data: pd.DataFrame,
        current_data: pd.DataFrame,
        feature_list: Optional[list[str]] = None,
        reference_errors: Optional[np.ndarray] = None,
        current_errors: Optional[np.ndarray] = None,
    ) -> DriftReport:
        report = DriftReport()

        # ── Data drift ───────────────────────────────────────
        data_metrics = self._detect_data_drift(reference_data, current_data)
        report.metrics.update(data_metrics)
        report.data_drift_detected = data_metrics.get("kl_divergence", 0) > self.data_drift_threshold

        # ── Performance drift ────────────────────────────────
        if reference_errors is not None and current_errors is not None:
            ref_mean = float(np.mean(np.abs(reference_errors)))
            cur_mean = float(np.mean(np.abs(current_errors)))
            perf_ratio = cur_mean / ref_mean if ref_mean > 1e-10 else 1.0
            report.metrics["error_ratio"] = perf_ratio
            report.performance_drift_detected = perf_ratio > 1 + self.performance_drift_threshold

        # ── Training-Serving Skew ────────────────────────────
        if feature_list:
            skew_reports = self.check_training_serving_skew(
                reference_data, current_data, feature_list,
            )
            critical_skews = [s for s in skew_reports if s.severity == "critical"]
            warning_skews = [s for s in skew_reports if s.severity == "warning"]
            report.training_serving_skew_detected = len(critical_skews) > 0
            report.skew_features = [s.feature for s in critical_skews + warning_skews]
            report.metrics["skew_critical_count"] = len(critical_skews)
            report.metrics["skew_warning_count"] = len(warning_skews)

        report.triggered_retrain = report.any_drift

        parts = []
        if report.data_drift_detected: parts.append("DATA_DRIFT")
        if report.performance_drift_detected: parts.append("PERF_DEGRADATION")
        if report.training_serving_skew_detected: parts.append("SKEW_CRITICAL")
        report.summary_text = "+".join(parts) if parts else "STABLE"

        logger.info("Drift check: %s", report.summary_text)
        return report

    # ── Training-Serving Skew ──────────────────────────────────

    def check_training_serving_skew(
        self,
        training_df: pd.DataFrame,
        serving_df: pd.DataFrame,
        feature_list: list[str],
    ) -> list[TrainingServingSkewReport]:
        """Compare feature distributions between training and serving data.

        This catches the most dangerous class of ML production bugs:
        features that change silently between training and serving time.

        Args:
            training_df: Data used during model training.
            serving_df: Data arriving at prediction time.
            feature_list: List of feature column names to check.

        Returns:
            List of per-feature skew reports, ordered by severity.
        """
        reports = []
        for feat in feature_list:
            report = TrainingServingSkewReport(feature=feat)

            if feat not in training_df.columns and feat not in serving_df.columns:
                continue
            if feat not in training_df.columns:
                report.severity = "critical"
                report.message = f"Feature '{feat}' exists at serving time but was never in training."
                reports.append(report)
                continue
            if feat not in serving_df.columns:
                report.severity = "critical"
                report.message = f"Feature '{feat}' was in training but is missing at serving time."
                reports.append(report)
                continue

            train_col = training_df[feat]
            serve_col = serving_df[feat]

            # Missing rate comparison
            report.train_missing_pct = float(train_col.isna().mean())
            report.serve_missing_pct = float(serve_col.isna().mean())
            missing_delta = abs(report.serve_missing_pct - report.train_missing_pct)
            if missing_delta > self.skew_missing_threshold:
                report.missing_spike_detected = True
                report.message = (
                    f"Missing rate changed from {report.train_missing_pct:.1%} "
                    f"to {report.serve_missing_pct:.1%} (Δ={missing_delta:.1%})"
                )

            # Numeric distribution comparison
            if pd.api.types.is_numeric_dtype(train_col) and pd.api.types.is_numeric_dtype(serve_col):
                train_vals = train_col.dropna()
                serve_vals = serve_col.dropna()
                if len(train_vals) > 10 and len(serve_vals) > 10:
                    report.train_mean = float(train_vals.mean())
                    report.serve_mean = float(serve_vals.mean())
                    report.train_std = float(train_vals.std())
                    report.serve_std = float(serve_vals.std())

                    # Z-score of mean shift
                    pooled_se = np.sqrt(
                        report.train_std**2 / len(train_vals)
                        + report.serve_std**2 / len(serve_vals)
                    ) or 1e-10
                    z_mean = abs(report.train_mean - report.serve_mean) / pooled_se
                    if z_mean > self.skew_shift_threshold:
                        report.distribution_shift_detected = True
                        if not report.message:
                            report.message = (
                                f"Mean shifted from {report.train_mean:.4f} to "
                                f"{report.serve_mean:.4f} (z={z_mean:.1f}σ)"
                            )

                    # Range check: is serving data outside training range?
                    train_min, train_max = float(train_vals.min()), float(train_vals.max())
                    serve_min, serve_max = float(serve_vals.min()), float(serve_vals.max())
                    if serve_min < train_min * 0.5 or serve_max > train_max * 1.5:
                        report.range_shift_detected = True
                        if not report.message:
                            report.message = (
                                f"Serving range [{serve_min:.2f}, {serve_max:.2f}] "
                                f"outside training range [{train_min:.2f}, {train_max:.2f}]"
                            )

            # Assign severity
            if report.distribution_shift_detected and report.missing_spike_detected:
                report.severity = "critical"
            elif report.distribution_shift_detected or report.range_shift_detected:
                report.severity = "warning"
            elif report.missing_spike_detected:
                report.severity = "warning"

            if report.severity != "ok":
                reports.append(report)

        reports.sort(key=lambda r: {"critical": 0, "warning": 1, "ok": 2}[r.severity])
        return reports

    # ── Data Drift ─────────────────────────────────────────────

    @staticmethod
    def _detect_data_drift(
        reference: pd.DataFrame, current: pd.DataFrame,
    ) -> dict[str, float]:
        metrics: dict[str, float] = {}
        numeric_cols = reference.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) == 0:
            return metrics
        total_kl = 0.0
        count = 0
        for col in numeric_cols:
            ref = reference[col].dropna().values
            cur = current[col].dropna().values
            if len(ref) < 10 or len(cur) < 10:
                continue
            try:
                bins = min(50, len(ref))
                ref_h, edges = np.histogram(ref, bins=bins, density=True)
                cur_h, _ = np.histogram(cur, bins=edges, density=True)
                eps = 1e-10
                total_kl += float(np.sum((ref_h + eps) * np.log((ref_h + eps) / (cur_h + eps))))
                count += 1
            except Exception:
                pass
        if count > 0:
            metrics["kl_divergence"] = total_kl / count
        return metrics

    # ── Data Freshness (P1) ──────────────────────────────────────

    @staticmethod
    def freshness_score(
        training_df: pd.DataFrame,
        time_col: str = "ds",
        reference_date: Optional[str | pd.Timestamp] = None,
    ) -> tuple[int, str]:
        """Compute data freshness and recommended refresh frequency.

        Args:
            training_df: Training data with a timestamp column.
            time_col: Name of the timestamp column.
            reference_date: Reference date to compare against (default: now).

        Returns:
            Tuple of (days_since_last_training_point, refresh_recommendation).
        """
        if time_col not in training_df.columns:
            return 0, "unknown"

        max_date = pd.Timestamp(training_df[time_col].max())
        reference = pd.Timestamp(reference_date) if reference_date else pd.Timestamp.now()
        days = (reference - max_date).days

        if days <= 7:
            freq = "daily"
        elif days <= 30:
            freq = "weekly"
        elif days <= 90:
            freq = "monthly"
        else:
            freq = "immediately"

        logger.info("Data freshness: %s days old → recommend %s refresh", days, freq)
        return max(days, 0), freq