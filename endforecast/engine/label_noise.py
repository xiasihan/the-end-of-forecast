"""
Label noise detection — identify potentially mislabeled training data.

Implements P1 of the prediction problem detection framework.
Uses two complementary approaches:
1. Consistency check: same features, different labels
2. Distribution check: anomalous class balance spikes

Real-world impact: label noise is one of the most common data quality
issues in classification tasks. Even 5% label noise can significantly
degrade model performance (Northcutt et al., 2021).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class LabelNoiseReport:
    """Result of label noise detection."""

    noise_detected: bool = False
    noise_ratio_estimate: float = 0.0
    inconsistent_samples: int = 0
    total_samples: int = 0
    class_imbalance_warning: bool = False
    dominant_class_ratio: float = 0.0
    severity: str = "ok"
    message: str = ""


class LabelNoiseDetector:
    """Detect label noise in classification datasets.

    Usage:
        >>> detector = LabelNoiseDetector()
        >>> report = detector.detect(df, target_col="y")
        >>> if report.noise_detected:
        ...     print(report.message)
    """

    def __init__(
        self,
        imbalance_threshold: float = 0.95,
        inconsistency_threshold: float = 0.05,
    ) -> None:
        self.imbalance_threshold = imbalance_threshold
        self.inconsistency_threshold = inconsistency_threshold

    def detect(
        self,
        df: pd.DataFrame,
        target_col: str = "y",
        feature_cols: Optional[list[str]] = None,
    ) -> LabelNoiseReport:
        """Detect label noise in a dataset.

        Args:
            df: Input DataFrame.
            target_col: Target column name.
            feature_cols: Feature columns to check for inconsistency.
                          If None, uses all numeric columns.

        Returns:
            LabelNoiseReport with detection results.
        """
        report = LabelNoiseReport(total_samples=len(df))

        if target_col not in df.columns:
            return report

        y = df[target_col].dropna()
        if len(y) < 10:
            return report

        # ── Class Imbalance Check ──────────────────────────────
        class_counts = y.value_counts()
        if len(class_counts) > 0:
            dominant_ratio = class_counts.iloc[0] / len(y)
            report.dominant_class_ratio = dominant_ratio
            if dominant_ratio > self.imbalance_threshold:
                report.class_imbalance_warning = True
                report.severity = "warning"
                report.message = (
                    f"Dominant class comprises {dominant_ratio:.1%} of labels. "
                    f"Model may learn to always predict the majority class."
                )

        # ── Consistency Check ──────────────────────────────────
        if feature_cols is None:
            feature_cols = list(df.select_dtypes(include=[np.number]).columns)
            feature_cols = [c for c in feature_cols if c != target_col]

        if len(feature_cols) > 0 and len(feature_cols) <= 5:
            # Group by all feature columns and check label consistency
            group_cols = [c for c in feature_cols if c in df.columns]
            if group_cols:
                groups = df.groupby(group_cols)[target_col].nunique()
                inconsistent_groups = (groups > 1).sum()
                if inconsistent_groups > 0:
                    report.inconsistent_samples = int(inconsistent_groups)

                    # Crude noise ratio estimate
                    total_groups = len(groups)
                    raw_ratio = inconsistent_groups / total_groups if total_groups > 0 else 0
                    # Apply a conservative scaling factor (not all inconsistency = noise)
                    report.noise_ratio_estimate = round(raw_ratio * 0.3, 4)

                    if report.noise_ratio_estimate > self.inconsistency_threshold:
                        report.noise_detected = True
                        report.severity = "warning"
                        existing = report.message
                        report.message = (
                            f"{existing} | "
                            f"Label inconsistency: {inconsistent_groups}/{total_groups} "
                            f"groups have mixed labels. Estimated noise ~{report.noise_ratio_estimate:.1%}."
                        ).strip()

        if report.noise_detected or report.class_imbalance_warning:
            logger.info("Label noise: %s", report.message)

        return report