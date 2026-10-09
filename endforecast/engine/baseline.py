"""
Baseline Runner — mandatory baseline establishment (Phase 4).

Industry best practice: NEVER skip baselines. A complex model that
barely beats a naive baseline is likely overfit or the data has no
predictive signal. The baseline runner establishes a set of simple
reference models before any complex experimentation begins.

Supported baselines:
- Classification: majority class, stratified random, linear (logistic)
- Regression: mean, median, linear (OLS)
- Time Series: naive (last value), seasonal naive, mean forecast
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd

from endforecast.engine.trial import Trial, TrialResult, TrialState
from endforecast.evaluation.metrics import MetricCalculator, MetricResult

logger = logging.getLogger(__name__)


@dataclass
class BaselineResult:
    """Results from running all applicable baselines.

    The best_baseline provides a floor that any complex model must beat.
    If complex models fail to beat the best baseline, the data likely
    has no exploitable predictive signal or there is data leakage.
    """
    results: list[TrialResult] = field(default_factory=list)
    best: Optional[TrialResult] = None
    metrics_calculator: Optional[MetricCalculator] = None

    @property
    def baseline_floor(self) -> float:
        """The metric value that any complex model MUST beat."""
        if self.best is None:
            return float("inf")
        return self.best.metric_value

    def summary(self) -> str:
        if not self.results:
            return "No baselines run."
        parts = [f"Baselines ({len(self.results)}):"]
        for r in self.results:
            icon = "\u2705" if r.is_successful else "\u274c"
            parts.append(f"  {icon} {r.trial.model}: {r.trial.metric.upper()}={r.metric_value:.4f}")
        if self.best:
            parts.append(f"  → Best baseline: {self.best.trial.model} ({self.best.trial.metric.upper()}={self.best.metric_value:.4f})")
        return "\n".join(parts)


class BaselineRunner:
    """Establish mandatory baselines before any experimentation.

    Usage:
        >>> runner = BaselineRunner()
        >>> result = runner.run(df, target_col="y", time_col="ds", task_type="timeseries")
        >>> print(result.summary())
        >>> print(f"Any complex model must beat: {result.baseline_floor}")
    """

    def __init__(self, metrics_calculator: MetricCalculator | None = None) -> None:
        self.metrics_calc = metrics_calculator or MetricCalculator()

    def run(
        self,
        df: pd.DataFrame,
        target_col: str = "y",
        time_col: Optional[str] = "ds",
        id_col: Optional[str] = None,
        task_type: str = "regression",
        metric: str = "auto",
    ) -> BaselineResult:
        """Run all applicable baselines for the given task type.

        Args:
            df: Training data.
            target_col: Target column name.
            time_col: Time column (required for time series baselines).
            id_col: Series ID column.
            task_type: "classification", "regression", or "timeseries".
            metric: Primary evaluation metric or "auto" for task default.

        Returns:
            BaselineResult with all baseline performances.
        """
        logger.info("Running baselines for task_type=%s", task_type)

        if metric == "auto":
            metric = self.metrics_calc._default_metrics(task_type)[0]

        y = df[target_col].values
        results: list[TrialResult] = []

        # ── Task-specific baselines ─────────────────────────────
        if task_type == "classification":
            results.extend(self._classification_baselines(y, metric))
        elif task_type == "regression":
            results.extend(self._regression_baselines(y, metric))
        elif task_type == "timeseries":
            results.extend(self._timeseries_baselines(y, metric, len(df)))

        # ── Always include a simple linear model ─────────────────
        results.append(self._linear_baseline(df, target_col, time_col, task_type, metric))

        # ── Find best ───────────────────────────────────────────
        direction = self.metrics_calc._DIRECTION.get(metric, "minimize")
        successful = [r for r in results if r.is_successful and not np.isnan(r.metric_value)]

        if successful:
            if direction == "minimize":
                best = min(successful, key=lambda r: r.metric_value)
            else:
                best = max(successful, key=lambda r: r.metric_value)
        else:
            best = None

        result = BaselineResult(
            results=results,
            best=best,
            metrics_calculator=self.metrics_calc,
        )
        logger.info(result.summary())
        return result

    # ── Classification Baselines ────────────────────────────────

    def _classification_baselines(
        self, y: np.ndarray, metric: str,
    ) -> list[TrialResult]:
        """Classification baselines: majority, stratified random."""
        results = []

        # Majority class
        from collections import Counter
        majority_label = Counter(y.tolist()).most_common(1)[0][0]
        y_majority = np.full_like(y, majority_label, dtype=float)
        acc = float(np.mean(y_majority == y))

        results.append(TrialResult(
            trial=Trial(id="baseline_majority", model="majority_class", task_type="classification", metric=metric),
            state=TrialState.COMPLETED, metric_value=acc,
            metrics_detail={"accuracy": acc},
        ))

        # Stratified random
        from collections import Counter
        class_probs = {k: v / len(y) for k, v in Counter(y.tolist()).items()}
        y_random = np.random.choice(list(class_probs.keys()), size=len(y), p=list(class_probs.values()))
        acc_random = float(np.mean(y_random == y))

        results.append(TrialResult(
            trial=Trial(id="baseline_random", model="stratified_random", task_type="classification", metric=metric),
            state=TrialState.COMPLETED, metric_value=acc_random,
            metrics_detail={"accuracy": acc_random},
        ))

        return results

    # ── Regression Baselines ────────────────────────────────────

    def _regression_baselines(
        self, y: np.ndarray, metric: str,
    ) -> list[TrialResult]:
        """Regression baselines: mean, median."""
        results = []

        # Mean
        y_mean = np.full_like(y, np.mean(y), dtype=float)
        mae_mean = float(np.mean(np.abs(y - y_mean)))

        results.append(TrialResult(
            trial=Trial(id="baseline_mean", model="mean", task_type="regression", metric=metric),
            state=TrialState.COMPLETED, metric_value=mae_mean,
            metrics_detail={"mae": mae_mean},
        ))

        # Median
        y_median = np.full_like(y, np.median(y), dtype=float)
        mae_median = float(np.mean(np.abs(y - y_median)))

        results.append(TrialResult(
            trial=Trial(id="baseline_median", model="median", task_type="regression", metric=metric),
            state=TrialState.COMPLETED, metric_value=mae_median,
            metrics_detail={"mae": mae_median},
        ))

        return results

    # ── Time Series Baselines ───────────────────────────────────

    def _timeseries_baselines(
        self, y: np.ndarray, metric: str, n: int,
    ) -> list[TrialResult]:
        """Time series baselines: naive, seasonal naive, mean."""
        results = []

        # Naive (last value / persistence forecast)
        if n > 1:
            y_naive = np.roll(y, shift=1)
            y_naive[0] = y[0]
            mae_naive = float(np.mean(np.abs(y[1:] - y_naive[1:])))
        else:
            mae_naive = 0.0

        results.append(TrialResult(
            trial=Trial(id="baseline_naive", model="naive", task_type="timeseries", metric=metric),
            state=TrialState.COMPLETED, metric_value=mae_naive,
            metrics_detail={"mae": mae_naive},
        ))

        # Seasonal naive (copy from same hour/day of previous cycle)
        season = min(7, max(1, n // 2))  # Default: weekly cycle
        if n > season:
            y_seasonal = np.roll(y, shift=season)
            y_seasonal[:season] = y[:season]
            mae_seasonal = float(np.mean(np.abs(y[season:] - y_seasonal[season:])))
        else:
            mae_seasonal = float("inf")

        results.append(TrialResult(
            trial=Trial(id="baseline_seasonal_naive", model="seasonal_naive", task_type="timeseries", metric=metric),
            state=TrialState.COMPLETED, metric_value=mae_seasonal,
            metrics_detail={"mae": mae_seasonal},
        ))

        # Mean
        y_mean = np.full_like(y, np.mean(y), dtype=float)
        mae_mean = float(np.mean(np.abs(y - y_mean)))

        results.append(TrialResult(
            trial=Trial(id="baseline_mean", model="mean", task_type="timeseries", metric=metric),
            state=TrialState.COMPLETED, metric_value=mae_mean,
            metrics_detail={"mae": mae_mean},
        ))

        return results

    # ── Linear Baseline ─────────────────────────────────────────

    def _linear_baseline(
        self, df: pd.DataFrame, target_col: str, time_col: Optional[str],
        task_type: str, metric: str,
    ) -> TrialResult:
        """Simple linear/logistic regression as a learning baseline."""
        try:
            from sklearn.linear_model import LinearRegression, LogisticRegression
            from sklearn.model_selection import train_test_split

            y = df[target_col].values
            numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
            feature_cols = [c for c in numeric_cols if c != target_col]

            if len(feature_cols) < 1:
                return TrialResult(
                    trial=Trial(id="baseline_linear", model="linear", task_type=task_type, metric=metric),
                    state=TrialState.COMPLETED, metric_value=float("nan"),
                    error_message="No numeric features available.",
                )

            X = df[feature_cols].fillna(0).values
            X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

            if task_type == "classification":
                model = LogisticRegression(max_iter=1000).fit(X_train, y_train)
                y_pred = model.predict_proba(X_val)[:, 1] if model.classes_.size > 1 else model.predict(X_val)
                # Binary threshold at 0.5
                if len(y_pred.shape) > 0 and y_pred.ndim == 1:
                    y_pred_binary = (y_pred >= 0.5).astype(int)
                else:
                    y_pred_binary = y_pred
                score = float(np.mean(y_pred_binary == y_val))
            else:
                model = LinearRegression().fit(X_train, y_train)
                y_pred = model.predict(X_val)
                score = float(np.mean(np.abs(y_val - y_pred)))

            return TrialResult(
                trial=Trial(id="baseline_linear", model="linear", task_type=task_type, metric=metric),
                state=TrialState.COMPLETED, metric_value=score,
                metrics_detail={metric: score},
            )

        except Exception as exc:
            return TrialResult(
                trial=Trial(id="baseline_linear", model="linear", task_type=task_type, metric=metric),
                state=TrialState.FAILED, metric_value=float("nan"),
                error_message=str(exc),
            )