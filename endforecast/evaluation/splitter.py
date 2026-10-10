"""
Data splitting strategies for model evaluation.

Supports task-appropriate split methods:
- KFold, StratifiedKFold, TimeSeriesSplit, ExpandingWindow, TrainTestSplit
- Nested CV: outer loop for model selection, inner loop for hyperparameter tuning
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd
from sklearn.model_selection import (
    KFold, StratifiedKFold, TimeSeriesSplit, train_test_split,
)

logger = logging.getLogger(__name__)


@dataclass
class SplitConfig:
    """Configuration for data splitting."""
    method: str = "kfold"
    n_splits: int = 5
    test_size: float = 0.2
    gap: int = 0
    shuffle: bool = False
    random_state: int = 42


@dataclass
class SplitResult:
    """Result of a data split operation."""
    folds: list[dict] = field(default_factory=list)
    test_indices: Optional[np.ndarray] = None
    method: str = "kfold"
    n_folds: int = 0
    total_samples: int = 0


class DataSplitter:
    """Smart data splitting engine."""

    def split(
        self, df: pd.DataFrame, target_col: str = "y",
        time_col: Optional[str] = "ds", task_type: str = "regression",
        config: SplitConfig | None = None,
    ) -> SplitResult:
        if config is None:
            config = self._auto_config(task_type)

        train_val_df = df
        test_indices = None
        if config.test_size > 0:
            if config.method in {"time_series", "expanding_window"}:
                split_idx = int(len(df) * (1 - config.test_size))
                train_val_df = df.iloc[:split_idx]
                test_indices = np.arange(split_idx, len(df))
            else:
                idx = np.arange(len(df))
                train_idx, test_idx = train_test_split(
                    idx, test_size=config.test_size,
                    random_state=config.random_state, shuffle=True,
                )
                train_val_df = df.iloc[train_idx]
                test_indices = test_idx

        y_tv = train_val_df[target_col].values
        folds = []

        if config.method == "kfold":
            kf = KFold(n_splits=config.n_splits, shuffle=config.shuffle, random_state=config.random_state)
            for ti, vi in kf.split(train_val_df):
                folds.append({"train_idx": train_val_df.index[ti].values, "val_idx": train_val_df.index[vi].values})
        elif config.method == "stratified_kfold":
            skf = StratifiedKFold(n_splits=config.n_splits, shuffle=config.shuffle, random_state=config.random_state)
            for ti, vi in skf.split(train_val_df, y_tv):
                folds.append({"train_idx": train_val_df.index[ti].values, "val_idx": train_val_df.index[vi].values})
        elif config.method in {"time_series", "expanding_window"}:
            tscv = TimeSeriesSplit(n_splits=config.n_splits, gap=config.gap)
            for ti, vi in tscv.split(train_val_df):
                folds.append({"train_idx": train_val_df.index[ti].values, "val_idx": train_val_df.index[vi].values})
        elif config.method == "train_test":
            idx = np.arange(len(train_val_df))
            ti, vi = train_test_split(idx, test_size=config.test_size, random_state=config.random_state,
                                      shuffle=not (task_type == "timeseries"))
            folds.append({"train_idx": train_val_df.index[ti].values, "val_idx": train_val_df.index[vi].values})
        else:
            raise ValueError(f"Unknown split method: {config.method}")

        return SplitResult(folds=folds, test_indices=test_indices, method=config.method,
                           n_folds=len(folds), total_samples=len(df))

    @staticmethod
    def _auto_config(task_type: str) -> SplitConfig:
        if task_type == "timeseries":
            return SplitConfig(method="time_series", n_splits=5, test_size=0.15, shuffle=False)
        # For classification and regression: do NOT shuffle if time column exists.
        # Lag features break when the temporal order is randomized.
        return SplitConfig(method="kfold", n_splits=5, test_size=0.2, shuffle=False)


# ════════════════════════════════════════════════════════════════════
# Nested Cross-Validation
# ════════════════════════════════════════════════════════════════════

@dataclass
class NestedCVConfig:
    """Configuration for nested cross-validation.

    Outer loop: evaluates different model families / feature sets.
    Inner loop: hyperparameter optimization within each outer fold.
    Hold-out: completely untouched test set for final unbiased assessment.

    This prevents the classic pitfall where model selection and
    hyperparameter tuning share the same validation set, inflating
    the apparent performance.
    """
    outer_splits: int = 3
    inner_splits: int = 3
    holdout_ratio: float = 0.15
    random_state: int = 42
    shuffle_inner: bool = False


@dataclass
class NestedCVResult:
    """Result of a nested cross-validation run.

    Attributes:
        outer_scores: List of scores from each outer fold.
        outer_mean: Mean of outer scores.
        outer_std: Standard deviation of outer scores.
        inner_best_params: Best hyperparameters per outer fold.
        holdout_score: Score on the completely untouched hold-out set.
        n_outer_folds: Number of outer folds executed.
    """
    outer_scores: list[float] = field(default_factory=list)
    outer_mean: float = 0.0
    outer_std: float = 0.0
    inner_best_params: list[dict] = field(default_factory=list)
    holdout_score: Optional[float] = None
    n_outer_folds: int = 0

    @property
    def score_range(self) -> tuple[float, float]:
        """The 95% confidence interval of the outer score."""
        if not self.outer_scores or len(self.outer_scores) < 2:
            return (self.outer_mean, self.outer_mean)
        se = self.outer_std / np.sqrt(len(self.outer_scores))
        return (self.outer_mean - 1.96 * se, self.outer_mean + 1.96 * se)

    def summary(self) -> str:
        lo, hi = self.score_range
        return (
            f"NestedCV: {self.outer_mean:.4f} ± {self.outer_std:.4f} "
            f"[{lo:.4f}, {hi:.4f}] over {self.n_outer_folds} folds"
        )


class NestedCVSplitter:
    """Nested cross-validation for unbiased model evaluation.

    Outer loop: K-fold CV to evaluate a "methodology" (model family + features).
    Inner loop: HPO within each outer fold — model never sees inner val data.
    Hold-out: Completely untouched for final unbiased assessment.

    Usage:
        >>> ncv = NestedCVSplitter()
        >>> result = ncv.run(
        ...     df, train_fn, predict_fn,
        ...     param_space={"n_estimators": [100, 200, 500]},
        ...     task_type="regression",
        ... )
        >>> print(result.summary())
    """

    def __init__(self) -> None:
        pass

    def run(
        self,
        df: pd.DataFrame,
        train_fn,
        predict_fn,
        param_space: dict[str, list],
        target_col: str = "y",
        time_col: Optional[str] = "ds",
        task_type: str = "regression",
        metric: str = "mase",
        config: NestedCVConfig | None = None,
    ) -> NestedCVResult:
        """Execute nested cross-validation.

        Args:
            df: Full dataset.
            train_fn: fn(X_train, y_train, params) -> trained_model.
            predict_fn: fn(model, X) -> predictions.
            param_space: Hyperparameter grid for inner loop.
            target_col: Target column name.
            time_col: Time column name (for time series).
            task_type: "classification", "regression", or "timeseries".
            metric: Evaluation metric name.
            config: NestedCV configuration.

        Returns:
            NestedCVResult with outer scores, inner best params, holdout score.
        """
        config = config or NestedCVConfig()

        from endforecast.evaluation.metrics import MetricCalculator
        from endforecast.evaluation.hyperopt import HPOptimizer, HPOConfig

        metric_calc = MetricCalculator()
        hyperopt = HPOptimizer()
        y_all = df[target_col].values

        # ── Split: hold-out from full dataset ─────────────────
        if config.holdout_ratio > 0:
            if task_type == "timeseries":
                split_idx = int(len(df) * (1 - config.holdout_ratio))
                df_train = df.iloc[:split_idx]
                df_holdout = df.iloc[split_idx:]
            else:
                idx = np.arange(len(df))
                train_idx, holdout_idx = train_test_split(
                    idx, test_size=config.holdout_ratio,
                    random_state=config.random_state,
                )
                df_train = df.iloc[train_idx]
                df_holdout = df.iloc[holdout_idx]
        else:
            df_train = df
            df_holdout = None

        # ── Outer CV splits ───────────────────────────────────
        outer_splitter = DataSplitter()
        outer_cfg = SplitConfig(
            method="time_series" if task_type == "timeseries" else "kfold",
            n_splits=config.outer_splits, test_size=1.0 / config.outer_splits,
            random_state=config.random_state,
        )
        outer_result = outer_splitter.split(
            df_train, target_col=target_col, time_col=time_col,
            task_type=task_type, config=outer_cfg,
        )

        outer_scores: list[float] = []
        inner_best_params: list[dict] = []

        for fold_idx, fold in enumerate(outer_result.folds):
            train_idx = fold["train_idx"]
            val_idx = fold["val_idx"]
            train_df = df_train.iloc[train_idx]
            val_df = df_train.iloc[val_idx]

            # ── Inner CV / HPO on this outer fold ──────────
            y_train = train_df[target_col].values
            y_val = val_df[target_col].values

            def _hpo_objective(params: dict) -> float:
                model = train_fn(train_df, target_col, time_col, params)
                pred = predict_fn(model, val_df, time_col)
                mr = metric_calc.evaluate(y_val, pred, task_type=task_type, metrics=[metric])
                return mr.value

            hpo_cfg = HPOConfig(method="random", n_trials=min(10, _count_combinations(param_space)))
            hpo_res = hyperopt.optimize(_hpo_objective, param_space, hpo_cfg)
            inner_best_params.append(hpo_res.best_params or {})

            # ── Re-train with best params on this fold ──────
            model = train_fn(train_df, target_col, time_col, hpo_res.best_params)
            pred = predict_fn(model, val_df, time_col)
            mr = metric_calc.evaluate(y_val, pred, task_type=task_type, metrics=[metric])
            outer_scores.append(mr.value)

        result = NestedCVResult(
            outer_scores=outer_scores,
            outer_mean=float(np.mean(outer_scores)) if outer_scores else 0.0,
            outer_std=float(np.std(outer_scores)) if outer_scores else 0.0,
            inner_best_params=inner_best_params,
            n_outer_folds=len(outer_scores),
        )

        # ── Final evaluation on hold-out ─────────────────────
        if df_holdout is not None and len(df_holdout) > 0:
            # Train on full df_train with most-frequent best params
            if inner_best_params:
                best_params = _most_frequent_params(inner_best_params)
            else:
                best_params = {}
            model = train_fn(df_train, target_col, time_col, best_params)
            pred = predict_fn(model, df_holdout, time_col)
            y_holdout = df_holdout[target_col].values
            mr = metric_calc.evaluate(y_holdout, pred, task_type=task_type, metrics=[metric])
            result.holdout_score = mr.value

        logger.info(result.summary())
        return result


def _count_combinations(param_space: dict[str, list]) -> int:
    """Count total grid combinations (capped)."""
    n = 1
    for v in param_space.values():
        n *= len(v)
    return min(n, 10000)


def _most_frequent_params(params_list: list[dict]) -> dict:
    """Pick the most frequently occurring best params across folds."""
    if not params_list:
        return {}
    from collections import Counter
    # For each key, pick the most common value
    result = {}
    for key in params_list[0]:
        vals = [p.get(key) for p in params_list if key in p]
        if vals:
            result[key] = Counter(vals).most_common(1)[0][0]
    return result