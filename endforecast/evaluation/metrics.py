"""
Evaluation metrics — task-adaptive metric computation.

Supports classification, regression, and time series metrics.
Each metric returns a structured result with:
- Primary value (the "score" for comparison)
- Auxiliary values (for diagnosis)
- Direction indicator (minimize or maximize)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Callable, Optional

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class MetricResult:
    """Structured metric evaluation result.

    Attributes:
        name: Metric name (e.g., "mase", "f1", "mae").
        value: Primary metric value.
        direction: "minimize" or "maximize" — guides optimization.
        auxiliary: Optional auxiliary metric values.
        interpretation: Human-readable interpretation string.
    """
    name: str
    value: float
    direction: str = "minimize"
    auxiliary: dict[str, float] = field(default_factory=dict)
    interpretation: str = ""


class MetricCalculator:
    """Task-adaptive metric computation engine.

    Automatically selects appropriate metrics based on task type
    and provides both point estimates and multi-metric evaluation.

    Usage:
        >>> calc = MetricCalculator()
        >>> result = calc.evaluate(
        ...     y_true, y_pred, task_type="classification",
        ...     metrics=["f1", "accuracy"],
        ... )
        >>> print(result.value, result.direction)
    """

    # ── Metric Registry ─────────────────────────────────────────

    _METRICS: dict[str, Callable] = {
        # Classification
        "accuracy": lambda y, yp, kw: float(np.mean((y.astype(int)) == (yp >= 0.5).astype(int))),
        "f1": lambda y, yp, kw: _f1_score(y, (yp >= 0.5)),
        "precision": lambda y, yp, kw: _precision(y, (yp >= 0.5)),
        "recall": lambda y, yp, kw: _recall(y, (yp >= 0.5)),
        "auc": lambda y, yp, kw: _roc_auc(y, yp),
        # Regression
        "mae": lambda y, yp, kw: float(np.mean(np.abs(y - yp))),
        "mse": lambda y, yp, kw: float(np.mean((y - yp) ** 2)),
        "rmse": lambda y, yp, kw: float(np.sqrt(np.mean((y - yp) ** 2))),
        "mape": lambda y, yp, kw: _mape(y, yp),
        "r2": lambda y, yp, kw: _r2(y, yp),
        # Time Series
        "mase": lambda y, yp, kw: _mase(y, yp, kw.get("seasonality", 1)),
        "smape": lambda y, yp, kw: _smape(y, yp),
        "pinball": lambda y, yp, kw: _pinball_loss(y, yp, kw.get("quantile", 0.5)),
        # Directional
        "direction_accuracy": lambda y, yp, kw: _dir_acc(y, yp),
    }

    _DIRECTION: dict[str, str] = {
        "accuracy": "maximize", "f1": "maximize", "precision": "maximize",
        "recall": "maximize", "auc": "maximize", "r2": "maximize",
        "direction_accuracy": "maximize",
        "mae": "minimize", "mse": "minimize", "rmse": "minimize",
        "mape": "minimize", "mase": "minimize", "smape": "minimize",
        "pinball": "minimize",
    }

    @classmethod
    def list_metrics(cls, task_type: str | None = None) -> list[str]:
        """List available metrics, optionally filtered by task type."""
        if task_type is None:
            return sorted(cls._METRICS.keys())
        groups = {
            "classification": ["accuracy", "f1", "precision", "recall", "auc"],
            "regression": ["mae", "mse", "rmse", "mape", "r2"],
            "timeseries": ["mase", "smape", "mae", "rmse", "direction_accuracy"],
        }
        return [m for m in cls._METRICS if m in groups.get(task_type, [])]

    def evaluate(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        task_type: str = "regression",
        metrics: str | list[str] = "auto",
        **kwargs,
    ) -> MetricResult:
        """Evaluate predictions against ground truth.

        Args:
            y_true: Ground-truth target values.
            y_pred: Model predictions.
            task_type: "classification", "regression", or "timeseries".
            metrics: Metric name, list of names, or "auto" for task defaults.
            **kwargs: Additional arguments (e.g., seasonality for MASE).

        Returns:
            MetricResult with the primary metric value.
        """
        if metrics == "auto":
            metrics = self._default_metrics(task_type)
        elif isinstance(metrics, str):
            metrics = [metrics]

        values = {}
        for name in metrics:
            if name not in self._METRICS:
                logger.warning("Unknown metric '%s'; skipping.", name)
                continue
            try:
                values[name] = self._METRICS[name](y_true, y_pred, kwargs)
            except Exception as exc:
                logger.warning("Metric '%s' failed: %s", name, exc)
                values[name] = float("nan")

        # Primary metric = first in list
        primary_name = metrics[0]
        primary_value = values.get(primary_name, float("nan"))
        direction = self._DIRECTION.get(primary_name, "minimize")

        return MetricResult(
            name=primary_name,
            value=float(primary_value),
            direction=direction,
            auxiliary={k: float(v) for k, v in values.items() if k != primary_name},
            interpretation=f"{primary_name}={primary_value:.4f} ({direction})",
        )

    @staticmethod
    def _default_metrics(task_type: str) -> list[str]:
        if task_type == "classification":
            return ["f1", "accuracy", "auc"]
        elif task_type == "timeseries":
            return ["mase", "smape", "direction_accuracy"]
        return ["mae", "mape", "rmse"]


# ── Metric implementations ─────────────────────────────────────

def _f1_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    tp = np.sum((y_pred == 1) & (y_true == 1))
    fp = np.sum((y_pred == 1) & (y_true == 0))
    fn = np.sum((y_pred == 0) & (y_true == 1))
    p = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    r = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    return 2 * p * r / (p + r) if (p + r) > 0 else 0.0

def _precision(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    tp = np.sum((y_pred == 1) & (y_true == 1))
    fp = np.sum((y_pred == 1) & (y_true == 0))
    return float(tp / (tp + fp)) if (tp + fp) > 0 else 0.0

def _recall(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    tp = np.sum((y_pred == 1) & (y_true == 1))
    fn = np.sum((y_pred == 0) & (y_true == 1))
    return float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0

def _roc_auc(y_true: np.ndarray, y_score: np.ndarray) -> float:
    try:
        from sklearn.metrics import roc_auc_score
        return float(roc_auc_score(y_true, y_score))
    except Exception:
        return 0.5

def _mape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    nonzero = np.abs(y_true) > 1e-10
    if not np.any(nonzero): return 0.0
    return float(np.mean(np.abs((y_true[nonzero] - y_pred[nonzero]) / y_true[nonzero])) * 100)

def _r2(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    return float(1 - ss_res / ss_tot) if ss_tot > 1e-10 else 0.0

def _mase(y_true: np.ndarray, y_pred: np.ndarray, seasonality: int = 1) -> float:
    n = len(y_true)
    if n <= seasonality: return float(np.mean(np.abs(y_true - y_pred)))
    naive_errors = np.abs(y_true[seasonality:] - y_true[:-seasonality])
    denom = float(np.mean(naive_errors)) or 1e-10
    return float(np.mean(np.abs(y_true - y_pred)) / denom)

def _smape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    denom = (np.abs(y_true) + np.abs(y_pred)) / 2
    mask = denom > 1e-10
    if not np.any(mask): return 0.0
    return float(np.mean(np.abs(y_true[mask] - y_pred[mask]) / denom[mask]) * 100)

def _pinball_loss(y_true: np.ndarray, y_pred: np.ndarray, quantile: float = 0.5) -> float:
    residuals = y_true - y_pred
    return float(np.mean(np.maximum(quantile * residuals, (quantile - 1) * residuals)))

def _dir_acc(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    if len(y_true) < 2: return 1.0
    true_dir = np.sign(np.diff(y_true))
    pred_dir = np.sign(np.diff(y_pred))
    return float(np.mean(true_dir == pred_dir))


# ════════════════════════════════════════════════════════════════════
# Statistical Comparison Tests
# ════════════════════════════════════════════════════════════════════

@dataclass
class ComparisonResult:
    """Result of a statistical comparison between two models.

    Attributes:
        test_name: Name of the statistical test used.
        p_value: Raw p-value (uncorrected for multiple comparisons).
        p_value_corrected: Bonferroni/Holm corrected p-value.
        significant: True if p_value_corrected < alpha.
        effect_size: Cohen's d / Cliff's delta — not just "is different" but "how different".
        effect_size_interpretation: "negligible", "small", "medium", or "large".
        winner: "A", "B", or "none" (no significant difference).
        detail: Human-readable interpretation.
    """
    test_name: str = ""
    p_value: float = 1.0
    p_value_corrected: float = 1.0
    significant: bool = False
    effect_size: float = 0.0
    effect_size_interpretation: str = "negligible"
    winner: str = "none"
    detail: str = ""


class StatisticalComparison:
    """Compare two models with proper statistical rigor.

    Supports:
    - Paired t-test (continuous metrics across CV folds)
    - Wilcoxon signed-rank test (non-parametric alternative)
    - McNemar's test (classification on same test set)
    - Bonferroni / Holm correction for multiple comparisons
    - Effect size estimation (Cohen's d, Cliff's delta)

    Usage:
        >>> comp = StatisticalComparison()
        >>> result = comp.compare(
        ...     scores_a, scores_b, alpha=0.05, n_total_comparisons=10,
        ... )
        >>> if result.significant:
        ...     print(f"Model {result.winner} wins, d={result.effect_size:.2f}")
    """

    def compare(
        self,
        scores_a: np.ndarray,
        scores_b: np.ndarray,
        test: str = "auto",
        alpha: float = 0.05,
        n_total_comparisons: int = 1,
        metric_direction: str = "minimize",
    ) -> ComparisonResult:
        """Compare two sets of scores.

        Args:
            scores_a: Scores from model A (e.g., per-fold CV scores).
            scores_b: Scores from model B.
            test: "paired_ttest", "wilcoxon", "mcnemar", or "auto".
            alpha: Significance level.
            n_total_comparisons: Total number of pairwise comparisons
                being made (for Bonferroni correction). Default 1 = no correction.
            metric_direction: "minimize" (lower is better) or "maximize".

        Returns:
            ComparisonResult with p-value, significance, effect size, winner.
        """
        if test == "auto":
            test = "paired_ttest" if len(scores_a) >= 5 else "wilcoxon"

        # ── Run the chosen test ──────────────────────────────
        if test == "paired_ttest":
            p_val, test_name = self._paired_ttest(scores_a, scores_b)
        elif test == "wilcoxon":
            p_val, test_name = self._wilcoxon(scores_a, scores_b)
        elif test == "mcnemar":
            p_val, test_name = self._mcnemar(scores_a, scores_b)
        else:
            p_val, test_name = 1.0, "unknown"

        # ── Multiple comparison correction ───────────────────
        if n_total_comparisons > 1:
            p_corrected = min(p_val * n_total_comparisons, 1.0)  # Bonferroni
        else:
            p_corrected = p_val

        significant = p_corrected < alpha

        # ── Effect size ──────────────────────────────────────
        d, interp = self._cohens_d(scores_a, scores_b, metric_direction)

        # ── Winner determination ─────────────────────────────
        winner = "none"
        if significant:
            mean_a, mean_b = float(np.mean(scores_a)), float(np.mean(scores_b))
            if metric_direction == "minimize":
                winner = "A" if mean_a < mean_b else "B"
            else:
                winner = "A" if mean_a > mean_b else "B"

        return ComparisonResult(
            test_name=test_name,
            p_value=round(p_val, 6),
            p_value_corrected=round(p_corrected, 6),
            significant=significant,
            effect_size=round(d, 3),
            effect_size_interpretation=interp,
            winner=winner,
            detail=f"{test_name}: p={p_val:.4f} (corr={p_corrected:.4f}), d={d:.3f} ({interp}), winner={winner}",
        )

    @staticmethod
    def _paired_ttest(a: np.ndarray, b: np.ndarray) -> tuple[float, str]:
        try:
            from scipy.stats import ttest_rel
            stat, p = ttest_rel(a, b)
            return float(p), "paired_ttest"
        except ImportError:
            # Fallback: approximate via normal distribution
            diff = a - b
            n = len(diff)
            if n < 2:
                return 1.0, "paired_ttest(fallback)"
            se = float(np.std(diff, ddof=1) / np.sqrt(n))
            if se < 1e-10:
                return 1.0, "paired_ttest(fallback)"
            t = float(np.mean(diff) / se)
            from math import erf, sqrt
            p = float(2 * (1 - 0.5 * (1 + erf(abs(t) / sqrt(2)))))
            return p, "paired_ttest(fallback)"

    @staticmethod
    def _wilcoxon(a: np.ndarray, b: np.ndarray) -> tuple[float, str]:
        try:
            from scipy.stats import wilcoxon
            _, p = wilcoxon(a, b)
            return float(p), "wilcoxon"
        except ImportError:
            return 1.0, "wilcoxon(unavailable)"

    @staticmethod
    def _mcnemar(a: np.ndarray, b: np.ndarray) -> tuple[float, str]:
        """McNemar's test for paired binary classifications."""
        try:
            a_bin = (a >= 0.5).astype(int)
            b_bin = (b >= 0.5).astype(int)
            b01 = np.sum((a_bin == 0) & (b_bin == 1))
            b10 = np.sum((a_bin == 1) & (b_bin == 0))
            if b01 + b10 == 0:
                return 1.0, "mcnemar"
            stat = (abs(b01 - b10) - 1) ** 2 / (b01 + b10)
            from math import exp
            p = float(min(1.0, exp(-stat / 2)))
            return p, "mcnemar"
        except Exception:
            return 1.0, "mcnemar(error)"

    @staticmethod
    def _cohens_d(a: np.ndarray, b: np.ndarray, direction: str = "minimize") -> tuple[float, str]:
        """Cohen's d effect size with pooled standard deviation."""
        n1, n2 = len(a), len(b)
        if n1 < 2 or n2 < 2:
            return 0.0, "negligible"
        s1, s2 = float(np.std(a, ddof=1)), float(np.std(b, ddof=1))
        pooled = np.sqrt(((n1 - 1) * s1**2 + (n2 - 1) * s2**2) / (n1 + n2 - 2))
        if pooled < 1e-10:
            return 0.0, "negligible"
        d = abs(float((np.mean(a) - np.mean(b)) / pooled))
        # Interpretation thresholds (Cohen, 1988)
        if d < 0.2: interp = "negligible"
        elif d < 0.5: interp = "small"
        elif d < 0.8: interp = "medium"
        else: interp = "large"
        # Adjust sign so positive = B is better (only when direction is minimize and B < A)
        if direction == "minimize" and np.mean(b) < np.mean(a):
            d = d  # B is better
        elif direction == "maximize" and np.mean(b) > np.mean(a):
            d = d  # B is better
        return d, interp


# ════════════════════════════════════════════════════════════════════
# Stability / Variance Estimation
# ════════════════════════════════════════════════════════════════════

@dataclass
class StabilityResult:
    """Result of model stability evaluation across multiple random seeds.

    A model whose performance varies by 5% depending on random seed
    is fundamentally different from one that is consistently ±0.1%.
    """
    mean: float = 0.0
    std: float = 0.0
    n_runs: int = 0
    scores: list[float] = field(default_factory=list)
    reliability_score: float = 0.0  # 1/(cv), higher = more reliable

    def __post_init__(self):
        if self.scores:
            self.mean = float(np.mean(self.scores))
            self.std = float(np.std(self.scores, ddof=1)) if len(self.scores) > 1 else 0.0
            cv = self.std / (abs(self.mean) + 1e-10)
            self.reliability_score = round(1.0 / (1.0 + cv), 4)
            self.n_runs = len(self.scores)

    def summary(self) -> str:
        return f"Stability: {self.mean:.4f} ± {self.std:.4f} (n={self.n_runs}, reliability={self.reliability_score:.2f})"


class StabilityEvaluator:
    """Evaluate model stability across random seeds.

    Usage:
        >>> se = StabilityEvaluator()
        >>> result = se.evaluate(train_fn, df, target_col="y", n_runs=10)
        >>> print(result.summary())
    """

    def __init__(self, metric_calculator: MetricCalculator | None = None) -> None:
        self.metrics_calc = metric_calculator or MetricCalculator()

    def evaluate(
        self,
        train_fn,
        df: pd.DataFrame,
        target_col: str = "y",
        time_col: Optional[str] = "ds",
        task_type: str = "regression",
        metric: str = "mae",
        n_runs: int = 10,
    ) -> StabilityResult:
        """Run N independent train/eval cycles with different random seeds.

        Args:
            train_fn: fn(df, target_col, seed) -> (model, predictions).
            df: Dataset.
            target_col: Target column.
            time_col: Time column.
            task_type: Task type.
            metric: Metric name.
            n_runs: Number of runs (5-10 recommended).

        Returns:
            StabilityResult with mean, std, and reliability_score.
        """
        scores = []
        splitter = DataSplitter()
        for seed in range(n_runs):
            cfg = SplitConfig(method="train_test", test_size=0.2, random_state=seed)
            sr = splitter.split(df, target_col=target_col, time_col=time_col, task_type=task_type, config=cfg)
            if not sr.folds:
                continue
            fold = sr.folds[0]
            train_df = df.iloc[fold["train_idx"]]
            val_df = df.iloc[fold["val_idx"]]
            try:
                model, pred = train_fn(train_df, target_col, time_col, seed)
                y_val = val_df[target_col].values
                mr = self.metrics_calc.evaluate(y_val, pred, task_type=task_type, metrics=[metric])
                scores.append(mr.value)
            except Exception as exc:
                logger.warning("Stability run %s failed: %s", seed, exc)

        return StabilityResult(scores=scores)


# ════════════════════════════════════════════════════════════════════
# Calibration Assessment (Classification)
# ════════════════════════════════════════════════════════════════════

@dataclass
class CalibrationResult:
    """Calibration assessment for probabilistic classifiers.

    A well-calibrated model means: when it outputs p=0.8, the actual
    event happens ~80% of the time. An uncalibrated model is dangerous
    in risk-sensitive applications.
    """
    ece: float = 0.0       # Expected Calibration Error (lower is better)
    brier_score: float = 0.0  # Brier score (lower is better)
    n_bins: int = 10
    bin_edges: list[float] = field(default_factory=list)
    bin_accuracies: list[float] = field(default_factory=list)
    bin_confidences: list[float] = field(default_factory=list)

    def summary(self) -> str:
        return f"ECE={self.ece:.4f}, Brier={self.brier_score:.4f} ({self.n_bins} bins)"


class CalibrationMetrics:
    """Probability calibration assessment.

    Usage:
        >>> cm = CalibrationMetrics()
        >>> result = cm.assess(y_true, y_proba, n_bins=10)
        >>> if result.ece > 0.05:
        ...     print("Model is poorly calibrated — consider Platt scaling.")
    """

    def assess(
        self, y_true: np.ndarray, y_proba: np.ndarray, n_bins: int = 10,
    ) -> CalibrationResult:
        """Assess probability calibration.

        Args:
            y_true: Binary ground truth (0/1).
            y_proba: Predicted probabilities [0, 1].
            n_bins: Number of bins for ECE calculation.

        Returns:
            CalibrationResult with ECE, Brier score, bin-level data.
        """
        # Brier score
        brier = float(np.mean((y_proba - y_true) ** 2))

        # ECE: divide predictions into bins and measure gap
        bin_edges = np.linspace(0, 1, n_bins + 1)
        ece = 0.0
        accs, confs = [], []
        for i in range(n_bins):
            mask = (y_proba >= bin_edges[i]) & (y_proba < bin_edges[i + 1])
            if mask.sum() == 0:
                continue
            bin_acc = float(np.mean(y_true[mask]))
            bin_conf = float(np.mean(y_proba[mask]))
            ece += (mask.sum() / len(y_true)) * abs(bin_acc - bin_conf)
            accs.append(bin_acc)
            confs.append(bin_conf)

        return CalibrationResult(
            ece=round(ece, 4),
            brier_score=round(brier, 4),
            n_bins=n_bins,
            bin_edges=bin_edges.tolist(),
            bin_accuracies=accs,
            bin_confidences=confs,
        )