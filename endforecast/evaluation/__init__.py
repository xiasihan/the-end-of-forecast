"""
Evaluation module — metrics, splitting, and hyperparameter optimization.

Provides task-appropriate evaluation strategies:
- Data splitting (KFold, TimeSeriesSplit, StratifiedKFold, NestedCV)
- Metric computation (classification, regression, time series) + statistical tests
- Hyperparameter optimization (grid, random, Bayesian via Optuna)
- Calibration assessment (ECE, Brier score)
- Stability evaluation (multi-seed variance)
"""

from endforecast.evaluation.splitter import (
    DataSplitter, SplitConfig, SplitResult,
    NestedCVSplitter, NestedCVConfig, NestedCVResult,
)
from endforecast.evaluation.metrics import (
    MetricCalculator, MetricResult,
    StatisticalComparison, ComparisonResult,
    StabilityEvaluator, StabilityResult,
    CalibrationMetrics, CalibrationResult,
)
from endforecast.evaluation.hyperopt import HPOptimizer, HPOConfig, HPOResult

__all__ = [
    "DataSplitter", "SplitConfig", "SplitResult",
    "NestedCVSplitter", "NestedCVConfig", "NestedCVResult",
    "MetricCalculator", "MetricResult",
    "StatisticalComparison", "ComparisonResult",
    "StabilityEvaluator", "StabilityResult",
    "CalibrationMetrics", "CalibrationResult",
    "HPOptimizer", "HPOConfig", "HPOResult",
]