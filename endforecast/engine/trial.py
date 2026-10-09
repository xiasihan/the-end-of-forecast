"""
Single trial execution and result data structures.

Includes:
- OptimizationBudget: Tracks degrees of freedom consumed during
  data-driven decisions (prevents data snooping).
- Trial / TrialResult: Enhanced with hypothesis, parent, multi-run stats.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Optional

import numpy as np
import pandas as pd


class TrialState(Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"


# ════════════════════════════════════════════════════════════════════
# Optimization Budget — Degrees of Freedom Tracking
# ════════════════════════════════════════════════════════════════════

@dataclass
class OptimizationBudget:
    """Tracks optimization degrees of freedom consumed during experimentation.

    Every data-driven decision (parameter selection, strategy adjustment,
    convergence check) consumes one "degree of freedom". When the budget
    is exhausted, further decisions risk data snooping — the reported
    performance becomes biased and unreproducible.

    Usage:
        >>> budget = OptimizationBudget(max_decisions=20)
        >>> budget.consume("Diagnostician suggested switching to LightGBM")
        >>> budget.consume("Convergence check: round 5")
        >>> if budget.is_exhausted:
        ...     print("Warning: optimization budget depleted.")
        ...     print(budget.summary())
    """
    max_decisions: int = 20
    decisions_made: int = 0
    decision_log: list[str] = field(default_factory=list)

    @property
    def remaining(self) -> int:
        return max(0, self.max_decisions - self.decisions_made)

    @property
    def is_exhausted(self) -> bool:
        return self.decisions_made >= self.max_decisions

    @property
    def is_low(self) -> bool:
        """Budget is below 30% — warning zone."""
        return self.remaining < self.max_decisions * 0.3

    def consume(self, reason: str) -> bool:
        """Attempt to consume one decision. Returns False if budget exhausted.

        Args:
            reason: Human-readable reason for the decision.

        Returns:
            True if the decision was allowed, False if budget is exhausted.
        """
        if self.is_exhausted:
            return False
        self.decisions_made += 1
        self.decision_log.append(f"[{self.decisions_made}/{self.max_decisions}] {reason}")
        return True

    def summary(self) -> str:
        status = "🔴 EXHAUSTED" if self.is_exhausted else ("🟡 LOW" if self.is_low else "🟢 OK")
        return (
            f"OptimizationBudget: {status} | "
            f"{self.decisions_made}/{self.max_decisions} decisions used | "
            f"{self.remaining} remaining"
        )


# ════════════════════════════════════════════════════════════════════
# Trial & TrialResult
# ════════════════════════════════════════════════════════════════════

@dataclass
class Trial:
    """Configuration for a single experiment trial."""
    id: str
    round: int = 1
    preprocess: list[str] = field(default_factory=list)
    features: list[str] = field(default_factory=list)
    model: str = "ridge"
    model_params: dict[str, Any] = field(default_factory=dict)
    cv_strategy: str = "rolling_window"
    cv_params: dict[str, Any] = field(default_factory=dict)
    refinement: str = "none"
    task_type: str = "regression"
    metric: str = "mase"
    state: TrialState = TrialState.PENDING
    # Provenance fields
    hypothesis: str = ""             # What hypothesis motivated this trial?
    parent_trial_id: str = ""        # Which prior trial inspired this?

    @property
    def pipeline_id(self) -> str:
        return f"r{self.round}_{self.id}"

    def to_dict(self) -> dict:
        return {k: v for k, v in self.__dict__.items() if not k.startswith("_")}


@dataclass
class TrialResult:
    """Collected results from a completed trial.

    Enhanced with:
    - Multi-run stats (mean, std across seeds)
    - Diagnosis provenance (which diagnosis triggered this)
    """
    trial: Trial
    state: TrialState = TrialState.PENDING
    metric_value: float = 0.0           # Mean metric across runs (point estimate)
    metric_std: float = 0.0             # Standard deviation across runs
    metric_n_runs: int = 1              # Number of stability runs
    metrics_detail: dict[str, float] = field(default_factory=dict)
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    duration_seconds: float = 0.0
    predictions: Optional[pd.DataFrame] = None
    evaluation_df: Optional[pd.DataFrame] = None
    diagnosis: Optional[Any] = None
    diagnosis_applied: str = ""         # Which diagnosis triggered this trial?
    parent_trial_id: str = ""           # Which prior trial inspired this?
    error_message: Optional[str] = None
    warnings: list[str] = field(default_factory=list)

    @property
    def is_successful(self) -> bool:
        return self.state == TrialState.COMPLETED and self.error_message is None

    @property
    def reliability_score(self) -> float:
        """Higher = more reliable (inverse of CV across seeds)."""
        if self.metric_n_runs < 2 or self.metric_std == 0:
            return 1.0
        cv = self.metric_std / (abs(self.metric_value) + 1e-10)
        return round(1.0 / (1.0 + cv), 4)

    def summary(self) -> str:
        icon = "\u2705" if self.is_successful else "\u274c"
        base = (f"{icon} {self.trial.pipeline_id}: "
                f"{self.trial.metric.upper()}={self.metric_value:.4f}"
                f"±{self.metric_std:.4f} (n={self.metric_n_runs})")
        if self.diagnosis:
            base += f" | diag: {self.diagnosis.summary}"
        return base

    @property
    def improvement_vs(self):
        def _improvement(other: "TrialResult") -> float:
            if other.metric_value == 0:
                return 0.0
            return (self.metric_value - other.metric_value) / abs(other.metric_value)
        return _improvement