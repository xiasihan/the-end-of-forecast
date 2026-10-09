"""Requirements Gatherer Agent — elicits and formalizes prediction requirements.

This is the FIRST agent invoked. It ensures the prediction problem
is well-defined before any data exploration begins. Includes a
Heuristic First check: per Google Rules of ML #1-3, we should
not use ML if simple heuristics suffice.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Optional

logger = logging.getLogger(__name__)


@dataclass
class RequirementsSpec:
    """Structured prediction requirements gathered from the user."""
    business_purpose: str = ""
    target_description: str = ""
    task_type_hint: str = "auto"
    prediction_horizon: str = ""
    success_metrics: list[str] = field(default_factory=list)
    acceptable_min_performance: Optional[float] = None
    data_constraints: dict = field(default_factory=dict)
    deployment_preferences: list[str] = field(default_factory=list)
    domain_constraints: list[str] = field(default_factory=list)
    # Per Google Rules of ML #9: encode existing heuristics as features
    heuristic_rules: list[str] = field(default_factory=list)
    # Risk detection flags (P0)
    feedback_loop_possible: bool = False
    proxy_label_warning: bool = False
    # Special event timeline (P2)
    event_timeline: list[dict] = field(default_factory=list)
    user_notes: str = ""

    def to_dict(self) -> dict:
        return {k: v for k, v in self.__dict__.items()}


class RequirementsGatherer:
    """Gathers and formalizes prediction requirements.

    Usage:
        >>> gatherer = RequirementsGatherer()
        >>> spec = gatherer.gather(
        ...     "Predict tomorrow's electricity price for Guangdong",
        ...     heuristic_rules=["weekends are cheaper", "summer peaks at 2pm"],
        ... )
    """

    def gather(
        self,
        description: str = "",
        available_data: str = "",
        constraints: str = "",
        success_metric: str = "auto",
        deployment: str = "auto",
        domain_notes: str = "",
        heuristic_rules: Optional[list[str]] = None,
    ) -> RequirementsSpec:
        """Elicit and formalize prediction requirements.

        Args:
            description: User's goal in natural language.
            available_data: Description of available datasets.
            constraints: Known data limitations or rules.
            success_metric: Preferred evaluation metric or "auto".
            deployment: How predictions will be consumed (api/code/tool).
            domain_notes: Known domain-specific constraints.
            heuristic_rules: Existing business heuristics that could become features.

        Returns:
            RequirementsSpec for downstream agents.
        """
        spec = RequirementsSpec(
            business_purpose=description,
            user_notes=domain_notes,
            heuristic_rules=heuristic_rules or [],
        )

        # ── Task type heuristic ───────────────────────────────
        desc_lower = description.lower()
        if any(kw in desc_lower for kw in [
            "classify", "classification", "category", "label", "binary",
            "churn", "fraud", "spam", "sentiment",
        ]):
            spec.task_type_hint = "classification"
        elif any(kw in desc_lower for kw in [
            "predict tomorrow", "forecast", "time series", "next week",
            "next month", "next hour", "price prediction", "load forecast",
            "demand forecast", "stock", "sales forecast",
        ]):
            spec.task_type_hint = "timeseries"
        elif any(kw in desc_lower for kw in [
            "regression", "estimate", "predict value", "price", "revenue",
            "score", "rating",
        ]):
            spec.task_type_hint = "regression"

        # ── P0: Feedback Loop Detection ────────────────────────
        feedback_keywords = [
            "recommend", "rank", "pricing", "bid", "auction", "targeting",
            "personalize", "personalize", "ad", "allocation", "trade",
            "loan", "credit", "approve",
        ]
        spec.feedback_loop_possible = any(kw in desc_lower for kw in feedback_keywords)

        # ── P0: Proxy Label Detection ──────────────────────────
        proxy_keywords = [
            "click", "engagement", "view", "impression", "like", "share",
            "response", "complaint", "review",
        ]
        spec.proxy_label_warning = any(kw in desc_lower for kw in proxy_keywords)

        # ── Success metrics ────────────────────────────────────
        if success_metric == "auto":
            if spec.task_type_hint == "classification":
                spec.success_metrics = ["f1", "accuracy"]
            elif spec.task_type_hint == "timeseries":
                spec.success_metrics = ["mase", "smape"]
            else:
                spec.success_metrics = ["mae", "mape"]
        else:
            spec.success_metrics = [success_metric]

        # ── Data constraints ────────────────────────────────────
        if constraints:
            spec.data_constraints = {"user_notes": constraints}

        # ── Deployment preferences ──────────────────────────────
        if deployment == "auto":
            spec.deployment_preferences = ["config", "code"]
        else:
            spec.deployment_preferences = [deployment]

        # ── Domain constraints ──────────────────────────────────
        if domain_notes:
            spec.domain_constraints.append(domain_notes)

        # ── Encode heuristics as suggested features (Google Rule #9) ─
        if spec.heuristic_rules:
            spec.suggested_heuristic_features = _encode_heuristics(
                spec.heuristic_rules, spec.task_type_hint,
            )

        logger.info("Requirements: task=%s metrics=%s heuristics=%s",
                     spec.task_type_hint, spec.success_metrics,
                     len(spec.heuristic_rules))
        return spec


# ── Heuristic-to-feature encoder ───────────────────────────────

def _encode_heuristics(rules: list[str], task_type: str) -> list[str]:
    """Translate business heuristics into feature names.

    Per Google Rules of ML #9: "Turn existing heuristics into features."
    These domain rules represent years of accumulated knowledge and
    often encode the most predictive patterns in the data.
    """
    features = []
    for rule in rules:
        r = rule.lower()
        if "weekend" in r:
            features.append("is_weekend")
        if "weekday" in r or "workday" in r:
            features.append("is_weekday")
        if "summer" in r:
            features.append("is_summer")
        if "winter" in r:
            features.append("is_winter")
        if "holiday" in r or "festival" in r:
            features.append("days_to_nearest_holiday")
        if "hour" in r or "peak" in r:
            features.append("hour_binned_peak")
        if "season" in r or "spring" in r:
            features.append("season")
        if "month" in r:
            features.append("month")
        if "year" in r:
            features.append("year")
        if "lag" in r:
            features.append("lag_7")
        if "moving" in r or "rolling" in r or "average" in r:
            features.append("rolling_mean_7")
        if "recent" in r:
            features.append("acc_recent")
        if "trend" in r:
            features.append("trend_strength")
    # Deduplicate
    return sorted(set(features))