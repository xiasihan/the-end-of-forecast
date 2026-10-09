"""
Model Card generator — automated model documentation per Google Model Card Toolkit.

Implements automated Model Card generation (Phase 8+ of the methodology).
Every deployed model gets a structured, audit-ready documentation card
covering: model details, intended use, training data, evaluation results,
limitations, and ethical considerations.

This is a compliance requirement for regulated industries (finance, healthcare)
and a trust-building artifact for all prediction deployments.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

logger = logging.getLogger(__name__)


@dataclass
class ModelCard:
    """Auto-generated model documentation card.

    Follows the structure proposed by Google's Model Card Toolkit
    (Mitchell et al., 2019) and adapted for EndForecast pipelines.
    """

    # Section 1: Model Details
    pipeline_id: str = ""
    model_family: str = ""
    task_type: str = ""
    created_at: str = ""
    framework_version: str = "endforecast 0.1.0"

    # Section 2: Intended Use
    intended_use: str = ""
    out_of_scope_uses: list[str] = field(default_factory=list)

    # Section 3: Training Data
    training_data_hash: str = ""
    n_training_rows: int = 0
    training_date_range: str = ""
    n_features: int = 0
    feature_list: list[str] = field(default_factory=list)
    data_sources: str = ""

    # Section 4: Evaluation Results
    primary_metric: str = ""
    metric_value: float = 0.0
    metric_std: float = 0.0
    metric_n_runs: int = 0
    baseline_comparison: str = ""
    optimization_rounds: int = 0
    data_snooping_warning: bool = False
    heuristic_first_warning: bool = False

    # Section 5: Limitations
    limitations: list[str] = field(default_factory=list)
    known_biases: list[str] = field(default_factory=list)
    event_windows_excluded: list[dict] = field(default_factory=list)

    # Section 6: Ethical Considerations
    feedback_loop_warning: bool = False
    proxy_label_warning: bool = False
    fairness_notes: str = ""
    ethical_review_required: bool = False

    def to_markdown(self) -> str:
        """Render the model card as a Markdown document."""
        lines = [
            f"# Model Card: {self.pipeline_id}",
            "",
            "## Model Details",
            f"- Pipeline ID: `{self.pipeline_id}`",
            f"- Model Family: {self.model_family}",
            f"- Task Type: {self.task_type}",
            f"- Created: {self.created_at}",
            f"- Framework: {self.framework_version}",
            "",
            "## Intended Use",
            f"{self.intended_use}",
            "",
            "### Out of Scope Uses",
        ]
        for u in self.out_of_scope_uses or ["Not specified"]:
            lines.append(f"- {u}")

        lines.extend([
            "",
            "## Training Data",
            f"- Data Hash (SHA256, first 16 chars): `{self.training_data_hash}`",
            f"- Training Rows: {self.n_training_rows:,}",
            f"- Date Range: {self.training_date_range}",
            f"- Number of Features: {self.n_features}",
            f"- Feature List: {', '.join(self.feature_list) if self.feature_list else 'N/A'}",
            f"- Data Sources: {self.data_sources or 'User-provided CSV/Parquet'}",
            "",
            "## Evaluation Results",
            f"- Primary Metric: {self.primary_metric} = {self.metric_value:.4f} ± {self.metric_std:.4f} (n={self.metric_n_runs})",
            f"- Baseline Comparison: {self.baseline_comparison}",
            f"- Optimization Rounds: {self.optimization_rounds}",
        ])

        if self.data_snooping_warning:
            lines.append("- ⚠ DATA SNOOPING WARNING: Validation score may be optimistic due to repeated tuning.")
        if self.heuristic_first_warning:
            lines.append("- ⚠ HEURISTIC SUFFICIENT: A simple baseline already meets performance targets. ML may not be necessary.")

        lines.extend([
            "",
            "## Limitations",
        ])
        for lim in self.limitations or ["Automated model — manual review recommended before production use."]:
            lines.append(f"- {lim}")
        if self.known_biases:
            lines.append("\n### Known Biases")
            for b in self.known_biases:
                lines.append(f"- {b}")
        if self.event_windows_excluded:
            lines.append(f"\n### Excluded Event Windows ({len(self.event_windows_excluded)} events)")
            for e in self.event_windows_excluded:
                lines.append(f"- {e.get('date', '?')}: {e.get('label', '?')} (severity={e.get('severity', '?')})")

        lines.extend([
            "",
            "## Ethical Considerations",
        ])
        if self.feedback_loop_warning:
            lines.append("- ⚠ FEEDBACK LOOP: Predictions may influence future data. Results could become self-reinforcing over time.")
        if self.proxy_label_warning:
            lines.append("- ⚠ PROXY LABEL: The target column may be a proxy for the true objective, not the objective itself.")
        if self.fairness_notes:
            lines.append(f"- {self.fairness_notes}")
        if self.ethical_review_required:
            lines.append("- ⚠ ETHICAL REVIEW REQUIRED before production deployment.")

        lines.extend([
            "",
            "---",
            f"*Auto-generated by EndForecast on {self.created_at}*",
        ])
        return "\n".join(lines)

    def save(self, path: str | Path) -> None:
        """Save the model card to a Markdown file."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(self.to_markdown(), encoding="utf-8")
        logger.info("Model card saved to %s", path)


class ModelCardGenerator:
    """Generate a ModelCard from a completed EndForecast RunResult.

    Usage:
        >>> generator = ModelCardGenerator()
        >>> card = generator.generate(result)
        >>> card.save("./artifacts/model_card.md")
    """

    def generate(self, result: Any) -> ModelCard:
        """Produce a ModelCard from a RunResult.

        Args:
            result: An EndForecast RunResult object.

        Returns:
            ModelCard populated from the run's lineage and evaluation data.
        """
        tiers = result.metric_tiers or {}
        report = result.report
        best = result.best_trial
        requirements = result.requirements
        req = requirements or (type('_R', (), {
            'business_purpose': '', 'feedback_loop_possible': False,
            'proxy_label_warning': False, 'event_timeline': [],
            'heuristic_rules': [],
        })())

        # Section 1: Model Details
        card = ModelCard(
            pipeline_id=tiers.get("pipeline_id", "unknown"),
            model_family=best.trial.model if best else "unknown",
            task_type=report.task_type if report else "unknown",
            created_at=datetime.now(timezone.utc).isoformat(timespec="seconds"),
        )

        # Section 2: Intended Use
        card.intended_use = getattr(req, 'business_purpose', '') or "Automated prediction pipeline"
        card.out_of_scope_uses = [
            "High-stakes decisions without human review" if not card.intended_use else "",
            "Use with data from different distributions without retraining",
            "Real-time safety-critical systems",
        ]
        card.out_of_scope_uses = [u for u in card.out_of_scope_uses if u]

        # Section 3: Training Data
        card.training_data_hash = tiers.get("training_data_hash", "")
        card.n_training_rows = tiers.get("n_training_rows", 0)
        card.training_date_range = tiers.get("training_date_range", "")
        card.feature_list = tiers.get("feature_list", [])
        card.n_features = len(card.feature_list)

        # Section 4: Evaluation
        if best:
            card.primary_metric = best.trial.metric
            card.metric_value = best.metric_value
            card.metric_std = best.metric_std
            card.metric_n_runs = best.metric_n_runs
        if result.baseline_result:
            card.baseline_comparison = f"Best model ({card.metric_value:.4f}) vs. baseline floor ({result.baseline_result.baseline_floor:.4f})"
        card.optimization_rounds = tiers.get("optimization_rounds", 0)
        card.data_snooping_warning = tiers.get("data_snooping_warning", False)
        card.heuristic_first_warning = tiers.get("heuristic_first_warning", False)

        # Section 5: Limitations
        card.limitations = []
        if report and report.fingerprint:
            fp = report.fingerprint
            if fp.n_obs < 500:
                card.limitations.append(f"Small dataset ({fp.n_obs} observations). Model generalization may be limited.")
            if fp.missing_ratio > 0.05:
                card.limitations.append(f"Data had {fp.missing_ratio:.1%} missing values. Imputation was applied.")
        if tiers.get("cold_start_detected"):
            card.limitations.append("Some groups have very few samples — predictions for these groups may be unreliable.")
        card.event_windows_excluded = getattr(req, 'event_timeline', []) or []

        # Section 6: Ethical
        card.feedback_loop_warning = getattr(req, 'feedback_loop_possible', False)
        card.proxy_label_warning = getattr(req, 'proxy_label_warning', False)
        if card.feedback_loop_warning or card.proxy_label_warning:
            card.ethical_review_required = True
            card.fairness_notes = "Manual review recommended before production deployment."

        logger.info("Model card generated for %s", card.pipeline_id)
        return card