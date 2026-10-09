"""Diagnostician Agent — failure analysis and strategy adjustment.

Architecture: The Diagnostician is the ANALYTICAL INTERFACE for the
LLM. When LLM is available, the DIAGNOSTICIAN_SYSTEM_PROMPT in
prompts.py drives the full diagnostic analysis. When LLM is NOT
available, diagnose() returns None (signal to orchestrator to skip
this round's diagnosis).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional

from endforecast.engine.trial import TrialResult

logger = logging.getLogger(__name__)


@dataclass
class DiagnosisReport:
    """Structured diagnosis produced by LLM or fallback.

    When LLM is available, all fields are populated by the
    DIAGNOSTICIAN_SYSTEM_PROMPT's output.
    """
    summary: str = ""
    observed_signal: str = ""
    root_cause: str = ""
    adjustment: str = ""
    confidence: str = "medium"
    findings: list[str] = []
    convergence_opinion: str = "continue"


class Diagnostician:
    """Diagnostician runs in two modes:

    - LLM-driven (primary): the LLM analyzes TrialResults and produces
      a free-form diagnostic report with rich reasoning.
    - No-LLM fallback: returns None (orchestrator skips diagnosis).

    The old _DIAGNOSTIC_RULES hardcoded table has been removed. All
    diagnostic logic now lives in DIAGNOSTICIAN_SYSTEM_PROMPT.
    """

    def diagnose(
        self,
        results: list[TrialResult],
        baseline_metric: float | None = None,
        previous_best: float | None = None,
        llm_output: Optional[dict] = None,
    ) -> Optional[DiagnosisReport]:
        """Produce a diagnosis from trial results.

        Args:
            results: Trial results from the current round.
            baseline_metric: Baseline metric for comparison.
            previous_best: Previous round's best metric.
            llm_output: Optional pre-parsed LLM output dict. If None,
                tries llm-driven analysis; if that returns nothing,
                returns None (skip diagnosis).

        Returns:
            DiagnosisReport or None (no diagnosis available).
        """
        if not results:
            return None

        # Check if LLM provided structured output
        if llm_output and isinstance(llm_output, dict):
            report = DiagnosisReport(
                summary=llm_output.get("summary", ""),
                observed_signal=llm_output.get("observed_signal", ""),
                root_cause=llm_output.get("root_cause", ""),
                adjustment=llm_output.get("adjustment", ""),
                confidence=llm_output.get("confidence", "medium"),
                findings=llm_output.get("findings", []),
                convergence_opinion=llm_output.get("convergence_opinion", "continue"),
            )
            logger.info("Diagnosis (LLM): %s", report.summary)
            return report

        # No LLM available — basic sanity check only
        successful = [r for r in results if r.is_successful]
        if not successful:
            report = DiagnosisReport(
                summary="All trials failed",
                observed_signal="Execution errors across all trials",
                root_cause="Systematic pipeline or data error",
                adjustment="Check data integrity and pipeline configuration",
                confidence="high",
                convergence_opinion="continue",
            )
            logger.info("Diagnosis (basic): %s", report.summary)
            return report

        # All trials succeeded but no LLM analysis — skip
        logger.debug("No LLM available for diagnosis. Skipping.")
        return None