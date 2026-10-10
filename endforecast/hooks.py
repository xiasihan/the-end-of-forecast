"""
Pipeline hooks — callback interface for streaming/progress reporting.

Allows external systems (web UI, CLI monitor, logging framework) to
observe and react to pipeline execution in real-time without modifying
the orchestrator's internal logic.

Usage:
    >>> from endforecast.hooks import PipelineHooks, PhaseEvent
    >>> hooks = PipelineHooks()
    >>> hooks.on_phase_complete = lambda e: print(f"Phase {e.phase} done")
    >>> result = ef.run("data.csv", hooks=hooks)
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Callable, Optional


@dataclass
class PhaseEvent:
    """Emitted when a pipeline phase starts or completes."""

    phase: int
    name: str
    status: str  # "running" | "completed" | "failed"
    data: Optional[dict[str, Any]] = None
    elapsed_ms: int = 0
    logs: list[str] = field(default_factory=list)
    timestamp: float = 0.0

    def __post_init__(self) -> None:
        if self.timestamp == 0.0:
            self.timestamp = time.time()


@dataclass
class RoundEvent:
    """Emitted at the start/end of each experiment round."""

    round_num: int
    n_trials: int = 0
    status: str = "running"  # "running" | "completed"
    best_model: Optional[str] = None
    best_metric: Optional[float] = None
    best_metric_name: Optional[str] = None
    trial_summaries: list[dict[str, Any]] = field(default_factory=list)
    diagnosis: Optional[dict[str, Any]] = None
    elapsed_ms: int = 0
    timestamp: float = 0.0

    def __post_init__(self) -> None:
        if self.timestamp == 0.0:
            self.timestamp = time.time()


@dataclass
class LLMEvent:
    """Emitted whenever an LLM agent makes a decision."""

    agent: str  # "planner" | "diagnostician" | "refiner" | "ensemble" | "configurator"
    decision_summary: str  # 1-line summary of what the LLM decided
    rationale: str = ""
    timestamp: float = 0.0

    def __post_init__(self) -> None:
        if self.timestamp == 0.0:
            self.timestamp = time.time()


class PipelineHooks:
    """Callback hooks observed by external systems during pipeline execution.

    All callbacks are optional — if None, the hook is silently skipped.
    Implementations must be fast and non-blocking (run in a background
    thread or queue if expensive). Hooks must never raise exceptions that
    propagate back to the orchestrator.

    Example:
        >>> class MyHooks(PipelineHooks):
        ...     def on_phase_complete(self, event: PhaseEvent) -> None:
        ...         print(f"[{event.phase}] {event.name}: {event.status}")
    """

    on_phase_start: Optional[Callable[[int, str], None]] = None
    on_phase_complete: Optional[Callable[[PhaseEvent], None]] = None

    on_round_start: Optional[Callable[[int, int], None]] = None
    on_round_complete: Optional[Callable[[RoundEvent], None]] = None

    on_llm_decision: Optional[Callable[[LLMEvent], None]] = None

    on_error: Optional[Callable[[int, str, str], None]] = None  # phase, name, message

    def fire_phase_start(self, phase: int, name: str) -> None:
        """Called when a phase begins execution."""
        if self.on_phase_start:
            try:
                self.on_phase_start(phase, name)
            except Exception:
                pass

    def fire_phase_complete(
        self,
        phase: int,
        name: str,
        status: str = "completed",
        data: Optional[dict[str, Any]] = None,
        elapsed_ms: int = 0,
        logs: Optional[list[str]] = None,
    ) -> None:
        """Called when a phase finishes execution."""
        if self.on_phase_complete:
            try:
                self.on_phase_complete(PhaseEvent(
                    phase=phase, name=name, status=status,
                    data=data, elapsed_ms=elapsed_ms,
                    logs=logs or [],
                ))
            except Exception:
                pass

    def fire_round_start(self, round_num: int, n_trials: int) -> None:
        """Called when an experiment round begins."""
        if self.on_round_start:
            try:
                self.on_round_start(round_num, n_trials)
            except Exception:
                pass

    def fire_round_complete(
        self,
        round_num: int,
        n_trials: int,
        best_model: Optional[str] = None,
        best_metric: Optional[float] = None,
        best_metric_name: Optional[str] = None,
        trial_summaries: Optional[list[dict[str, Any]]] = None,
        diagnosis: Optional[dict[str, Any]] = None,
        elapsed_ms: int = 0,
    ) -> None:
        """Called when an experiment round finishes."""
        if self.on_round_complete:
            try:
                self.on_round_complete(RoundEvent(
                    round_num=round_num, n_trials=n_trials,
                    status="completed",
                    best_model=best_model, best_metric=best_metric,
                    best_metric_name=best_metric_name,
                    trial_summaries=trial_summaries or [],
                    diagnosis=diagnosis, elapsed_ms=elapsed_ms,
                ))
            except Exception:
                pass

    def fire_llm_decision(
        self, agent: str, decision_summary: str, rationale: str = "",
    ) -> None:
        """Called when an LLM agent makes a decision."""
        if self.on_llm_decision:
            try:
                self.on_llm_decision(LLMEvent(
                    agent=agent, decision_summary=decision_summary,
                    rationale=rationale,
                ))
            except Exception:
                pass

    def fire_error(self, phase: int, name: str, message: str) -> None:
        """Called when a phase encounters a non-fatal error."""
        if self.on_error:
            try:
                self.on_error(phase, name, message)
            except Exception:
                pass