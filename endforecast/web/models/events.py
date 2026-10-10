"""SSE event types for streaming pipeline execution."""

from __future__ import annotations

from typing import Any, Optional
from pydantic import BaseModel


class SSEPhaseStart(BaseModel):
    event: str = "phase_start"
    phase: int
    name: str
    timestamp: float


class SSEPhaseComplete(BaseModel):
    event: str = "phase_complete"
    phase: int
    status: str  # "completed" | "failed"
    data: Optional[dict[str, Any]] = None
    elapsed_ms: int = 0
    logs: list[str] = []
    timestamp: float


class SSERoundStart(BaseModel):
    event: str = "round_start"
    round_num: int
    n_trials: int
    timestamp: float


class SSERoundComplete(BaseModel):
    event: str = "round_complete"
    round_num: int
    status: str = "completed"
    best_model: Optional[str] = None
    best_metric: Optional[float] = None
    best_metric_name: Optional[str] = None
    trial_summaries: list[dict[str, Any]] = []
    diagnosis: Optional[dict[str, Any]] = None
    elapsed_ms: int = 0
    timestamp: float


class SSELLMDecision(BaseModel):
    event: str = "llm_decision"
    agent: str  # "planner" | "diagnostician" | "refiner" | "ensemble" | "configurator"
    decision_summary: str
    rationale: str = ""
    timestamp: float


class SSEError(BaseModel):
    event: str = "error"
    phase: Optional[int] = None
    message: str
    timestamp: float


class SSEComplete(BaseModel):
    event: str = "complete"
    run_id: str
    summary: str
    holdout_score: Optional[float] = None
    validation_score: Optional[float] = None
    best_model: Optional[str] = None
    best_model_metric: Optional[str] = None
    best_model_value: Optional[float] = None
    timestamp: float