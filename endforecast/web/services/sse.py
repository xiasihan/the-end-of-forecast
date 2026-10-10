"""SSE event emitter for streaming pipeline execution to clients."""

from __future__ import annotations

import asyncio
import json
import logging
import time
from typing import Any, Optional

from endforecast.hooks import PipelineHooks, PhaseEvent, RoundEvent, LLMEvent

logger = logging.getLogger(__name__)


class SSEHooks(PipelineHooks):
    """PipelineHooks implementation that emits SSE events to an asyncio Queue.

    The web endpoint reads from the queue and sends SSE messages to the client.
    """

    def __init__(self, queue: asyncio.Queue) -> None:
        super().__init__()
        self.queue = queue
        self._phase_timers: dict[int, float] = {}

    def _emit(self, event_type: str, data: dict[str, Any]) -> None:
        """Push an event to the async queue (non-blocking)."""
        payload = json.dumps(data, default=str, ensure_ascii=False)
        try:
            self.queue.put_nowait({"event": event_type, "data": payload})
        except asyncio.QueueFull:
            logger.warning("SSE queue full, dropping event: %s", event_type)

    def fire_phase_start(self, phase: int, name: str) -> None:
        self._phase_timers[phase] = time.time()
        self._emit("phase_start", {
            "phase": phase,
            "name": name,
            "timestamp": time.time(),
        })

    def fire_phase_complete(
        self,
        phase: int,
        name: str,
        status: str = "completed",
        data: Optional[dict[str, Any]] = None,
        elapsed_ms: int = 0,
        logs: Optional[list[str]] = None,
    ) -> None:
        actual_elapsed = elapsed_ms or int((time.time() - self._phase_timers.get(phase, time.time())) * 1000)
        self._emit("phase_complete", {
            "phase": phase,
            "status": status,
            "data": data,
            "elapsed_ms": actual_elapsed,
            "logs": logs or [],
            "timestamp": time.time(),
        })

    def fire_round_start(self, round_num: int, n_trials: int) -> None:
        self._emit("round_start", {
            "round_num": round_num,
            "n_trials": n_trials,
            "timestamp": time.time(),
        })

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
        self._emit("round_complete", {
            "round_num": round_num,
            "status": "completed",
            "best_model": best_model,
            "best_metric": best_metric,
            "best_metric_name": best_metric_name,
            "trial_summaries": trial_summaries or [],
            "diagnosis": diagnosis,
            "elapsed_ms": elapsed_ms,
            "timestamp": time.time(),
        })

    def fire_llm_decision(
        self, agent: str, decision_summary: str, rationale: str = "",
    ) -> None:
        self._emit("llm_decision", {
            "agent": agent,
            "decision_summary": decision_summary,
            "rationale": rationale,
            "timestamp": time.time(),
        })

    def fire_error(self, phase: int, name: str, message: str) -> None:
        self._emit("error", {
            "phase": phase,
            "message": message,
            "timestamp": time.time(),
        })