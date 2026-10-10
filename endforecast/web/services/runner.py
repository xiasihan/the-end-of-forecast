"""Pipeline runner — wraps EndForecast.run() with SSE hooks for web streaming."""

from __future__ import annotations

import asyncio
import json
import logging
import time
import traceback
from typing import Optional

import numpy as np
import pandas as pd

from endforecast.orchestrator import EndForecast, RunResult
from endforecast.web.services.sse import SSEHooks

logger = logging.getLogger(__name__)


def _serialize_result(run_result: RunResult) -> dict:
    """Convert RunResult to a JSON-safe dict for storage/streaming.
    
    Sanitizes NaN, Infinity values that are not JSON-compliant.
    """
    def _sanitize(obj):
        if isinstance(obj, float):
            if not np.isfinite(obj):
                return None
            return obj
        if isinstance(obj, dict):
            return {k: _sanitize(v) for k, v in obj.items()}
        if isinstance(obj, (list, tuple)):
            return [_sanitize(v) for v in obj]
        return obj

    data: dict = {
        "success": run_result.success,
        "errors": run_result.errors,
        "metric_tiers": _sanitize(run_result.metric_tiers),
    }

    if run_result.report:
        data["task_type"] = run_result.report.task_type
        data["n_rows"] = run_result.report.n_rows
        data["n_cols"] = run_result.report.n_cols
        data["quality_flags"] = run_result.report.quality_flags

    if run_result.baseline_result and run_result.baseline_result.best:
        data["baseline_floor"] = run_result.baseline_result.baseline_floor
        data["baseline_model"] = run_result.baseline_result.best.trial.model

    if run_result.best_trial:
        bt = run_result.best_trial
        data["best_model"] = bt.trial.model
        data["best_metric_name"] = bt.trial.metric
        data["best_metric_value"] = bt.metric_value
        data["best_metric_std"] = bt.metric_std
        data["best_features"] = bt.trial.features

    if run_result.pipeline:
        data["pipeline_id"] = run_result.pipeline.config.pipeline_id

    data["elapsed_seconds"] = 0  # Will be set by caller
    data["summary"] = run_result.summary()

    return data


def run_pipeline_sync(
    data_path: str,
    target_col: str,
    time_col: Optional[str],
    id_col: Optional[str],
    requirements: str,
    budget: str,
    hooks: SSEHooks,
) -> dict:
    """Run EndForecast synchronously with SSE hooks. Called from a thread pool.

    Returns a JSON-safe dict of the final result.
    """
    ef = EndForecast()
    t0 = time.time()

    try:
        result = ef.run(
            data=data_path,
            target_col=target_col or "y",
            time_col=time_col or "ds",
            id_col=id_col if id_col else None,
            requirements=requirements or "Predict target",
            budget=budget or "auto",
            hooks=hooks,
        )
        elapsed = time.time() - t0
        data = _serialize_result(result)
        data["elapsed_seconds"] = elapsed
        return data
    except Exception as exc:
        elapsed = time.time() - t0
        hooks.fire_error(0, "Pipeline", f"{type(exc).__name__}: {exc}")
        return {
            "success": False,
            "errors": [f"{type(exc).__name__}: {exc}\n{traceback.format_exc()}"],
            "elapsed_seconds": elapsed,
        }


async def run_pipeline_async(
    data_path: str,
    target_col: str,
    time_col: Optional[str],
    id_col: Optional[str],
    requirements: str,
    budget: str,
    queue: asyncio.Queue,
) -> dict:
    """Run EndForecast in a thread pool, streaming events to the queue."""
    hooks = SSEHooks(queue)
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(
        None,
        run_pipeline_sync,
        data_path, target_col, time_col, id_col, requirements, budget, hooks,
    )