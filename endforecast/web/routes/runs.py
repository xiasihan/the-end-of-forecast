"""Pipeline run routes — start, SSE stream, query runs.

Uses a shared per-run asyncio.Queue so the background pipeline runner
and the SSE streaming endpoint see the same events.
"""

from __future__ import annotations

import asyncio
import json
import logging
import math
import time
from typing import Optional

from fastapi import APIRouter, HTTPException, Query
from starlette.responses import StreamingResponse

from endforecast.web.services.session import SessionManager
from endforecast.web.services.runner import run_pipeline_async

logger = logging.getLogger(__name__)

router = APIRouter()
sessions: Optional[SessionManager] = None  # Injected by app.py

# Shared queues: run_id → asyncio.Queue[dict]
_run_queues: dict[str, asyncio.Queue] = {}


def _sanitize_nan(obj):
    """Recursively replace NaN/Infinity with None for JSON compliance."""
    if isinstance(obj, float):
        return None if (math.isnan(obj) or math.isinf(obj)) else obj
    if isinstance(obj, dict):
        return {k: _sanitize_nan(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_sanitize_nan(v) for v in obj]
    return obj


@router.post("/projects/{proj_id}/runs")
async def start_run(
    proj_id: str,
    requirements: Optional[str] = Query(None),
    budget: Optional[str] = Query(None),
):
    """Start a new pipeline run. Returns run_id immediately."""
    meta = sessions.get_project(proj_id)
    if not meta:
        raise HTTPException(404, "Project not found")

    data_path = sessions.project_data_path(proj_id)
    if not data_path or not data_path.exists():
        raise HTTPException(400, "Data file not found")

    run_id = sessions.create_run(proj_id)
    queue: asyncio.Queue = asyncio.Queue(maxsize=256)
    _run_queues[run_id] = queue

    async def _bg_run():
        try:
            result = await run_pipeline_async(
                data_path=str(data_path),
                target_col=meta.get("target_col", "y"),
                time_col=meta.get("time_col"),
                id_col=meta.get("id_col"),
                requirements=requirements or meta.get("description", "Predict target"),
                budget=budget or meta.get("budget", "auto"),
                queue=queue,
            )
            sessions.complete_run(proj_id, run_id, result)
            await queue.put({
                "event": "complete",
                "data": json.dumps({
                    "run_id": run_id,
                    "summary": result.get("summary", ""),
                    "holdout_score": result.get("metric_tiers", {}).get("holdout_score"),
                    "validation_score": result.get("metric_tiers", {}).get("validation_score"),
                    "best_model": result.get("best_model"),
                    "best_model_metric": result.get("best_metric_name"),
                    "best_model_value": result.get("best_metric_value"),
                    "timestamp": time.time(),
                }, default=str, ensure_ascii=False),
            })
        except Exception as exc:
            logger.exception("Pipeline run %s failed", run_id)
            sessions.fail_run(proj_id, run_id, str(exc))
            await queue.put({
                "event": "error",
                "data": json.dumps({"message": str(exc), "timestamp": time.time()}, ensure_ascii=False),
            })
        finally:
            await queue.put(None)  # Sentinel
            _run_queues.pop(run_id, None)

    asyncio.create_task(_bg_run())
    return {"run_id": run_id, "project_id": proj_id, "status": "running"}


@router.get("/projects/{proj_id}/runs/{run_id}/stream")
async def stream_run(proj_id: str, run_id: str):
    """SSE stream of pipeline execution events."""
    run = sessions.get_run(proj_id, run_id)
    if not run:
        raise HTTPException(404, "Run not found")

    # If run already done, serve cached result
    if run.get("status") in ("completed", "failed"):
        result = run.get("result", {})
        async def _cached():
            yield f"event: complete\ndata: {json.dumps(result, default=str)}\n\n"
        return StreamingResponse(_cached(), media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})

    # If still running, read from shared queue
    queue = _run_queues.get(run_id)
    if not queue:
        raise HTTPException(400, "Run queue not found — may have already completed")

    async def _stream():
        while True:
            try:
                msg = await asyncio.wait_for(queue.get(), timeout=60)
            except asyncio.TimeoutError:
                yield f": heartbeat\n\n"
                # Check if run completed externally
                current = sessions.get_run(proj_id, run_id)
                if current and current.get("status") in ("completed", "failed"):
                    result = current.get("result", {})
                    yield f"event: complete\ndata: {json.dumps(result, default=str)}\n\n"
                    return
                continue

            if msg is None:
                return
            yield f"event: {msg['event']}\ndata: {msg['data']}\n\n"

    return StreamingResponse(
        _stream(), media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "Connection": "keep-alive", "X-Accel-Buffering": "no"},
    )


@router.get("/projects/{proj_id}/runs")
async def list_runs(proj_id: str):
    runs = sessions.list_runs(proj_id)
    return {"runs": runs}


@router.get("/projects/{proj_id}/runs/{run_id}")
async def get_run(proj_id: str, run_id: str):
    run = sessions.get_run(proj_id, run_id)
    if not run:
        raise HTTPException(404, "Run not found")
    return _sanitize_nan(run)


@router.post("/projects/{proj_id}/runs/{run_id}/cancel")
async def cancel_run(proj_id: str, run_id: str):
    q = _run_queues.pop(run_id, None)
    if q:
        await q.put(None)
    sessions.fail_run(proj_id, run_id, "Cancelled by user")
    return {"status": "cancelled"}