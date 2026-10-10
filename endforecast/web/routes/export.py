"""Export routes — download predictor.py, manage API keys, production predict."""

from __future__ import annotations

import json
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, Header, Request
from fastapi.responses import FileResponse, JSONResponse

from endforecast.web.services.session import SessionManager
from endforecast.web.models.project import PredictRequest, PredictResponse, ApiKeyCreate, ApiKeyResponse

router = APIRouter()
sessions: Optional[SessionManager] = None  # Injected by app.py


@router.get("/projects/{proj_id}/runs/{run_id}/export")
async def export_predictor(proj_id: str, run_id: str):
    """Download the exported predictor.py for a completed run."""
    pred_path = sessions.run_dir(proj_id, run_id) / "predictor.py"
    if not pred_path.exists():
        raise HTTPException(404, "Predictor not found. Pipeline may not have completed successfully.")
    return FileResponse(pred_path, filename=f"predictor_{proj_id}.py", media_type="text/x-python")


@router.post("/projects/{proj_id}/predict", response_model=PredictResponse)
async def predict(
    proj_id: str,
    body: PredictRequest,
    authorization: Optional[str] = Header(None),
):
    """Production prediction endpoint. Requires API key."""
    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(401, "API key required: Bearer <key>")
    raw_key = authorization[len("Bearer "):]
    if not sessions.verify_api_key(raw_key):
        raise HTTPException(403, "Invalid API key")

    meta = sessions.get_project(proj_id)
    if not meta:
        raise HTTPException(404, "Project not found")

    # Load latest run's pipeline
    runs = sessions.list_runs(proj_id)
    completed = [r for r in runs if r.get("status") == "completed"]
    if not completed:
        raise HTTPException(400, "No completed pipeline found. Run the pipeline first.")

    latest_run = sessions.get_run(proj_id, completed[0]["id"])
    result = latest_run.get("result", {}) if latest_run else {}

    # Use last known pipeline to predict
    import pandas as pd
    from endforecast.engine.pipeline import Pipeline, PipelineConfig

    df = pd.DataFrame(body.data)
    # For now return a placeholder — full production predict needs pipeline deserialization
    return PredictResponse(
        predictions=[0.0] * len(df),
        pipeline_id=result.get("pipeline_id", f"{proj_id}_v1"),
        model=result.get("best_model", "ridge"),
        metric=result.get("best_metric_name", "mase"),
    )


@router.post("/keys", response_model=ApiKeyResponse)
async def create_api_key(body: ApiKeyCreate):
    """Create a new API key for production access."""
    key_dict, raw_key = sessions.create_api_key(body.name)
    return ApiKeyResponse(
        id=key_dict["id"], name=key_dict["name"],
        prefix=key_dict["prefix"], created_at=key_dict["created_at"],
        # Include raw key only on creation — never again
        **{"raw_key": raw_key},
    )


@router.get("/keys")
async def list_api_keys():
    """List all API keys (prefix only, raw key never shown)."""
    keys = sessions.list_api_keys()
    return {"keys": keys}


@router.delete("/keys/{key_id}")
async def delete_api_key(key_id: str):
    """Revoke an API key."""
    sessions.delete_api_key(key_id)
    return {"status": "revoked"}