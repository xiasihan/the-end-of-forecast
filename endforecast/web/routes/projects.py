"""Project CRUD routes."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

from fastapi import APIRouter, File, Form, HTTPException, UploadFile

from endforecast.web.models.project import (
    ProjectCreate,
    ProjectResponse,
    ProjectListResponse,
)
from endforecast.web.services.session import SessionManager

router = APIRouter()
sessions: Optional[SessionManager] = None  # Injected by app.py


@router.post("/projects", response_model=ProjectResponse)
async def create_project(
    name: str = Form(...),
    description: str = Form(""),
    target_col: str = Form("y"),
    time_col: Optional[str] = Form(None),
    id_col: Optional[str] = Form(None),
    budget: str = Form("auto"),
    file: UploadFile = File(...),
):
    """Create a new project with uploaded data file."""
    allowed_extensions = {".csv", ".parquet", ".pqt", ".xlsx", ".xls"}
    suffix = Path(file.filename or "data.csv").suffix.lower()
    if suffix not in allowed_extensions:
        raise HTTPException(400, f"Unsupported file type: {suffix}. Allowed: {allowed_extensions}")

    data_bytes = await file.read()
    meta = sessions.create_project(
        name=name, description=description,
        target_col=target_col, time_col=time_col,
        id_col=id_col, budget=budget,
        data_bytes=data_bytes, data_filename=file.filename or "data.csv",
    )
    return ProjectResponse(
        id=meta["id"], name=meta["name"], description=meta["description"],
        status="idle", created_at=meta["created_at"], updated_at=meta["updated_at"],
    )


@router.get("/projects", response_model=ProjectListResponse)
async def list_projects():
    """List all projects."""
    projs = sessions.list_projects()
    return ProjectListResponse(
        projects=[
            ProjectResponse(
                id=p["id"], name=p["name"], description=p.get("description", ""),
                status=p.get("status", "idle"),
                created_at=p.get("created_at", ""), updated_at=p.get("updated_at", ""),
                n_rows=p.get("n_rows"), n_cols=p.get("n_cols"),
                task_type=p.get("task_type"),
                latest_holdout_score=p.get("latest_holdout_score"),
            )
            for p in projs
        ]
    )


@router.get("/projects/{proj_id}", response_model=ProjectResponse)
async def get_project(proj_id: str):
    """Get a project by ID."""
    p = sessions.get_project(proj_id)
    if not p:
        raise HTTPException(404, "Project not found")
    return ProjectResponse(
        id=p["id"], name=p["name"], description=p.get("description", ""),
        status=p.get("status", "idle"),
        created_at=p.get("created_at", ""), updated_at=p.get("updated_at", ""),
        n_rows=p.get("n_rows"), n_cols=p.get("n_cols"),
        task_type=p.get("task_type"),
        latest_holdout_score=p.get("latest_holdout_score"),
    )


@router.put("/projects/{proj_id}")
async def update_project(proj_id: str, name: Optional[str] = None, description: Optional[str] = None):
    """Update project name or description."""
    p = sessions.get_project(proj_id)
    if not p:
        raise HTTPException(404, "Project not found")
    updates = {}
    if name is not None:
        updates["name"] = name
    if description is not None:
        updates["description"] = description
    if updates:
        sessions.update_project(proj_id, **updates)
    return {"status": "ok"}


@router.delete("/projects/{proj_id}")
async def delete_project(proj_id: str):
    """Delete a project and all its data."""
    p = sessions.get_project(proj_id)
    if not p:
        raise HTTPException(404, "Project not found")
    sessions.delete_project(proj_id)
    return {"status": "ok"}