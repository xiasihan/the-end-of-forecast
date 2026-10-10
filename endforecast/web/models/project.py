"""Pydantic models for the web platform."""

from __future__ import annotations

from datetime import datetime
from typing import Optional

from pydantic import BaseModel, Field


class ProjectCreate(BaseModel):
    name: str = Field(..., min_length=1, max_length=128, description="Project name")
    description: str = Field("", description="Natural-language prediction goal")
    target_col: str = Field("y", description="Target column name")
    time_col: Optional[str] = Field("ds", description="Time column name")
    id_col: Optional[str] = Field(None, description="ID column name")
    budget: str = Field("auto", description="auto | fast | standard | deep")


class ProjectResponse(BaseModel):
    id: str
    name: str
    description: str
    status: str  # "idle" | "running" | "completed" | "failed"
    created_at: str
    updated_at: str
    n_rows: Optional[int] = None
    n_cols: Optional[int] = None
    task_type: Optional[str] = None
    latest_holdout_score: Optional[float] = None


class ProjectListResponse(BaseModel):
    projects: list[ProjectResponse]


class RunResponse(BaseModel):
    id: str
    project_id: str
    status: str  # "running" | "completed" | "failed" | "cancelled"
    started_at: Optional[str] = None
    completed_at: Optional[str] = None
    elapsed_seconds: Optional[float] = None
    summary: Optional[str] = None


class RunListResponse(BaseModel):
    runs: list[RunResponse]


class PredictRequest(BaseModel):
    data: list[dict] = Field(..., description="List of row dicts with feature columns")


class PredictResponse(BaseModel):
    predictions: list[float]
    pipeline_id: str
    model: str
    metric: str


class ApiKeyCreate(BaseModel):
    name: str = Field(..., min_length=1, max_length=64)


class ApiKeyResponse(BaseModel):
    id: str
    name: str
    prefix: str  # First 8 chars of key
    created_at: str
    last_used: Optional[str] = None
    raw_key: Optional[str] = None  # Only returned on creation, never again