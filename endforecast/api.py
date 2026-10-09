"""
FastAPI application for serving EndForecast predictions as a REST API.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from endforecast.engine.pipeline import Pipeline, PipelineConfig
from endforecast import __version__


class PredictRequest(BaseModel):
    pipeline_id: str
    data_path: Optional[str] = None
    data: Optional[list[dict]] = None


class PredictResponse(BaseModel):
    pipeline_id: str
    predictions: list[float]
    raw_predictions: Optional[list[float]] = None
    metrics: dict[str, float] = {}
    warnings: list[str] = []


_PIPELINES: dict[str, Pipeline] = {}


def create_app(pipeline_dir: Optional[str | Path] = None) -> FastAPI:
    app = FastAPI(
        title="EndForecast — Predict API",
        description="AI agent-driven adaptive prediction service",
        version=__version__,
    )
    app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

    if pipeline_dir:
        _load_pipelines(Path(pipeline_dir))

    @app.get("/health")
    async def health():
        return {"status": "ok", "version": __version__, "pipelines_loaded": len(_PIPELINES)}

    @app.post("/predict", response_model=PredictResponse)
    async def predict(req: PredictRequest):
        import pandas as pd
        if req.pipeline_id not in _PIPELINES:
            raise HTTPException(404, f"Pipeline '{req.pipeline_id}' not found.")
        if req.data_path:
            path = Path(req.data_path)
            df = pd.read_csv(path) if path.suffix == ".csv" else pd.read_parquet(path)
        elif req.data:
            df = pd.DataFrame(req.data)
        else:
            raise HTTPException(400, "Provide data_path or data.")
        result = _PIPELINES[req.pipeline_id].predict(df)
        return PredictResponse(
            pipeline_id=req.pipeline_id,
            predictions=result.predictions.tolist(),
            raw_predictions=result.raw_predictions.tolist() if result.raw_predictions is not None else None,
            metrics=result.metrics, warnings=result.warnings,
        )

    @app.post("/pipelines/{pipeline_id}")
    async def register_pipeline(pipeline_id: str, config_path: str):
        cfg = PipelineConfig.load(config_path)
        _PIPELINES[pipeline_id] = Pipeline(cfg)
        return {"pipeline_id": pipeline_id, "status": "registered"}

    @app.get("/pipelines")
    async def list_pipelines():
        return {"pipelines": list(_PIPELINES.keys())}

    return app


def _load_pipelines(directory: Path) -> None:
    for cf in directory.glob("*.json"):
        try:
            cfg = PipelineConfig.load(cf)
            _PIPELINES[cfg.pipeline_id] = Pipeline(cfg)
        except Exception:
            pass