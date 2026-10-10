"""FastAPI application factory with shared SessionManager."""

from __future__ import annotations

from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from endforecast.web.services.session import SessionManager
from endforecast.web.routes import projects, runs, export, monitor

# Shared singleton — all routes use the same manager + same SQLite DB
sessions = SessionManager(base_dir="./endforecast_sessions")


def create_app() -> FastAPI:
    app = FastAPI(
        title="EndForecast Platform",
        description="Production ML prediction platform console",
        version="0.2.0",
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Inject shared sessions into route modules
    projects.sessions = sessions
    runs.sessions = sessions
    export.sessions = sessions

    app.include_router(projects.router, prefix="/api", tags=["projects"])
    app.include_router(runs.router, prefix="/api", tags=["runs"])
    app.include_router(export.router, prefix="/api", tags=["export"])
    app.include_router(monitor.router, prefix="/api", tags=["monitor"])

    @app.get("/api/health")
    async def health():
        return {"status": "ok", "version": "0.2.0"}

    return app


app = create_app()