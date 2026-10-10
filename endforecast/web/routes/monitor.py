"""Monitor routes — check run status, project health."""

from __future__ import annotations

from fastapi import APIRouter

router = APIRouter()


@router.get("/monitor/status")
async def monitor_status():
    """Global platform status."""
    return {
        "status": "healthy",
        "uptime_seconds": 0,
        "active_runs": 0,
        "total_projects": 0,
    }