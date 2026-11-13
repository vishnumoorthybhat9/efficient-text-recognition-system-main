"""Health and system-information endpoints."""

from __future__ import annotations

import logging
from datetime import datetime, timezone

from fastapi import APIRouter
from sqlalchemy import text

from backend.api.deps import RuntimeDep, SessionDep, SettingsDep
from backend.schemas.common import HealthResponse, SystemInfoResponse

__all__ = ["router"]

logger = logging.getLogger(__name__)

router = APIRouter(tags=["system"])


@router.get("/health", response_model=HealthResponse, summary="Liveness and readiness")
def health(session: SessionDep, runtime: RuntimeDep, settings: SettingsDep) -> HealthResponse:
    """Report whether the API, database, models and dataset are ready.

    ``status`` is ``"ok"`` only when the database responds *and* at least one
    model is usable. A degraded install (no trained model yet) reports
    ``"degraded"`` rather than failing, because the UI is still useful for
    dataset preparation and training.
    """
    database_ready = True
    try:
        session.execute(text("SELECT 1"))
    except Exception as exc:  # pragma: no cover - only on a broken database
        database_ready = False
        logger.error("Database health check failed: %s", exc)

    models_available = sum(1 for entry in runtime.registry.visible_entries() if entry.available)
    dataset_ready = runtime.dataset_ready()

    status = "ok" if (database_ready and models_available) else "degraded"

    return HealthResponse(
        status=status,
        app_name=settings.app_name,
        version=settings.app_version,
        environment=settings.environment,
        device=str(runtime.device),
        models_available=models_available,
        database_ready=database_ready,
        dataset_ready=dataset_ready,
        uptime_seconds=runtime.uptime_seconds,
        timestamp=datetime.now(timezone.utc),
    )


@router.get(
    "/system/info",
    response_model=SystemInfoResponse,
    summary="Hardware and software report",
)
def system_info(runtime: RuntimeDep) -> SystemInfoResponse:
    """GPU, CPU, memory and library versions, as detected at start-up."""
    return SystemInfoResponse(**runtime.system_info())
