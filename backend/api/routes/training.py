"""Training-job endpoints.

Training never starts on its own: a job exists only because this API was
asked for one. Only one job runs at a time -- see
:mod:`backend.services.training_service` for why.
"""

from __future__ import annotations

import logging
from datetime import datetime
from typing import Annotated, Any

from fastapi import APIRouter, Query
from pydantic import Field

from backend.api.deps import RuntimeDep, SessionDep
from backend.schemas.common import ApiModel, MessageResponse
from backend.services import training_service

__all__ = ["router"]

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/training", tags=["training"])


class StartTrainingIn(ApiModel):
    """Request body for ``POST /api/training/jobs``."""

    job: str = Field(description="One of the keys from GET /api/training/jobs/available")
    epochs: int | None = Field(default=None, ge=1, le=500, description="Override the config")
    batch_size: int | None = Field(default=None, ge=1, le=512)
    max_samples: int | None = Field(
        default=None, ge=1, description="Cap the labeled training subset"
    )


class TrainingJobOut(ApiModel):
    """A training job's state."""

    id: int
    created_at: datetime
    started_at: datetime | None = None
    finished_at: datetime | None = None
    kind: str
    config_name: str
    run_name: str
    status: str
    epochs_total: int
    epochs_completed: int
    current_epoch: int
    progress: float
    latest_metrics: dict[str, Any] = Field(default_factory=dict)
    history: list[Any] = Field(default_factory=list)
    checkpoint_dir: str | None = None
    log_path: str | None = None
    error: str | None = None
    device: str = ""


@router.get("/jobs/available", summary="Training jobs this project offers")
def available_jobs(runtime: RuntimeDep) -> dict[str, Any]:
    """The three stages of the experiment, with their prerequisites.

    ``ready`` says whether a job can start right now; when it cannot,
    ``blocked_reason`` explains what is missing.
    """
    manager = training_service.get_job_manager()
    jobs: list[dict[str, Any]] = []

    for key, spec in training_service.KNOWN_JOBS.items():
        ready = True
        reason = ""

        if not runtime.dataset_ready():
            ready = False
            reason = (
                "The dataset is not prepared. Run `python scripts/make_synthetic_data.py` "
                "or import a real one with `python scripts/prepare_dataset.py`."
            )
        elif key == "simclr_finetune":
            encoder = training_service.resolve_encoder_checkpoint(
                "artifacts/checkpoints/simclr_pretrain/encoder_best.pt"
            )
            if not encoder.is_file():
                ready = False
                reason = "Run the SimCLR pretraining job first; this one starts from its encoder."

        jobs.append(
            {
                "key": key,
                "kind": spec["kind"],
                "label": spec["label"],
                "description": spec["description"],
                "config": spec["config"],
                "ready": ready,
                "blocked_reason": reason,
            }
        )

    return {
        "jobs": jobs,
        "busy": manager.is_running,
        "active_job_id": manager.active_job_id,
        "device": str(runtime.device),
    }


@router.post("/jobs", response_model=TrainingJobOut, summary="Start a training job")
def start_job(
    session: SessionDep, runtime: RuntimeDep, payload: StartTrainingIn
) -> TrainingJobOut:
    """Start a training job in the background.

    Returns immediately with the job row; poll ``GET /api/training/jobs/{id}``
    for progress. Returns 409 if a job is already running, and 503 if the
    dataset or the stage-1 encoder is missing.
    """
    job = training_service.get_job_manager().start(
        session,
        runtime,
        job_key=payload.job,
        epochs=payload.epochs,
        batch_size=payload.batch_size,
        max_samples=payload.max_samples,
    )
    return TrainingJobOut.model_validate(job)


@router.get("/jobs", summary="Recent training jobs")
def list_jobs(
    session: SessionDep,
    limit: Annotated[int, Query(ge=1, le=200)] = 50,
) -> dict[str, Any]:
    """Training jobs, newest first."""
    manager = training_service.get_job_manager()
    jobs = training_service.list_jobs(session, limit=limit)
    return {
        "items": [TrainingJobOut.model_validate(job).model_dump() for job in jobs],
        "total": len(jobs),
        "busy": manager.is_running,
        "active_job_id": manager.active_job_id,
    }


@router.get("/jobs/{job_id}", response_model=TrainingJobOut, summary="One training job")
def get_job(session: SessionDep, job_id: int) -> TrainingJobOut:
    """A job's current status, progress and per-epoch metrics."""
    return TrainingJobOut.model_validate(training_service.get_job(session, job_id))


@router.get("/jobs/{job_id}/log", summary="Tail of a job's log file")
def job_log(
    session: SessionDep,
    job_id: int,
    lines: Annotated[int, Query(ge=10, le=2000)] = 200,
) -> dict[str, Any]:
    """The last *lines* of the job's log file."""
    job = training_service.get_job(session, job_id)
    return {
        "job_id": job_id,
        "status": job.status,
        "log": training_service.read_job_log(job, lines=lines),
        "path": job.log_path,
    }


@router.post("/jobs/{job_id}/cancel", response_model=MessageResponse, summary="Cancel a job")
def cancel_job(session: SessionDep, job_id: int) -> MessageResponse:
    """Request cancellation.

    Cancellation is cooperative: the trainer stops at its next batch boundary
    and saves its checkpoint and history first, so no work is lost.
    """
    job = training_service.get_job_manager().cancel(session, job_id)
    return MessageResponse(
        message=(
            f"Cancellation requested for job {job_id}. It will stop at the next batch "
            "boundary, after saving its current checkpoint."
        ),
        detail={"id": job_id, "status": job.status},
    )
