"""Evaluation, model comparison and chart endpoints."""

from __future__ import annotations

import logging
from typing import Annotated

from fastapi import APIRouter, Query

from backend.api.deps import PaginationDep, RuntimeDep, SessionDep
from backend.core.errors import BadRequest
from backend.schemas.evaluation import (
    ChartResponse,
    EvaluationListResponse,
    EvaluationRunOut,
    ExperimentComparisonOut,
    StartEvaluationIn,
    TrainingCurveOut,
)
from backend.services import chart_service, evaluation_service
from ml.engine.common import TrainingHistory
from ml.utils.paths import resolve_path

__all__ = ["router"]

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/evaluation", tags=["evaluation"])


@router.post("/run", response_model=EvaluationListResponse, summary="Benchmark one or more models")
def run_evaluation(
    session: SessionDep, runtime: RuntimeDep, payload: StartEvaluationIn
) -> EvaluationListResponse:
    """Evaluate models on a dataset split and store the results.

    Runs synchronously: a benchmark over the default 1,000-sample test split
    takes seconds to a couple of minutes, which is short enough that a job
    queue would add more complexity than it removes. Use ``max_samples`` to
    bound it further.

    Each model produces one :class:`EvaluationRun` row. A model that fails is
    reported in the response rather than aborting the others.
    """
    results: list[EvaluationRunOut] = []
    failures: list[dict[str, str]] = []

    for model_id in payload.model_ids:
        try:
            _, row = evaluation_service.run_evaluation(
                session=session,
                runtime=runtime,
                model_id=model_id,
                dataset=payload.dataset,
                split=payload.split,
                max_samples=payload.max_samples,
                batch_size=payload.batch_size,
                decode_strategy=payload.decode_strategy,
                beam_width=payload.beam_width,
            )
            if row is not None:
                results.append(EvaluationRunOut.model_validate(row))
        except Exception as exc:
            logger.exception("Evaluation of %s failed", model_id)
            failures.append({"model_id": model_id, "error": str(exc)[:400]})

    if not results and failures:
        raise BadRequest(
            "Every requested model failed to evaluate. First error: "
            + failures[0]["error"],
            detail={"failures": failures},
        )

    if failures:
        logger.warning("%d model(s) failed during evaluation: %s", len(failures), failures)

    return EvaluationListResponse(items=results, total=len(results))


@router.get("/runs", response_model=EvaluationListResponse, summary="List evaluation runs")
def list_runs(
    session: SessionDep,
    pagination: PaginationDep,
    model_id: Annotated[str | None, Query()] = None,
    dataset: Annotated[str | None, Query()] = None,
    split: Annotated[str | None, Query()] = None,
) -> EvaluationListResponse:
    """Stored evaluation runs, newest first."""
    rows, total = evaluation_service.list_evaluations(
        session,
        model_id=model_id,
        dataset=dataset,
        split=split,
        limit=pagination.limit,
        offset=pagination.offset,
    )
    return EvaluationListResponse(
        items=[EvaluationRunOut.model_validate(row) for row in rows], total=total
    )


@router.get(
    "/comparison",
    response_model=ExperimentComparisonOut,
    summary="The self-supervised comparison",
)
def comparison(
    session: SessionDep,
    runtime: RuntimeDep,
    dataset: Annotated[str | None, Query()] = None,
    split: Annotated[str, Query()] = "test",
) -> ExperimentComparisonOut:
    """The headline result: SimCLR-pretrained against trained-from-scratch.

    An improvement figure is reported only when both arms were evaluated on
    the same dataset and split; otherwise ``comparable`` is false and
    ``summary`` explains what is missing.
    """
    payload = evaluation_service.build_comparison(session, runtime, dataset=dataset, split=split)
    return ExperimentComparisonOut(**payload)


@router.get("/runs/{run_id}", response_model=EvaluationRunOut, summary="One evaluation run")
def get_run(session: SessionDep, run_id: int) -> EvaluationRunOut:
    """Full detail of one benchmark, including its worst-scoring samples."""
    return EvaluationRunOut.model_validate(evaluation_service.get_evaluation(session, run_id))


@router.get(
    "/training-history/{run_name}",
    response_model=TrainingCurveOut,
    summary="Per-epoch metrics of a training run",
)
def training_history(run_name: str) -> TrainingCurveOut:
    """Read ``artifacts/checkpoints/<run_name>/history.json``.

    Raises 404 when that run has not been trained on this machine.
    """
    path = resolve_path(f"artifacts/checkpoints/{run_name}/history.json")
    if path is None or not path.is_file():
        from backend.core.errors import NotFound

        raise NotFound(
            f"No training history for {run_name!r}. Train it first with "
            "`python scripts/train_crnn.py`.",
            detail={"run_name": run_name},
        )

    history = TrainingHistory.load(path)
    records = history.get("epochs", [])
    epochs = [int(r["epoch"]) for r in records]

    # Collect every numeric metric present across the epochs.
    keys: set[str] = set()
    for record in records:
        keys.update(
            k
            for k, v in record.items()
            if k not in ("epoch", "timestamp") and isinstance(v, (int, float))
        )

    series = {key: [record.get(key) for record in records] for key in sorted(keys)}

    return TrainingCurveOut(
        run_name=run_name,
        epochs=epochs,
        series=series,
        best_epoch=history.get("best_epoch"),
        monitor=history.get("monitor"),
        total_duration_s=history.get("total_duration_s"),
        notes=history.get("notes") or {},
    )


@router.get("/charts", summary="List available charts")
def list_charts() -> dict[str, list[dict[str, str]]]:
    """Chart ids and titles, so the UI can enumerate them from the API."""
    return {"charts": chart_service.available_charts()}


@router.get("/charts/{chart_id}", response_model=ChartResponse, summary="One Plotly figure")
def get_chart(
    session: SessionDep,
    runtime: RuntimeDep,
    chart_id: str,
    split: Annotated[str, Query()] = "test",
) -> ChartResponse:
    """Build a chart server-side and return it as Plotly JSON.

    A chart with no data returns ``empty=true`` and a reason rather than an
    error, so the UI can show guidance instead of blank axes.
    """
    try:
        payload = chart_service.build_chart(chart_id, session, runtime, split=split)
    except KeyError as exc:
        raise BadRequest(str(exc).strip("'\""), detail={"chart_id": chart_id}) from exc

    return ChartResponse(**payload)
