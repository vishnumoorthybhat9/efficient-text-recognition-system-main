"""OCR history endpoints."""

from __future__ import annotations

import logging
from typing import Annotated

from fastapi import APIRouter, Query
from fastapi.responses import FileResponse

from backend.api.deps import PaginationDep, SessionDep, SettingsDep
from backend.core.errors import NotFound
from backend.schemas.common import MessageResponse, Page
from backend.schemas.history import (
    GroundTruthIn,
    HistoryDetailOut,
    HistoryItemOut,
    HistoryStatsOut,
)
from backend.services import history_service
from backend.services.storage import resolve_stored_path

__all__ = ["router"]

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/history", tags=["history"])


def _to_item(row) -> HistoryItemOut:
    item = HistoryItemOut.model_validate(row)
    item.has_image = bool(row.stored_filename)
    return item


@router.get("", response_model=Page[HistoryItemOut], summary="List past recognitions")
def list_history(
    session: SessionDep,
    pagination: PaginationDep,
    model_id: Annotated[str | None, Query(description="Filter by model")] = None,
    search: Annotated[str | None, Query(description="Substring of the text or filename")] = None,
    scored_only: Annotated[bool, Query(description="Only entries with a ground truth")] = False,
) -> Page[HistoryItemOut]:
    """Recognition history, newest first."""
    rows, total = history_service.list_history(
        session,
        limit=pagination.limit,
        offset=pagination.offset,
        model_id=model_id,
        search=search,
        scored_only=scored_only,
    )
    return Page[HistoryItemOut](
        items=[_to_item(row) for row in rows],
        total=total,
        limit=pagination.limit,
        offset=pagination.offset,
    )


@router.get("/stats", response_model=HistoryStatsOut, summary="History aggregates")
def stats(session: SessionDep) -> HistoryStatsOut:
    """Totals and averages for the dashboard tiles."""
    return HistoryStatsOut(**history_service.history_stats(session))


@router.get("/{item_id}", response_model=HistoryDetailOut, summary="One past recognition")
def get_item(session: SessionDep, item_id: int) -> HistoryDetailOut:
    """A history entry with its per-line results and preprocessing detail."""
    row = history_service.get_history_item(session, item_id)
    detail = HistoryDetailOut.model_validate(row)
    detail.has_image = bool(row.stored_filename)
    return detail


@router.get("/{item_id}/image", summary="The image that was recognised", response_class=FileResponse)
def get_image(session: SessionDep, settings: SettingsDep, item_id: int) -> FileResponse:
    """Serve the stored upload for a history entry.

    Raises 404 when the entry was created with history saving off, or when the
    file has since been pruned.
    """
    row = history_service.get_history_item(session, item_id)
    path = resolve_stored_path(row.stored_filename, settings)

    if path is None:
        raise NotFound(
            f"The image for history entry {item_id} is no longer stored. It was either "
            "recognised with history saving disabled, or pruned by the retention policy.",
            detail={"id": item_id},
        )

    return FileResponse(path, filename=row.original_filename or path.name)


@router.put("/{item_id}/ground-truth", response_model=HistoryDetailOut, summary="Score an entry")
def set_ground_truth(
    session: SessionDep, item_id: int, payload: GroundTruthIn
) -> HistoryDetailOut:
    """Attach a correct transcription and recompute CER/WER for this entry.

    Sending an empty string clears the ground truth and the scores.
    """
    row = history_service.set_ground_truth(session, item_id, payload.ground_truth)
    detail = HistoryDetailOut.model_validate(row)
    detail.has_image = bool(row.stored_filename)
    return detail


@router.delete("/{item_id}", response_model=MessageResponse, summary="Delete one entry")
def delete_item(session: SessionDep, settings: SettingsDep, item_id: int) -> MessageResponse:
    """Delete a history entry and its stored image."""
    history_service.delete_history_item(session, item_id, settings)
    return MessageResponse(message=f"Deleted history entry {item_id}.", detail={"id": item_id})


@router.delete("", response_model=MessageResponse, summary="Clear the whole history")
def clear(
    session: SessionDep,
    settings: SettingsDep,
    confirm: Annotated[bool, Query(description="Must be true; guards against an accidental call")] = False,
    delete_images: Annotated[bool, Query()] = True,
) -> MessageResponse:
    """Delete every history entry.

    Requires ``?confirm=true``: this is destructive and a bare DELETE on a
    collection URL is too easy to issue by accident.
    """
    if not confirm:
        from backend.core.errors import BadRequest

        raise BadRequest(
            "Clearing the history is irreversible. Repeat the request with ?confirm=true.",
            detail={"required_parameter": "confirm=true"},
        )

    count = history_service.clear_history(session, settings, delete_images=delete_images)
    return MessageResponse(
        message=f"Cleared {count} history entry(ies).",
        detail={"deleted": count, "images_deleted": delete_images},
    )
