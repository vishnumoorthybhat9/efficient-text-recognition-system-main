"""OCR history: listing, detail, re-scoring and deletion."""

from __future__ import annotations

import logging
from datetime import timedelta
from typing import Any

from sqlalchemy import delete, func, select
from sqlalchemy.orm import Session

from backend.core.config import Settings
from backend.core.errors import NotFound
from backend.db.models import OCRRequest, utcnow
from backend.services.ocr_service import score_against_ground_truth
from backend.services.storage import resolve_stored_path

__all__ = [
    "list_history",
    "get_history_item",
    "set_ground_truth",
    "delete_history_item",
    "clear_history",
    "history_stats",
]

logger = logging.getLogger(__name__)


def list_history(
    session: Session,
    *,
    limit: int = 50,
    offset: int = 0,
    model_id: str | None = None,
    search: str | None = None,
    scored_only: bool = False,
) -> tuple[list[OCRRequest], int]:
    """Return a page of history rows, newest first.

    Args:
        limit: page size.
        offset: rows to skip.
        model_id: restrict to one model.
        search: case-insensitive substring match on the recognised text or the
            original filename.
        scored_only: only rows that have a user-supplied ground truth.

    Returns:
        ``(rows, total_matching)`` -- the total ignores pagination so the UI
        can render a page count.
    """
    filters = []
    if model_id:
        filters.append(OCRRequest.model_id == model_id)
    if scored_only:
        filters.append(OCRRequest.ground_truth.is_not(None))
    if search and search.strip():
        pattern = f"%{search.strip()}%"
        filters.append(
            func.lower(OCRRequest.text).like(func.lower(pattern))
            | func.lower(OCRRequest.original_filename).like(func.lower(pattern))
        )

    total = session.scalar(select(func.count()).select_from(OCRRequest).where(*filters)) or 0

    rows = list(
        session.scalars(
            select(OCRRequest)
            .where(*filters)
            .order_by(OCRRequest.created_at.desc(), OCRRequest.id.desc())
            .limit(limit)
            .offset(offset)
        )
    )
    return rows, int(total)


def get_history_item(session: Session, item_id: int) -> OCRRequest:
    """Fetch one history row.

    Raises:
        NotFound: no such row.
    """
    row = session.get(OCRRequest, item_id)
    if row is None:
        raise NotFound(
            f"No history entry with id {item_id}. It may have been deleted.",
            detail={"id": item_id},
        )
    return row


def set_ground_truth(session: Session, item_id: int, ground_truth: str) -> OCRRequest:
    """Attach or update a ground truth and recompute CER/WER.

    Passing an empty string clears the ground truth and the scores, so a
    mistyped reference can be undone.
    """
    row = get_history_item(session, item_id)
    cleaned = ground_truth.strip()

    if not cleaned:
        row.ground_truth = None
        row.cer = None
        row.wer = None
        row.exact_match = None
        logger.info("Cleared the ground truth on history entry %d", item_id)
    else:
        scores = score_against_ground_truth(row.text, cleaned)
        row.ground_truth = cleaned
        row.cer = scores["cer"]
        row.wer = scores["wer"]
        row.exact_match = scores["exact_match"]
        logger.info(
            "Scored history entry %d against a ground truth: CER %.4f, WER %.4f",
            item_id,
            row.cer,
            row.wer,
        )

    session.commit()
    session.refresh(row)
    return row


def delete_history_item(session: Session, item_id: int, settings: Settings) -> None:
    """Delete a history row and its stored image."""
    row = get_history_item(session, item_id)

    path = resolve_stored_path(row.stored_filename, settings)
    session.delete(row)
    session.commit()

    if path is not None:
        try:
            path.unlink()
        except OSError as exc:
            # The row is already gone; a leftover file is not worth failing on.
            logger.warning("Deleted history entry %d but could not remove %s: %s", item_id, path.name, exc)

    logger.info("Deleted history entry %d", item_id)


def clear_history(session: Session, settings: Settings, *, delete_images: bool = True) -> int:
    """Delete every history row.  Returns how many were removed."""
    if delete_images:
        for stored in session.scalars(
            select(OCRRequest.stored_filename).where(OCRRequest.stored_filename.is_not(None))
        ):
            path = resolve_stored_path(stored, settings)
            if path is not None:
                try:
                    path.unlink()
                except OSError as exc:
                    logger.warning("Could not remove %s: %s", path.name, exc)

    count = session.scalar(select(func.count()).select_from(OCRRequest)) or 0
    session.execute(delete(OCRRequest))
    session.commit()

    logger.info("Cleared %d history entry(ies)", count)
    return int(count)


def history_stats(session: Session) -> dict[str, Any]:
    """Aggregates for the dashboard tiles."""
    total = session.scalar(select(func.count()).select_from(OCRRequest)) or 0

    if not total:
        return {
            "total_requests": 0,
            "total_characters": 0,
            "average_confidence": 0.0,
            "average_total_ms": 0.0,
            "requests_by_model": {},
            "requests_last_7_days": 0,
            "scored_requests": 0,
            "average_cer": None,
            "average_wer": None,
        }

    characters = session.scalar(select(func.sum(OCRRequest.character_count))) or 0
    confidence = session.scalar(select(func.avg(OCRRequest.confidence))) or 0.0
    duration = session.scalar(select(func.avg(OCRRequest.total_ms))) or 0.0

    by_model = {
        str(model_id): int(count)
        for model_id, count in session.execute(
            select(OCRRequest.model_id, func.count())
            .group_by(OCRRequest.model_id)
            .order_by(func.count().desc())
        )
    }

    recent = (
        session.scalar(
            select(func.count())
            .select_from(OCRRequest)
            .where(OCRRequest.created_at >= utcnow() - timedelta(days=7))
        )
        or 0
    )

    # CER/WER averages cover only the rows that actually have a ground truth;
    # averaging over all rows would silently treat unscored ones as perfect.
    scored = (
        session.scalar(
            select(func.count()).select_from(OCRRequest).where(OCRRequest.cer.is_not(None))
        )
        or 0
    )
    average_cer = session.scalar(select(func.avg(OCRRequest.cer))) if scored else None
    average_wer = session.scalar(select(func.avg(OCRRequest.wer))) if scored else None

    return {
        "total_requests": int(total),
        "total_characters": int(characters),
        "average_confidence": float(confidence),
        "average_total_ms": float(duration),
        "requests_by_model": by_model,
        "requests_last_7_days": int(recent),
        "scored_requests": int(scored),
        "average_cer": float(average_cer) if average_cer is not None else None,
        "average_wer": float(average_wer) if average_wer is not None else None,
    }
