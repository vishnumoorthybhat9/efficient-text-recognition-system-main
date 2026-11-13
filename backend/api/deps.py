"""FastAPI dependencies shared by the route modules."""

from __future__ import annotations

import json
import logging
from typing import Annotated, Any, TypeVar

from fastapi import Depends, Form, Query
from pydantic import BaseModel, ValidationError
from sqlalchemy.orm import Session

from backend.core.config import Settings, get_settings
from backend.core.errors import BadRequest
from backend.db.session import get_db
from backend.services.runtime import MLRuntime, get_runtime

__all__ = [
    "SessionDep",
    "SettingsDep",
    "RuntimeDep",
    "PaginationDep",
    "Pagination",
    "parse_json_form",
]

logger = logging.getLogger(__name__)

T = TypeVar("T", bound=BaseModel)

SessionDep = Annotated[Session, Depends(get_db)]
SettingsDep = Annotated[Settings, Depends(get_settings)]
RuntimeDep = Annotated[MLRuntime, Depends(get_runtime)]


class Pagination:
    """Validated ``limit`` / ``offset`` pair.

    The upper bound comes from settings rather than being hard-coded, so a
    deployment can tighten it without a code change.
    """

    __slots__ = ("limit", "offset")

    def __init__(
        self,
        limit: Annotated[int, Query(ge=1, le=1000, description="Rows per page")] = 50,
        offset: Annotated[int, Query(ge=0, description="Rows to skip")] = 0,
        settings: SettingsDep = None,  # type: ignore[assignment]
    ) -> None:
        cap = settings.max_page_size if settings is not None else 200
        if limit > cap:
            logger.debug("Clamping requested page size %d to %d", limit, cap)
        self.limit = min(limit, cap)
        self.offset = offset


PaginationDep = Annotated[Pagination, Depends(Pagination)]


def parse_json_form(raw: str | None, model: type[T], *, field_name: str = "options") -> T:
    """Parse a JSON string sent as a multipart form field into *model*.

    A multipart request cannot also carry a JSON body, so structured options
    travel as a JSON string alongside the file.  An absent or empty value
    yields the model's defaults, which keeps the simple "just upload a file"
    case free of ceremony.

    Raises:
        BadRequest: the string is not valid JSON, or fails validation.
    """
    if raw is None or not raw.strip():
        return model()

    try:
        payload = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise BadRequest(
            f"The {field_name!r} form field is not valid JSON: {exc.msg} "
            f"(line {exc.lineno}, column {exc.colno}).",
            detail={"field": field_name},
        ) from exc

    if not isinstance(payload, dict):
        raise BadRequest(
            f"The {field_name!r} form field must be a JSON object, got {type(payload).__name__}.",
            detail={"field": field_name},
        )

    try:
        return model(**payload)
    except ValidationError as exc:
        problems = [
            {
                "field": ".".join(str(p) for p in error.get("loc", ())),
                "problem": error.get("msg", "invalid"),
            }
            for error in exc.errors()
        ]
        summary = "; ".join(f"{p['field']}: {p['problem']}" for p in problems[:4])
        raise BadRequest(
            f"Invalid {field_name}: {summary}", detail={"field": field_name, "problems": problems}
        ) from exc


OptionsForm = Annotated[
    str | None,
    Form(description="JSON object of recognition options; omit to use defaults"),
]


def dump_for_response(payload: dict[str, Any]) -> dict[str, Any]:
    """Round float values so API responses stay readable."""
    return {
        key: (round(value, 6) if isinstance(value, float) else value)
        for key, value in payload.items()
    }
