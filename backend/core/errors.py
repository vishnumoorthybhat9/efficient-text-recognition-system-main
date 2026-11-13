"""Application exceptions and the handlers that turn them into HTTP responses.

Every error the API returns has the same JSON shape::

    {
      "error": "model_unavailable",
      "message": "TrOCR ... is not in the local cache ...",
      "detail": {"model_id": "trocr_printed"},
      "request_id": "..."
    }

``message`` is written for a person to read -- the frontend displays it
verbatim -- so it says what went wrong *and* what to do about it.  Raising a
bare ``HTTPException`` with a terse string is deliberately avoided.
"""

from __future__ import annotations

import logging
import uuid
from typing import Any

from fastapi import FastAPI, Request, status
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from starlette.exceptions import HTTPException as StarletteHTTPException

__all__ = [
    "AppError",
    "BadRequest",
    "NotFound",
    "ModelUnavailable",
    "InvalidImage",
    "PayloadTooLarge",
    "ConflictError",
    "DependencyMissing",
    "register_exception_handlers",
]

logger = logging.getLogger(__name__)

# Starlette renamed these constants; resolve by value so the project works on
# both old and new releases without emitting a deprecation warning on import.
_HTTP_422 = getattr(status, "HTTP_422_UNPROCESSABLE_CONTENT", 422)
_HTTP_413 = getattr(status, "HTTP_413_CONTENT_TOO_LARGE", 413)


class AppError(Exception):
    """Base class for every expected application error.

    Args:
        message: human-readable explanation, shown to the user.
        detail: structured context for the client.
        code: stable machine-readable slug.
        status_code: HTTP status to return.
    """

    code = "internal_error"
    status_code = status.HTTP_500_INTERNAL_SERVER_ERROR

    def __init__(
        self,
        message: str,
        *,
        detail: dict[str, Any] | None = None,
        code: str | None = None,
        status_code: int | None = None,
    ) -> None:
        super().__init__(message)
        self.message = message
        self.detail = detail or {}
        if code is not None:
            self.code = code
        if status_code is not None:
            self.status_code = status_code

    def to_payload(self, request_id: str | None = None) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "error": self.code,
            "message": self.message,
            "detail": self.detail,
        }
        if request_id:
            payload["request_id"] = request_id
        return payload


class BadRequest(AppError):
    """The request was malformed or asked for something impossible."""

    code = "bad_request"
    status_code = status.HTTP_400_BAD_REQUEST


class NotFound(AppError):
    """The requested resource does not exist."""

    code = "not_found"
    status_code = status.HTTP_404_NOT_FOUND


class ConflictError(AppError):
    """The request conflicts with the current state (e.g. a job already running)."""

    code = "conflict"
    status_code = status.HTTP_409_CONFLICT


class InvalidImage(BadRequest):
    """The uploaded file is not a decodable image."""

    code = "invalid_image"
    status_code = _HTTP_422


class PayloadTooLarge(AppError):
    """The upload exceeded the configured size limit."""

    code = "payload_too_large"
    status_code = _HTTP_413


class ModelUnavailable(AppError):
    """A model exists in the registry but cannot be used right now.

    503 rather than 404: the model is a known resource that is temporarily
    unusable (not downloaded, out of memory, not yet trained), and retrying
    after fixing the cause is the right behaviour.
    """

    code = "model_unavailable"
    status_code = status.HTTP_503_SERVICE_UNAVAILABLE


class DependencyMissing(AppError):
    """An optional dependency or artefact the endpoint needs is absent."""

    code = "dependency_missing"
    status_code = status.HTTP_503_SERVICE_UNAVAILABLE


def register_exception_handlers(app: FastAPI) -> None:
    """Install handlers so every error leaves the API in the same shape."""

    @app.exception_handler(AppError)
    async def _handle_app_error(request: Request, exc: AppError) -> JSONResponse:
        request_id = getattr(request.state, "request_id", None)
        # 5xx is a bug or an environment problem and deserves a stack trace;
        # 4xx is the client's doing and would only add noise.
        if exc.status_code >= 500:
            logger.error(
                "%s on %s %s: %s", exc.code, request.method, request.url.path, exc.message,
                exc_info=True,
            )
        else:
            logger.info("%s on %s %s: %s", exc.code, request.method, request.url.path, exc.message)

        return JSONResponse(status_code=exc.status_code, content=exc.to_payload(request_id))

    @app.exception_handler(RequestValidationError)
    async def _handle_validation_error(
        request: Request, exc: RequestValidationError
    ) -> JSONResponse:
        fields = []
        for error in exc.errors():
            location = ".".join(str(part) for part in error.get("loc", ()) if part != "body")
            fields.append({"field": location or "body", "problem": error.get("msg", "invalid")})

        summary = "; ".join(f"{f['field']}: {f['problem']}" for f in fields[:4])
        return JSONResponse(
            status_code=_HTTP_422,
            content={
                "error": "validation_error",
                "message": f"The request parameters are not valid. {summary}",
                "detail": {"fields": fields},
                "request_id": getattr(request.state, "request_id", None),
            },
        )

    @app.exception_handler(StarletteHTTPException)
    async def _handle_http_exception(
        request: Request, exc: StarletteHTTPException
    ) -> JSONResponse:
        return JSONResponse(
            status_code=exc.status_code,
            content={
                "error": "http_error",
                "message": str(exc.detail),
                "detail": {},
                "request_id": getattr(request.state, "request_id", None),
            },
            headers=getattr(exc, "headers", None),
        )

    @app.exception_handler(Exception)
    async def _handle_unexpected(request: Request, exc: Exception) -> JSONResponse:
        request_id = getattr(request.state, "request_id", None) or uuid.uuid4().hex[:12]
        logger.exception(
            "Unhandled %s on %s %s [request_id=%s]",
            type(exc).__name__,
            request.method,
            request.url.path,
            request_id,
        )
        # The exception text may contain file paths; return the request id
        # instead and keep the detail in the server log.
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content={
                "error": "internal_error",
                "message": (
                    "The server hit an unexpected error. The full traceback is in "
                    f"artifacts/logs/api.log under request id {request_id}."
                ),
                "detail": {"type": type(exc).__name__},
                "request_id": request_id,
            },
        )
