"""Shared response schemas."""

from __future__ import annotations

from datetime import datetime
from typing import Any, Generic, TypeVar

from pydantic import BaseModel, ConfigDict, Field

__all__ = [
    "ApiModel",
    "ErrorResponse",
    "HealthResponse",
    "GpuInfoOut",
    "SystemInfoResponse",
    "Page",
    "MessageResponse",
]

T = TypeVar("T")


class ApiModel(BaseModel):
    """Base for every schema.

    ``from_attributes`` lets a response be built straight from a SQLAlchemy
    row, so the route layer does not need a hand-written field-by-field copy.

    ``protected_namespaces=()`` is required: this domain is full of fields
    named ``model_id`` and ``model_name``, and pydantic otherwise reserves the
    ``model_`` prefix for its own API and warns on every one of them.  Renaming
    the fields would be the tail wagging the dog -- "model" is the domain term.
    """

    model_config = ConfigDict(
        from_attributes=True,
        populate_by_name=True,
        protected_namespaces=(),
    )


class ErrorResponse(ApiModel):
    """The shape every error returns (see :mod:`backend.core.errors`)."""

    error: str = Field(description="Stable machine-readable error slug")
    message: str = Field(description="Human-readable explanation, safe to show the user")
    detail: dict[str, Any] = Field(default_factory=dict)
    request_id: str | None = None


class MessageResponse(ApiModel):
    """A simple acknowledgement."""

    message: str
    detail: dict[str, Any] = Field(default_factory=dict)


class HealthResponse(ApiModel):
    """Liveness and readiness."""

    status: str = Field(description="'ok' when every subsystem is ready")
    app_name: str
    version: str
    environment: str
    device: str
    models_available: int
    database_ready: bool
    dataset_ready: bool
    uptime_seconds: float
    timestamp: datetime


class GpuInfoOut(ApiModel):
    """One CUDA device."""

    index: int
    name: str
    total_memory_mb: int
    capability: str
    multi_processor_count: int


class SystemInfoResponse(ApiModel):
    """Hardware and software report, shown on the dashboard."""

    device: str
    device_type: str
    cuda_available: bool
    amp_supported: bool
    torch_version: str
    cuda_version: str | None = None
    cudnn_version: int | None = None
    gpus: list[GpuInfoOut] = Field(default_factory=list)
    cpu_name: str = ""
    cpu_count: int = 0
    total_ram_mb: int = 0
    platform: str = ""
    settings: dict[str, Any] = Field(default_factory=dict)


class Page(ApiModel, Generic[T]):
    """A page of results."""

    items: list[T]
    total: int = Field(description="Total rows matching the query, ignoring pagination")
    limit: int
    offset: int

    @property
    def has_more(self) -> bool:
        return self.offset + len(self.items) < self.total
