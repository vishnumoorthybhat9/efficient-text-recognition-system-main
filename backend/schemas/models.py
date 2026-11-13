"""Schemas describing the available recognition models."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import Field

from backend.schemas.common import ApiModel

__all__ = ["TrainingSummaryOut", "ModelInfoOut", "ModelListResponse"]

Provenance = Literal["trained_in_project", "pretrained_third_party"]


class TrainingSummaryOut(ApiModel):
    """Headline numbers from a model's own training run."""

    epochs_completed: int | None = None
    best_epoch: int | None = None
    monitor: str | None = None
    best_value: float | None = None
    val_cer: float | None = None
    val_wer: float | None = None
    val_accuracy: float | None = None
    train_loss: float | None = None
    training_seconds: float | None = None
    train_samples: int | None = None
    device: str | None = None
    parameters: int | None = None
    encoder_transfer: dict[str, Any] | None = Field(
        default=None,
        description="How much of the SimCLR encoder transferred, when this run used one",
    )


class ModelInfoOut(ApiModel):
    """One registered model."""

    id: str
    name: str
    family: str = Field(description="crnn_attention | cnn_transformer | mamba_ocr | trocr")
    architecture: str | None = None
    provenance: Provenance = Field(
        description=(
            "trained_in_project: a result of this work. "
            "pretrained_third_party: downloaded weights, shown as a baseline only."
        )
    )
    description: str = ""
    available: bool = True
    loaded: bool = False
    unavailable_reason: str | None = None
    checkpoint: str | None = None
    experiment_arm: str | None = Field(
        default=None, description="control | treatment | reference | other"
    )
    supports_beam_search: bool = False
    parameters: dict[str, Any] = Field(default_factory=dict)
    training_summary: TrainingSummaryOut | None = None
    hf_model_id: str | None = None
    downloads_on_first_use: bool = False


class ModelListResponse(ApiModel):
    """Result of ``GET /api/models``."""

    models: list[ModelInfoOut]
    default_model_id: str | None = None
    loaded_ids: list[str] = Field(default_factory=list)
    device: str = ""
    comparison_ready: bool = Field(
        default=False,
        description="True when both arms of the self-supervised experiment are present",
    )
