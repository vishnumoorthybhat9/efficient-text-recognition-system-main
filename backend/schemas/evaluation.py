"""Schemas for model evaluation and the comparison dashboard."""

from __future__ import annotations

from datetime import datetime
from typing import Any, Literal

from pydantic import Field

from backend.schemas.common import ApiModel

__all__ = [
    "EvaluationRunOut",
    "EvaluationListResponse",
    "StartEvaluationIn",
    "ComparisonRowOut",
    "ExperimentComparisonOut",
    "ChartResponse",
    "TrainingCurveOut",
]


class EvaluationRunOut(ApiModel):
    """One benchmark of one model over a dataset split."""

    id: int
    created_at: datetime
    model_id: str
    model_name: str
    provenance: str
    experiment_arm: str = "other"
    dataset: str
    split: str
    num_samples: int

    # accuracy
    cer: float
    wer: float
    sequence_accuracy: float
    character_accuracy: float
    cer_normalized: float

    # efficiency
    avg_inference_ms: float
    median_inference_ms: float
    throughput_images_per_s: float
    peak_memory_mb: float
    parameter_count: int
    model_size_mb: float

    device: str
    decode_strategy: str
    batch_size: int
    metrics: dict[str, Any] = Field(default_factory=dict)
    worst_samples: list[Any] = Field(default_factory=list)
    report_path: str | None = None
    notes: str = ""


class EvaluationListResponse(ApiModel):
    """Result of ``GET /api/evaluation/runs``."""

    items: list[EvaluationRunOut]
    total: int


class StartEvaluationIn(ApiModel):
    """Request body for ``POST /api/evaluation/run``."""

    model_ids: list[str] = Field(
        min_length=1,
        max_length=8,
        description="Models to evaluate; each produces its own evaluation run row",
    )
    split: Literal["train", "val", "test"] = "test"
    dataset: str = Field(default="synthetic", description="Dataset name under data/processed")
    max_samples: int | None = Field(
        default=None,
        ge=1,
        description="Evaluate only this many samples; null uses the whole split",
    )
    batch_size: int = Field(default=32, ge=1, le=256)
    decode_strategy: Literal["greedy", "beam"] = "greedy"
    beam_width: int = Field(default=10, ge=1, le=50)


class ComparisonRowOut(ApiModel):
    """One model's row in the comparison table."""

    model_id: str
    model_name: str
    provenance: str
    experiment_arm: str
    architecture: str = ""
    trained_on: str = Field(
        default="unknown",
        description=(
            "Dataset the model was trained on. When it differs from the evaluation "
            "dataset the row is a cross-domain generalisation result, not a like-for-like "
            "comparison."
        ),
    )
    cer: float | None = None
    wer: float | None = None
    sequence_accuracy: float | None = None
    character_accuracy: float | None = None
    # Case- and punctuation-insensitive variants. These matter for a fair
    # reading: a model trained on an all-caps corpus (TrOCR on SROIE receipts)
    # is penalised heavily by the strict metric for a convention difference
    # rather than for misreading the text.
    cer_normalized: float | None = None
    sequence_accuracy_normalized: float | None = None
    avg_inference_ms: float | None = None
    throughput_images_per_s: float | None = None
    parameter_count: int | None = None
    model_size_mb: float | None = None
    num_samples: int | None = None
    evaluated_at: datetime | None = None
    evaluation_id: int | None = None


class ExperimentComparisonOut(ApiModel):
    """The headline result: SimCLR pretraining vs. training from scratch.

    ``cer_improvement_absolute`` and ``..._relative`` are only populated when
    *both* arms have been evaluated on the same dataset and split, because a
    comparison across different test sets would be meaningless.
    """

    rows: list[ComparisonRowOut]
    control: ComparisonRowOut | None = None
    treatment: ComparisonRowOut | None = None
    dataset: str | None = None
    split: str | None = None
    comparable: bool = Field(
        default=False,
        description="True when both arms were evaluated on the same dataset and split",
    )
    cer_improvement_absolute: float | None = Field(
        default=None, description="control.cer - treatment.cer; positive means SimCLR helped"
    )
    cer_improvement_relative: float | None = Field(
        default=None, description="Fractional CER reduction, e.g. 0.18 means 18% lower CER"
    )
    accuracy_improvement_absolute: float | None = None
    summary: str = Field(
        default="",
        description="A one-paragraph, numbers-only statement of what the experiment showed",
    )


class TrainingCurveOut(ApiModel):
    """Per-epoch metrics of one training run, for the dashboard charts."""

    run_name: str
    epochs: list[int] = Field(default_factory=list)
    series: dict[str, list[float | None]] = Field(
        default_factory=dict, description="metric name -> value per epoch (null where absent)"
    )
    best_epoch: int | None = None
    monitor: str | None = None
    total_duration_s: float | None = None
    notes: dict[str, Any] = Field(default_factory=dict)


class ChartResponse(ApiModel):
    """A Plotly figure, ready to hand to ``react-plotly``."""

    id: str
    title: str
    description: str = ""
    figure: dict[str, Any] = Field(description="Plotly figure spec: {data: [...], layout: {...}}")
    empty: bool = Field(default=False, description="True when there is no data to plot yet")
    empty_reason: str = ""
