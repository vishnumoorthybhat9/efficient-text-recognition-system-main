"""Schemas for the OCR history."""

from __future__ import annotations

from datetime import datetime
from typing import Any

from pydantic import Field

from backend.schemas.common import ApiModel

__all__ = [
    "HistoryItemOut",
    "HistoryDetailOut",
    "GroundTruthIn",
    "HistoryStatsOut",
]


class HistoryItemOut(ApiModel):
    """A history row as shown in the list.

    Deliberately omits the per-line detail and the preprocessing metadata:
    the list can hold hundreds of rows and the detail view fetches them.
    """

    id: int
    created_at: datetime
    model_id: str
    model_name: str
    original_filename: str
    text: str
    confidence: float
    line_count: int
    word_count: int
    character_count: int
    total_ms: float
    inference_ms: float
    device: str
    image_width: int
    image_height: int
    has_image: bool = Field(default=False, description="Whether the uploaded image was kept")
    ground_truth: str | None = None
    cer: float | None = None
    wer: float | None = None
    exact_match: bool | None = None


class HistoryDetailOut(HistoryItemOut):
    """A history row with its full structured detail."""

    decode_strategy: str = "greedy"
    preprocess_options: dict[str, Any] = Field(default_factory=dict)
    preprocessing_meta: dict[str, Any] = Field(default_factory=dict)
    segmentation: dict[str, Any] = Field(default_factory=dict)
    lines: list[Any] = Field(default_factory=list)
    preprocess_ms: float = 0.0
    segment_ms: float = 0.0
    content_type: str = ""
    file_size_bytes: int = 0


class GroundTruthIn(ApiModel):
    """Attach a correct transcription to a past recognition.

    This is what turns the history into an error-analysis tool: CER and WER
    are recomputed against the text the user supplies, so accuracy can be
    measured on real uploads and not only on the held-out test split.
    """

    ground_truth: str = Field(min_length=0, max_length=20000)


class HistoryStatsOut(ApiModel):
    """Aggregates for the dashboard tiles."""

    total_requests: int = 0
    total_characters: int = 0
    average_confidence: float = 0.0
    average_total_ms: float = 0.0
    requests_by_model: dict[str, int] = Field(default_factory=dict)
    requests_last_7_days: int = 0
    scored_requests: int = Field(
        default=0, description="History rows that have a user-supplied ground truth"
    )
    average_cer: float | None = Field(
        default=None, description="Mean CER over scored rows only; null when none are scored"
    )
    average_wer: float | None = None
