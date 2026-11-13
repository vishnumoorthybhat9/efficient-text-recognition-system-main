"""Schemas for recognition, preprocessing preview and model comparison."""

from __future__ import annotations

from datetime import datetime
from typing import Any, Literal

from pydantic import Field, field_validator

from backend.schemas.common import ApiModel
from ml.preprocessing.pipeline import BINARIZE_METHODS, DENOISE_METHODS, PreprocessOptions

__all__ = [
    "PreprocessOptionsIn",
    "RecognizeOptionsIn",
    "RegionOut",
    "LineOut",
    "RecognitionResponse",
    "PreprocessStageOut",
    "PreprocessPreviewResponse",
    "CompareModelResult",
    "CompareResponse",
]


class PreprocessOptionsIn(ApiModel):
    """Image-enhancement toggles.

    Mirrors :class:`ml.preprocessing.pipeline.PreprocessOptions`; kept as a
    separate schema so the HTTP contract can validate ranges and is documented
    in the generated OpenAPI spec.
    """

    grayscale: bool = True
    deskew: bool = True
    perspective: bool = Field(
        default=False,
        description="Flatten a photographed page to a head-on view. No-op if no page outline is found.",
    )
    denoise: bool = False
    denoise_strength: int = Field(default=7, ge=1, le=30)
    denoise_method: Literal["nlmeans", "bilateral", "median"] = "nlmeans"
    clahe: bool = False
    clahe_clip_limit: float = Field(default=2.0, ge=0.1, le=10.0)
    sharpen: bool = False
    sharpen_amount: float = Field(default=1.0, ge=0.0, le=3.0)
    binarize: Literal["none", "otsu", "adaptive", "sauvola"] = "none"
    binarize_block_size: int = Field(default=31, ge=3, le=199)
    binarize_offset: int = Field(default=10, ge=-50, le=50)
    invert_if_dark: bool = True
    normalize: bool = True

    @field_validator("binarize_block_size")
    @classmethod
    def _force_odd(cls, value: int) -> int:
        """OpenCV requires an odd neighbourhood size."""
        return value if value % 2 == 1 else value + 1

    def to_domain(self) -> PreprocessOptions:
        """Convert to the ML layer's dataclass."""
        return PreprocessOptions(**self.model_dump())

    @classmethod
    def describe_choices(cls) -> dict[str, list[str]]:
        """Valid enum values, so the UI can build its controls from the API."""
        return {
            "binarize": list(BINARIZE_METHODS),
            "denoise_method": list(DENOISE_METHODS),
        }


class RecognizeOptionsIn(ApiModel):
    """Per-request recognition settings, sent as a JSON string form field."""

    single_line: bool = Field(
        default=False,
        description="Treat the image as one pre-cropped text line and skip segmentation",
    )
    decode_strategy: Literal["greedy", "beam"] = "greedy"
    beam_width: int = Field(default=10, ge=1, le=50)
    max_lines: int = Field(
        default=200, ge=1, le=500,
        description="Cap on detected lines. A dense invoice can legitimately have 100+.",
    )
    min_confidence: float = Field(
        default=0.0, ge=0.0, le=1.0, description="Drop lines below this confidence"
    )
    handwritten: bool = Field(
        default=True,
        description=(
            "The image is a handwritten page (notebook, letter, prescription): remove "
            "shadows and ruled lines and use the handwriting line detector"
        ),
    )
    line_detector: Literal["auto", "projection", "doctr"] = Field(
        default="auto",
        description=(
            "Line detector for handwritten pages. 'doctr' needs the python-doctr package; "
            "'auto' uses it when installed and the built-in projection detector otherwise"
        ),
    )
    preprocess: PreprocessOptionsIn = Field(default_factory=PreprocessOptionsIn)
    save_to_history: bool = True


class RegionOut(ApiModel):
    """Bounding box of a detected text line."""

    index: int
    x: int
    y: int
    width: int
    height: int
    ink_ratio: float = 0.0


class LineOut(ApiModel):
    """One recognised line."""

    index: int
    text: str
    confidence: float
    region: RegionOut | None = None
    inference_ms: float = 0.0


class RecognitionResponse(ApiModel):
    """Result of ``POST /api/ocr/recognize``."""

    id: int | None = Field(default=None, description="History row id, null when not saved")
    text: str
    lines: list[LineOut]
    model_id: str
    model_name: str
    confidence: float
    line_count: int
    word_count: int
    character_count: int
    timings_ms: dict[str, float]
    segmentation: dict[str, Any]
    preprocessing: dict[str, Any]
    device: str
    image_width: int
    image_height: int
    created_at: datetime | None = None

    # Populated only when the request carried a ground truth.
    ground_truth: str | None = None
    cer: float | None = None
    wer: float | None = None
    exact_match: bool | None = None


class PreprocessStageOut(ApiModel):
    """One captured intermediate image, PNG encoded as a data URL."""

    name: str
    label: str
    detail: str = ""
    duration_ms: float
    image: str = Field(description="data:image/png;base64,... ready for an <img> src")
    width: int
    height: int


class PreprocessPreviewResponse(ApiModel):
    """Result of ``POST /api/preprocess/preview``."""

    stages: list[PreprocessStageOut]
    metadata: dict[str, Any]
    options: dict[str, Any]
    detected_lines: list[RegionOut] = Field(default_factory=list)
    overlay: str | None = Field(
        default=None, description="Input image with detected line boxes drawn on it"
    )


class CompareModelResult(ApiModel):
    """One model's result inside a comparison."""

    model_id: str
    model_name: str
    provenance: str
    text: str = ""
    confidence: float = 0.0
    total_ms: float = 0.0
    inference_ms: float = 0.0
    line_count: int = 0
    cer: float | None = None
    wer: float | None = None
    exact_match: bool | None = None
    error: str | None = Field(
        default=None, description="Set when this model failed; the others still returned"
    )


class CompareResponse(ApiModel):
    """Result of ``POST /api/ocr/compare``."""

    id: int | None = None
    results: list[CompareModelResult]
    ground_truth: str | None = None
    fastest_model_id: str | None = None
    most_accurate_model_id: str | None = Field(
        default=None, description="Lowest CER; null when no ground truth was supplied"
    )
    image_width: int = 0
    image_height: int = 0
    device: str = ""
    created_at: datetime | None = None
