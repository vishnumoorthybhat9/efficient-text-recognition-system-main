"""Database schema (SQLAlchemy 2.0 declarative).

Four tables, each answering a question the UI asks:

``ocr_requests``
    every recognition performed, with its result, timings and (optionally) a
    ground truth the user supplied afterwards so CER/WER can be computed on
    real uploads rather than only on the test set.
``comparison_runs`` / ``comparison_results``
    one image run through several models side by side.
``evaluation_runs``
    a full benchmark of one model over a dataset split -- the rows behind the
    model-comparison dashboard.
``training_jobs``
    training started from the API, so a run survives a page reload.

JSON columns hold structured detail (per-line results, preprocessing options,
per-epoch metrics).  SQLite stores them as TEXT and SQLAlchemy handles the
serialisation; they are read back whole, never queried into, which is what
makes that acceptable here.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from sqlalchemy import (
    JSON,
    Boolean,
    DateTime,
    Float,
    ForeignKey,
    Index,
    Integer,
    String,
    Text,
    func,
)
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship

__all__ = [
    "Base",
    "OCRRequest",
    "ComparisonRun",
    "ComparisonResult",
    "EvaluationRun",
    "TrainingJob",
    "utcnow",
]


def utcnow() -> datetime:
    """Timezone-aware UTC now.

    ``datetime.utcnow`` is deprecated and returns a *naive* datetime, which
    then compares incorrectly against aware ones. Used as the Python-side
    default so the value is consistent regardless of the database backend.
    """
    return datetime.now(timezone.utc)


class Base(DeclarativeBase):
    """Declarative base for every model."""

    type_annotation_map = {dict[str, Any]: JSON, list[Any]: JSON}


class OCRRequest(Base):
    """One recognition request and its result."""

    __tablename__ = "ocr_requests"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=utcnow, server_default=func.now(), index=True
    )

    # --- model ---
    model_id: Mapped[str] = mapped_column(String(100), index=True)
    model_name: Mapped[str] = mapped_column(String(200), default="")
    device: Mapped[str] = mapped_column(String(32), default="cpu")

    # --- input ---
    original_filename: Mapped[str] = mapped_column(String(400), default="")
    stored_filename: Mapped[str | None] = mapped_column(String(400), nullable=True)
    content_type: Mapped[str] = mapped_column(String(100), default="")
    file_size_bytes: Mapped[int] = mapped_column(Integer, default=0)
    image_width: Mapped[int] = mapped_column(Integer, default=0)
    image_height: Mapped[int] = mapped_column(Integer, default=0)

    # --- output ---
    text: Mapped[str] = mapped_column(Text, default="")
    confidence: Mapped[float] = mapped_column(Float, default=0.0)
    line_count: Mapped[int] = mapped_column(Integer, default=0)
    word_count: Mapped[int] = mapped_column(Integer, default=0)
    character_count: Mapped[int] = mapped_column(Integer, default=0)

    # --- timings (milliseconds) ---
    total_ms: Mapped[float] = mapped_column(Float, default=0.0)
    preprocess_ms: Mapped[float] = mapped_column(Float, default=0.0)
    segment_ms: Mapped[float] = mapped_column(Float, default=0.0)
    inference_ms: Mapped[float] = mapped_column(Float, default=0.0)

    # --- settings and structured detail ---
    decode_strategy: Mapped[str] = mapped_column(String(32), default="greedy")
    preprocess_options: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict)
    lines: Mapped[list[Any]] = mapped_column(JSON, default=list)
    segmentation: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict)
    preprocessing_meta: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict)

    # --- optional ground truth, supplied by the user after the fact ---
    ground_truth: Mapped[str | None] = mapped_column(Text, nullable=True)
    cer: Mapped[float | None] = mapped_column(Float, nullable=True)
    wer: Mapped[float | None] = mapped_column(Float, nullable=True)
    exact_match: Mapped[bool | None] = mapped_column(Boolean, nullable=True)

    __table_args__ = (
        # The history list is "newest first, optionally filtered by model".
        Index("ix_ocr_requests_model_created", "model_id", "created_at"),
    )

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return f"<OCRRequest id={self.id} model={self.model_id!r} chars={self.character_count}>"


class ComparisonRun(Base):
    """One image recognised by several models for side-by-side comparison."""

    __tablename__ = "comparison_runs"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=utcnow, server_default=func.now(), index=True
    )

    original_filename: Mapped[str] = mapped_column(String(400), default="")
    stored_filename: Mapped[str | None] = mapped_column(String(400), nullable=True)
    image_width: Mapped[int] = mapped_column(Integer, default=0)
    image_height: Mapped[int] = mapped_column(Integer, default=0)
    ground_truth: Mapped[str | None] = mapped_column(Text, nullable=True)
    preprocess_options: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict)
    device: Mapped[str] = mapped_column(String(32), default="cpu")

    results: Mapped[list["ComparisonResult"]] = relationship(
        back_populates="run",
        cascade="all, delete-orphan",
        order_by="ComparisonResult.id",
        lazy="selectin",
    )

    def __repr__(self) -> str:  # pragma: no cover
        return f"<ComparisonRun id={self.id} models={len(self.results)}>"


class ComparisonResult(Base):
    """One model's result within a :class:`ComparisonRun`."""

    __tablename__ = "comparison_results"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    run_id: Mapped[int] = mapped_column(
        ForeignKey("comparison_runs.id", ondelete="CASCADE"), index=True
    )

    model_id: Mapped[str] = mapped_column(String(100), index=True)
    model_name: Mapped[str] = mapped_column(String(200), default="")
    provenance: Mapped[str] = mapped_column(String(50), default="")

    text: Mapped[str] = mapped_column(Text, default="")
    confidence: Mapped[float] = mapped_column(Float, default=0.0)
    total_ms: Mapped[float] = mapped_column(Float, default=0.0)
    inference_ms: Mapped[float] = mapped_column(Float, default=0.0)
    line_count: Mapped[int] = mapped_column(Integer, default=0)

    # Populated only when the run carries a ground truth.
    cer: Mapped[float | None] = mapped_column(Float, nullable=True)
    wer: Mapped[float | None] = mapped_column(Float, nullable=True)
    exact_match: Mapped[bool | None] = mapped_column(Boolean, nullable=True)

    # Set when this model failed while the others succeeded, so a partial
    # comparison is still usable instead of failing the whole request.
    error: Mapped[str | None] = mapped_column(Text, nullable=True)

    run: Mapped[ComparisonRun] = relationship(back_populates="results")

    def __repr__(self) -> str:  # pragma: no cover
        return f"<ComparisonResult run={self.run_id} model={self.model_id!r}>"


class EvaluationRun(Base):
    """A benchmark of one model over a dataset split."""

    __tablename__ = "evaluation_runs"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=utcnow, server_default=func.now(), index=True
    )

    model_id: Mapped[str] = mapped_column(String(100), index=True)
    model_name: Mapped[str] = mapped_column(String(200), default="")
    provenance: Mapped[str] = mapped_column(String(50), default="")
    experiment_arm: Mapped[str] = mapped_column(String(50), default="other")

    dataset: Mapped[str] = mapped_column(String(120), default="synthetic")
    split: Mapped[str] = mapped_column(String(50), default="test")
    num_samples: Mapped[int] = mapped_column(Integer, default=0)

    # --- accuracy ---
    cer: Mapped[float] = mapped_column(Float, default=0.0)
    wer: Mapped[float] = mapped_column(Float, default=0.0)
    sequence_accuracy: Mapped[float] = mapped_column(Float, default=0.0)
    character_accuracy: Mapped[float] = mapped_column(Float, default=0.0)
    cer_normalized: Mapped[float] = mapped_column(Float, default=0.0)

    # --- efficiency ---
    avg_inference_ms: Mapped[float] = mapped_column(Float, default=0.0)
    median_inference_ms: Mapped[float] = mapped_column(Float, default=0.0)
    throughput_images_per_s: Mapped[float] = mapped_column(Float, default=0.0)
    peak_memory_mb: Mapped[float] = mapped_column(Float, default=0.0)
    parameter_count: Mapped[int] = mapped_column(Integer, default=0)
    model_size_mb: Mapped[float] = mapped_column(Float, default=0.0)

    device: Mapped[str] = mapped_column(String(32), default="cpu")
    decode_strategy: Mapped[str] = mapped_column(String(32), default="greedy")
    batch_size: Mapped[int] = mapped_column(Integer, default=32)

    metrics: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict)
    worst_samples: Mapped[list[Any]] = mapped_column(JSON, default=list)
    report_path: Mapped[str | None] = mapped_column(String(500), nullable=True)
    notes: Mapped[str] = mapped_column(Text, default="")

    __table_args__ = (
        Index("ix_evaluation_model_dataset_split", "model_id", "dataset", "split"),
    )

    def __repr__(self) -> str:  # pragma: no cover
        return f"<EvaluationRun id={self.id} model={self.model_id!r} cer={self.cer:.4f}>"


class TrainingJob(Base):
    """A training or pretraining run started through the API."""

    __tablename__ = "training_jobs"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=utcnow, server_default=func.now(), index=True
    )
    started_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    finished_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)

    #: "simclr_pretrain" or "crnn_train".
    kind: Mapped[str] = mapped_column(String(40), index=True)
    config_name: Mapped[str] = mapped_column(String(200), default="")
    run_name: Mapped[str] = mapped_column(String(120), default="")

    #: queued | running | completed | failed | cancelled
    status: Mapped[str] = mapped_column(String(20), default="queued", index=True)

    epochs_total: Mapped[int] = mapped_column(Integer, default=0)
    epochs_completed: Mapped[int] = mapped_column(Integer, default=0)
    current_epoch: Mapped[int] = mapped_column(Integer, default=0)
    progress: Mapped[float] = mapped_column(Float, default=0.0)

    latest_metrics: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict)
    history: Mapped[list[Any]] = mapped_column(JSON, default=list)
    checkpoint_dir: Mapped[str | None] = mapped_column(String(500), nullable=True)
    log_path: Mapped[str | None] = mapped_column(String(500), nullable=True)
    error: Mapped[str | None] = mapped_column(Text, nullable=True)
    device: Mapped[str] = mapped_column(String(32), default="cpu")

    @property
    def is_active(self) -> bool:
        return self.status in ("queued", "running")

    def __repr__(self) -> str:  # pragma: no cover
        return f"<TrainingJob id={self.id} kind={self.kind!r} status={self.status!r}>"
