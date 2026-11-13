"""Benchmarking: accuracy *and* cost, measured on a held-out split.

The synopsis asks for an "optimized OCR system with lower computational
complexity", so a results table with only CER in it would answer half the
question.  Every benchmark therefore reports accuracy (CER / WER / exact
match) alongside cost (latency, throughput, peak memory, parameter count,
checkpoint size).

Measurement rules followed here
-------------------------------
* **Warm-up batches are discarded.** The first CUDA forward pass pays for
  kernel autotuning and allocator setup and can be 10-100x steady state.
* **CUDA is synchronised** around each timed region, otherwise the clock only
  measures how long it took to *enqueue* the work.
* **Image loading is timed separately** from inference. Mixing them would make
  a fast model look slow because of disk I/O that every model shares.
* Latency is reported **per image**, and both mean and median are kept: the
  median is the honest "typical" figure when one outlier batch skews the mean.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Sequence

import numpy as np
import torch

from ml.datasets.manifest import ManifestEntry
from ml.evaluation.metrics import (
    AggregateMetrics,
    SampleMetrics,
    evaluate_predictions,
    worst_samples,
)
from ml.inference.predictor import BasePredictor, RecognizeOptions
from ml.utils.image_io import ImageDecodeError, read_image
from ml.utils.timing import stopwatch

__all__ = ["BenchmarkResult", "benchmark_predictor"]

logger = logging.getLogger(__name__)

#: Batches run before timing starts.
DEFAULT_WARMUP_BATCHES = 2


@dataclass(slots=True)
class BenchmarkResult:
    """Accuracy and cost for one model on one dataset split."""

    model_id: str
    model_name: str
    dataset: str
    split: str
    num_samples: int
    metrics: AggregateMetrics
    samples: list[SampleMetrics] = field(default_factory=list)

    # --- cost ---
    avg_inference_ms: float = 0.0
    median_inference_ms: float = 0.0
    p95_inference_ms: float = 0.0
    total_inference_s: float = 0.0
    throughput_images_per_s: float = 0.0
    avg_load_ms: float = 0.0
    peak_memory_mb: float = 0.0
    parameter_count: int = 0
    model_size_mb: float = 0.0

    device: str = "cpu"
    decode_strategy: str = "greedy"
    batch_size: int = 32
    failed_samples: int = 0
    notes: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "model_id": self.model_id,
            "model_name": self.model_name,
            "dataset": self.dataset,
            "split": self.split,
            "num_samples": self.num_samples,
            "accuracy": self.metrics.to_dict(),
            "cost": {
                "avg_inference_ms": round(self.avg_inference_ms, 3),
                "median_inference_ms": round(self.median_inference_ms, 3),
                "p95_inference_ms": round(self.p95_inference_ms, 3),
                "total_inference_s": round(self.total_inference_s, 3),
                "throughput_images_per_s": round(self.throughput_images_per_s, 2),
                "avg_load_ms": round(self.avg_load_ms, 3),
                "peak_memory_mb": round(self.peak_memory_mb, 2),
                "parameter_count": self.parameter_count,
                "model_size_mb": round(self.model_size_mb, 2),
            },
            "device": self.device,
            "decode_strategy": self.decode_strategy,
            "batch_size": self.batch_size,
            "failed_samples": self.failed_samples,
            "notes": self.notes,
            "worst_samples": [s.to_dict() for s in worst_samples(self.samples, 20)],
        }

    def summary(self) -> str:
        return (
            f"{self.model_name}: CER={self.metrics.cer:.4f} WER={self.metrics.wer:.4f} "
            f"exact={self.metrics.sequence_accuracy:.4f} | "
            f"{self.avg_inference_ms:.2f} ms/img ({self.throughput_images_per_s:.0f} img/s) "
            f"on {self.device}"
        )


def _model_size_mb(predictor: BasePredictor) -> tuple[int, float]:
    """Parameter count and on-disk size for *predictor*.

    Falls back to ``params x 4 bytes`` when there is no checkpoint file --
    the case for TrOCR, whose weights live in the Hugging Face cache.
    """
    info = predictor.describe()

    parameters = 0
    params_field = info.get("parameters")
    if isinstance(params_field, dict):
        parameters = int(params_field.get("total") or 0)
    elif isinstance(params_field, int):
        parameters = params_field

    checkpoint = info.get("checkpoint")
    if checkpoint:
        path = Path(str(checkpoint))
        if path.is_file():
            return parameters, path.stat().st_size / (1024 * 1024)

    return parameters, (parameters * 4) / (1024 * 1024)


def _reset_peak_memory(device: torch.device) -> None:
    if device.type == "cuda" and torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats(device)


def _peak_memory_mb(device: torch.device) -> float:
    if device.type == "cuda" and torch.cuda.is_available():
        return torch.cuda.max_memory_allocated(device) / (1024 * 1024)
    return 0.0


def benchmark_predictor(
    predictor: BasePredictor,
    entries: Sequence[ManifestEntry],
    *,
    dataset: str = "synthetic",
    split: str = "test",
    batch_size: int = 32,
    decode_strategy: str = "greedy",
    beam_width: int = 10,
    warmup_batches: int = DEFAULT_WARMUP_BATCHES,
    progress: Callable[[int, int], None] | None = None,
    should_stop: Callable[[], bool] | None = None,
) -> BenchmarkResult:
    """Run *predictor* over *entries* and measure accuracy and cost.

    The manifest entries are already cropped text lines, so segmentation is
    skipped and :meth:`BasePredictor.recognize_lines` is called directly.

    Args:
        predictor: the model to benchmark.
        entries: labeled samples.
        dataset / split: recorded in the result for provenance.
        batch_size: images per forward pass.
        decode_strategy: ``"greedy"`` or ``"beam"``.
        beam_width: beam size when decoding with beam search.
        warmup_batches: batches run untimed before measurement starts.
        progress: called with ``(done, total)`` after each batch.
        should_stop: polled per batch; returning True ends the run early and
            scores whatever was completed.

    Returns:
        A :class:`BenchmarkResult`.

    Raises:
        ValueError: *entries* is empty.
    """
    if not entries:
        raise ValueError("benchmark_predictor received no samples to evaluate")

    device = predictor.device
    options = RecognizeOptions(decode_strategy=decode_strategy, beam_width=beam_width)

    references: list[str] = []
    predictions: list[str] = []
    sample_ids: list[str] = []
    batch_times_ms: list[float] = []
    batch_sizes: list[int] = []
    load_times_ms: list[float] = []
    failed = 0

    total = len(entries)
    batch_count = (total + batch_size - 1) // batch_size
    effective_warmup = min(max(0, warmup_batches), max(0, batch_count - 1))

    logger.info(
        "Benchmarking %s on %s/%s: %d sample(s), batch=%d, decode=%s, %d warm-up batch(es)",
        predictor.model_id,
        dataset,
        split,
        total,
        batch_size,
        decode_strategy,
        effective_warmup,
    )

    _reset_peak_memory(device)
    started = time.perf_counter()

    for batch_index in range(batch_count):
        if should_stop is not None and should_stop():
            logger.warning("Benchmark cancelled after %d/%d sample(s)", len(predictions), total)
            break

        chunk = entries[batch_index * batch_size : (batch_index + 1) * batch_size]

        # --- load (timed separately: shared cost, not the model's) ---
        images: list[np.ndarray] = []
        chunk_refs: list[str] = []
        chunk_ids: list[str] = []

        with stopwatch() as load_timer:
            for entry in chunk:
                try:
                    images.append(read_image(entry.path, grayscale=True))
                    chunk_refs.append(entry.label)
                    chunk_ids.append(entry.relative_path)
                except (FileNotFoundError, ImageDecodeError) as exc:
                    failed += 1
                    logger.warning("Skipping %s: %s", entry.relative_path, exc)

        if not images:
            continue
        load_times_ms.append(load_timer.elapsed_ms)

        # --- inference (timed, CUDA-synchronised) ---
        with stopwatch(device) as inference_timer:
            outputs = predictor.recognize_lines(images, options)

        is_warmup = batch_index < effective_warmup
        if not is_warmup:
            batch_times_ms.append(inference_timer.elapsed_ms)
            batch_sizes.append(len(images))

        references.extend(chunk_refs)
        predictions.extend(text for text, _ in outputs)
        sample_ids.extend(chunk_ids)

        if progress is not None:
            progress(len(predictions), total)

    wall_clock = time.perf_counter() - started

    # --- accuracy ---
    aggregate, samples = evaluate_predictions(references, predictions, sample_ids=sample_ids)

    # --- cost ---
    per_image_ms: list[float] = []
    for duration, size in zip(batch_times_ms, batch_sizes):
        per_image_ms.extend([duration / size] * size)

    if per_image_ms:
        array = np.asarray(per_image_ms, dtype=np.float64)
        avg_ms = float(array.mean())
        median_ms = float(np.median(array))
        p95_ms = float(np.percentile(array, 95))
        total_inference_s = float(sum(batch_times_ms) / 1000.0)
        throughput = sum(batch_sizes) / total_inference_s if total_inference_s > 0 else 0.0
    else:
        # Everything was warm-up: too few batches to time honestly.
        avg_ms = median_ms = p95_ms = total_inference_s = throughput = 0.0
        logger.warning(
            "No timed batches (every batch was warm-up). Latency figures are reported as 0; "
            "evaluate more samples or lower --warmup-batches for a usable measurement."
        )

    parameters, size_mb = _model_size_mb(predictor)

    result = BenchmarkResult(
        model_id=predictor.model_id,
        model_name=predictor.model_name,
        dataset=dataset,
        split=split,
        num_samples=len(predictions),
        metrics=aggregate,
        samples=samples,
        avg_inference_ms=avg_ms,
        median_inference_ms=median_ms,
        p95_inference_ms=p95_ms,
        total_inference_s=total_inference_s,
        throughput_images_per_s=throughput,
        avg_load_ms=float(np.mean(load_times_ms) / batch_size) if load_times_ms else 0.0,
        peak_memory_mb=_peak_memory_mb(device),
        parameter_count=parameters,
        model_size_mb=size_mb,
        device=str(device),
        decode_strategy=decode_strategy,
        batch_size=batch_size,
        failed_samples=failed,
        notes=(
            f"wall clock {wall_clock:.1f}s including image loading; "
            f"{effective_warmup} warm-up batch(es) excluded from timing"
        ),
    )

    logger.info("%s", result.summary())
    return result
