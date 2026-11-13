"""Recognition and model-comparison use cases.

Sits between the HTTP routes and the ML layer: translates request options into
:class:`ml.inference.predictor.RecognizeOptions`, runs the predictor, scores
against a ground truth when one was supplied, and persists the result.
"""

from __future__ import annotations

import logging
from typing import Any, Sequence

import numpy as np
from sqlalchemy.orm import Session

from backend.core.errors import BadRequest, ModelUnavailable
from backend.db.models import ComparisonResult, ComparisonRun, OCRRequest
from backend.schemas.ocr import RecognizeOptionsIn
from backend.services.runtime import MLRuntime
from backend.services.storage import StoredUpload
from ml.evaluation.metrics import character_error_rate, normalize_text, word_error_rate
from ml.inference.predictor import RecognitionResult, RecognizeOptions
from ml.inference.registry import ModelLoadError, ModelNotFound
from ml.models.trocr_wrapper import TrOCRUnavailable

__all__ = [
    "recognize_image",
    "compare_models",
    "score_against_ground_truth",
    "resolve_model_id",
]

logger = logging.getLogger(__name__)

#: Comparing more than this many models in one request would hold several
#: models in VRAM at once, which a 4 GB card cannot do.
MAX_COMPARISON_MODELS = 6


def _oom_error_types() -> tuple[type[BaseException], ...]:
    """Exception types meaning "out of memory", resolved once at import."""
    import torch

    types: list[type[BaseException]] = []
    if hasattr(torch, "OutOfMemoryError"):
        types.append(torch.OutOfMemoryError)
    cuda_oom = getattr(torch.cuda, "OutOfMemoryError", None)
    if cuda_oom is not None and cuda_oom not in types:
        types.append(cuda_oom)
    # `except ()` matches nothing, which is the right behaviour on a build
    # that exposes neither symbol.
    return tuple(types)


OOM_ERRORS: tuple[type[BaseException], ...] = _oom_error_types()


def resolve_model_id(runtime: MLRuntime, model_id: str | None) -> str:
    """Return a usable model id, falling back to the registry's default.

    Raises:
        ModelUnavailable: no model is usable at all (nothing trained yet and
            no pretrained model reachable).
        BadRequest: the requested id does not exist.
    """
    if model_id:
        if model_id not in runtime.registry:
            available = ", ".join(entry.id for entry in runtime.registry.entries())
            raise BadRequest(
                f"Unknown model {model_id!r}. Available models: {available or 'none'}.",
                detail={"requested": model_id, "available": available.split(", ") if available else []},
            )
        return model_id

    default = runtime.registry.default_model_id()
    if default is None:
        raise ModelUnavailable(
            "No recognition model is available yet. Train one with:\n"
            "  python scripts/pretrain_simclr.py --config configs/experiments/simclr_pretrain.yaml\n"
            "  python scripts/train_crnn.py --config configs/experiments/baseline_scratch.yaml\n"
            "Alternatively, install `transformers` so the pretrained TrOCR baseline can be used.",
            detail={"models": []},
        )
    return default


def _to_domain_options(options: RecognizeOptionsIn) -> RecognizeOptions:
    """Translate the HTTP schema into the ML layer's options object."""
    return RecognizeOptions(
        preprocess=options.preprocess.to_domain(),
        single_line=options.single_line,
        decode_strategy=options.decode_strategy,
        beam_width=options.beam_width,
        max_lines=options.max_lines,
        min_confidence=options.min_confidence,
        handwritten=options.handwritten,
        line_detector=options.line_detector,
    )


def _run_predictor(
    runtime: MLRuntime,
    model_id: str,
    image: np.ndarray,
    options: RecognizeOptions,
) -> RecognitionResult:
    """Load the model if needed and recognise *image*.

    Raises:
        ModelUnavailable: the model could not be loaded or used.
    """
    try:
        predictor = runtime.registry.get(model_id)
    except ModelNotFound as exc:
        raise BadRequest(str(exc), detail={"model_id": model_id}) from exc
    except ModelLoadError as exc:
        raise ModelUnavailable(str(exc), detail={"model_id": model_id}) from exc

    try:
        return predictor.recognize(image, options)
    except TrOCRUnavailable as exc:
        raise ModelUnavailable(str(exc), detail={"model_id": model_id}) from exc
    except OOM_ERRORS as exc:
        # Free whatever we can so the next request has a chance of succeeding.
        runtime.registry.unload(model_id)
        raise ModelUnavailable(
            f"Ran out of GPU memory while running {model_id!r}. The model has been unloaded. "
            "Try a smaller image, or set OCR_DEVICE=cpu.",
            detail={"model_id": model_id, "error": str(exc)[:200]},
        ) from exc


def score_against_ground_truth(prediction: str, ground_truth: str) -> dict[str, Any]:
    """Compute CER, WER and exact match for one prediction.

    Whitespace is normalised on both sides first: leading or trailing space is
    an artefact of line cropping, not a recognition error.
    """
    reference = normalize_text(ground_truth)
    hypothesis = normalize_text(prediction)
    return {
        "cer": character_error_rate(reference, hypothesis),
        "wer": word_error_rate(reference, hypothesis),
        "exact_match": reference == hypothesis,
    }


def recognize_image(
    *,
    session: Session,
    runtime: MLRuntime,
    upload: StoredUpload,
    model_id: str | None,
    options: RecognizeOptionsIn,
    ground_truth: str | None = None,
) -> tuple[RecognitionResult, OCRRequest | None]:
    """Recognise an uploaded image and optionally persist the result.

    Returns:
        ``(result, row_or_None)``.  The row is ``None`` when
        ``options.save_to_history`` is false.
    """
    resolved_id = resolve_model_id(runtime, model_id)
    domain_options = _to_domain_options(options)

    result = _run_predictor(runtime, resolved_id, upload.image, domain_options)

    logger.info(
        "Recognised %r with %s: %d line(s), %d char(s), confidence %.3f, %.0f ms",
        upload.original_filename,
        resolved_id,
        result.line_count,
        result.character_count,
        result.confidence,
        result.timings_ms.get("total", 0.0),
    )

    if not options.save_to_history:
        return result, None

    row = OCRRequest(
        model_id=result.model_id,
        model_name=result.model_name,
        device=result.device,
        original_filename=upload.original_filename,
        stored_filename=upload.stored_filename,
        content_type=upload.content_type,
        file_size_bytes=upload.size_bytes,
        image_width=result.image_size[0],
        image_height=result.image_size[1],
        text=result.text,
        confidence=result.confidence,
        line_count=result.line_count,
        word_count=result.word_count,
        character_count=result.character_count,
        total_ms=result.timings_ms.get("total", 0.0),
        preprocess_ms=result.timings_ms.get("preprocess", 0.0),
        segment_ms=result.timings_ms.get("segment", 0.0),
        inference_ms=result.timings_ms.get("inference", 0.0),
        decode_strategy=options.decode_strategy,
        preprocess_options=options.preprocess.model_dump(),
        lines=[line.to_dict() for line in result.lines],
        segmentation=result.segmentation,
        preprocessing_meta=result.preprocessing,
    )

    if ground_truth is not None and ground_truth.strip():
        scores = score_against_ground_truth(result.text, ground_truth)
        row.ground_truth = ground_truth
        row.cer = scores["cer"]
        row.wer = scores["wer"]
        row.exact_match = scores["exact_match"]

    session.add(row)
    session.commit()
    session.refresh(row)

    return result, row


def compare_models(
    *,
    session: Session,
    runtime: MLRuntime,
    upload: StoredUpload,
    model_ids: Sequence[str],
    options: RecognizeOptionsIn,
    ground_truth: str | None = None,
    save: bool = True,
) -> tuple[list[dict[str, Any]], ComparisonRun | None]:
    """Run one image through several models.

    A model that fails is recorded with its error and the others still return:
    a partial comparison is far more useful than a failed request, especially
    when the failure is "TrOCR is not downloaded yet".

    Raises:
        BadRequest: no models given, too many given, or an id is unknown.
    """
    unique_ids = list(dict.fromkeys(model_ids))
    if not unique_ids:
        raise BadRequest("Select at least one model to compare.", detail={"model_ids": []})
    if len(unique_ids) > MAX_COMPARISON_MODELS:
        raise BadRequest(
            f"At most {MAX_COMPARISON_MODELS} models can be compared in one request "
            f"(received {len(unique_ids)}). More than that will not fit in GPU memory together.",
            detail={"limit": MAX_COMPARISON_MODELS, "requested": len(unique_ids)},
        )

    for model_id in unique_ids:
        if model_id not in runtime.registry:
            available = ", ".join(entry.id for entry in runtime.registry.entries())
            raise BadRequest(
                f"Unknown model {model_id!r}. Available models: {available or 'none'}.",
                detail={"requested": model_id},
            )

    domain_options = _to_domain_options(options)
    reference = normalize_text(ground_truth) if ground_truth and ground_truth.strip() else None

    payloads: list[dict[str, Any]] = []
    rows: list[ComparisonResult] = []

    for model_id in unique_ids:
        entry = runtime.registry.entry(model_id)
        payload: dict[str, Any] = {
            "model_id": model_id,
            "model_name": entry.name,
            "provenance": entry.provenance,
            "text": "",
            "confidence": 0.0,
            "total_ms": 0.0,
            "inference_ms": 0.0,
            "line_count": 0,
            "cer": None,
            "wer": None,
            "exact_match": None,
            "error": None,
        }

        try:
            result = _run_predictor(runtime, model_id, upload.image, domain_options)
            payload.update(
                text=result.text,
                confidence=result.confidence,
                total_ms=result.timings_ms.get("total", 0.0),
                inference_ms=result.timings_ms.get("inference", 0.0),
                line_count=result.line_count,
            )
            if reference is not None:
                scores = score_against_ground_truth(result.text, ground_truth or "")
                payload.update(cer=scores["cer"], wer=scores["wer"], exact_match=scores["exact_match"])
        except (ModelUnavailable, BadRequest) as exc:
            # Record and continue: the point of a comparison is the models that
            # did work.
            payload["error"] = exc.message
            logger.warning("Model %s failed during comparison: %s", model_id, exc.message)

        payloads.append(payload)
        rows.append(
            ComparisonResult(
                model_id=payload["model_id"],
                model_name=payload["model_name"],
                provenance=payload["provenance"],
                text=payload["text"],
                confidence=payload["confidence"],
                total_ms=payload["total_ms"],
                inference_ms=payload["inference_ms"],
                line_count=payload["line_count"],
                cer=payload["cer"],
                wer=payload["wer"],
                exact_match=payload["exact_match"],
                error=payload["error"],
            )
        )

    run: ComparisonRun | None = None
    if save:
        run = ComparisonRun(
            original_filename=upload.original_filename,
            stored_filename=upload.stored_filename,
            image_width=upload.width,
            image_height=upload.height,
            ground_truth=ground_truth,
            preprocess_options=options.preprocess.model_dump(),
            device=str(runtime.device),
            results=rows,
        )
        session.add(run)
        session.commit()
        session.refresh(run)

    return payloads, run


def summarise_comparison(payloads: Sequence[dict[str, Any]]) -> dict[str, str | None]:
    """Pick the fastest and the most accurate model from a comparison.

    Only successful runs are considered, and "most accurate" is reported only
    when a ground truth produced CER values.
    """
    successes = [p for p in payloads if not p.get("error") and p.get("text")]

    fastest = min(successes, key=lambda p: p["total_ms"])["model_id"] if successes else None

    scored = [p for p in successes if p.get("cer") is not None]
    most_accurate = min(scored, key=lambda p: p["cer"])["model_id"] if scored else None

    return {"fastest_model_id": fastest, "most_accurate_model_id": most_accurate}
