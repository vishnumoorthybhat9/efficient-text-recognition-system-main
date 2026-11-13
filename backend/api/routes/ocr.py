"""Recognition, preprocessing preview and model comparison endpoints."""

from __future__ import annotations

import base64
import logging
import time
from typing import Annotated, Any

from fastapi import APIRouter, File, Form, Response, UploadFile
from pydantic import Field

from backend.api.deps import RuntimeDep, SessionDep, SettingsDep, parse_json_form
from backend.core.errors import DependencyMissing
from backend.schemas.common import ApiModel
from backend.schemas.ocr import (
    CompareResponse,
    PreprocessPreviewResponse,
    PreprocessStageOut,
    RecognitionResponse,
    RecognizeOptionsIn,
    RegionOut,
)
from backend.services import ocr_service
from backend.services.export_service import (
    DOCX_AVAILABLE,
    build_docx,
    build_txt,
    export_metadata,
    safe_filename,
)
from backend.services.storage import read_upload
from ml.inference.handwriting import clean_handwritten_page, detect_handwritten_lines
from ml.inference.segmentation import detect_text_lines, draw_regions
from ml.preprocessing import ops
from ml.preprocessing.pipeline import PreprocessPipeline

__all__ = ["router"]

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/ocr", tags=["recognition"])

#: Stage previews are downscaled before encoding: a 4000 px scan would
#: otherwise produce several megabytes of base64 per stage.
_PREVIEW_MAX_WIDTH = 900


def _data_url(image) -> tuple[str, int, int]:
    """Encode an image as a PNG data URL, downscaled for preview."""
    import cv2

    preview = image
    if preview.shape[1] > _PREVIEW_MAX_WIDTH:
        scale = _PREVIEW_MAX_WIDTH / preview.shape[1]
        preview = cv2.resize(
            preview,
            (_PREVIEW_MAX_WIDTH, max(1, int(preview.shape[0] * scale))),
            interpolation=cv2.INTER_AREA,
        )

    encoded = base64.b64encode(ops.encode_png(preview)).decode("ascii")
    return f"data:image/png;base64,{encoded}", int(preview.shape[1]), int(preview.shape[0])


@router.post(
    "/recognize",
    response_model=RecognitionResponse,
    summary="Recognise text in an uploaded image",
)
async def recognize(
    session: SessionDep,
    runtime: RuntimeDep,
    settings: SettingsDep,
    file: Annotated[UploadFile, File(description="Image to recognise")],
    model_id: Annotated[str | None, Form(description="Model id; omitted uses the default")] = None,
    options: Annotated[str | None, Form(description="JSON object of recognition options")] = None,
    ground_truth: Annotated[str | None, Form(description="Optional reference, scores the result")] = None,
) -> RecognitionResponse:
    """Run the full pipeline: enhance, segment into lines, recognise, assemble.

    Supplying ``ground_truth`` computes CER and WER for this image immediately
    and stores them with the history entry.
    """
    parsed = parse_json_form(options, RecognizeOptionsIn)
    upload = await read_upload(file, settings, store=parsed.save_to_history)

    result, row = ocr_service.recognize_image(
        session=session,
        runtime=runtime,
        upload=upload,
        model_id=model_id,
        options=parsed,
        ground_truth=ground_truth,
    )

    payload = result.to_dict()
    payload["id"] = row.id if row else None
    payload["created_at"] = row.created_at if row else None

    if ground_truth and ground_truth.strip():
        # Score even when the result was not persisted, so "save to history
        # off" does not silently drop the metrics the caller asked for.
        scores = (
            {"cer": row.cer, "wer": row.wer, "exact_match": row.exact_match}
            if row is not None
            else ocr_service.score_against_ground_truth(result.text, ground_truth)
        )
        payload.update(ground_truth=ground_truth, **scores)

    return RecognitionResponse(**payload)


@router.post(
    "/compare",
    response_model=CompareResponse,
    summary="Run one image through several models",
)
async def compare(
    session: SessionDep,
    runtime: RuntimeDep,
    settings: SettingsDep,
    file: Annotated[UploadFile, File(description="Image to recognise")],
    model_ids: Annotated[str, Form(description="Comma-separated model ids")],
    options: Annotated[str | None, Form()] = None,
    ground_truth: Annotated[str | None, Form()] = None,
) -> CompareResponse:
    """Compare models on the same image, with identical preprocessing.

    A model that fails (for example TrOCR with no network and an empty cache)
    is returned with an ``error`` field set while the others still produce
    results -- a partial comparison is more useful than none.
    """
    parsed = parse_json_form(options, RecognizeOptionsIn)
    requested = [item.strip() for item in model_ids.split(",") if item.strip()]

    upload = await read_upload(file, settings, store=parsed.save_to_history)

    payloads, run = ocr_service.compare_models(
        session=session,
        runtime=runtime,
        upload=upload,
        model_ids=requested,
        options=parsed,
        ground_truth=ground_truth,
        save=parsed.save_to_history,
    )
    summary = ocr_service.summarise_comparison(payloads)

    return CompareResponse(
        id=run.id if run else None,
        results=payloads,  # type: ignore[arg-type]
        ground_truth=ground_truth,
        image_width=upload.width,
        image_height=upload.height,
        device=str(runtime.device),
        created_at=run.created_at if run else None,
        **summary,  # type: ignore[arg-type]
    )


@router.post(
    "/preprocess/preview",
    response_model=PreprocessPreviewResponse,
    summary="Show every preprocessing stage",
    tags=["preprocessing"],
)
async def preprocess_preview(
    settings: SettingsDep,
    file: Annotated[UploadFile, File(description="Image to preprocess")],
    options: Annotated[str | None, Form(description="JSON object of recognition options")] = None,
    detect_lines: Annotated[bool, Form(description="Also run line segmentation")] = True,
) -> PreprocessPreviewResponse:
    """Apply the enhancement pipeline and return each intermediate image.

    This is what makes the preprocessing controls honest: the user sees what
    each toggle actually did to their image rather than being told.
    """
    parsed = parse_json_form(options, RecognizeOptionsIn)
    upload = await read_upload(file, settings, store=False)

    pipeline = PreprocessPipeline(parsed.preprocess.to_domain())
    result = pipeline.run(upload.image, capture_stages=True)

    stages: list[PreprocessStageOut] = []
    for stage in result.stages:
        data_url, width, height = _data_url(stage.image)
        stages.append(
            PreprocessStageOut(
                name=stage.name,
                label=stage.label,
                detail=stage.detail,
                duration_ms=stage.duration_ms,
                image=data_url,
                width=width,
                height=height,
            )
        )

    regions: list[RegionOut] = []
    overlay: str | None = None

    page = result.image
    if parsed.handwritten and not parsed.single_line:
        # Show the cleaned page as its own stage, so the user can see the
        # shadows and ruled lines that were removed.
        started = time.perf_counter()
        cleaned = clean_handwritten_page(result.image)
        page = cleaned.image
        data_url, width, height = _data_url(page)
        info = cleaned.metadata()
        stages.append(
            PreprocessStageOut(
                name="handwriting_clean",
                label="Handwriting clean-up",
                detail=(
                    f"shadows flattened, {info['horizontal_rules_removed']} ruled line(s) and "
                    f"{info['vertical_rules_removed']} margin line(s) removed"
                ),
                duration_ms=(time.perf_counter() - started) * 1000.0,
                image=data_url,
                width=width,
                height=height,
            )
        )

    if detect_lines and not parsed.single_line:
        if parsed.handwritten:
            segmentation, _ = detect_handwritten_lines(
                page, detector=parsed.line_detector, max_lines=parsed.max_lines
            )
        else:
            segmentation = detect_text_lines(page, max_lines=parsed.max_lines)
        regions = [RegionOut(**region.to_dict()) for region in segmentation.regions]
        overlay, _, _ = _data_url(draw_regions(page, segmentation.regions))

    logger.info(
        "Preprocessing preview for %r: %d stage(s), %d line(s) detected",
        upload.original_filename,
        len(stages),
        len(regions),
    )

    return PreprocessPreviewResponse(
        stages=stages,
        metadata=result.metadata(),
        options=parsed.preprocess.model_dump(),
        detected_lines=regions,
        overlay=overlay,
    )


class ExportRequest(ApiModel):
    """Body for the export endpoints.

    The text is taken from the request rather than re-recognised, so whatever
    the user edited in the browser is what gets saved.
    """

    text: str = Field(description="Exactly the text to write; it is not modified")
    filename: str = Field(default="extracted-text", max_length=120)
    title: str = Field(default="Extracted text", max_length=200)
    model_name: str | None = None
    device: str | None = None
    total_ms: float | None = None
    confidence: float | None = None
    lines: list[dict[str, Any]] = Field(
        default_factory=list,
        description="Per-line records with `text` and `confidence`, used to mark uncertain lines",
    )
    low_confidence_threshold: float = Field(default=0.0, ge=0.0, le=1.0)


@router.post("/export/txt", summary="Download the text as a .txt file")
def export_txt(payload: ExportRequest) -> Response:
    """Return the text verbatim as UTF-8 with a BOM.

    Nothing is re-wrapped or corrected: the file contains exactly the
    characters supplied.
    """
    name = safe_filename(payload.filename)
    return Response(
        content=build_txt(payload.text),
        media_type="text/plain; charset=utf-8",
        headers={"Content-Disposition": f'attachment; filename="{name}.txt"'},
    )


@router.post("/export/docx", summary="Download the text as a Word document")
def export_docx(payload: ExportRequest) -> Response:
    """Render the text into a .docx, preserving its line breaks.

    Lines whose confidence falls below ``low_confidence_threshold`` are
    highlighted rather than presented as certain.
    """
    if not DOCX_AVAILABLE:
        raise DependencyMissing(
            "DOCX export needs the python-docx package. Install it with:  pip install python-docx",
            detail={"package": "python-docx"},
        )

    document = build_docx(
        payload.text,
        title=payload.title,
        metadata=export_metadata(
            model_name=payload.model_name,
            device=payload.device,
            total_ms=payload.total_ms,
            confidence=payload.confidence,
        ),
        lines=payload.lines,
        low_confidence_threshold=payload.low_confidence_threshold,
    )

    name = safe_filename(payload.filename)
    return Response(
        content=document,
        media_type="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
        headers={"Content-Disposition": f'attachment; filename="{name}.docx"'},
    )
