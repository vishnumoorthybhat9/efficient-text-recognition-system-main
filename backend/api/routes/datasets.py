"""Dataset status, provenance and demo-image endpoints."""

from __future__ import annotations

import logging
from typing import Annotated, Any

from fastapi import APIRouter, Query
from fastapi.responses import FileResponse

from backend.api.deps import RuntimeDep
from backend.services import dataset_service

__all__ = ["router"]

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/datasets", tags=["datasets"])


@router.get("", summary="Prepared datasets and official download links")
def list_datasets(runtime: RuntimeDep) -> dict[str, Any]:
    """What data is on this machine, and how to obtain the public benchmarks.

    ``prepared`` lists datasets ready to train on; ``external`` carries the
    official URLs, sizes, licence terms and preparation commands for the
    datasets named in the synopsis (IAM, IIIT5K, MJSynth, SynthText), none of
    which are redistributed with this project.
    """
    return dataset_service.list_datasets(runtime)


@router.get("/demo", summary="Bundled demo images")
def demo_images(runtime: RuntimeDep) -> dict[str, Any]:
    """Sample images so the application is usable without an upload.

    Returns an empty list when the demo set has not been generated, which the
    UI treats as "hide the section" rather than an error.
    """
    images = dataset_service.list_demo_images(runtime)
    return {"images": images, "count": len(images)}


@router.get("/demo/{filename}", summary="One demo image", response_class=FileResponse)
def demo_image(runtime: RuntimeDep, filename: str) -> FileResponse:
    """Serve a bundled demo image by filename."""
    path = dataset_service.resolve_demo_image(runtime, filename)
    return FileResponse(path, media_type="image/png", filename=filename)


@router.get("/{name}", summary="One prepared dataset's detail")
def dataset_detail(
    runtime: RuntimeDep,
    name: str,
    split: Annotated[str, Query(description="train | val | test")] = "train",
    limit: Annotated[int, Query(ge=1, le=200)] = 20,
) -> dict[str, Any]:
    """Label statistics and sample rows for one split.

    ``unsupported_characters`` is the field to watch: a non-empty result means
    the dataset's labels contain characters the configured charset cannot
    represent, and those samples will be dropped during training.
    """
    return dataset_service.dataset_detail(runtime, name, split=split, limit=limit)
