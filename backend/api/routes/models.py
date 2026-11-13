"""Model registry endpoints."""

from __future__ import annotations

import logging

from fastapi import APIRouter, Path

from backend.api.deps import RuntimeDep
from backend.core.errors import BadRequest, ModelUnavailable
from backend.schemas.common import MessageResponse
from backend.schemas.models import ModelInfoOut, ModelListResponse
from ml.inference.registry import ModelLoadError, ModelNotFound

__all__ = ["router"]

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/models", tags=["models"])


@router.get("", response_model=ModelListResponse, summary="List available models")
def list_models(runtime: RuntimeDep) -> ModelListResponse:
    """Every discovered model, with provenance and training results.

    Discovery is cheap -- it reads the small ``history.json`` beside each
    checkpoint, not the weights -- so this endpoint never loads a model.
    """
    payloads = runtime.registry.describe_all()
    arms = {entry.extra.get("experiment_arm") for entry in runtime.registry.entries()}

    return ModelListResponse(
        models=[ModelInfoOut(**payload) for payload in payloads],
        default_model_id=runtime.registry.default_model_id(),
        loaded_ids=runtime.registry.loaded_ids,
        device=str(runtime.device),
        comparison_ready={"control", "treatment"}.issubset(arms),
    )


@router.get("/{model_id}", response_model=ModelInfoOut, summary="One model's details")
def get_model(runtime: RuntimeDep, model_id: str = Path(description="Model id")) -> ModelInfoOut:
    """Metadata for a single model.

    Raises 400 when the id is unknown.
    """
    try:
        entry = runtime.registry.entry(model_id)
    except ModelNotFound as exc:
        raise BadRequest(str(exc), detail={"model_id": model_id}) from exc

    payload = entry.to_dict()
    payload["loaded"] = model_id in runtime.registry.loaded_ids
    return ModelInfoOut(**payload)


@router.post("/{model_id}/load", response_model=MessageResponse, summary="Load a model into memory")
def load_model(runtime: RuntimeDep, model_id: str) -> MessageResponse:
    """Load a model eagerly.

    Useful before a demo: the first request to a cold model pays for reading
    the checkpoint (and, for TrOCR, downloading several hundred megabytes),
    which would otherwise look like a slow recognition.
    """
    try:
        predictor = runtime.registry.get(model_id)
    except ModelNotFound as exc:
        raise BadRequest(str(exc), detail={"model_id": model_id}) from exc
    except ModelLoadError as exc:
        raise ModelUnavailable(str(exc), detail={"model_id": model_id}) from exc

    return MessageResponse(
        message=f"{predictor.model_name} is loaded on {predictor.device} and ready.",
        detail=predictor.describe(),
    )


@router.post("/{model_id}/unload", response_model=MessageResponse, summary="Free a model's memory")
def unload_model(runtime: RuntimeDep, model_id: str) -> MessageResponse:
    """Unload a model, freeing its GPU memory.

    Relevant on a small card: TrOCR and the project's own recogniser together
    can exhaust 4 GB.
    """
    if model_id not in runtime.registry:
        raise BadRequest(f"Unknown model {model_id!r}.", detail={"model_id": model_id})

    was_loaded = runtime.registry.unload(model_id)
    return MessageResponse(
        message=(
            f"Unloaded {model_id!r}; its memory has been released."
            if was_loaded
            else f"{model_id!r} was not loaded, so nothing was freed."
        ),
        detail={"model_id": model_id, "was_loaded": was_loaded},
    )


@router.post("/refresh", response_model=ModelListResponse, summary="Re-scan for new checkpoints")
def refresh_models(runtime: RuntimeDep) -> ModelListResponse:
    """Re-scan the checkpoint directory.

    Called after training finishes so a newly trained model becomes selectable
    without restarting the server.
    """
    runtime.refresh_models()
    logger.info("Model registry refreshed on request")
    return list_models(runtime)
