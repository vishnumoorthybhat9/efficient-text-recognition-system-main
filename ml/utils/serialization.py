"""Safe checkpoint loading.

Every ``torch.load`` in this project goes through :func:`safe_load`, which
keeps ``weights_only=True``. That setting uses a restricted unpickler and is
what stops a shared ``.pt`` file from executing arbitrary code on load -- a
real concern for a project whose checkpoints are meant to be handed to a
marker or pushed to a repository.

``weights_only=True`` rejects any class it does not know, including a few
harmless ones that PyTorch itself puts in a checkpoint. Those are allowlisted
here, explicitly and narrowly, rather than by turning the protection off.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import torch

__all__ = ["safe_load", "register_safe_globals"]

logger = logging.getLogger(__name__)

_REGISTERED = False


def register_safe_globals() -> None:
    """Allowlist the benign PyTorch types that appear in our checkpoints.

    ``torch.torch_version.TorchVersion`` is a ``str`` subclass; it reaches the
    checkpoint because the training config snapshot records the torch version
    that produced it. Allowlisting it is strictly safer than the usual
    workaround of setting ``weights_only=False``, which would permit *any*
    class.

    Idempotent, and tolerant of older torch builds that lack the API.
    """
    global _REGISTERED
    if _REGISTERED:
        return

    add_safe_globals = getattr(torch.serialization, "add_safe_globals", None)
    if add_safe_globals is None:  # pragma: no cover - torch < 2.2
        _REGISTERED = True
        return

    allowed: list[Any] = []
    version_type = getattr(getattr(torch, "torch_version", None), "TorchVersion", None)
    if version_type is not None:
        allowed.append(version_type)

    if allowed:
        try:
            add_safe_globals(allowed)
        except Exception:  # pragma: no cover - API drift
            logger.debug("Could not register safe globals for torch.load", exc_info=True)

    _REGISTERED = True


def safe_load(
    path: str | Path,
    *,
    map_location: str | torch.device = "cpu",
) -> Any:
    """Load a checkpoint with the restricted unpickler.

    Args:
        path: the ``.pt`` file.
        map_location: where to place the tensors.

    Returns:
        Whatever the checkpoint contains, normally a dict.

    Raises:
        FileNotFoundError: *path* is not a file.
        RuntimeError: the file could not be loaded, with a message naming the
            likely cause.
    """
    file = Path(path)
    if not file.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {file}")

    register_safe_globals()

    try:
        return torch.load(file, map_location=map_location, weights_only=True)
    except Exception as exc:
        message = str(exc)
        if "WeightsUnpickler" in message or "weights_only" in message:
            raise RuntimeError(
                f"{file.name} contains an object the safe loader does not recognise. "
                "This project keeps weights_only=True so checkpoints cannot execute code "
                "on load. If this checkpoint came from this project, add its type to "
                "ml/utils/serialization.register_safe_globals(); if it came from elsewhere, "
                f"do not load it. Original error: {message[:300]}"
            ) from exc
        raise RuntimeError(f"Could not load {file.name}: {message[:300]}") from exc
