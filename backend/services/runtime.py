"""Process-wide ML runtime.

Models, the device and the resolved ML config are expensive to set up and must
be shared across requests -- loading a recogniser per request would be
unusable on a 4 GB GPU.  This module owns that state as a single object built
once during application start-up and reached through
:func:`get_runtime`.

Start-up is deliberately cheap: the registry only *discovers* models (reading
small JSON files), it does not load any weights.  The first request for a
model pays for loading it.
"""

from __future__ import annotations

import logging
import threading
import time
from pathlib import Path
from typing import Any

import torch

from backend.core.config import Settings, get_settings
from ml.config import Config, load_config
from ml.inference.registry import ModelRegistry
from ml.preprocessing.pipeline import PreprocessOptions
from ml.utils.device import describe_hardware, resolve_device
from ml.utils.paths import resolve_path

__all__ = ["MLRuntime", "get_runtime", "init_runtime", "shutdown_runtime"]

logger = logging.getLogger(__name__)

_runtime: "MLRuntime | None" = None
_lock = threading.Lock()


class MLRuntime:
    """Shared ML state for the API process."""

    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.started_at = time.time()

        self.config: Config = load_config(settings.ml_config_path)
        self.device: torch.device = resolve_device(settings.device)
        self.hardware = describe_hardware(settings.device)

        if not settings.allow_amp:
            logger.info("Mixed precision disabled by configuration")

        self.registry = ModelRegistry(self.config, self.device)
        self.default_preprocess = PreprocessOptions.from_config(self.config.get("preprocess"))

        logger.info(
            "ML runtime ready on %s with %d model(s) discovered",
            self.device,
            len(self.registry),
        )
        if self.device.type == "cpu" and self.hardware.cuda_available:
            logger.warning("CUDA is available but OCR_DEVICE pinned the runtime to the CPU.")
        elif self.device.type == "cpu":
            logger.info("Running on CPU. Recognition works but is several times slower.")

    # -- derived state ------------------------------------------------------
    @property
    def uptime_seconds(self) -> float:
        return time.time() - self.started_at

    @property
    def dataset_dir(self) -> Path:
        path = resolve_path(self.config.get("data.processed_dir", "data/processed"))
        return path if path is not None else Path("data/processed")

    def dataset_ready(self, name: str = "synthetic") -> bool:
        """Whether a prepared dataset with a test split exists."""
        return (self.dataset_dir / name / "test.tsv").is_file()

    def demo_dir(self) -> Path:
        path = resolve_path(self.config.get("data.demo_dir", "data/demo"))
        return path if path is not None else Path("data/demo")

    def system_info(self) -> dict[str, Any]:
        """Hardware + settings report for the dashboard."""
        payload = self.hardware.to_dict()
        payload["settings"] = self.settings.summary()
        return payload

    def refresh_models(self) -> None:
        """Re-scan for checkpoints (after a training job finishes)."""
        self.registry.refresh()

    def close(self) -> None:
        """Release every loaded model."""
        self.registry.unload_all()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        logger.info("ML runtime shut down")


def init_runtime(settings: Settings | None = None) -> MLRuntime:
    """Create the runtime, once.  Called from the application lifespan hook."""
    global _runtime
    with _lock:
        if _runtime is None:
            _runtime = MLRuntime(settings or get_settings())
        return _runtime


def get_runtime() -> MLRuntime:
    """Return the runtime, creating it if start-up has not run yet.

    The lazy path exists for tests and scripts that import a service directly
    without going through the FastAPI lifespan.
    """
    if _runtime is None:
        return init_runtime()
    return _runtime


def shutdown_runtime() -> None:
    """Dispose of the runtime (application shutdown, and between tests)."""
    global _runtime
    with _lock:
        if _runtime is not None:
            _runtime.close()
        _runtime = None
