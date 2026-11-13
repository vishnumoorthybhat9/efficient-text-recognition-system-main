"""Shared pytest fixtures.

The API fixtures point the application at a temporary database and upload
directory, so running the suite never touches the real ``artifacts/ocr.db``
or the images in ``data/uploads``.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Iterator

import numpy as np
import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from ml.config import load_config  # noqa: E402
from ml.text.charset import Charset  # noqa: E402


@pytest.fixture(scope="session")
def project_root() -> Path:
    return PROJECT_ROOT


@pytest.fixture(scope="session")
def config():
    """The default ML config."""
    return load_config("default.yaml")


@pytest.fixture(scope="session")
def charset(config) -> Charset:
    return Charset.from_config(config.charset)


@pytest.fixture
def text_image() -> np.ndarray:
    """A synthetic grayscale image with one line of dark text on light paper."""
    import cv2

    image = np.full((64, 380), 240, dtype=np.uint8)
    cv2.putText(image, "Hello World 123", (12, 44), cv2.FONT_HERSHEY_SIMPLEX, 1.0, 25, 2, cv2.LINE_AA)
    return image


@pytest.fixture
def multiline_image() -> np.ndarray:
    """A page with four clearly separated text lines."""
    import cv2

    image = np.full((360, 760), 243, dtype=np.uint8)
    for index, text in enumerate(["First line here", "Second line now", "Third line too", "Fourth line end"]):
        cv2.putText(image, text, (28, 64 + index * 82), cv2.FONT_HERSHEY_SIMPLEX, 1.1, 22, 2, cv2.LINE_AA)
    return image


@pytest.fixture(scope="session")
def torch_available() -> bool:
    try:
        import torch  # noqa: F401

        return True
    except ImportError:
        return False


@pytest.fixture
def api_environment(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[Path]:
    """Point the application's settings at a throwaway directory.

    Caches are cleared on both sides of the fixture: ``get_settings`` is
    ``lru_cache``d and the database engine is a module-level singleton, so a
    leftover instance from another test would silently use the real database.
    """
    from backend.core.config import get_settings
    from backend.db.session import dispose_engine
    from backend.services.runtime import shutdown_runtime

    monkeypatch.setenv("OCR_DATABASE_URL", f"sqlite:///{(tmp_path / 'test.db').as_posix()}")
    monkeypatch.setenv("OCR_UPLOAD_DIR", str(tmp_path / "uploads"))
    monkeypatch.setenv("OCR_ARTIFACT_DIR", str(tmp_path / "artifacts"))
    monkeypatch.setenv("OCR_ENVIRONMENT", "development")
    monkeypatch.setenv("OCR_LOG_LEVEL", "WARNING")
    # Never let a test reach the network for model weights.
    monkeypatch.setenv("TRANSFORMERS_OFFLINE", "1")

    get_settings.cache_clear()
    dispose_engine()
    shutdown_runtime()

    yield tmp_path

    get_settings.cache_clear()
    dispose_engine()
    shutdown_runtime()


@pytest.fixture
def client(api_environment):
    """A TestClient with the application's lifespan run."""
    pytest.importorskip("torch", reason="the API imports the ML stack")
    from fastapi.testclient import TestClient

    from backend.main import create_app

    with TestClient(create_app()) as test_client:
        yield test_client
