"""Application settings, loaded from the environment and ``.env``.

Everything here has a working default, so the application runs with no
configuration at all.  ``.env`` (see ``.env.example``) overrides the defaults,
and real environment variables override ``.env``.

ML *hyperparameters* deliberately live in ``configs/*.yaml`` rather than here:
they belong to an experiment and must be versioned with it, while these
settings describe one deployment of the server.
"""

from __future__ import annotations

import logging
from functools import lru_cache
from pathlib import Path
from typing import Any, Literal

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

from ml.utils.paths import PROJECT_ROOT, ensure_dir, resolve_path

__all__ = ["Settings", "get_settings"]

logger = logging.getLogger(__name__)


class Settings(BaseSettings):
    """Runtime configuration for the FastAPI application."""

    model_config = SettingsConfigDict(
        env_prefix="OCR_",
        env_file=str(PROJECT_ROOT / ".env"),
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
    )

    # --- application ---
    app_name: str = "Efficient Text Recognition"
    app_version: str = "1.0.0"
    environment: Literal["development", "production"] = "development"
    log_level: str = "INFO"
    log_json: bool = False

    # --- server ---
    host: str = "127.0.0.1"
    port: int = 8000
    reload: bool = False
    cors_origins: str = "http://localhost:5173,http://127.0.0.1:5173"

    # --- storage ---
    data_dir: str = "data"
    upload_dir: str = "data/uploads"
    artifact_dir: str = "artifacts"
    database_url: str = "sqlite:///artifacts/ocr.db"

    # --- uploads ---
    max_upload_mb: float = Field(default=15.0, gt=0, le=200)

    # --- compute ---
    device: Literal["auto", "cuda", "cpu"] = "auto"
    allow_amp: bool = True
    num_workers: int = Field(default=0, ge=0, le=32)

    # --- ML config file used by the server ---
    ml_config: str = "default.yaml"

    # --- history retention ---
    #: Uploaded images older than this are pruned by the maintenance endpoint.
    history_retention_days: int = Field(default=30, ge=1)
    #: Cap on rows returned by a single history query.
    max_page_size: int = Field(default=200, ge=1, le=1000)

    @field_validator("log_level")
    @classmethod
    def _validate_log_level(cls, value: str) -> str:
        level = value.strip().upper()
        if level not in ("DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"):
            raise ValueError(
                f"log_level must be one of DEBUG, INFO, WARNING, ERROR, CRITICAL; got {value!r}"
            )
        return level

    # --- derived paths ---
    @property
    def data_path(self) -> Path:
        return ensure_dir(self.data_dir)

    @property
    def upload_path(self) -> Path:
        return ensure_dir(self.upload_dir)

    @property
    def artifact_path(self) -> Path:
        return ensure_dir(self.artifact_dir)

    @property
    def log_path(self) -> Path:
        return ensure_dir(Path(self.artifact_dir) / "logs") / "api.log"

    @property
    def max_upload_bytes(self) -> int:
        return int(self.max_upload_mb * 1024 * 1024)

    @property
    def cors_origin_list(self) -> list[str]:
        """CORS origins as a list.

        A single ``*`` allows everything, which is convenient in development
        but is rejected in production: with ``allow_credentials`` on, browsers
        refuse a wildcard anyway, so silently allowing it would be a trap.
        """
        origins = [origin.strip() for origin in self.cors_origins.split(",") if origin.strip()]
        if "*" in origins and self.environment == "production":
            logger.warning(
                "CORS is set to '*' but the environment is production; "
                "falling back to localhost origins."
            )
            return ["http://localhost:5173", "http://127.0.0.1:5173"]
        return origins or ["http://localhost:5173"]

    @property
    def resolved_database_url(self) -> str:
        """Database URL with any relative SQLite path resolved to absolute.

        Without this, the database file would land wherever the server happens
        to have been started from.
        """
        url = self.database_url
        prefix = "sqlite:///"
        if not url.startswith(prefix):
            return url

        raw = url[len(prefix) :]
        if raw == ":memory:" or raw.startswith(":memory:"):
            return url

        path = Path(raw)
        if not path.is_absolute():
            path = PROJECT_ROOT / path
        path.parent.mkdir(parents=True, exist_ok=True)
        return f"{prefix}{path.as_posix()}"

    @property
    def ml_config_path(self) -> Path:
        resolved = resolve_path(self.ml_config)
        if resolved is not None and resolved.is_file():
            return resolved
        return PROJECT_ROOT / "configs" / self.ml_config

    def summary(self) -> dict[str, Any]:
        """Non-secret settings, exposed by the health endpoint."""
        return {
            "app_name": self.app_name,
            "version": self.app_version,
            "environment": self.environment,
            "device_preference": self.device,
            "amp_allowed": self.allow_amp,
            "max_upload_mb": self.max_upload_mb,
            "ml_config": self.ml_config,
            "num_workers": self.num_workers,
        }


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return the singleton settings instance.

    Cached so that ``.env`` is read once and every dependency sees the same
    object.  Tests clear the cache with ``get_settings.cache_clear()``.
    """
    settings = Settings()
    logger.debug("Loaded settings for environment %r", settings.environment)
    return settings
