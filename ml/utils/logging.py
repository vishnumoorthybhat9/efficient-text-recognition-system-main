"""Logging configuration shared by the API server, the training scripts and
the test suite.

Call :func:`setup_logging` exactly once per process.  It is idempotent, so a
second call with the same arguments is harmless.
"""

from __future__ import annotations

import json
import logging
import logging.handlers
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from ml.utils.paths import ensure_dir

__all__ = ["setup_logging", "get_logger"]

_CONFIGURED = False

_PLAIN_FORMAT = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"

# Third-party loggers that are far too chatty at DEBUG/INFO level.
_NOISY = {
    "PIL": logging.WARNING,
    "matplotlib": logging.WARNING,
    "matplotlib.font_manager": logging.ERROR,
    "urllib3": logging.WARNING,
    "filelock": logging.WARNING,
    "asyncio": logging.WARNING,
    "watchfiles": logging.WARNING,
    "multipart": logging.WARNING,
}


class JsonFormatter(logging.Formatter):
    """Emit one JSON object per record (handy when shipping logs elsewhere)."""

    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, Any] = {
            "ts": datetime.fromtimestamp(record.created, tz=timezone.utc).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        if record.exc_info:
            payload["exception"] = self.formatException(record.exc_info)
        for key, value in getattr(record, "extra_fields", {}).items():
            payload[key] = value
        return json.dumps(payload, ensure_ascii=False)


class _SafeStreamHandler(logging.StreamHandler):
    """StreamHandler that survives a console which cannot encode a character.

    Windows consoles default to cp1252; a log line containing a decoded OCR
    prediction can easily carry something outside that code page.  Dropping the
    character is far better than crashing the request that produced it.
    """

    def emit(self, record: logging.LogRecord) -> None:
        try:
            super().emit(record)
        except UnicodeEncodeError:
            try:
                stream = self.stream
                encoding = getattr(stream, "encoding", None) or "ascii"
                message = self.format(record).encode(encoding, errors="replace").decode(encoding)
                stream.write(message + self.terminator)
                self.flush()
            except Exception:  # pragma: no cover - last resort
                self.handleError(record)


def setup_logging(
    level: str | int = "INFO",
    *,
    log_file: str | Path | None = None,
    json_format: bool = False,
    force: bool = False,
) -> None:
    """Configure the root logger.

    Args:
        level: textual or numeric log level.
        log_file: optional path for a rotating file handler (5 MB x 3).
        json_format: emit JSON instead of human-readable lines.
        force: reconfigure even if this function already ran.
    """
    global _CONFIGURED
    if _CONFIGURED and not force:
        return

    if isinstance(level, str):
        level = getattr(logging, level.strip().upper(), logging.INFO)

    root = logging.getLogger()
    root.setLevel(level)
    for handler in list(root.handlers):
        root.removeHandler(handler)
        handler.close()

    formatter: logging.Formatter = (
        JsonFormatter() if json_format else logging.Formatter(_PLAIN_FORMAT, _DATE_FORMAT)
    )

    console = _SafeStreamHandler(stream=sys.stdout)
    console.setFormatter(formatter)
    console.setLevel(level)
    root.addHandler(console)

    if log_file is not None:
        path = Path(log_file)
        ensure_dir(path.parent)
        file_handler = logging.handlers.RotatingFileHandler(
            path, maxBytes=5 * 1024 * 1024, backupCount=3, encoding="utf-8"
        )
        file_handler.setFormatter(formatter)
        file_handler.setLevel(level)
        root.addHandler(file_handler)

    for name, noisy_level in _NOISY.items():
        logging.getLogger(name).setLevel(noisy_level)

    _CONFIGURED = True
    logging.getLogger(__name__).debug("Logging configured at level %s", logging.getLevelName(level))


def get_logger(name: str) -> logging.Logger:
    """Return a logger, configuring logging with defaults if nobody has yet."""
    if not _CONFIGURED:
        setup_logging()
    return logging.getLogger(name)
