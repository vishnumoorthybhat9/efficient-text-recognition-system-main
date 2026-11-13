"""Upload validation and storage.

Uploads are the one place where untrusted input enters the application, so
this module is deliberately strict:

* the size limit is enforced **while streaming**, not after the whole body is
  in memory -- otherwise the limit is decorative;
* the stored filename is generated, never derived from the client's, which
  removes path traversal and collisions in one step;
* the file is decoded as an image before it is accepted, so a ``.png`` that is
  not a PNG is rejected at the boundary rather than deep inside OpenCV.
"""

from __future__ import annotations

import logging
import uuid
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, BinaryIO

import numpy as np
from fastapi import UploadFile

from backend.core.config import Settings
from backend.core.errors import InvalidImage, PayloadTooLarge
from ml.utils.image_io import SUPPORTED_EXTENSIONS, ImageDecodeError, decode_image_bytes, write_image

__all__ = ["StoredUpload", "read_upload", "store_image", "prune_uploads", "resolve_stored_path"]

logger = logging.getLogger(__name__)

#: Read the upload in chunks of this size while checking the limit.
_CHUNK_BYTES = 1024 * 1024

#: Largest image this application will process. Beyond this, segmentation and
#: preprocessing allocate enough memory to threaten a 16 GB machine, and the
#: result would be unusable anyway.
MAX_PIXELS = 50_000_000


class StoredUpload:
    """An accepted upload: its decoded pixels and where it was stored."""

    __slots__ = ("image", "original_filename", "stored_filename", "content_type", "size_bytes")

    def __init__(
        self,
        image: np.ndarray,
        *,
        original_filename: str,
        stored_filename: str | None,
        content_type: str,
        size_bytes: int,
    ) -> None:
        self.image = image
        self.original_filename = original_filename
        self.stored_filename = stored_filename
        self.content_type = content_type
        self.size_bytes = size_bytes

    @property
    def width(self) -> int:
        return int(self.image.shape[1])

    @property
    def height(self) -> int:
        return int(self.image.shape[0])

    def to_dict(self) -> dict[str, Any]:
        return {
            "original_filename": self.original_filename,
            "stored_filename": self.stored_filename,
            "content_type": self.content_type,
            "size_bytes": self.size_bytes,
            "width": self.width,
            "height": self.height,
        }


def _read_limited(stream: BinaryIO, limit_bytes: int, *, filename: str) -> bytes:
    """Read *stream* up to *limit_bytes*, raising if it is exceeded.

    Raises:
        PayloadTooLarge: the stream is bigger than the limit.
    """
    chunks: list[bytes] = []
    total = 0

    while True:
        chunk = stream.read(_CHUNK_BYTES)
        if not chunk:
            break
        total += len(chunk)
        if total > limit_bytes:
            raise PayloadTooLarge(
                f"{filename!r} is larger than the {limit_bytes / (1024 * 1024):.0f} MB upload "
                "limit. Resize the image, or raise OCR_MAX_UPLOAD_MB.",
                detail={"limit_bytes": limit_bytes, "filename": filename},
            )
        chunks.append(chunk)

    return b"".join(chunks)


async def read_upload(
    upload: UploadFile,
    settings: Settings,
    *,
    store: bool = True,
) -> StoredUpload:
    """Validate, decode and optionally persist an uploaded image.

    Args:
        upload: the multipart file.
        settings: supplies the size limit and the upload directory.
        store: write the bytes to ``data/uploads``.  Turned off when the
            caller is not saving the request to history.

    Returns:
        A :class:`StoredUpload`.

    Raises:
        InvalidImage: no file, unsupported extension, or undecodable content.
        PayloadTooLarge: the file exceeds the configured limit.
    """
    original = (upload.filename or "upload").strip() or "upload"
    suffix = Path(original).suffix.lower()

    if suffix and suffix not in SUPPORTED_EXTENSIONS:
        raise InvalidImage(
            f"{original!r} has an unsupported extension. Accepted formats: "
            + ", ".join(sorted(e.lstrip(".") for e in SUPPORTED_EXTENSIONS)),
            detail={"filename": original, "extension": suffix},
        )

    # `upload.file` is a SpooledTemporaryFile; reading it in chunks keeps the
    # size check meaningful for a large body.
    await upload.seek(0)
    data = _read_limited(upload.file, settings.max_upload_bytes, filename=original)

    if not data:
        raise InvalidImage(
            f"{original!r} is empty (0 bytes).", detail={"filename": original}
        )

    try:
        image = decode_image_bytes(data, grayscale=False)
    except ImageDecodeError as exc:
        raise InvalidImage(
            f"{original!r} could not be read as an image. It may be corrupt, or its "
            f"extension may not match its contents. ({exc})",
            detail={"filename": original, "size_bytes": len(data)},
        ) from exc

    pixels = int(image.shape[0]) * int(image.shape[1])
    if pixels > MAX_PIXELS:
        raise PayloadTooLarge(
            f"{original!r} is {image.shape[1]}x{image.shape[0]} ({pixels / 1e6:.0f} megapixels), "
            f"above the {MAX_PIXELS / 1e6:.0f} MP processing limit. Downscale it first.",
            detail={"width": int(image.shape[1]), "height": int(image.shape[0])},
        )

    stored_filename: str | None = None
    if store:
        stored_filename = _write(data, suffix or ".png", settings)

    logger.info(
        "Accepted upload %r (%dx%d, %.1f KB)%s",
        original,
        image.shape[1],
        image.shape[0],
        len(data) / 1024,
        f" stored as {stored_filename}" if stored_filename else " (not stored)",
    )

    return StoredUpload(
        image=image,
        original_filename=original,
        stored_filename=stored_filename,
        content_type=upload.content_type or "",
        size_bytes=len(data),
    )


def _write(data: bytes, suffix: str, settings: Settings) -> str:
    """Write *data* under a generated name; return that name.

    The name is ``<utc-date>_<uuid4-hex-12><suffix>`` -- sortable by day, and
    with no component taken from the client.
    """
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d")
    name = f"{stamp}_{uuid.uuid4().hex[:12]}{suffix}"
    path = settings.upload_path / name
    path.write_bytes(data)
    return name


def store_image(image: np.ndarray, settings: Settings, *, suffix: str = ".png") -> str:
    """Store a numpy image (e.g. a generated demo) and return its filename."""
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d")
    name = f"{stamp}_{uuid.uuid4().hex[:12]}{suffix}"
    write_image(settings.upload_path / name, image)
    return name


def resolve_stored_path(stored_filename: str | None, settings: Settings) -> Path | None:
    """Resolve a stored filename to a path inside the upload directory.

    Returns ``None`` when the name is empty, escapes the upload directory, or
    does not exist.  The containment check matters because the filename
    reaches this function from the database, and a row could in principle hold
    anything.
    """
    if not stored_filename:
        return None

    candidate = (settings.upload_path / stored_filename).resolve()
    try:
        candidate.relative_to(settings.upload_path.resolve())
    except ValueError:
        logger.warning("Refusing stored path outside the upload directory: %r", stored_filename)
        return None

    return candidate if candidate.is_file() else None


def prune_uploads(settings: Settings, *, older_than_days: int | None = None) -> dict[str, int]:
    """Delete stored uploads older than the retention window.

    Returns:
        ``{"deleted": n, "freed_bytes": n, "kept": n}``.
    """
    days = older_than_days if older_than_days is not None else settings.history_retention_days
    cutoff = datetime.now(timezone.utc) - timedelta(days=days)

    deleted = freed = kept = 0
    for path in settings.upload_path.iterdir():
        if not path.is_file():
            continue
        try:
            modified = datetime.fromtimestamp(path.stat().st_mtime, tz=timezone.utc)
            if modified < cutoff:
                size = path.stat().st_size
                path.unlink()
                deleted += 1
                freed += size
            else:
                kept += 1
        except OSError as exc:
            logger.warning("Could not prune %s: %s", path.name, exc)
            kept += 1

    logger.info("Pruned %d upload(s), freeing %.1f MB; %d kept", deleted, freed / 1e6, kept)
    return {"deleted": deleted, "freed_bytes": freed, "kept": kept}
