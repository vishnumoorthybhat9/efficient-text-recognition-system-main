"""Image reading and writing that works with Windows paths.

``cv2.imread`` and ``cv2.imwrite`` pass the filename to the OS as a byte
string, so they fail silently -- returning ``None`` -- on any path containing a
non-ASCII character.  A project directory like ``D:\\MAJORR PROJECTT`` is fine,
but a user folder with an accented name is not, and the failure mode (a
``None`` that surfaces three frames later) is genuinely hard to diagnose.
Every read and write in this project therefore goes through
``np.fromfile`` / ``tofile``, which use the Unicode path API.
"""

from __future__ import annotations

import logging
from pathlib import Path

import cv2
import numpy as np

__all__ = [
    "ImageDecodeError",
    "read_image",
    "write_image",
    "decode_image_bytes",
    "encode_image",
    "SUPPORTED_EXTENSIONS",
]

logger = logging.getLogger(__name__)

#: Extensions the application accepts for upload and for dataset ingestion.
SUPPORTED_EXTENSIONS: frozenset[str] = frozenset(
    {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp", ".ppm", ".pgm"}
)


class ImageDecodeError(Exception):
    """Raised when bytes or a file cannot be decoded as an image."""


def decode_image_bytes(data: bytes, *, grayscale: bool = False) -> np.ndarray:
    """Decode raw image *data* into a numpy array.

    Args:
        data: the encoded bytes (PNG, JPEG, ...).
        grayscale: decode directly to single channel.

    Returns:
        ``uint8`` array: 2-D when *grayscale*, else ``(H, W, 3)`` BGR.

    Raises:
        ImageDecodeError: the bytes are empty or not a supported image.
    """
    if not data:
        raise ImageDecodeError("Received 0 bytes; nothing to decode")

    buffer = np.frombuffer(data, dtype=np.uint8)
    flags = cv2.IMREAD_GRAYSCALE if grayscale else cv2.IMREAD_COLOR
    image = cv2.imdecode(buffer, flags)

    if image is None:
        raise ImageDecodeError(
            f"Could not decode {len(data)} byte(s) as an image. Supported formats: "
            + ", ".join(sorted(e.lstrip('.') for e in SUPPORTED_EXTENSIONS))
        )
    return image


def read_image(path: str | Path, *, grayscale: bool = True) -> np.ndarray:
    """Read an image from disk, tolerating non-ASCII paths.

    Raises:
        FileNotFoundError: *path* is not a file.
        ImageDecodeError: the file exists but is not a decodable image.
    """
    file = Path(path)
    if not file.is_file():
        raise FileNotFoundError(f"Image not found: {file}")

    try:
        data = np.fromfile(str(file), dtype=np.uint8)
    except OSError as exc:
        raise ImageDecodeError(f"Could not read {file}: {exc}") from exc

    if data.size == 0:
        raise ImageDecodeError(f"{file} is empty (0 bytes)")

    flags = cv2.IMREAD_GRAYSCALE if grayscale else cv2.IMREAD_COLOR
    image = cv2.imdecode(data, flags)
    if image is None:
        raise ImageDecodeError(f"{file} is not a decodable image (size {data.size} bytes)")
    return image


def encode_image(image: np.ndarray, *, extension: str = ".png", quality: int = 95) -> bytes:
    """Encode *image* to bytes.

    Args:
        extension: target format, e.g. ``".png"`` or ``".jpg"``.
        quality: JPEG/WebP quality, ignored for PNG.

    Raises:
        ValueError: the extension is not supported.
        RuntimeError: the encoder failed.
    """
    suffix = extension if extension.startswith(".") else f".{extension}"
    if suffix.lower() not in SUPPORTED_EXTENSIONS:
        raise ValueError(f"Unsupported image extension {extension!r}")

    params: list[int] = []
    if suffix.lower() in (".jpg", ".jpeg"):
        params = [int(cv2.IMWRITE_JPEG_QUALITY), int(quality)]
    elif suffix.lower() == ".webp":
        params = [int(cv2.IMWRITE_WEBP_QUALITY), int(quality)]

    success, buffer = cv2.imencode(suffix, image, params)
    if not success:
        raise RuntimeError(f"Failed to encode image as {suffix}")
    return buffer.tobytes()


def write_image(path: str | Path, image: np.ndarray, *, quality: int = 95) -> Path:
    """Write *image* to *path*, creating parent directories.

    Raises:
        RuntimeError: encoding or writing failed.
    """
    file = Path(path)
    file.parent.mkdir(parents=True, exist_ok=True)

    data = encode_image(image, extension=file.suffix or ".png", quality=quality)
    try:
        np.frombuffer(data, dtype=np.uint8).tofile(str(file))
    except OSError as exc:
        raise RuntimeError(f"Could not write {file}: {exc}") from exc
    return file
