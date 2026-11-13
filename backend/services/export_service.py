"""Export extracted text to downloadable files.

The guiding rule is *faithfulness*: an export must contain exactly the
characters that were recognised. Nothing is re-wrapped, auto-corrected or
tidied, because the point of the feature is a transcription of the image, not
a readable document derived from one.

Blank lines are preserved, so paragraph breaks survive the round trip.
"""

from __future__ import annotations

import io
import logging
import re
from datetime import datetime, timezone
from typing import Any, Sequence

__all__ = ["build_txt", "build_docx", "safe_filename", "DOCX_AVAILABLE"]

logger = logging.getLogger(__name__)

try:  # pragma: no cover - depends on the environment
    from docx import Document
    from docx.enum.text import WD_ALIGN_PARAGRAPH
    from docx.shared import Pt

    DOCX_AVAILABLE = True
except ImportError:  # pragma: no cover
    Document = None  # type: ignore[assignment]
    DOCX_AVAILABLE = False
    logger.debug("python-docx is not installed; DOCX export is unavailable")

_UNSAFE = re.compile(r"[^A-Za-z0-9._-]+")

#: XML 1.0 forbids most control characters. python-docx would raise on them,
#: and a recognised control character is a bug upstream anyway.
_CONTROL = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f]")


def safe_filename(name: str, *, fallback: str = "extracted-text", max_length: int = 60) -> str:
    """Reduce *name* to a filename safe on every platform."""
    stem = _UNSAFE.sub("-", str(name).strip()).strip("-._")
    stem = stem[:max_length].strip("-._")
    return stem or fallback


def build_txt(text: str) -> bytes:
    """Encode *text* as UTF-8 with a BOM.

    The BOM is deliberate: without it Notepad and Excel on Windows still guess
    the legacy code page and mangle anything outside ASCII.
    """
    # Normalise line endings to CRLF so Notepad shows the line breaks.
    normalised = str(text).replace("\r\n", "\n").replace("\r", "\n").replace("\n", "\r\n")
    return normalised.encode("utf-8-sig")


def build_docx(
    text: str,
    *,
    title: str = "Extracted text",
    metadata: dict[str, Any] | None = None,
    lines: Sequence[dict[str, Any]] | None = None,
    low_confidence_threshold: float = 0.0,
) -> bytes:
    """Render *text* into a .docx document.

    Args:
        text: the transcription, exactly as it should appear.
        title: document heading.
        metadata: model, device and timing, written as a small provenance
            block so a saved transcription says what produced it.
        lines: optional per-line records with ``text`` and ``confidence``.
            When given with a positive *low_confidence_threshold*, lines below
            it are highlighted rather than silently presented as certain.
        low_confidence_threshold: confidence below which a line is marked.

    Returns:
        The .docx file as bytes.

    Raises:
        RuntimeError: python-docx is not installed.
    """
    if not DOCX_AVAILABLE:
        raise RuntimeError(
            "DOCX export needs the python-docx package. Install it with:  pip install python-docx"
        )

    document = Document()

    # A monospaced body keeps columns and alignment recognisable.
    normal = document.styles["Normal"]
    normal.font.name = "Consolas"
    normal.font.size = Pt(10.5)

    heading = document.add_heading(title, level=1)
    heading.alignment = WD_ALIGN_PARAGRAPH.LEFT

    if metadata:
        parts = [f"{key}: {value}" for key, value in metadata.items() if value not in (None, "")]
        if parts:
            info = document.add_paragraph(" | ".join(parts))
            info.runs[0].font.size = Pt(8)
            info.runs[0].font.italic = True

    document.add_paragraph()

    marked = 0
    by_text: dict[str, float] = {}
    if lines and low_confidence_threshold > 0:
        for record in lines:
            value = record.get("text")
            if isinstance(value, str):
                by_text[value] = float(record.get("confidence") or 0.0)

    for raw_line in _CONTROL.sub("", str(text)).split("\n"):
        paragraph = document.add_paragraph()
        # Keep the transcription's own line breaks exactly.
        paragraph.paragraph_format.space_after = Pt(0)

        if not raw_line:
            continue

        run = paragraph.add_run(raw_line)
        confidence = by_text.get(raw_line)
        if confidence is not None and confidence < low_confidence_threshold:
            # Highlighting rather than omitting: the reader must be able to
            # see which text the model was unsure of.
            run.font.highlight_color = 7  # WD_COLOR_INDEX.YELLOW
            marked += 1

    if marked:
        document.add_paragraph()
        note = document.add_paragraph(
            f"{marked} line(s) highlighted: the model's confidence was below "
            f"{low_confidence_threshold:.0%}. Check them against the image."
        )
        note.runs[0].font.size = Pt(8)
        note.runs[0].font.italic = True

    buffer = io.BytesIO()
    document.save(buffer)
    return buffer.getvalue()


def export_metadata(
    *,
    model_name: str | None = None,
    device: str | None = None,
    total_ms: float | None = None,
    confidence: float | None = None,
) -> dict[str, Any]:
    """Build the provenance block written into an exported document."""
    payload: dict[str, Any] = {"exported": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")}
    if model_name:
        payload["model"] = model_name
    if device:
        payload["device"] = device
    if total_ms is not None:
        payload["time"] = f"{total_ms:.0f} ms"
    if confidence is not None:
        payload["mean confidence"] = f"{confidence:.1%}"
    return payload
