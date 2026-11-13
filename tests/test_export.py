"""Tests for text export and perspective correction.

The export rule being protected: a file must contain exactly the characters
it was given. An exporter that re-wraps, trims or "tidies" the text would
quietly turn a faithful transcription into an interpretation of one.
"""

from __future__ import annotations

import io

import cv2
import numpy as np
import pytest

from backend.services.export_service import (
    DOCX_AVAILABLE,
    build_docx,
    build_txt,
    export_metadata,
    safe_filename,
)
from ml.preprocessing import ops

SAMPLE = "Invoice No: INV-44718\nTax at 18% $1,250.00\n\nNew paragraph (after a blank line)."


# ---------------------------------------------------------------------------
# txt
# ---------------------------------------------------------------------------
def test_txt_round_trips_exactly() -> None:
    decoded = build_txt(SAMPLE).decode("utf-8-sig").replace("\r\n", "\n")
    assert decoded == SAMPLE


def test_txt_preserves_blank_lines() -> None:
    decoded = build_txt("a\n\n\nb").decode("utf-8-sig").replace("\r\n", "\n")
    assert decoded == "a\n\n\nb"


def test_txt_uses_crlf_for_windows_notepad() -> None:
    assert b"\r\n" in build_txt("one\ntwo")


def test_txt_has_a_bom() -> None:
    # Without it Windows tools guess the legacy code page.
    assert build_txt("x").startswith(b"\xef\xbb\xbf")


def test_txt_keeps_symbols_and_punctuation() -> None:
    text = 'Cost: $1,250.00 (50% off) <tag> {ref} | "quoted" ~ok'
    assert build_txt(text).decode("utf-8-sig").replace("\r\n", "\n") == text


# ---------------------------------------------------------------------------
# filenames
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("report 2026", "report-2026"),
        ("../../etc/passwd", "etc-passwd"),
        ("a:b*c?d", "a-b-c-d"),
        ("", "extracted-text"),
        ("   ", "extracted-text"),
    ],
)
def test_safe_filename(raw: str, expected: str) -> None:
    assert safe_filename(raw) == expected


# ---------------------------------------------------------------------------
# docx
# ---------------------------------------------------------------------------
@pytest.mark.skipif(not DOCX_AVAILABLE, reason="python-docx is not installed")
def test_docx_contains_every_line() -> None:
    from docx import Document

    document = Document(io.BytesIO(build_docx(SAMPLE)))
    rendered = [p.text for p in document.paragraphs]

    for line in SAMPLE.split("\n"):
        if line:
            assert line in rendered, f"{line!r} missing from the document"


@pytest.mark.skipif(not DOCX_AVAILABLE, reason="python-docx is not installed")
def test_docx_preserves_symbols() -> None:
    from docx import Document

    text = "Tax at 18% $1,250.00"
    document = Document(io.BytesIO(build_docx(text)))
    assert any(text == p.text for p in document.paragraphs)


@pytest.mark.skipif(not DOCX_AVAILABLE, reason="python-docx is not installed")
def test_docx_highlights_low_confidence_lines() -> None:
    from docx import Document

    payload = build_docx(
        "sure line\nunsure line",
        lines=[
            {"text": "sure line", "confidence": 0.98},
            {"text": "unsure line", "confidence": 0.30},
        ],
        low_confidence_threshold=0.6,
    )
    document = Document(io.BytesIO(payload))

    highlighted = {
        p.text for p in document.paragraphs if any(r.font.highlight_color for r in p.runs)
    }
    assert "unsure line" in highlighted
    assert "sure line" not in highlighted


@pytest.mark.skipif(not DOCX_AVAILABLE, reason="python-docx is not installed")
def test_docx_without_threshold_highlights_nothing() -> None:
    from docx import Document

    document = Document(
        io.BytesIO(
            build_docx(
                "a\nb",
                lines=[{"text": "a", "confidence": 0.1}],
                low_confidence_threshold=0.0,
            )
        )
    )
    assert not any(any(r.font.highlight_color for r in p.runs) for p in document.paragraphs)


def test_export_metadata_omits_missing_fields() -> None:
    payload = export_metadata(model_name="M", device=None, total_ms=None, confidence=0.9)
    assert payload["model"] == "M"
    assert "device" not in payload and "time" not in payload
    assert payload["mean confidence"] == "90.0%"


# ---------------------------------------------------------------------------
# perspective correction
# ---------------------------------------------------------------------------
def _keystoned_page() -> tuple[np.ndarray, np.ndarray]:
    """A rectangular page warped as if photographed at an angle."""
    page = np.full((420, 600), 250, np.uint8)
    for index in range(5):
        cv2.putText(
            page, f"Line number {index} of the page", (40, 80 + index * 70),
            cv2.FONT_HERSHEY_SIMPLEX, 0.9, 25, 2, cv2.LINE_AA,
        )
    cv2.rectangle(page, (8, 8), (591, 411), 0, 3)

    canvas = np.full((520, 760), 235, np.uint8)
    source = np.float32([[0, 0], [600, 0], [600, 420], [0, 420]])
    destination = np.float32([[120, 40], [690, 110], [640, 470], [70, 400]])
    matrix = cv2.getPerspectiveTransform(source, destination)
    warped = cv2.warpPerspective(page, matrix, (760, 520), borderValue=235)
    return warped, page


def test_perspective_correction_flattens_a_keystoned_page() -> None:
    warped, _ = _keystoned_page()
    corrected, applied = ops.perspective_correct(warped)

    assert applied, "the page outline should have been found"
    # A flattened page is close to the original aspect ratio (600x420 = 1.43).
    ratio = corrected.shape[1] / corrected.shape[0]
    assert 1.1 < ratio < 1.8


def test_perspective_correction_is_a_no_op_without_an_outline(text_image: np.ndarray) -> None:
    """A wrong warp is worse than none, so an unconvincing page is left alone."""
    corrected, applied = ops.perspective_correct(text_image)
    assert not applied
    assert np.array_equal(corrected, text_image)


def test_perspective_correction_does_not_mutate_its_input() -> None:
    warped, _ = _keystoned_page()
    before = warped.copy()
    ops.perspective_correct(warped)
    assert np.array_equal(warped, before)


def test_find_document_quad_returns_ordered_corners() -> None:
    warped, _ = _keystoned_page()
    quad = ops.find_document_quad(warped)

    assert quad is not None and quad.shape == (4, 2)
    top_left, top_right, bottom_right, bottom_left = quad
    assert top_left[0] < top_right[0]
    assert top_left[1] < bottom_left[1]
