"""Tests for handwritten-document support (ml/inference/handwriting.py).

The page builder below draws what made the original pipeline fail on a real
notebook photo: full-width ruled lines touching the writing, a margin line,
a dark desk edge and a lighting gradient. Before handwriting mode existed,
that page came back as a single "line" -- the whole image.
"""

from __future__ import annotations

import cv2
import numpy as np
import pytest

from ml.inference import handwriting
from ml.inference.handwriting import (
    clean_handwritten_page,
    detect_handwritten_lines,
    group_words_into_lines,
    remove_ruled_lines,
    split_tall_regions,
    whiten_dark_borders,
)
from ml.inference.segmentation import TextRegion, detect_text_lines

LINES = [
    "Capital budgeting is a long",
    "term investment decision",
    "Payback period method",
    "Initial investment",
    "Annual cash inflows",
]
SPACING = 70


def _notebook_page(lines=LINES, *, shadow: bool = True, desk_edge: bool = True) -> np.ndarray:
    width, height = 1000, 120 + SPACING * len(lines)
    page = np.full((height, width), 238, np.uint8)

    # ruled lines every SPACING px, the writing sits on them
    for y in range(60, height, SPACING):
        cv2.line(page, (0, y), (width, y), 150, 2)
    # margin line
    cv2.line(page, (90, 0), (90, height), 120, 2)

    for i, text in enumerate(lines):
        baseline = 60 + SPACING * (i + 1) - 4
        cv2.putText(page, text, (110, baseline), cv2.FONT_HERSHEY_SCRIPT_SIMPLEX, 1.6, 30, 3, cv2.LINE_AA)

    if shadow:
        gradient = np.linspace(1.0, 0.55, width, dtype=np.float32)[None, :]
        page = (page.astype(np.float32) * gradient).astype(np.uint8)
    if desk_edge:
        page[:, -50:] = 0
    return page


def test_original_detector_fails_on_ruled_page() -> None:
    """Documents the problem this module solves."""
    result = detect_text_lines(_notebook_page())
    assert len(result.regions) < len(LINES)


def test_ruled_lines_are_removed() -> None:
    page = _notebook_page(shadow=False, desk_edge=False)
    _, horizontal, vertical = remove_ruled_lines(page)
    assert horizontal >= len(LINES)
    assert vertical >= 1


def test_clean_page_reports_what_it_did() -> None:
    cleaned = clean_handwritten_page(_notebook_page())
    info = cleaned.metadata()
    assert info["background_flattened"] is True
    assert info["horizontal_rules_removed"] >= len(LINES)
    assert cleaned.image.dtype == np.uint8
    assert cleaned.image.shape == _notebook_page().shape


def test_desk_edge_is_whitened() -> None:
    page = _notebook_page(shadow=False)
    out = whiten_dark_borders(page)
    assert out[:, -40:].mean() > 250
    # the writing survives
    assert (out[:, 100:600] < 100).sum() > 0


@pytest.mark.parametrize("shadow", [False, True])
def test_every_line_is_found(shadow: bool) -> None:
    cleaned = clean_handwritten_page(_notebook_page(shadow=shadow))
    result, used = detect_handwritten_lines(cleaned.image, detector="projection")
    assert used == "projection"
    assert len(result.regions) == len(LINES)
    # top to bottom, one per ruled row
    centres = [r.centre_y for r in result.regions]
    assert centres == sorted(centres)


def test_merged_lines_are_split() -> None:
    page = np.full((460, 600), 255, np.uint8)
    for i in range(5):
        cv2.putText(page, "handwriting", (20, 70 + i * 80), cv2.FONT_HERSHEY_SCRIPT_SIMPLEX, 1.5, 0, 3)
    regions = [
        TextRegion(10, 20 + i * 80, 500, 72, index=i) for i in (0, 1)
    ] + [
        TextRegion(10, 180, 500, 152, index=2),  # lines 3 and 4 in one box
        TextRegion(10, 340, 500, 72, index=3),
    ]
    out = split_tall_regions(page, regions)
    assert len(out) == 5
    assert [r.index for r in out] == [0, 1, 2, 3, 4]
    # the cut falls in the gap between the two lines, not through a word
    assert 230 <= out[2].y + out[2].height <= 285


def test_words_chain_into_drifting_lines() -> None:
    # line 1 drifts upward by 4 px per word; line 2 is level
    boxes = [(10 + 110 * i, 100 - 4 * i, 100, 40) for i in range(6)]
    boxes += [(10 + 110 * i, 170, 100, 40) for i in range(4)]
    lines = group_words_into_lines(boxes)
    assert len(lines) == 2
    assert sorted(lines[0]) == list(range(6))
    assert sorted(lines[1]) == list(range(6, 10))


def test_far_apart_words_start_new_line() -> None:
    boxes = [(10, 100, 80, 30), (100, 100, 80, 30), (900, 100, 80, 30)]
    assert len(group_words_into_lines(boxes)) == 2


def test_doctr_failure_falls_back(monkeypatch: pytest.MonkeyPatch) -> None:
    def broken(*args, **kwargs):
        raise RuntimeError("weights not downloadable")

    monkeypatch.setattr(handwriting, "_doctr_word_boxes", broken)
    cleaned = clean_handwritten_page(_notebook_page())
    result, used = detect_handwritten_lines(cleaned.image, detector="doctr")
    assert used == "projection (docTR unavailable)"
    assert len(result.regions) == len(LINES)


def test_auto_uses_projection_without_doctr(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(handwriting, "doctr_available", lambda: False)
    _, used = detect_handwritten_lines(_notebook_page(), detector="auto")
    assert used == "projection"


def test_doctr_boxes_are_grouped(monkeypatch: pytest.MonkeyPatch) -> None:
    """The docTR path, with the network replaced by fixed word boxes."""
    words = [(110 + 120 * i, 80 + 70 * row, 100, 40) for row in range(3) for i in range(4)]
    monkeypatch.setattr(handwriting, "_doctr_word_boxes", lambda gray: words)
    result, used = detect_handwritten_lines(np.full((400, 700), 255, np.uint8), detector="doctr")
    assert used == "doctr"
    assert len(result.regions) == 3


def test_unknown_detector_is_rejected() -> None:
    with pytest.raises(ValueError):
        detect_handwritten_lines(_notebook_page(), detector="tesseract")
