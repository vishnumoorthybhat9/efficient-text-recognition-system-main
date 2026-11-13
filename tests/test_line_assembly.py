"""Tests for multi-line document handling.

These cover two bugs that made the application lose text on real documents:

1. **Long lines were cropped.** Inference reused the 32x256 training canvas,
   so anything wider was cut off -- a full sentence lost more than half its
   characters, and the truncated result looked like a complete transcription.
2. **Table rows were emitted as separate lines.** Segmentation keeps a row's
   columns as separate crops (which recognise better), but nothing put the
   row back together afterwards.
"""

from __future__ import annotations

import cv2
import numpy as np
import pytest

from ml.inference.segmentation import TextRegion, detect_text_lines, group_into_rows
from ml.preprocessing import ops
from ml.preprocessing.pipeline import (
    MAX_INFERENCE_WIDTH,
    prepare_variable_width_batch,
)


# ---------------------------------------------------------------------------
# page builders
# ---------------------------------------------------------------------------
def _invoice() -> tuple[np.ndarray, int]:
    """A page with tabular rows: wide gaps between columns on one baseline."""
    lines = [
        "INVOICE",
        "Acme Corporation Limited",
        "42 Industrial Estate, Mangalore",
        "",
        "Description            Qty    Amount",
        "Laptop stand            2     51.25",
        "USB-C cable             3     18.00",
        "",
        "Subtotal                     111.75",
        "Total                        131.87",
    ]
    height, width = 40 * len(lines) + 80, 900
    page = np.full((height, width), 248, np.uint8)
    y = 60
    for text in lines:
        if text:
            cv2.putText(page, text, (50, y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, 30, 2, cv2.LINE_AA)
        y += 40
    return page, sum(1 for t in lines if t)


def _two_column() -> tuple[np.ndarray, int]:
    """A two-column page: boxes share baselines but belong to different columns."""
    left = [
        "Self supervised learning", "has emerged as a", "promising approach for",
        "learning useful", "representations from", "unlabeled data which",
        "reduces annotation", "effort considerably",
    ]
    right = [
        "The proposed system", "combines a CNN based", "encoder with a BiLSTM",
        "sequence model and a", "CTC decoder to produce", "accurate transcriptions",
        "from both printed and", "handwritten inputs",
    ]
    height, width = 40 * len(left) + 70, 980
    page = np.full((height, width), 248, np.uint8)
    y = 55
    for a, b in zip(left, right):
        cv2.putText(page, a, (40, y), cv2.FONT_HERSHEY_SIMPLEX, 0.55, 30, 1, cv2.LINE_AA)
        cv2.putText(page, b, (530, y), cv2.FONT_HERSHEY_SIMPLEX, 0.55, 30, 1, cv2.LINE_AA)
        y += 40
    return page, len(left) * 2


def _paragraph() -> tuple[np.ndarray, int]:
    lines = [
        "Text recognition from images is a fundamental problem in computer",
        "vision and pattern recognition playing a vital role in applications",
        "such as document digitization and automatic form processing.",
        "Traditional systems rely on handcrafted features and rule based",
        "techniques which often fail with noisy images or handwritten text.",
    ]
    height, width = 44 * len(lines) + 60, 980
    page = np.full((height, width), 248, np.uint8)
    y = 50
    for text in lines:
        cv2.putText(page, text, (40, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, 30, 1, cv2.LINE_AA)
        y += 44
    return page, len(lines)


def _wide_line(text: str) -> np.ndarray:
    """A tightly cropped single line, far wider than the training canvas."""
    (width, height), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.9, 2)
    image = np.full((height + 24, width + 20), 245, np.uint8)
    cv2.putText(image, text, (10, height + 8), cv2.FONT_HERSHEY_SIMPLEX, 0.9, 25, 2, cv2.LINE_AA)
    return image


# ---------------------------------------------------------------------------
# truncation is detected and avoided
# ---------------------------------------------------------------------------
def test_fixed_canvas_reports_truncation() -> None:
    """The fixed-width path must admit when it cut text off.

    It previously reported content_ratio 1.0 after cropping, so no caller
    could tell that half the line had been discarded.
    """
    image = _wide_line("The quick brown fox jumps over the lazy dog near the river bank")
    _, info = ops.resize_with_padding(image, height=32, width=256)

    assert info.truncated is True
    assert info.natural_width > 256
    assert info.lost_fraction > 0.3


def test_short_line_is_not_marked_truncated() -> None:
    image = _wide_line("short")
    _, info = ops.resize_with_padding(image, height=32, width=256)
    assert info.truncated is False
    assert info.lost_fraction == 0.0


def test_variable_width_batch_fits_the_whole_line() -> None:
    """The inference path must size its canvas to the content, not to 256."""
    image = _wide_line("The quick brown fox jumps over the lazy dog near the river bank")
    batch, ratios, infos = prepare_variable_width_batch([image], height=32)

    assert batch.shape[0] == 1 and batch.shape[2] == 32
    assert batch.shape[3] > 256, "canvas should have grown past the training width"
    assert infos[0].truncated is False
    assert 0 < ratios[0] <= 1


def test_variable_width_batch_shares_one_canvas() -> None:
    """Mixed widths must pad to a common canvas, with honest per-item ratios."""
    images = [_wide_line("short"), _wide_line("a considerably longer line of text here")]
    batch, ratios, _ = prepare_variable_width_batch(images, height=32)

    assert batch.shape[0] == 2
    # The short line occupies proportionally less of the shared canvas.
    assert ratios[0] < ratios[1]


def test_absurdly_wide_line_is_compressed_not_cropped() -> None:
    """Beyond the cap, squeeze rather than discard: distorted beats missing."""
    image = _wide_line("word " * 160)
    batch, _, infos = prepare_variable_width_batch([image], height=32)

    assert batch.shape[3] <= MAX_INFERENCE_WIDTH
    # Everything is still present, just horizontally compressed.
    assert infos[0].natural_width > MAX_INFERENCE_WIDTH


def test_variable_width_respects_the_minimum() -> None:
    batch, _, _ = prepare_variable_width_batch([_wide_line("hi")], height=32, min_width=256)
    assert batch.shape[3] >= 256


def test_empty_batch_is_rejected() -> None:
    with pytest.raises(ValueError, match="no images"):
        prepare_variable_width_batch([])


# ---------------------------------------------------------------------------
# segmentation across layouts
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    ("builder", "label"),
    [(_invoice, "invoice/table"), (_two_column, "two-column"), (_paragraph, "paragraph")],
)
def test_output_row_count_across_layouts(builder, label: str) -> None:
    """The number of *output lines* must match the page.

    `detect_text_lines` deliberately returns fragments -- a table row's
    columns stay separate crops, because the recogniser reads a tight crop far
    better than one spanning a wide blank gap. `group_into_rows` is what turns
    those fragments back into one line per visual row, so that is what gets
    asserted here.
    """
    page, expected = builder()
    result = detect_text_lines(page)
    rows = group_into_rows(result.regions, gutters=result.column_gutters)

    assert len(rows) == expected, (
        f"{label}: {len(result.regions)} region(s) grouped into {len(rows)} row(s), "
        f"expected {expected}"
    )


def test_table_columns_stay_separate_crops(builder=_invoice) -> None:
    """Fragments, not merged boxes: a wide blank gap wrecks recognition."""
    page, lines = builder()
    result = detect_text_lines(page)
    assert len(result.regions) > lines, "table columns should remain separate crops"


def test_two_column_page_detects_a_gutter() -> None:
    page, _ = _two_column()
    result = detect_text_lines(page)
    assert result.column_gutters, "the gap between the two columns should be found"


def test_table_page_has_no_false_gutter() -> None:
    """A table's column gaps must not be mistaken for a page gutter.

    They sit at different x positions on different rows and are crossed by the
    full-width header lines, unlike a real column separator.
    """
    page, _ = _invoice()
    assert detect_text_lines(page).column_gutters == []


# ---------------------------------------------------------------------------
# row assembly
# ---------------------------------------------------------------------------
def test_row_grouping_rejoins_a_table_row() -> None:
    regions = [
        TextRegion(40, 100, 120, 24, index=0),   # "Laptop stand"
        TextRegion(240, 100, 20, 24, index=1),   # "2"
        TextRegion(320, 100, 70, 24, index=2),   # "51.25"
        TextRegion(40, 150, 110, 24, index=3),   # next row
    ]
    rows = group_into_rows(regions)

    assert len(rows) == 2
    assert rows[0] == [0, 1, 2]
    assert rows[1] == [3]


def test_row_grouping_orders_left_to_right() -> None:
    regions = [
        TextRegion(320, 100, 70, 24, index=0),
        TextRegion(40, 100, 120, 24, index=1),
        TextRegion(240, 100, 20, 24, index=2),
    ]
    assert group_into_rows(regions)[0] == [1, 2, 0]


def test_row_grouping_orders_top_to_bottom() -> None:
    regions = [
        TextRegion(40, 300, 100, 24, index=0),
        TextRegion(40, 100, 100, 24, index=1),
        TextRegion(40, 200, 100, 24, index=2),
    ]
    rows = group_into_rows(regions)
    assert [row[0] for row in rows] == [1, 2, 0]


def test_row_grouping_never_joins_across_a_gutter() -> None:
    """Two columns sharing a baseline must stay on separate output lines."""
    regions = [
        TextRegion(40, 100, 200, 24, index=0),   # left column
        TextRegion(520, 100, 200, 24, index=1),  # right column
    ]
    rows = group_into_rows(regions, gutters=[(300, 500)])

    assert len(rows) == 2, "a column gutter must prevent joining"


def test_row_grouping_handles_an_empty_input() -> None:
    assert group_into_rows([]) == []


def test_every_region_appears_exactly_once() -> None:
    page, _ = _invoice()
    result = detect_text_lines(page)
    rows = group_into_rows(result.regions, gutters=result.column_gutters)

    flat = [index for row in rows for index in row]
    assert sorted(flat) == list(range(len(result.regions)))


# ---------------------------------------------------------------------------
# real-document failures found on user-supplied images
# ---------------------------------------------------------------------------
def _paragraph_page(*, border: bool = False, line_gap: int = 6, font_scale: float = 0.62):
    """A tight paragraph, optionally inside a printed frame."""
    lines = [
        "Optical character recognition or optical character",
        "reader (OCR) is the electronic or mechanical",
        "conversion of images of typed, handwritten or",
        "printed text into machine-encoded text, whether",
        "from a scanned document, a photo of a document,",
        "a scene-photo (for example the text on signs and",
        "billboards in a landscape photo) or from subtitle",
    ]
    text_height = 20
    step = text_height + line_gap
    width = 620
    page = np.full((step * len(lines) + 70, width), 250, np.uint8)

    y = 45
    for text in lines:
        cv2.putText(page, text, (24, y), cv2.FONT_HERSHEY_SIMPLEX, font_scale, 25, 2, cv2.LINE_AA)
        y += step

    if border:
        cv2.rectangle(page, (6, 6), (width - 7, page.shape[0] - 7), 0, 5)
    return page, len(lines)


def test_tight_paragraph_is_not_fused_into_one_line() -> None:
    """Normal leading must not be mistaken for a diacritic gap.

    The band merger used to join any gap below half the band height, which is
    exactly the leading of body text -- so a whole paragraph became one band
    and the page was handed to the recogniser as a single crop.
    """
    page, expected = _paragraph_page()
    result = detect_text_lines(page)
    rows = group_into_rows(result.regions, gutters=result.column_gutters)
    assert len(rows) == expected


def test_printed_page_border_does_not_fuse_the_page() -> None:
    """A frame puts ink in every row and used to destroy line detection."""
    page, expected = _paragraph_page(border=True)
    result = detect_text_lines(page)
    rows = group_into_rows(result.regions, gutters=result.column_gutters)

    assert not result.used_whole_image
    assert len(rows) == expected


def test_small_text_is_still_segmented() -> None:
    """Tiny glyphs must not fall below the pixel thresholds.

    An invoice with ~7 px text collapsed from ~30 lines to 3 before detection
    learned to upscale first.
    """
    lines = [f"Line number {i} of the small print" for i in range(8)]
    page = np.full((18 * len(lines) + 30, 420), 250, np.uint8)
    y = 18
    for text in lines:
        cv2.putText(page, text, (12, y), cv2.FONT_HERSHEY_SIMPLEX, 0.3, 25, 1, cv2.LINE_AA)
        y += 18

    result = detect_text_lines(page)
    rows = group_into_rows(result.regions, gutters=result.column_gutters)
    assert len(rows) >= len(lines) - 1


def test_drop_cap_does_not_swallow_its_lines() -> None:
    """An oversized initial spans two lines and used to hide both."""
    page = np.full((260, 620), 250, np.uint8)
    # A large initial spanning the first two text lines.
    cv2.putText(page, "M", (20, 95), cv2.FONT_HERSHEY_SIMPLEX, 3.0, 20, 5, cv2.LINE_AA)
    cv2.putText(page, "r Oliver, an Anglo-Indian teacher", (105, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.6, 25, 2, cv2.LINE_AA)
    cv2.putText(page, "returning to his school late at night", (105, 95), cv2.FONT_HERSHEY_SIMPLEX, 0.6, 25, 2, cv2.LINE_AA)
    cv2.putText(page, "on the outskirts of the hill station", (20, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.6, 25, 2, cv2.LINE_AA)
    cv2.putText(page, "of Simla, before Kipling's time.", (20, 195), cv2.FONT_HERSHEY_SIMPLEX, 0.6, 25, 2, cv2.LINE_AA)

    result = detect_text_lines(page)
    rows = group_into_rows(result.regions, gutters=result.column_gutters)
    # Four text lines; without the split the first two collapse into one.
    assert len(rows) >= 4
