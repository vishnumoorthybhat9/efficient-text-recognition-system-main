"""Tests for text-line segmentation.

Segmentation decides how the image is cut up before recognition, so an error
here shows up as mangled output rather than as a crash. The cases below cover
the failure that actually occurred during development: a single pre-cropped
line being split into word-sized fragments once binarisation shattered the
glyphs into separate components.
"""

from __future__ import annotations

import cv2
import numpy as np
import pytest

from ml.inference.segmentation import crop_region, detect_text_lines, draw_regions
from ml.preprocessing.pipeline import PreprocessOptions, PreprocessPipeline


def _page(lines: list[str], *, width: int = 760, spacing: int = 82) -> np.ndarray:
    image = np.full((spacing * len(lines) + 60, width), 243, dtype=np.uint8)
    for index, text in enumerate(lines):
        cv2.putText(
            image, text, (28, 60 + index * spacing), cv2.FONT_HERSHEY_SIMPLEX, 1.1, 22, 2, cv2.LINE_AA
        )
    return image


@pytest.mark.parametrize("count", [1, 2, 3, 5])
def test_line_count_is_detected(count: int) -> None:
    page = _page([f"Line number {i}" for i in range(count)])
    assert len(detect_text_lines(page)) == count


@pytest.mark.parametrize("method", ["none", "otsu", "adaptive", "sauvola"])
def test_single_line_stays_one_region_after_binarisation(text_image: np.ndarray, method: str) -> None:
    """A cropped line must not be split, whatever binarisation ran first.

    Sauvola in particular fragments glyphs into stroke pieces, which collapsed
    the median component height and made one line look like several.
    """
    processed = PreprocessPipeline(PreprocessOptions(binarize=method)).run(text_image).image
    result = detect_text_lines(processed)

    assert len(result) == 1
    assert result.regions[0].width >= processed.shape[1] * 0.9


@pytest.mark.parametrize("method", ["none", "sauvola"])
def test_multi_line_survives_binarisation(multiline_image: np.ndarray, method: str) -> None:
    processed = PreprocessPipeline(PreprocessOptions(binarize=method)).run(multiline_image).image
    assert len(detect_text_lines(processed)) == 4


def test_regions_are_ordered_top_to_bottom() -> None:
    result = detect_text_lines(_page(["one", "two", "three", "four"]))
    tops = [region.y for region in result.regions]
    assert tops == sorted(tops)
    assert [region.index for region in result.regions] == [0, 1, 2, 3]


def test_blank_image_falls_back_to_the_whole_image() -> None:
    result = detect_text_lines(np.full((120, 400), 240, dtype=np.uint8))
    assert len(result) == 1
    assert result.used_whole_image


def test_tiny_image_falls_back_to_the_whole_image() -> None:
    result = detect_text_lines(np.full((4, 4), 128, dtype=np.uint8))
    assert len(result) == 1 and result.used_whole_image


def test_words_on_one_line_are_not_split() -> None:
    # Wide inter-word gaps are exactly what used to defeat the dilation step.
    image = np.full((80, 900), 243, dtype=np.uint8)
    cv2.putText(image, "WIDELY    SPACED    WORDS", (20, 55), cv2.FONT_HERSHEY_SIMPLEX, 1.2, 22, 2, cv2.LINE_AA)
    assert len(detect_text_lines(image)) == 1


def test_max_lines_is_respected() -> None:
    page = _page([f"Line {i}" for i in range(8)])
    assert len(detect_text_lines(page, max_lines=3)) <= 3


def test_light_on_dark_input_is_handled() -> None:
    page = 255 - _page(["Inverted text here", "Second inverted line"])
    assert len(detect_text_lines(page)) == 2


def test_regions_stay_inside_the_image() -> None:
    page = _page(["Edge case", "Second line"])
    height, width = page.shape[:2]
    for region in detect_text_lines(page).regions:
        assert 0 <= region.x and 0 <= region.y
        assert region.x + region.width <= width
        assert region.y + region.height <= height


def test_crop_region_returns_the_right_patch() -> None:
    page = _page(["First line text", "Second line text"])
    result = detect_text_lines(page)
    crop = crop_region(page, result.regions[0])

    assert crop.shape[0] == result.regions[0].height
    assert crop.shape[1] == result.regions[0].width
    # The crop must contain ink, not just paper.
    assert crop.min() < 128


def test_crop_region_clamps_an_out_of_bounds_box() -> None:
    from ml.inference.segmentation import TextRegion

    page = np.full((50, 50), 255, dtype=np.uint8)
    crop = crop_region(page, TextRegion(-10, -10, 200, 200))
    assert crop.shape[0] <= 50 and crop.shape[1] <= 50 and crop.size > 0


def test_draw_regions_returns_a_colour_overlay() -> None:
    page = _page(["Draw me"])
    result = detect_text_lines(page)
    overlay = draw_regions(page, result.regions)

    assert overlay.ndim == 3 and overlay.shape[2] == 3
    assert overlay.shape[:2] == page.shape[:2]


def test_result_is_json_serialisable() -> None:
    import json

    json.dumps(detect_text_lines(_page(["a line", "another"])).to_dict())
