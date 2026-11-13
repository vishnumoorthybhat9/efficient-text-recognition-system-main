"""Tests for display-oriented page extraction from handwritten PDFs."""

from __future__ import annotations

import io
import json
from pathlib import Path

import pymupdf
import numpy as np
from PIL import Image

from scripts.prepare_handwritten_pdf import _orient_image, _rotation_from_transform
from scripts.transcribe_handwriting import _write_page
from ml.inference.segmentation import TextRegion, crop_region


def test_pdf_image_transform_maps_to_clockwise_rotation() -> None:
    transform = pymupdf.Matrix(0, 792, -498, 0, 498, 0)
    assert _rotation_from_transform(transform) == 90


def test_pdf_image_transform_maps_to_half_turn() -> None:
    transform = pymupdf.Matrix(-612, 0, 0, -177, 612, 177)
    assert _rotation_from_transform(transform) == 180


def test_orient_image_rotates_without_resizing() -> None:
    source = Image.new("RGB", (4, 2))
    source.putpixel((0, 0), (255, 0, 0))
    stream = io.BytesIO()
    source.save(stream, format="PNG")

    image = _orient_image(stream.getvalue(), 90)

    assert image.size == (2, 4)
    assert image.getpixel((1, 0)) == (255, 0, 0)


def test_write_page_saves_matching_text_json_and_line_crop(tmp_path: Path) -> None:
    page = np.full((32, 48), 255, dtype=np.uint8)
    region = TextRegion(x=4, y=5, width=30, height=16, index=1)
    crop = crop_region(page, region)

    record = _write_page(
        source=Path("page_0001.jpg"),
        output_dir=tmp_path,
        model_id="trocr_base",
        regions=[region],
        crops=[crop],
        predictions=[("Capital budgeting", 0.91)],
        detector="doctr",
        elapsed_seconds=1.5,
    )

    assert (tmp_path / "page_0001.txt").read_text(encoding="utf-8") == "Capital budgeting\n"
    saved = json.loads((tmp_path / "page_0001.json").read_text(encoding="utf-8"))
    assert saved["text"] == record["text"] == "Capital budgeting"
    assert saved["status"] == "ocr_draft_unverified"
    assert (tmp_path / saved["lines"][0]["crop"]).is_file()
