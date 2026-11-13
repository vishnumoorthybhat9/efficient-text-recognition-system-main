"""Extract correctly oriented scanned pages from a handwritten PDF.

Usage:
    python scripts/prepare_handwritten_pdf.py "C:\\data\\notes.pdf"
    python scripts/prepare_handwritten_pdf.py "C:\\data\\notes.pdf" --force
"""

from __future__ import annotations

import argparse
import io
import json
import logging
import re
import sys
import tempfile
from pathlib import Path
from typing import Sequence

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pymupdf  # noqa: E402
from PIL import Image  # noqa: E402

from ml.utils.paths import PROJECT_ROOT, relative_to_root  # noqa: E402

logger = logging.getLogger("prepare_handwritten_pdf")
_PAGE_IMAGE_NAME = re.compile(r"page_\d{4}\.jpg$")


def _rotation_from_transform(matrix: pymupdf.Matrix) -> int:
    """Return the clockwise quarter-turn needed to orient an embedded scan."""
    a, b, c, d = (float(value) for value in matrix[:4])
    if abs(b) > abs(a):
        return 90 if b > 0 else 270
    if a < 0 and d < 0:
        return 180
    return 0


def _orient_image(image_bytes: bytes, rotation: int) -> Image.Image:
    """Decode and orient a PDF scan without resampling its pixels."""
    with Image.open(io.BytesIO(image_bytes)) as source:
        image = source.convert("RGB")
    transpose = {
        0: None,
        90: Image.Transpose.ROTATE_270,
        180: Image.Transpose.ROTATE_180,
        270: Image.Transpose.ROTATE_90,
    }[rotation]
    return image.transpose(transpose) if transpose is not None else image


def extract_pdf(pdf_path: Path, output_dir: Path, *, force: bool = False) -> list[Path]:
    """Extract one display-oriented image per PDF page."""
    if not pdf_path.is_file():
        raise FileNotFoundError(f"PDF not found: {pdf_path}")

    doc = pymupdf.open(pdf_path)
    if not len(doc):
        raise ValueError(f"PDF has no pages: {pdf_path}")

    existing = (
        [path for path in output_dir.glob("page_*.jpg") if _PAGE_IMAGE_NAME.fullmatch(path.name)]
        if output_dir.exists()
        else []
    )
    if existing and not force:
        raise FileExistsError(
            f"{output_dir} already contains page images. Use --force to replace generated page_*.jpg files."
        )

    output_dir.parent.mkdir(parents=True, exist_ok=True)
    staged: list[Path] = []
    with tempfile.TemporaryDirectory(prefix="handwritten_pdf_", dir=output_dir.parent) as temp:
        stage_dir = Path(temp)
        for page_index, page in enumerate(doc, start=1):
            image_refs = page.get_images(full=True)
            if len(image_refs) != 1:
                raise ValueError(
                    f"Expected one scanned image on PDF page {page_index}, found {len(image_refs)}"
                )
            xref = int(image_refs[0][0])
            rects = page.get_image_rects(xref, transform=True)
            if len(rects) != 1:
                raise ValueError(
                    f"Expected one placement for the scanned image on PDF page {page_index}, "
                    f"found {len(rects)}"
                )
            image_data = doc.extract_image(xref)
            rotation = _rotation_from_transform(rects[0][1])
            image = _orient_image(image_data["image"], rotation)
            target = stage_dir / f"page_{page_index:04d}.jpg"
            image.save(target, format="JPEG", quality=95, optimize=True)
            staged.append(target)

        for source in staged:
            target = output_dir / source.name
            target.parent.mkdir(parents=True, exist_ok=True)
            if target.exists() and not force:
                raise FileExistsError(f"Refusing to overwrite {target}; use --force.")
        if force:
            for old_page in existing:
                old_page.unlink()
        written: list[Path] = []
        for source in staged:
            target = output_dir / source.name
            source.replace(target)
            written.append(target)

    image_metadata = []
    for page_index, path in enumerate(written):
        page = doc[page_index]
        xref = int(page.get_images(full=True)[0][0])
        transform = page.get_image_rects(xref, transform=True)[0][1]
        with Image.open(path) as image:
            width, height = image.size
        image_metadata.append(
            {
                "file": path.name,
                "width": width,
                "height": height,
                "rotation_applied": _rotation_from_transform(transform),
            }
        )
    metadata = {
        "source_pdf": str(pdf_path.resolve()),
        "page_count": len(written),
        "images": image_metadata,
        "labels_available": False,
        "notes": "Images are full-page scans. Transcriptions are required for supervised training and accuracy evaluation.",
    }
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    (output_dir.parent / "dataset_info.json").write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    logger.info(
        "Extracted %d upright page image(s) from %s into %s",
        len(written),
        pdf_path.name,
        relative_to_root(output_dir),
    )
    return written


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("pdf", type=Path, help="source PDF containing scanned handwritten pages")
    parser.add_argument(
        "--out",
        type=Path,
        default=PROJECT_ROOT / "data" / "handwritten" / "images",
        help="page image output directory",
    )
    parser.add_argument("--force", action="store_true", help="replace existing page_*.jpg files")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    try:
        extract_pdf(args.pdf, args.out, force=args.force)
    except (FileNotFoundError, FileExistsError, ValueError, OSError) as exc:
        logger.error("%s", exc)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
