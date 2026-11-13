"""Render a small document benchmark with exact ground truth.

Measures the *whole* pipeline -- preprocessing, line detection, recognition
and row assembly -- against the exact text of a page, not just against
pre-cropped line images. That distinction matters: a model can score well on
cropped lines and still produce unusable output on a real page because lines
were missed, split or merged.

Pages are rendered rather than photographed so the ground truth is exact.
They deliberately include what real documents contain and the earlier
synthetic corpus did not: currency and percent signs, long sentences, mixed
case, and tabular rows.

Usage
-----
::

    python scripts/make_document_benchmark.py
    python scripts/make_document_benchmark.py --out data/processed/docbench --count 12
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Sequence

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import cv2  # noqa: E402
import numpy as np  # noqa: E402

from ml.datasets.synthetic import discover_fonts  # noqa: E402
from ml.utils.image_io import write_image  # noqa: E402
from ml.utils.logging import setup_logging  # noqa: E402
from ml.utils.paths import PROJECT_ROOT, ensure_dir, relative_to_root  # noqa: E402

logger = logging.getLogger("make_document_benchmark")

#: Each entry is one page: a list of text lines. Blank strings are spacing.
PAGES: list[list[str]] = [
    [
        "INVOICE",
        "Acme Corporation Limited",
        "42 Industrial Estate, Mangalore 575001",
        "",
        "Invoice No: INV-44718",
        "Date: 22 March 2026",
        "",
        "Laptop stand 51.25",
        "USB-C cable 18.00",
        "Wireless keyboard 42.50",
        "",
        "Subtotal 111.75",
        "Tax at 18% 20.12",
        "Total 131.87",
    ],
    [
        "Department of Information Science and Engineering",
        "Major Project Synopsis",
        "",
        "Efficient Text Recognition from Images Using Deep",
        "Learning and Self Supervised Learning",
        "",
        "Project Code: 42",
        "Guide: Dr Rashmi Naveen",
        "Submitted on 30 October 2026",
    ],
    [
        "Text recognition from images is a fundamental problem",
        "in computer vision and pattern recognition, playing a",
        "vital role in applications such as document",
        "digitization and automatic form processing.",
        "",
        "Traditional OCR systems rely on handcrafted features",
        "and rule based techniques, which often fail when",
        "dealing with noisy images or handwritten text.",
    ],
    [
        "MEETING NOTES",
        "Date: 14 Feb 2026 at 10:30",
        "Present: Tejaswi, Simran, Pranava",
        "",
        "1. Reviewed the baseline results.",
        "2. Agreed to add self supervised pretraining.",
        "3. Next review on 28 Feb 2026.",
        "",
        "Action: prepare the comparison table.",
    ],
    [
        "Quarterly Summary 2026",
        "",
        "Revenue grew by 18% to 1,250,000 rupees",
        "Operating cost fell by 7% to 430,500 rupees",
        "Headcount increased from 42 to 55 people",
        "",
        "Customer retention held at 94%",
        "Average response time was 2.4 hours",
    ],
    [
        "Certificate of Completion",
        "",
        "This is to certify that",
        "Pranava Kumara K",
        "has successfully completed the course",
        "Deep Learning for Computer Vision",
        "",
        "Issued on 30 October 2026",
    ],
]


def _render_page(
    lines: Sequence[str],
    *,
    font_scale: float,
    thickness: int,
    width: int = 1100,
    line_height: int = 52,
    margin: int = 60,
) -> np.ndarray:
    height = line_height * len(lines) + margin * 2
    page = np.full((height, width), 250, np.uint8)

    y = margin + line_height // 2
    for text in lines:
        if text:
            cv2.putText(
                page, text, (margin, y), cv2.FONT_HERSHEY_SIMPLEX, font_scale, 28, thickness, cv2.LINE_AA
            )
        y += line_height
    return page


def _degrade(page: np.ndarray, rng: np.random.Generator, level: str) -> np.ndarray:
    """Apply a realistic capture degradation."""
    if level == "clean":
        return page

    out = page
    height, width = out.shape[:2]

    if level in ("scan", "photo"):
        angle = float(rng.uniform(-2.5, 2.5)) if level == "scan" else float(rng.uniform(-5.0, 5.0))
        matrix = cv2.getRotationMatrix2D((width / 2, height / 2), angle, 1.0)
        out = cv2.warpAffine(out, matrix, (width, height), borderValue=250)

    if level == "photo":
        # Uneven illumination across the page, as from a phone camera.
        gradient = np.linspace(-55, 30, width, dtype=np.float32)[None, :]
        out = np.clip(out.astype(np.float32) + gradient, 0, 255).astype(np.uint8)
        out = cv2.GaussianBlur(out, (0, 0), 0.7)

    noise_sigma = 3.0 if level == "scan" else 6.0
    out = np.clip(
        out.astype(np.float32) + rng.normal(0, noise_sigma, out.shape).astype(np.float32), 0, 255
    ).astype(np.uint8)

    quality = 70 if level == "scan" else 45
    encoded = cv2.imencode(".jpg", out, [int(cv2.IMWRITE_JPEG_QUALITY), quality])[1]
    return cv2.imdecode(encoded, cv2.IMREAD_GRAYSCALE)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Render a document benchmark with exact ground truth.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--out", default="data/processed/docbench")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--log-level", default="INFO")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    setup_logging(args.log_level)

    out = Path(args.out)
    if not out.is_absolute():
        out = PROJECT_ROOT / out
    ensure_dir(out / "pages")

    rng = np.random.default_rng(args.seed)
    discover_fonts()  # warm the cache / surface a warning if none are installed

    entries = []
    # Each page is rendered at three capture qualities, so the benchmark says
    # how much of any error is the model and how much is the capture.
    for page_index, lines in enumerate(PAGES):
        for level in ("clean", "scan", "photo"):
            scale = float(rng.uniform(0.75, 0.95))
            page = _render_page(lines, font_scale=scale, thickness=2)
            page = _degrade(page, rng, level)

            name = f"page_{page_index:02d}_{level}.png"
            write_image(out / "pages" / name, page)
            entries.append(
                {
                    "file": f"pages/{name}",
                    "page": page_index,
                    "capture": level,
                    "lines": [t for t in lines if t],
                    "text": "\n".join(t for t in lines if t),
                }
            )

    (out / "benchmark.json").write_text(
        json.dumps(
            {
                "name": "document_benchmark",
                "seed": args.seed,
                "pages": len(PAGES),
                "captures": ["clean", "scan", "photo"],
                "count": len(entries),
                "notes": (
                    "Rendered pages with exact ground truth. Measures the whole pipeline "
                    "including line detection and row assembly, not just cropped-line "
                    "recognition."
                ),
                "entries": entries,
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    total_lines = sum(len(e["lines"]) for e in entries)
    logger.info(
        "Wrote %d page image(s) covering %d text line(s) to %s",
        len(entries), total_lines, relative_to_root(out),
    )
    logger.info("Evaluate with:  python scripts/evaluate_documents.py")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
