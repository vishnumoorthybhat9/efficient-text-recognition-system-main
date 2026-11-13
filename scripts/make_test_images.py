"""Assemble a small, varied folder of images for trying the system out.

Everything here already exists elsewhere in the project; this gathers a
representative selection into one place with an index of the expected text, so
the output can be judged rather than just looked at.

Usage
-----
::

    python scripts/make_test_images.py
    python scripts/make_test_images.py --out data/test_images --handwriting 8

Produces ``data/test_images/`` containing:

``01_line_printed_*``      single printed lines -- the easiest case
``02_line_handwritten_*``  real photographed handwriting from the Kaggle set
``03_page_clean_*``        full document pages, cleanly rendered
``04_page_scan_*``         the same pages, lightly skewed and noisy
``05_page_photo_*``        the same pages, rotated with uneven lighting
``06_hard_*``              deliberately difficult: low contrast, heavy skew,
                           inverted, blurred
``EXPECTED.md``            the exact text of every image
"""

from __future__ import annotations

import argparse
import json
import logging
import shutil
import sys
from pathlib import Path
from typing import Sequence

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import cv2  # noqa: E402
import numpy as np  # noqa: E402

from ml.datasets.manifest import read_manifest  # noqa: E402
from ml.utils.image_io import read_image, write_image  # noqa: E402
from ml.utils.logging import setup_logging  # noqa: E402
from ml.utils.paths import PROJECT_ROOT, ensure_dir, relative_to_root  # noqa: E402

logger = logging.getLogger("make_test_images")


def _hard_cases(source: np.ndarray, text: str) -> list[tuple[str, np.ndarray, str, str]]:
    """Degrade one clean page four different ways.

    Each case targets a specific part of the pipeline, so when one fails you
    know which control to reach for.
    """
    height, width = source.shape[:2]
    cases: list[tuple[str, np.ndarray, str, str]] = []

    # 1. Low contrast -- tests CLAHE.
    faint = (source.astype(np.float32) * 0.28 + 170).clip(0, 255).astype(np.uint8)
    cases.append(("06_hard_low_contrast.png", faint, text, "Very faint. Try Enhance contrast (CLAHE)."))

    # 2. Heavy skew -- tests deskew and line detection.
    matrix = cv2.getRotationMatrix2D((width / 2, height / 2), -8.0, 1.0)
    skewed = cv2.warpAffine(source, matrix, (width, height), borderValue=250)
    cases.append(("06_hard_skewed.png", skewed, text, "Rotated 8 degrees. Deskew must be on."))

    # 3. Inverted -- tests polarity normalisation.
    cases.append(
        ("06_hard_inverted.png", 255 - source, text, "White text on black. Polarity is normalised automatically.")
    )

    # 4. Blurred and noisy -- the realistic bad-phone-photo case.
    blurred = cv2.GaussianBlur(source, (0, 0), 1.6)
    rng = np.random.default_rng(3)
    blurred = np.clip(
        blurred.astype(np.float32) + rng.normal(0, 9, blurred.shape).astype(np.float32), 0, 255
    ).astype(np.uint8)
    cases.append(("06_hard_blurred.png", blurred, text, "Out of focus and noisy. Expect errors here."))

    return cases


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Gather a varied set of images for trying the system out.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--out", default="data/test_images")
    parser.add_argument("--handwriting", type=int, default=8, help="real handwriting samples to include")
    parser.add_argument("--seed", type=int, default=11)
    parser.add_argument("--log-level", default="INFO")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    setup_logging(args.log_level)

    out = Path(args.out)
    if not out.is_absolute():
        out = PROJECT_ROOT / out
    if out.exists():
        shutil.rmtree(out)
    ensure_dir(out)

    rng = np.random.default_rng(args.seed)
    index: list[dict[str, str]] = []

    # --- 1. printed single lines (from the bundled demo set) ---------------
    demo_dir = PROJECT_ROOT / "data" / "demo"
    demo_index = {}
    demo_index_file = demo_dir / "demo_index.json"
    if demo_index_file.is_file():
        demo_index = {d["file"]: d for d in json.loads(demo_index_file.read_text(encoding="utf-8"))}

    printed = 0
    handwritten_lines = 0
    for path in sorted(demo_dir.glob("*.png")):
        meta = demo_index.get(path.name, {})
        kind = meta.get("kind", "printed")
        if kind == "printed":
            printed += 1
            name = f"01_line_printed_{printed:02d}.png"
            note = "Single printed line."
        else:
            handwritten_lines += 1
            name = f"02_line_script_{handwritten_lines:02d}.png"
            note = "Single line in a handwriting typeface."
        shutil.copy(path, out / name)
        index.append({"file": name, "expected": meta.get("text", "(unknown)"), "note": note})

    # --- 2. real photographed handwriting ----------------------------------
    handwriting_manifest = PROJECT_ROOT / "data/processed/handwriting/test.tsv"
    if handwriting_manifest.is_file():
        entries = read_manifest(handwriting_manifest, skip_missing=True)
        picked = rng.permutation(len(entries))[: args.handwriting]
        for position, entry_index in enumerate(sorted(picked), start=1):
            entry = entries[int(entry_index)]
            name = f"02_line_handwritten_{position:02d}.png"
            shutil.copy(entry.path, out / name)
            index.append(
                {
                    "file": name,
                    "expected": entry.label,
                    "note": "Real handwriting (Kaggle). See the caveats below the table.",
                }
            )
    else:
        logger.warning("Handwriting dataset not prepared; skipping real handwriting samples")

    # --- 3-5. document pages at three capture qualities --------------------
    bench_file = PROJECT_ROOT / "data/processed/docbench/benchmark.json"
    clean_source: np.ndarray | None = None
    clean_text = ""

    if bench_file.is_file():
        payload = json.loads(bench_file.read_text(encoding="utf-8"))
        prefix = {"clean": "03_page_clean", "scan": "04_page_scan", "photo": "05_page_photo"}
        counters = {"clean": 0, "scan": 0, "photo": 0}
        notes = {
            "clean": "Cleanly rendered page.",
            "scan": "Lightly skewed and noisy, like a flatbed scan.",
            "photo": "Rotated with uneven lighting, like a phone photo.",
        }

        for entry in payload["entries"]:
            capture = entry["capture"]
            counters[capture] += 1
            # Keep three of each so the folder stays browsable.
            if counters[capture] > 3:
                continue

            name = f"{prefix[capture]}_{counters[capture]:02d}.png"
            source_path = PROJECT_ROOT / "data/processed/docbench" / entry["file"]
            shutil.copy(source_path, out / name)
            index.append({"file": name, "expected": entry["text"], "note": notes[capture]})

            if capture == "clean" and clean_source is None:
                clean_source = read_image(source_path)
                clean_text = entry["text"]
    else:
        logger.warning("Document benchmark not built; run scripts/make_document_benchmark.py")

    # --- 6. deliberately hard cases ----------------------------------------
    if clean_source is not None:
        for name, image, text, note in _hard_cases(clean_source, clean_text):
            write_image(out / name, image)
            index.append({"file": name, "expected": text, "note": note})

    # --- index -------------------------------------------------------------
    lines = [
        "# Test images",
        "",
        f"{len(index)} images for trying the system out. Every one has known text,",
        "so you can judge the output rather than guess at it.",
        "",
        "Try them with:",
        "",
        "```bash",
        ".venv/Scripts/python.exe scripts/extract_text.py data/test_images/03_page_clean_01.png --model trocr_base",
        "```",
        "",
        "or drag one onto the **Recognise** page at http://localhost:5173",
        "",
        "| File | Note | Expected text |",
        "|---|---|---|",
    ]
    for item in index:
        expected = item["expected"].replace("\n", " ⏎ ")
        if len(expected) > 110:
            expected = expected[:107] + "..."
        lines.append(f"| `{item['file']}` | {item['note']} | {expected} |")

    lines += [
        "",
        "## Reading the results",
        "",
        "- `01_*` and `02_line_script_*` are single lines and should be near perfect.",
        "- `02_line_handwritten_*` are real photographed form fields, and they come",
        "  with two annotation quirks that make a *correct* reading look wrong:",
        "    - The crop often includes a printed label such as `NOM :`, which is **not**",
        "      in the expected text. A model that transcribes it is not misreading;",
        "      it has no way to know the convention.",
        "    - The annotators dropped accents. `02_line_handwritten_02.png` clearly",
        "      reads **LÉANNE** on the page but its label is `LEANNE`. Reading the",
        "      accent is right; the label is the simplification.",
        "- `03_*` to `05_*` are the same pages captured three ways, so you can see",
        "  how much of any error is the capture rather than the model.",
        "- `06_hard_*` are meant to fail in specific ways; each row says which",
        "  control to reach for.",
        "",
        "`trocr_base` is the most accurate model on these. `crnn_full` is much",
        "faster. Compare them on the **Compare models** page.",
    ]

    (out / "EXPECTED.md").write_text("\n".join(lines), encoding="utf-8")
    (out / "expected.json").write_text(json.dumps(index, indent=2, ensure_ascii=False), encoding="utf-8")

    logger.info("Wrote %d image(s) to %s", len(index), relative_to_root(out))
    logger.info("Index: %s", relative_to_root(out / "EXPECTED.md"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
