"""Create editable, line-level draft transcripts for handwritten page images.

The OCR output is unverified and must not be used as ground truth until
reviewed. Line crops and confidence values are saved for efficient correction.

Usage:
    python scripts/transcribe_handwriting.py data/handwritten/images
    python scripts/transcribe_handwriting.py data/handwritten/images --model trocr_large
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path
from typing import Any, Sequence

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from backend.core.config import get_settings  # noqa: E402
from backend.services.runtime import init_runtime, shutdown_runtime  # noqa: E402
from ml.inference.handwriting import clean_handwritten_page, detect_handwritten_lines  # noqa: E402
from ml.inference.predictor import RecognizeOptions  # noqa: E402
from ml.inference.segmentation import TextRegion, crop_region  # noqa: E402
from ml.preprocessing.pipeline import PreprocessOptions, PreprocessPipeline  # noqa: E402
from ml.utils.image_io import SUPPORTED_EXTENSIONS, read_image  # noqa: E402
from ml.utils.paths import PROJECT_ROOT, relative_to_root  # noqa: E402

logger = logging.getLogger("transcribe_handwriting")


def _collect(folder: Path) -> list[Path]:
    if not folder.is_dir():
        raise NotADirectoryError(f"Image folder does not exist: {folder}")
    return sorted(
        path
        for path in folder.iterdir()
        if path.is_file() and path.suffix.lower() in SUPPORTED_EXTENSIONS
    )


def _write_page(
    *,
    source: Path,
    output_dir: Path,
    model_id: str,
    regions: Sequence[TextRegion],
    crops: Sequence[Any],
    predictions: Sequence[tuple[str, float]],
    detector: str,
    elapsed_seconds: float,
) -> dict[str, Any]:
    import cv2

    crop_dir = output_dir / "line_crops" / source.stem
    crop_dir.mkdir(parents=True, exist_ok=True)
    lines: list[dict[str, Any]] = []

    if len(regions) != len(predictions) or len(regions) != len(crops):
        raise ValueError(
            f"Found {len(regions)} text regions, {len(crops)} crops, "
            f"and {len(predictions)} predictions"
        )
    text_lines: list[str] = []
    for index, (region, crop, (text, confidence)) in enumerate(
        zip(regions, crops, predictions, strict=True), start=1
    ):
        crop_name = f"line_{index:03d}.png"
        crop_path = crop_dir / crop_name
        if crop.size:
            if not cv2.imwrite(str(crop_path), crop):
                raise OSError(f"Could not write line crop: {crop_path}")
        else:
            crop_name = ""
        text_lines.append(text)
        lines.append(
            {
                "line": index,
                "text": text,
                "confidence": round(float(confidence), 6),
                "region": region.to_dict() if region is not None else None,
                "crop": (
                    (Path("line_crops") / source.stem / crop_name).as_posix()
                    if crop_name
                    else None
                ),
            }
        )

    page_record = {
        "source_image": source.name,
        "model_id": model_id,
        "status": "ocr_draft_unverified",
        "line_detector": detector,
        "line_count": len(lines),
        "elapsed_seconds": round(elapsed_seconds, 2),
        "text": "\n".join(text_lines).strip(),
        "lines": lines,
        "review_required": True,
    }
    json_path = output_dir / f"{source.stem}.json"
    text_path = output_dir / f"{source.stem}.txt"
    json_path.write_text(json.dumps(page_record, indent=2, ensure_ascii=False), encoding="utf-8")
    text_path.write_text(page_record["text"].rstrip() + "\n", encoding="utf-8")
    return page_record


def transcribe(
    image_dir: Path,
    output_dir: Path,
    *,
    model_id: str = "trocr_base",
    detector: str = "auto",
    batch_size: int = 16,
    overwrite: bool = False,
) -> dict[str, Any]:
    """Run local OCR on every page, saving resumable draft transcripts."""
    images = _collect(image_dir)
    if not images:
        raise ValueError(f"No supported page images found in {image_dir}")
    if batch_size < 1:
        raise ValueError("batch_size must be at least one")

    output_dir.mkdir(parents=True, exist_ok=True)
    runtime = init_runtime(get_settings())
    try:
        predictor = runtime.registry.get(model_id)
        options = RecognizeOptions(
            preprocess=PreprocessOptions(),
            handwritten=True,
            line_detector=detector,
            batch_size=batch_size,
        )
        aggregate: list[dict[str, Any]] = []
        failures: list[dict[str, str]] = []

        for index, source in enumerate(images, start=1):
            output_json = output_dir / f"{source.stem}.json"
            if output_json.is_file() and not overwrite:
                previous = json.loads(output_json.read_text(encoding="utf-8"))
                if previous.get("status") == "ocr_draft_unverified":
                    aggregate.append(previous)
                    logger.info("Skipping %s (%d/%d): draft already exists", source.name, index, len(images))
                    continue

            try:
                started = time.perf_counter()
                image = read_image(source, grayscale=False)
                processed = PreprocessPipeline(options.preprocess).run(image)
                cleaned = clean_handwritten_page(processed.image)
                segmentation, used_detector = detect_handwritten_lines(
                    cleaned.image,
                    detector=detector,
                    max_lines=options.max_lines,
                )
                crops = [crop_region(cleaned.image, region) for region in segmentation.regions]
                predictions = predictor.recognize_lines(crops, options)
                elapsed = time.perf_counter() - started
                record = _write_page(
                    source=source,
                    output_dir=output_dir,
                    model_id=model_id,
                    regions=segmentation.regions,
                    crops=crops,
                    predictions=predictions,
                    detector=used_detector,
                    elapsed_seconds=elapsed,
                )
                record["detected_line_count"] = len(segmentation.regions)
                (output_dir / f"{source.stem}.json").write_text(
                    json.dumps(record, indent=2, ensure_ascii=False), encoding="utf-8"
                )
                aggregate.append(record)
                logger.info(
                    "%s (%d/%d): %d line(s), %.1fs",
                    source.name,
                    index,
                    len(images),
                    record["line_count"],
                    elapsed,
                )
            except Exception as exc:
                logger.exception("Could not transcribe %s", source)
                failures.append(
                    {"file": source.name, "error": f"{type(exc).__name__}: {exc}"}
                )

        manifest = {
            "status": "ocr_draft_unverified",
            "source_directory": str(image_dir.resolve()),
            "output_directory": str(output_dir.resolve()),
            "model_id": model_id,
            "device": str(runtime.device),
            "processed_pages": len(aggregate),
            "failed_pages": failures,
            "review_required": True,
            "pages": [
                {
                    "source_image": record["source_image"],
                    "text_file": f"{Path(record['source_image']).stem}.txt",
                    "line_count": record["line_count"],
                }
                for record in aggregate
            ],
        }
        (output_dir / "index.json").write_text(
            json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8"
        )

        # Keep machine-generated guesses out of the training manifests.
        warning = (
            "# UNVERIFIED OCR DRAFT. Review every transcription before using as ground truth.\n"
            "# image_path<TAB>draft_text; this file is intentionally outside data/processed.\n"
        )
        rows = [
            f"{line['crop']}\t{line['text'].replace(chr(9), ' ').replace(chr(10), ' ')}"
            for record in aggregate
            for line in record["lines"]
            if line["crop"] and line["text"].strip()
        ]
        (output_dir / "draft_lines.tsv").write_text(
            warning + "\n".join(rows) + ("\n" if rows else ""),
            encoding="utf-8",
        )
        if failures:
            raise RuntimeError(
                f"Transcribed {len(aggregate)} page(s), but {len(failures)} page(s) failed. "
                f"See {relative_to_root(output_dir / 'index.json')} for details."
            )
        return manifest
    finally:
        shutdown_runtime()


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "images",
        nargs="?",
        type=Path,
        default=PROJECT_ROOT / "data" / "handwritten" / "images",
        help="directory containing upright handwritten page images",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=PROJECT_ROOT / "data" / "handwritten" / "transcriptions",
        help="output directory for editable drafts, line crops, and JSON",
    )
    parser.add_argument("--model", default="trocr_base", help="recognizer model id (default: trocr_base)")
    parser.add_argument(
        "--detector",
        choices=("auto", "projection", "doctr"),
        default="auto",
        help="handwritten line detector; auto uses docTR when installed",
    )
    parser.add_argument("--batch-size", type=int, default=16, help="OCR line crops per model batch")
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="re-run OCR for pages with existing draft JSON files",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    try:
        manifest = transcribe(
            args.images,
            args.out,
            model_id=args.model,
            detector=args.detector,
            batch_size=args.batch_size,
            overwrite=args.overwrite,
        )
    except Exception as exc:
        logger.error("%s: %s", type(exc).__name__, exc)
        return 1
    logger.info(
        "Saved %d unverified draft(s) to %s",
        manifest["processed_pages"],
        relative_to_root(args.out),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
