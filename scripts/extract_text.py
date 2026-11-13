"""Extract text from an image and print it.

The plainest possible entry point: give it a file, get the text on stdout.
No server, no browser.

Usage
-----
::

    # one image -> text on screen
    python scripts/extract_text.py page.png

    # choose a model
    python scripts/extract_text.py page.png --model trocr_base

    # a photographed handwritten page (notebook, letter, prescription)
    python scripts/extract_text.py notes.jpg --handwritten --model trocr_base

    # a whole folder, writing one .txt beside each image
    python scripts/extract_text.py scans/ --save-next-to-image

    # everything into one file
    python scripts/extract_text.py scans/*.png --out all.txt

    # structured output for scripting
    python scripts/extract_text.py page.png --json

Which model?
------------
``--model list`` prints what is installed. In short:

``crnn_full``   this project's own recogniser -- fast (a few ms per line)
``trocr_base``  most accurate on arbitrary documents, ~3 s per page, and
                downloads ~1.3 GB the first time it runs
"""

from __future__ import annotations

import argparse
import glob
import json
import logging
import sys
import time
from pathlib import Path
from typing import Any, Sequence

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from backend.core.config import get_settings  # noqa: E402
from backend.services.runtime import init_runtime  # noqa: E402
from ml.inference.predictor import RecognizeOptions  # noqa: E402
from ml.preprocessing.pipeline import PreprocessOptions  # noqa: E402
from ml.utils.image_io import SUPPORTED_EXTENSIONS, ImageDecodeError, read_image  # noqa: E402
from ml.utils.logging import setup_logging  # noqa: E402

logger = logging.getLogger("extract_text")


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Extract text from one or more images.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "inputs",
        nargs="*",
        help="image files, folders, or glob patterns. A folder is searched recursively.",
    )
    parser.add_argument(
        "--model",
        default=None,
        help="model id, or 'list' to show what is installed. Default: the best available.",
    )
    parser.add_argument("--out", default=None, help="write all text to this file instead of stdout")
    parser.add_argument(
        "--save-next-to-image",
        action="store_true",
        help="write <image>.txt beside each input image",
    )
    parser.add_argument("--json", action="store_true", help="emit structured JSON")

    group = parser.add_argument_group("recognition")
    group.add_argument(
        "--single-line",
        action="store_true",
        help="the image is one cropped line; skip line detection",
    )
    group.add_argument("--beam", action="store_true", help="beam-search decoding (slower, slightly better)")
    group.add_argument(
        "--handwritten",
        action="store_true",
        help="handwritten page: remove shadows and ruled lines, use the handwriting line detector",
    )
    group.add_argument(
        "--detector",
        choices=("auto", "projection", "doctr"),
        default="auto",
        help="line detector for --handwritten pages (doctr needs: pip install python-doctr)",
    )
    group.add_argument("--no-deskew", action="store_true", help="do not straighten the page")
    group.add_argument("--enhance", action="store_true", help="contrast-enhance (helps dim photographs)")
    group.add_argument(
        "--min-confidence",
        type=float,
        default=0.0,
        help="drop lines below this confidence (0 keeps everything)",
    )

    parser.add_argument("--quiet", action="store_true", help="text only, no headers or timings")
    parser.add_argument("--log-level", default="ERROR")
    return parser.parse_args(argv)


def _collect(inputs: Sequence[str]) -> list[Path]:
    """Expand files, folders and globs into a sorted list of image paths."""
    found: list[Path] = []

    for item in inputs:
        path = Path(item)
        if path.is_dir():
            found.extend(
                child
                for child in sorted(path.rglob("*"))
                if child.is_file() and child.suffix.lower() in SUPPORTED_EXTENSIONS
            )
        elif path.is_file():
            found.append(path)
        else:
            # Let the shell's failure to expand a glob still work.
            matches = [Path(m) for m in glob.glob(item)]
            found.extend(
                m for m in sorted(matches) if m.is_file() and m.suffix.lower() in SUPPORTED_EXTENSIONS
            )

    # Preserve order while removing duplicates.
    seen: dict[Path, None] = {}
    for path in found:
        seen.setdefault(path.resolve(), None)
    return list(seen)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    setup_logging(args.log_level)

    runtime = init_runtime(get_settings())

    # --- model listing -----------------------------------------------------
    if args.model == "list":
        print("\nInstalled models:\n")
        for entry in runtime.registry.entries():
            mark = "ok" if entry.available else "--"
            default = "  (default)" if entry.id == runtime.registry.default_model_id() else ""
            print(f"  [{mark}] {entry.id:22s} {entry.name}{default}")
            if not entry.available and entry.unavailable_reason:
                print(f"       unavailable: {entry.unavailable_reason[:90]}")
        print()
        return 0

    if not args.inputs:
        print("Nothing to do: give me an image file, a folder, or a glob pattern.\n")
        print("  python scripts/extract_text.py page.png")
        print("  python scripts/extract_text.py scans/ --save-next-to-image")
        print("  python scripts/extract_text.py --model list")
        return 1

    images = _collect(args.inputs)
    if not images:
        print(f"No readable image found in: {', '.join(args.inputs)}")
        print("Supported: " + ", ".join(sorted(e.lstrip('.') for e in SUPPORTED_EXTENSIONS)))
        return 1

    # --- model -------------------------------------------------------------
    model_id = args.model or runtime.registry.default_model_id()
    if model_id is None:
        print("No model is available. Train one, or install `transformers` for the TrOCR baseline.")
        return 1

    try:
        predictor = runtime.registry.get(model_id)
    except Exception as exc:
        print(f"Could not load model {model_id!r}: {exc}")
        print("Run with `--model list` to see what is installed.")
        return 1

    options = RecognizeOptions(
        preprocess=PreprocessOptions(deskew=not args.no_deskew, clahe=args.enhance),
        single_line=args.single_line,
        handwritten=args.handwritten,
        line_detector=args.detector,
        decode_strategy="beam" if args.beam else "greedy",
        min_confidence=args.min_confidence,
    )

    if not args.quiet and not args.json:
        print(f"Model: {predictor.model_name}  ({predictor.device})\n")

    # --- run ---------------------------------------------------------------
    records: list[dict[str, Any]] = []
    chunks: list[str] = []
    failures = 0

    for path in images:
        try:
            image = read_image(path, grayscale=False)
            started = time.perf_counter()
            result = predictor.recognize(image, options)
            elapsed = (time.perf_counter() - started) * 1000
        except (FileNotFoundError, ImageDecodeError) as exc:
            failures += 1
            print(f"[skipped] {path.name}: {exc}", file=sys.stderr)
            continue
        except Exception as exc:
            failures += 1
            print(f"[failed]  {path.name}: {type(exc).__name__}: {exc}", file=sys.stderr)
            continue

        records.append(
            {
                "file": str(path),
                "text": result.text,
                "lines": [line.text for line in result.lines],
                "confidence": round(result.confidence, 4),
                "line_count": result.line_count,
                "character_count": result.character_count,
                "ms": round(elapsed, 1),
                "model": predictor.model_id,
            }
        )

        if args.save_next_to_image:
            path.with_suffix(".txt").write_text(result.text, encoding="utf-8")

        if len(images) > 1 and not args.quiet and not args.json:
            chunks.append(f"===== {path.name} =====\n{result.text}")
        else:
            chunks.append(result.text)

    if not records:
        return 1

    # --- output ------------------------------------------------------------
    if args.json:
        payload = json.dumps(records if len(records) > 1 else records[0], indent=2, ensure_ascii=False)
        if args.out:
            Path(args.out).write_text(payload, encoding="utf-8")
            print(f"Wrote {args.out}")
        else:
            print(payload)
        return 0

    combined = "\n\n".join(chunks)

    if args.out:
        Path(args.out).write_text(combined, encoding="utf-8")
        print(f"Wrote {args.out}  ({sum(r['character_count'] for r in records)} characters)")
    else:
        print(combined)

    if not args.quiet:
        sys.stdout.flush()
        total_ms = sum(r["ms"] for r in records)
        mean_confidence = sum(r["confidence"] for r in records) / len(records)
        print(
            f"\n{len(records)} image(s), {sum(r['line_count'] for r in records)} line(s), "
            f"mean confidence {mean_confidence:.1%}, {total_ms:.0f} ms total"
            + (f", {failures} failed" if failures else ""),
            file=sys.stderr,
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
