"""Score the full pipeline on the document benchmark.

Unlike ``scripts/evaluate.py``, which feeds pre-cropped lines straight to a
model, this runs whole pages through preprocessing, line detection,
recognition and row assembly, then compares the assembled text against the
page's exact ground truth.

That is the number that matters for "extract the text from this document",
and it is always worse than cropped-line CER, because it also carries
segmentation errors.

Usage
-----
::

    python scripts/evaluate_documents.py
    python scripts/evaluate_documents.py --models crnn_full trocr_printed
    python scripts/evaluate_documents.py --capture clean
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
from backend.services.runtime import init_runtime  # noqa: E402
from ml.evaluation.metrics import evaluate_predictions, normalize_text  # noqa: E402
from ml.inference.predictor import RecognizeOptions  # noqa: E402
from ml.preprocessing.pipeline import PreprocessOptions  # noqa: E402
from ml.utils.image_io import read_image  # noqa: E402
from ml.utils.logging import setup_logging  # noqa: E402
from ml.utils.paths import PROJECT_ROOT, ensure_dir, relative_to_root  # noqa: E402

logger = logging.getLogger("evaluate_documents")


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Score whole-document extraction for one or more models.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--benchmark", default="data/processed/docbench/benchmark.json")
    parser.add_argument("--models", nargs="+", default=None, help="model ids; default is all available")
    parser.add_argument(
        "--capture",
        default=None,
        choices=["clean", "scan", "photo"],
        help="restrict to one capture quality",
    )
    parser.add_argument("--out", default="artifacts/reports/document_benchmark.json")
    parser.add_argument("--show", type=int, default=2, help="print this many example outputs")
    parser.add_argument("--log-level", default="WARNING")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    setup_logging(args.log_level)

    benchmark_path = Path(args.benchmark)
    if not benchmark_path.is_absolute():
        benchmark_path = PROJECT_ROOT / benchmark_path

    if not benchmark_path.is_file():
        logger.error(
            "Benchmark not found at %s. Create it with:\n"
            "  python scripts/make_document_benchmark.py",
            relative_to_root(benchmark_path),
        )
        return 1

    payload = json.loads(benchmark_path.read_text(encoding="utf-8"))
    entries = payload["entries"]
    if args.capture:
        entries = [e for e in entries if e["capture"] == args.capture]

    runtime = init_runtime(get_settings())
    model_ids = args.models or [e.id for e in runtime.registry.entries() if e.available]

    # Deskew matters a great deal on the skewed captures; use the app defaults.
    options = RecognizeOptions(preprocess=PreprocessOptions())

    results: dict[str, Any] = {}
    examples: list[dict[str, Any]] = []

    for model_id in model_ids:
        try:
            predictor = runtime.registry.get(model_id)
        except Exception as exc:
            logger.error("Skipping %s: %s", model_id, str(exc)[:160])
            continue

        references: list[str] = []
        predictions: list[str] = []
        per_capture: dict[str, list[tuple[str, str]]] = {}
        started = time.perf_counter()

        for entry in entries:
            image = read_image(PROJECT_ROOT / "data/processed/docbench" / entry["file"])
            try:
                result = predictor.recognize(image, options)
                text = result.text
            except Exception as exc:
                logger.warning("%s failed on %s: %s", model_id, entry["file"], str(exc)[:120])
                text = ""

            references.append(entry["text"])
            predictions.append(text)
            per_capture.setdefault(entry["capture"], []).append((entry["text"], text))

            if len(examples) < args.show and entry["capture"] == "clean":
                examples.append(
                    {"model": model_id, "file": entry["file"], "reference": entry["text"], "prediction": text}
                )

        elapsed = time.perf_counter() - started
        aggregate, _ = evaluate_predictions(references, predictions, return_samples=False)

        by_capture = {}
        for capture, pairs in per_capture.items():
            sub, _ = evaluate_predictions([a for a, _ in pairs], [b for _, b in pairs], return_samples=False)
            by_capture[capture] = {
                "cer": sub.cer,
                "wer": sub.wer,
                "sequence_accuracy": sub.sequence_accuracy,
                "cer_normalized": sub.cer_normalized,
            }

        entry_obj = runtime.registry.entry(model_id)
        results[model_id] = {
            "name": entry_obj.name,
            "provenance": entry_obj.provenance,
            "pages": len(entries),
            "cer": aggregate.cer,
            "wer": aggregate.wer,
            "sequence_accuracy": aggregate.sequence_accuracy,
            "character_accuracy": aggregate.character_accuracy,
            "cer_normalized": aggregate.cer_normalized,
            "seconds_per_page": elapsed / max(1, len(entries)),
            "by_capture": by_capture,
        }
        logger.warning(
            "%s: page CER %.4f, exact-page accuracy %.4f",
            model_id,
            aggregate.cer,
            aggregate.sequence_accuracy,
        )
        runtime.registry.unload(model_id)

    if not results:
        logger.error("No model could be evaluated.")
        return 1

    # --- report -----------------------------------------------------------
    print(f"\nWhole-document extraction -- {len(entries)} page(s)")
    print("(CER over the entire assembled page text, including segmentation errors)\n")

    header = (
        f"{'model':34s} {'CER':>8s} {'CER(ci)':>8s} {'WER':>8s} {'exact':>8s} "
        f"{'clean':>8s} {'scan':>8s} {'photo':>8s} {'s/page':>7s}"
    )
    print(header)
    print("-" * len(header))

    for model_id, row in sorted(results.items(), key=lambda kv: kv[1]["cer"]):
        caps = row["by_capture"]
        print(
            f"{row['name'][:34]:34s} {row['cer']:8.4f} {row['cer_normalized']:8.4f} "
            f"{row['wer']:8.4f} {row['sequence_accuracy']:8.4f} "
            f"{caps.get('clean', {}).get('cer', float('nan')):8.4f} "
            f"{caps.get('scan', {}).get('cer', float('nan')):8.4f} "
            f"{caps.get('photo', {}).get('cer', float('nan')):8.4f} "
            f"{row['seconds_per_page']:7.2f}"
        )
    print()

    for example in examples:
        print(f"--- {example['model']} on {example['file']} ---")
        print("reference :", example["reference"].replace("\n", " | ")[:180])
        print("prediction:", example["prediction"].replace("\n", " | ")[:180])
        print()

    out_path = Path(args.out)
    if not out_path.is_absolute():
        out_path = PROJECT_ROOT / out_path
    ensure_dir(out_path.parent)
    out_path.write_text(
        json.dumps({"pages": len(entries), "results": results, "examples": examples}, indent=2),
        encoding="utf-8",
    )
    print(f"Report written to {relative_to_root(out_path)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
