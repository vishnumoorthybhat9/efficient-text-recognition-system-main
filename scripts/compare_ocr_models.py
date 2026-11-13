"""Compare OCR accuracy and predictions on the same held-out samples."""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Sequence

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from backend.core.config import get_settings  # noqa: E402
from backend.services.evaluation_service import load_split_entries  # noqa: E402
from backend.services.runtime import init_runtime  # noqa: E402
from ml.evaluation.benchmark import benchmark_predictor  # noqa: E402
from ml.utils.logging import setup_logging  # noqa: E402
from ml.utils.paths import ensure_dir, relative_to_root  # noqa: E402

logger = logging.getLogger("compare_ocr_models")


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare OCR models on identical test images and print example outputs.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--models", nargs="+", required=True, help="registered model ids")
    parser.add_argument("--dataset", default="synthetic")
    parser.add_argument("--split", default="test", choices=["train", "val", "test"])
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--examples", type=int, default=10, help="number of aligned output examples")
    parser.add_argument("--out", default="artifacts/reports/recognizer_comparison.json")
    parser.add_argument("--log-level", default="INFO")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    setup_logging(args.log_level)
    runtime = init_runtime(get_settings())
    entries = load_split_entries(
        runtime,
        dataset=args.dataset,
        split=args.split,
        max_samples=args.max_samples,
    )

    results = {}
    predictions_by_sample: dict[str, dict[str, str]] = {}
    references_by_sample: dict[str, str] = {}

    try:
        for model_id in args.models:
            predictor = runtime.registry.get(model_id)
            result = benchmark_predictor(
                predictor,
                entries,
                dataset=args.dataset,
                split=args.split,
                batch_size=args.batch_size,
            )
            results[model_id] = {
                "name": result.model_name,
                "provenance": runtime.registry.entry(model_id).provenance,
                "samples": result.num_samples,
                "cer": result.metrics.cer,
                "character_accuracy": result.metrics.character_accuracy,
                "wer": result.metrics.wer,
                "exact_match_accuracy": result.metrics.sequence_accuracy,
                "avg_inference_ms": result.avg_inference_ms,
                "parameters": result.parameter_count,
            }
            for sample in result.samples:
                if sample.sample_id is None:
                    continue
                references_by_sample[sample.sample_id] = sample.reference
                predictions_by_sample.setdefault(sample.sample_id, {})[model_id] = sample.prediction
            runtime.registry.unload(model_id)
    finally:
        runtime.close()

    examples = []
    for sample_id, model_predictions in list(predictions_by_sample.items())[: max(0, args.examples)]:
        examples.append(
            {
                "sample_id": sample_id,
                "reference": references_by_sample[sample_id],
                "predictions": model_predictions,
            }
        )

    report = {
        "dataset": args.dataset,
        "split": args.split,
        "same_samples": len(predictions_by_sample),
        "results": results,
        "examples": examples,
    }
    output = Path(args.out)
    if not output.is_absolute():
        output = PROJECT_ROOT / output
    ensure_dir(output.parent)
    output.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"\n{args.dataset}/{args.split} — {len(predictions_by_sample)} identical samples")
    print(f"{'model':28s} {'CER':>8s} {'char acc':>9s} {'WER':>8s} {'exact':>8s} {'ms/img':>9s}")
    print("-" * 76)
    for model_id, row in results.items():
        print(
            f"{row['name'][:28]:28s} {row['cer']:8.4f} "
            f"{row['character_accuracy']:9.4f} {row['wer']:8.4f} "
            f"{row['exact_match_accuracy']:8.4f} {row['avg_inference_ms']:9.2f}"
        )
    print(f"\nAligned examples and metrics written to {relative_to_root(output)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
