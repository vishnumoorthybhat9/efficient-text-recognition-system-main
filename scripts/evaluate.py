"""Benchmark trained models on a held-out split.

Measures accuracy (CER, WER, exact match) **and** cost (latency, throughput,
peak memory, parameter count), stores each run in the database so the
dashboard can show it, and writes a JSON report to ``artifacts/reports/``.

Usage
-----
::

    # every available model on the synthetic test split
    python scripts/evaluate.py --all --split test

    # just the two experiment arms
    python scripts/evaluate.py --models baseline_scratch simclr_finetune

    # on the real handwriting dataset
    python scripts/evaluate.py --all --dataset handwriting --split test

    # quick check
    python scripts/evaluate.py --all --max-samples 200

Timing notes
------------
The first batches are discarded as warm-up and CUDA is synchronised around
every timed region, so the latency figures are steady-state and real. Image
loading is timed separately from inference: it is a cost every model shares,
and folding it in would flatter a slow model.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import Sequence

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from backend.core.config import get_settings  # noqa: E402
from backend.db.session import init_db, session_scope  # noqa: E402
from backend.services.evaluation_service import run_evaluation  # noqa: E402
from backend.services.runtime import init_runtime  # noqa: E402
from ml.utils.logging import setup_logging  # noqa: E402
from ml.utils.paths import ensure_dir, relative_to_root  # noqa: E402

logger = logging.getLogger("evaluate")


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Benchmark models for accuracy and computational cost.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--models", nargs="+", help="model ids to evaluate")
    group.add_argument("--all", action="store_true", help="evaluate every available model")

    parser.add_argument("--dataset", default="synthetic", help="dataset name under data/processed")
    parser.add_argument("--split", default="test", choices=["train", "val", "test"])
    parser.add_argument("--max-samples", type=int, default=None, help="cap the sample count")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--decode", default="greedy", choices=["greedy", "beam"])
    parser.add_argument("--beam-width", type=int, default=10)
    parser.add_argument(
        "--skip-pretrained",
        action="store_true",
        help="exclude third-party pretrained models (TrOCR)",
    )
    parser.add_argument("--no-store", action="store_true", help="do not write to the database")
    parser.add_argument("--log-level", default="INFO")
    return parser.parse_args(argv)


def _print_table(rows: list[dict[str, object]]) -> None:
    """Print a fixed-width results table."""
    if not rows:
        return

    headers = [
        ("model", 30),
        ("CER", 9),
        ("WER", 9),
        ("exact", 9),
        ("ms/img", 9),
        ("img/s", 8),
        ("params", 10),
        ("n", 7),
    ]

    line = "  ".join(h.ljust(w) for h, w in headers)
    print()
    print(line)
    print("-" * len(line))

    for row in rows:
        cells = [
            str(row["model"])[:30].ljust(30),
            f"{row['cer']:.4f}".ljust(9),
            f"{row['wer']:.4f}".ljust(9),
            f"{row['exact']:.4f}".ljust(9),
            f"{row['ms']:.2f}".ljust(9),
            f"{row['throughput']:.0f}".ljust(8),
            f"{row['params'] / 1e6:.2f}M".ljust(10),
            str(row["n"]).ljust(7),
        ]
        print("  ".join(cells))
    print()


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    setup_logging(args.log_level, log_file=ensure_dir("artifacts/logs") / "evaluate.log")

    settings = get_settings()
    init_db(settings)
    runtime = init_runtime(settings)

    entries = runtime.registry.entries()
    if args.all:
        model_ids = [e.id for e in entries if e.available]
        if args.skip_pretrained:
            model_ids = [
                e.id for e in entries if e.available and e.provenance == "trained_in_project"
            ]
    else:
        model_ids = list(args.models)

    if not model_ids:
        logger.error(
            "No models to evaluate. Train one first:\n"
            "  python scripts/train_crnn.py --config configs/experiments/baseline_scratch.yaml"
        )
        return 1

    logger.info(
        "Evaluating %d model(s) on %s/%s: %s",
        len(model_ids),
        args.dataset,
        args.split,
        ", ".join(model_ids),
    )

    table: list[dict[str, object]] = []
    failures: list[tuple[str, str]] = []

    for model_id in model_ids:
        logger.info("-" * 72)
        try:
            with session_scope() as session:
                result, row = run_evaluation(
                    session=session,
                    runtime=runtime,
                    model_id=model_id,
                    dataset=args.dataset,
                    split=args.split,
                    max_samples=args.max_samples,
                    batch_size=args.batch_size,
                    decode_strategy=args.decode,
                    beam_width=args.beam_width,
                    persist=not args.no_store,
                )

            table.append(
                {
                    "model": result.model_name,
                    "cer": result.metrics.cer,
                    "wer": result.metrics.wer,
                    "exact": result.metrics.sequence_accuracy,
                    "ms": result.avg_inference_ms,
                    "throughput": result.throughput_images_per_s,
                    "params": result.parameter_count,
                    "n": result.num_samples,
                }
            )

            # Free VRAM before the next model: on a 4 GB card two recognisers
            # plus TrOCR will not coexist.
            runtime.registry.unload(model_id)

        except Exception as exc:
            logger.error("Evaluation of %s failed: %s", model_id, exc)
            logger.debug("", exc_info=True)
            failures.append((model_id, str(exc)[:200]))

    if not table:
        logger.error("Every model failed to evaluate.")
        for model_id, error in failures:
            logger.error("  %s: %s", model_id, error)
        return 1

    table.sort(key=lambda r: r["cer"])  # type: ignore[arg-type,return-value]

    print(f"\nResults on {args.dataset}/{args.split} (device: {runtime.device})")
    _print_table(table)

    if failures:
        print("Models that could not be evaluated:")
        for model_id, error in failures:
            print(f"  - {model_id}: {error}")
        print()

    logger.info("Reports written to %s", relative_to_root(Path("artifacts/reports")))
    logger.info("Build the comparison report with:  python scripts/compare_models.py")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
