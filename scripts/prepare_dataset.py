"""Import a real, externally sourced dataset into the project's manifest format.

Once imported, a dataset is indistinguishable from the synthetic one to every
downstream stage -- training, evaluation and the API all read the same
manifests -- so the whole experiment can be re-run on real data by changing
one config path.

Usage
-----
::

    # Kaggle "Handwriting Recognition" (413k handwritten name fields), read
    # straight out of the downloaded zip:
    python scripts/prepare_dataset.py --dataset handwriting \\
        --source "D:/Downloads/archive (1).zip" \\
        --train 20000 --val 3000 --test 3000 --unlabeled 30000

    # IAM Handwriting Database (after extracting lines.tgz and ascii.tgz):
    python scripts/prepare_dataset.py --dataset iam --source data/raw/iam

    # IIIT 5K-Word:
    python scripts/prepare_dataset.py --dataset iiit5k --source data/raw/iiit5k

Where to get the data
---------------------
IAM      https://fki.tic.heia-fr.ch/databases/iam-handwriting-database  (registration)
IIIT5K   https://cvit.iiit.ac.in/research/projects/cvit-projects/the-iiit-5k-word-dataset
MJSynth  https://www.robots.ox.ac.uk/~vgg/data/text/
SynthText https://www.robots.ox.ac.uk/~vgg/data/scenetext/
Kaggle handwriting  https://www.kaggle.com/datasets/landlord/handwriting-recognition
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path
from typing import Sequence

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ml.config import load_config  # noqa: E402
from ml.datasets.external import (  # noqa: E402
    SplitCounts,
    prepare_iam,
    prepare_iiit5k,
    prepare_kaggle_handwriting,
)
from ml.datasets.manifest import validate_manifest  # noqa: E402
from ml.text.charset import Charset  # noqa: E402
from ml.utils.logging import setup_logging  # noqa: E402
from ml.utils.paths import PROJECT_ROOT, relative_to_root  # noqa: E402

logger = logging.getLogger("prepare_dataset")

DATASETS = ("handwriting", "iam", "iiit5k")


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Import a real OCR dataset into the project's manifest format.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--dataset", required=True, choices=DATASETS)
    parser.add_argument(
        "--source",
        required=True,
        help="zip file (handwriting) or extracted root directory (iam, iiit5k)",
    )
    parser.add_argument("--out", default=None, help="output directory (default: data/processed/<dataset>)")
    parser.add_argument("--config", default="default.yaml")
    parser.add_argument("--train", type=int, default=20000, help="max labeled training samples")
    parser.add_argument("--val", type=int, default=3000, help="max validation samples")
    parser.add_argument("--test", type=int, default=3000, help="max test samples")
    parser.add_argument(
        "--unlabeled",
        type=int,
        default=30000,
        help="size of the unlabeled pool for SimCLR (disjoint from every labeled split)",
    )
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--force", action="store_true", help="overwrite an existing output directory")
    parser.add_argument("--log-level", default="INFO")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    setup_logging(args.log_level)

    config = load_config(args.config)
    charset = Charset.from_config(config.charset)

    # The encoder emits width//4 - 1 time steps; CTC also needs a blank between
    # repeated characters, so cap labels well under that.
    time_steps = max(1, int(config.get("image.width", 256)) // 4 - 1)
    max_label_length = max(4, int(time_steps * 0.6))

    output_dir = Path(args.out) if args.out else PROJECT_ROOT / "data" / "processed" / args.dataset
    if not output_dir.is_absolute():
        output_dir = PROJECT_ROOT / output_dir

    if output_dir.exists() and any(output_dir.iterdir()) and not args.force:
        logger.error(
            "%s already exists and is not empty. Re-run with --force to overwrite it.",
            relative_to_root(output_dir),
        )
        return 1

    counts = SplitCounts(
        train=args.train if args.train > 0 else None,
        val=args.val if args.val > 0 else None,
        test=args.test if args.test > 0 else None,
        unlabeled=max(0, args.unlabeled),
    )

    logger.info(
        "Importing %r from %s into %s",
        args.dataset,
        args.source,
        relative_to_root(output_dir),
    )
    logger.info(
        "Charset: %d classes | max label length: %d characters (%d CTC time steps)",
        charset.num_classes,
        max_label_length,
        time_steps,
    )

    def progress(split: str, done: int, total: int) -> None:
        logger.info("  %s: %d/%d written", split, done, total)

    started = time.perf_counter()

    try:
        if args.dataset == "handwriting":
            report = prepare_kaggle_handwriting(
                args.source,
                output_dir,
                charset=charset,
                counts=counts,
                max_label_length=max_label_length,
                seed=args.seed,
                progress=progress,
            )
        elif args.dataset == "iam":
            report = prepare_iam(
                args.source,
                output_dir,
                charset=charset,
                counts=counts,
                max_label_length=max_label_length,
                seed=args.seed,
            )
        else:
            report = prepare_iiit5k(
                args.source,
                output_dir,
                charset=charset,
                counts=counts,
                max_label_length=max_label_length,
                seed=args.seed,
            )
    except (FileNotFoundError, ValueError, ImportError) as exc:
        logger.error("%s", exc)
        return 1

    elapsed = time.perf_counter() - started

    # --- validate what we wrote -------------------------------------------
    validations: dict[str, dict[str, object]] = {}
    all_ok = True
    for split in ("train", "val", "test"):
        manifest = output_dir / f"{split}.tsv"
        if not manifest.is_file():
            continue
        result = validate_manifest(manifest, charset, max_label_length=max_label_length)
        validations[split] = result.to_dict()
        logger.info("%s", result.summary())
        all_ok = all_ok and result.ok

    # --- provenance --------------------------------------------------------
    info = {
        "name": args.dataset,
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "generator": "scripts/prepare_dataset.py",
        "source": str(args.source),
        "seed": args.seed,
        "counts_requested": {
            "train": args.train,
            "val": args.val,
            "test": args.test,
            "unlabeled": args.unlabeled,
        },
        "max_label_length": max_label_length,
        "charset": charset.to_dict(),
        "preparation": report.to_dict(),
        "validation": validations,
        "elapsed_seconds": round(elapsed, 1),
        "notes": report.notes,
    }
    info_path = output_dir / "dataset_info.json"
    info_path.write_text(json.dumps(info, indent=2, ensure_ascii=False), encoding="utf-8")

    logger.info("-" * 72)
    logger.info("%s", report.summary())
    logger.info("  label length : %d-%d (mean %.1f)", report.label_length_min, report.label_length_max, report.label_length_mean)
    logger.info("  characters   : %r", report.observed_characters)
    if report.dropped_characters:
        logger.info("  dropped chars: %s", dict(list(report.dropped_characters.items())[:15]))
    logger.info("  elapsed      : %.1f s", elapsed)
    logger.info("  provenance   : %s", relative_to_root(info_path))
    logger.info("-" * 72)
    logger.info(
        "To run the experiment on this dataset, point a config's data.* paths at "
        "%s and re-run pretrain_simclr.py then train_crnn.py.",
        relative_to_root(output_dir),
    )
    return 0 if all_ok else 2


if __name__ == "__main__":
    raise SystemExit(main())
