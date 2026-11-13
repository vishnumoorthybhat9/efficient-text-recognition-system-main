"""Stage 1: SimCLR self-supervised pretraining of the CNN encoder.

Trains on the **unlabeled** image pool only -- no transcription is read at any
point.  The product is ``encoder_best.pt``, a bare encoder ``state_dict`` that
``scripts/train_crnn.py`` loads into the recogniser.

Usage
-----
::

    python scripts/pretrain_simclr.py --config configs/experiments/simclr_pretrain.yaml

    # faster, for a smoke test
    python scripts/pretrain_simclr.py --config configs/experiments/simclr_pretrain.yaml \\
        --epochs 2 --max-unlabeled 500

What to watch
-------------
``train_acc`` -- the fraction of anchors whose nearest neighbour is its true
positive.  It starts near ``1 / (2B - 1)`` (printed at start-up as the chance
level) and should rise steadily.  A loss that falls while ``train_acc`` stays
at chance means the model is collapsing to a constant representation.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import Sequence

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts._bootstrap import add_common_arguments, bootstrap, log_hardware  # noqa: E402

from ml.datasets.manifest import read_unlabeled_manifest  # noqa: E402
from ml.datasets.ocr_dataset import (  # noqa: E402
    ImageGeometry,
    UnlabeledLineDataset,
    build_simclr_dataloader,
)
from ml.engine.common import TrainSettings  # noqa: E402
from ml.engine.simclr_trainer import SimCLRTrainer  # noqa: E402
from ml.models.simclr import SimCLRModel  # noqa: E402
from ml.preprocessing.augment import SimCLRViewOptions  # noqa: E402
from ml.preprocessing.pipeline import PreprocessOptions  # noqa: E402
from ml.utils.paths import ensure_dir, relative_to_root, resolve_path  # noqa: E402

logger = logging.getLogger("pretrain_simclr")

#: Fraction of the unlabeled pool held out to check the encoder still separates
#: pairs it has not been trained on.
VALIDATION_FRACTION = 0.05


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="SimCLR self-supervised pretraining of the OCR CNN encoder.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    add_common_arguments(parser)
    parser.add_argument(
        "--max-unlabeled",
        type=int,
        default=None,
        help="cap the unlabeled pool size (useful for a quick check)",
    )
    parser.add_argument(
        "--temperature", type=float, default=None, help="override simclr.temperature"
    )
    parser.add_argument(
        "--no-validation",
        action="store_true",
        help="train on the whole pool instead of holding 5%% out",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)

    log_dir = ensure_dir("artifacts/logs")
    boot = bootstrap(args, log_file=log_dir / "pretrain_simclr.log")
    config, device = boot.config, boot.device

    hardware = log_hardware(device)

    settings = TrainSettings.from_config(config)
    geometry = ImageGeometry.from_config(config.image)
    temperature = float(
        args.temperature if args.temperature is not None else config.get("simclr.temperature", 0.1)
    )

    # --- data ---------------------------------------------------------------
    manifest = resolve_path(config.get("data.unlabeled_manifest"))
    if manifest is None or not manifest.is_file():
        logger.error(
            "Unlabeled manifest not found at %s.\nGenerate the dataset first:\n"
            "  python scripts/make_synthetic_data.py",
            relative_to_root(manifest) if manifest else "data.unlabeled_manifest",
        )
        return 1

    paths = read_unlabeled_manifest(manifest, skip_missing=True)
    if args.max_unlabeled is not None and 0 < args.max_unlabeled < len(paths):
        paths = paths[: args.max_unlabeled]

    if len(paths) < settings.batch_size * 2:
        logger.error(
            "Only %d unlabeled image(s) available but the batch size is %d. "
            "Contrastive learning needs many more negatives than that; "
            "generate more data or lower --batch-size.",
            len(paths),
            settings.batch_size,
        )
        return 1

    holdout = 0 if args.no_validation else max(settings.batch_size, int(len(paths) * VALIDATION_FRACTION))
    train_paths = paths[holdout:] if holdout else paths
    val_paths = paths[:holdout] if holdout else []

    view_options = SimCLRViewOptions.from_config(config.get("simclr.views"))
    preprocess = PreprocessOptions.from_config(config.get("preprocess"))

    train_dataset = UnlabeledLineDataset(
        paths=train_paths,
        geometry=geometry,
        preprocess=preprocess,
        views=view_options,
        seed=boot.seed,
    )
    train_loader = build_simclr_dataloader(
        train_dataset,
        batch_size=settings.batch_size,
        num_workers=settings.num_workers,
        device=device,
    )

    val_loader = None
    if val_paths:
        val_dataset = UnlabeledLineDataset(
            paths=val_paths,
            geometry=geometry,
            preprocess=preprocess,
            views=view_options,
            seed=boot.seed + 1,
        )
        val_loader = build_simclr_dataloader(
            val_dataset,
            batch_size=settings.batch_size,
            num_workers=settings.num_workers,
            device=device,
        )

    # --- model --------------------------------------------------------------
    model = SimCLRModel.from_config(config)
    logger.info(
        "SimCLR model: %s encoder (%.2f M params) + projection head (%.2f M)",
        type(model.encoder).__name__,
        model.encoder.num_parameters() / 1e6,
        sum(p.numel() for p in model.projection.parameters()) / 1e6,
    )

    snapshot = config.to_dict()
    snapshot["_hardware"] = hardware
    snapshot["_augmentation"] = view_options.to_dict()

    trainer = SimCLRTrainer(
        model,
        device=device,
        settings=settings,
        temperature=temperature,
        run_name=str(config.get("name", "simclr_pretrain")),
        geometry=geometry,
        config_snapshot=snapshot,
    )
    trainer.history.notes["holdout_images"] = len(val_paths)
    trainer.history.notes["unlabeled_manifest"] = relative_to_root(manifest)

    history = trainer.fit(train_loader, val_loader)

    # --- report -------------------------------------------------------------
    encoder_path = trainer.encoder_checkpoint
    if not encoder_path.is_file():
        logger.error(
            "Pretraining finished but no encoder checkpoint was written. This happens "
            "when every epoch was skipped; check the log above for non-finite losses."
        )
        return 1

    first = history.records[0].metrics if history.records else {}
    last = history.records[-1].metrics if history.records else {}
    chance = history.notes.get("chance_accuracy", 0.0)

    logger.info("-" * 72)
    logger.info("SimCLR pretraining complete")
    logger.info("  unlabeled images : %d train, %d held out", len(train_paths), len(val_paths))
    logger.info("  epochs           : %d", len(history.records))
    logger.info(
        "  contrastive acc  : %.4f -> %.4f   (chance %.4f)",
        first.get("train_acc", 0.0),
        last.get("train_acc", 0.0),
        chance,
    )
    logger.info(
        "  NT-Xent loss     : %.4f -> %.4f", first.get("train_loss", 0.0), last.get("train_loss", 0.0)
    )
    if "val_acc" in last:
        logger.info("  held-out acc     : %.4f", last["val_acc"])
    logger.info("  wall clock       : %.1f s", history.total_duration_s)
    logger.info("  encoder written  : %s", relative_to_root(encoder_path))
    logger.info("-" * 72)

    if last.get("train_acc", 0.0) <= chance * 2:
        logger.warning(
            "Contrastive accuracy barely moved above chance. The transferred encoder is "
            "unlikely to help the supervised stage; consider more epochs or a larger batch."
        )

    logger.info(
        "Next, train both arms of the comparison:\n"
        "  python scripts/train_crnn.py --config configs/experiments/baseline_scratch.yaml\n"
        "  python scripts/train_crnn.py --config configs/experiments/simclr_finetune.yaml"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
