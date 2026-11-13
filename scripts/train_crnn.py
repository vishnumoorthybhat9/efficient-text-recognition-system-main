"""Stage 2: supervised CTC training of the CNN + BiLSTM + Attention recogniser.

Runs one arm of the controlled experiment.  Which arm is decided entirely by
``model.init_encoder_from`` in the config:

``baseline_scratch.yaml``
    ``init_encoder_from: null`` -- the encoder starts from random weights.
``simclr_finetune.yaml``
    ``init_encoder_from: artifacts/checkpoints/simclr_pretrain/encoder_best.pt``

Everything else -- the labeled subset, the seed, the optimiser, the schedule,
the augmentation -- is identical between the two, which is what makes the
difference in final CER attributable to the pretraining and nothing else.

Usage
-----
::

    python scripts/train_crnn.py --config configs/experiments/baseline_scratch.yaml
    python scripts/train_crnn.py --config configs/experiments/simclr_finetune.yaml

    # train on the full labeled set as an upper reference point
    python scripts/train_crnn.py --config configs/default.yaml \\
        --run-name crnn_full --max-samples 0
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import Sequence

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts._bootstrap import add_common_arguments, bootstrap, log_hardware  # noqa: E402

from ml.datasets.manifest import validate_manifest  # noqa: E402
from ml.datasets.ocr_dataset import (  # noqa: E402
    ImageGeometry,
    OCRLineDataset,
    build_ocr_dataloader,
)
from ml.engine.common import TrainSettings  # noqa: E402
from ml.engine.ctc_trainer import CTCTrainer  # noqa: E402
from ml.models.cnn_transformer_ocr import CNNTransformerOCR  # noqa: E402
from ml.models.crnn_attention import CRNNAttentionOCR  # noqa: E402
from ml.models.mamba_ocr import MambaOCR  # noqa: E402
from ml.preprocessing.augment import AugmentOptions  # noqa: E402
from ml.preprocessing.pipeline import PreprocessOptions  # noqa: E402
from ml.text.charset import Charset  # noqa: E402
from ml.utils.paths import ensure_dir, relative_to_root, resolve_path  # noqa: E402

logger = logging.getLogger("train_crnn")


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train a configured CTC-based OCR recogniser.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    add_common_arguments(parser)
    parser.add_argument(
        "--init-encoder-from",
        default=None,
        help="path to a SimCLR encoder checkpoint (overrides model.init_encoder_from)",
    )
    parser.add_argument(
        "--freeze-encoder-epochs",
        type=int,
        default=None,
        help="hold the transferred encoder fixed for this many epochs",
    )
    parser.add_argument(
        "--require-encoder",
        action="store_true",
        help="fail instead of warning if the encoder checkpoint is missing",
    )
    parser.add_argument(
        "--skip-validation-check",
        action="store_true",
        help="skip the manifest sanity check (saves a few seconds on a large set)",
    )
    parser.add_argument(
        "--max-val-samples",
        type=int,
        default=None,
        help="cap validation samples for quick comparative runs",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)

    run_name = args.run_name or "crnn"
    log_dir = ensure_dir("artifacts/logs")
    boot = bootstrap(args, log_file=log_dir / f"train_{run_name}.log")
    config, device = boot.config, boot.device
    run_name = str(config.get("name", run_name))

    hardware = log_hardware(device)

    settings = TrainSettings.from_config(config)
    geometry = ImageGeometry.from_config(config.image)
    charset = Charset.from_config(config.charset)

    # --- data ---------------------------------------------------------------
    train_manifest = resolve_path(config.get("data.train_manifest"))
    val_manifest = resolve_path(config.get("data.val_manifest"))

    for name, path in (("train", train_manifest), ("val", val_manifest)):
        if path is None or not path.is_file():
            logger.error(
                "%s manifest not found at %s.\nGenerate the dataset first:\n"
                "  python scripts/make_synthetic_data.py",
                name,
                relative_to_root(path) if path else f"data.{name}_manifest",
            )
            return 1

    # `max_train_samples: 0` from the CLI means "use everything".
    max_train = config.get("data.max_train_samples")
    if max_train is not None and int(max_train) <= 0:
        max_train = None

    # The encoder emits width//4 - 1 time steps; a label longer than that
    # cannot be produced by CTC at all, so catch it now rather than as an
    # `inf` loss thirty seconds into training.
    time_steps = max(1, geometry.width // 4 - 1)
    if not args.skip_validation_check:
        report = validate_manifest(train_manifest, charset, max_label_length=time_steps)
        logger.info("Train manifest: %s", report.summary())
        if report.too_long:
            logger.warning(
                "%d training label(s) are longer than the %d available CTC time steps and "
                "will contribute no usable gradient (zero_infinity drops them).",
                len(report.too_long),
                time_steps,
            )

    preprocess = PreprocessOptions.from_config(config.get("preprocess"))
    augment = AugmentOptions.from_config(config.get("augment"))

    train_dataset = OCRLineDataset(
        charset=charset,
        manifest=train_manifest,
        geometry=geometry,
        preprocess=preprocess,
        augment=augment,
        train=True,
        max_samples=int(max_train) if max_train else None,
        seed=boot.seed,
    )
    val_dataset = OCRLineDataset(
        charset=charset,
        manifest=val_manifest,
        geometry=geometry,
        preprocess=preprocess,
        train=False,
        max_samples=args.max_val_samples,
        seed=boot.seed,
    )

    train_loader = build_ocr_dataloader(
        train_dataset,
        batch_size=settings.batch_size,
        num_workers=settings.num_workers,
        device=device,
    )
    val_loader = build_ocr_dataloader(
        val_dataset,
        batch_size=settings.batch_size,
        num_workers=settings.num_workers,
        device=device,
    )

    # --- model --------------------------------------------------------------
    model_family = str(config.get("model.family", "crnn_attention"))
    if model_family == "mamba_ocr":
        model = MambaOCR.from_config(config, charset)
    elif model_family == "cnn_transformer":
        model = CNNTransformerOCR.from_config(config, charset)
    elif model_family == "crnn_attention":
        model = CRNNAttentionOCR.from_config(config, charset)
    else:
        logger.error("Unsupported OCR model family %r.", model_family)
        return 1

    breakdown = model.parameter_breakdown()
    logger.info(
        "Recogniser: %s | %.2f M parameters (%s)",
        model.describe()["architecture"],
        breakdown["total"] / 1e6,
        ", ".join(f"{key} {value / 1e6:.2f} M" for key, value in breakdown.items() if key != "total"),
    )

    # --- transfer the self-supervised encoder, if this is the treatment arm --
    encoder_source = args.init_encoder_from or config.get("model.init_encoder_from")
    transfer_report = None

    if encoder_source:
        encoder_path = resolve_path(str(encoder_source))
        if encoder_path is None or not encoder_path.is_file():
            message = (
                f"Encoder checkpoint not found at {relative_to_root(encoder_path) if encoder_path else encoder_source}.\n"
                "Run the self-supervised stage first:\n"
                "  python scripts/pretrain_simclr.py --config configs/experiments/simclr_pretrain.yaml"
            )
            if args.require_encoder:
                logger.error(message)
                return 1
            logger.error(
                "%s\n\nRefusing to continue: this config is the SimCLR arm of the comparison, "
                "and training it from random weights would silently produce a duplicate of "
                "the control arm. Pass --init-encoder-from to point elsewhere, or use "
                "configs/experiments/baseline_scratch.yaml if you meant to train from scratch.",
                message,
            )
            return 1

        transfer_report = model.load_encoder_weights(encoder_path)
        if transfer_report.fraction < 1.0:
            logger.warning(
                "Only %.0f%% of the encoder transferred. The comparison is still valid but "
                "weaker than intended; check that both stages use the same crnn.encoder config.",
                transfer_report.fraction * 100,
            )
    else:
        logger.info("No encoder checkpoint configured: training the encoder from scratch (control arm).")

    freeze_epochs = int(
        args.freeze_encoder_epochs
        if args.freeze_encoder_epochs is not None
        else (config.get("model.freeze_encoder_epochs", 0) or 0)
    )
    if freeze_epochs and not encoder_source:
        logger.warning(
            "freeze_encoder_epochs=%d was requested but no pretrained encoder was loaded; "
            "freezing random weights would cripple the run, so it is ignored.",
            freeze_epochs,
        )
        freeze_epochs = 0

    # --- train --------------------------------------------------------------
    snapshot = config.to_dict()
    snapshot["_hardware"] = hardware

    trainer = CTCTrainer(
        model,
        charset,
        device=device,
        settings=settings,
        run_name=run_name,
        geometry=geometry,
        freeze_encoder_epochs=freeze_epochs,
        config_snapshot=snapshot,
    )
    trainer.history.notes["labeled_subset_size"] = len(train_dataset)
    trainer.history.notes["encoder_transfer"] = (
        transfer_report.to_dict() if transfer_report else None
    )
    # Describe how the encoder was initialised, and state the experiment role
    # only when the config actually claims one. Deriving "treatment" purely
    # from the presence of a pretrained encoder was wrong: the production
    # model (`crnn_full`) also starts from the SimCLR encoder but is not part
    # of the controlled comparison at all, and labelling it an arm would
    # misrepresent the experiment.
    initialisation = (
        "SimCLR-pretrained encoder" if encoder_source else "encoder trained from scratch"
    )
    declared_arm = config.get("model.experiment_arm")
    trainer.history.notes["encoder_init"] = initialisation
    trainer.history.notes["experiment_arm"] = (
        f"{declared_arm} ({initialisation})" if declared_arm else f"not an experiment arm ({initialisation})"
    )

    history = trainer.fit(train_loader, val_loader)

    # --- report -------------------------------------------------------------
    best = {}
    for record in history.records:
        if record.epoch == history.best_epoch:
            best = record.metrics
            break
    if not best and history.records:
        best = history.records[-1].metrics

    logger.info("-" * 72)
    logger.info("Training complete: %s", run_name)
    logger.info("  arm              : %s", trainer.history.notes["experiment_arm"])
    if transfer_report:
        logger.info("  encoder transfer : %s", transfer_report.summary())
    logger.info("  labeled samples  : %d train, %d val", len(train_dataset), len(val_dataset))
    logger.info("  epochs run       : %d (best epoch %s)", len(history.records), history.best_epoch)
    logger.info("  val CER          : %.4f", best.get("val_cer", float("nan")))
    logger.info("  val WER          : %.4f", best.get("val_wer", float("nan")))
    logger.info("  val exact match  : %.4f", best.get("val_accuracy", float("nan")))
    logger.info("  wall clock       : %.1f s", history.total_duration_s)
    logger.info("  checkpoint       : %s", relative_to_root(trainer.best_checkpoint))
    logger.info("-" * 72)
    logger.info(
        "Next, benchmark it on the held-out test split:\n"
        "  python scripts/evaluate.py --models %s --split test",
        run_name,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
