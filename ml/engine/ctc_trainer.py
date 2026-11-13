"""Supervised CTC trainer for :class:`~ml.models.crnn_attention.CRNNAttentionOCR`.

Drives one arm of the research experiment.  The *only* thing that differs
between the two arms is whether the encoder was initialised from a SimCLR
checkpoint (``model.init_encoder_from`` in the experiment config); the optimiser,
schedule, augmentation, data subset and seed are identical, which is what makes
the resulting CER difference attributable to the pretraining.
"""

from __future__ import annotations

import logging
import math
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Mapping

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from ml.datasets.ocr_dataset import ImageGeometry, OCRBatch
from ml.decoding.ctc import greedy_decode
from ml.engine.common import (
    AverageMeter,
    CheckpointManager,
    EarlyStopping,
    EpochRecord,
    TrainSettings,
    TrainingHistory,
    TrainingStopped,
    build_optimizer,
    build_scheduler,
    current_learning_rate,
)
from ml.evaluation.metrics import evaluate_predictions
from ml.models.crnn_attention import CRNNAttentionOCR
from ml.models.cnn_transformer_ocr import CNNTransformerOCR
from ml.models.mamba_ocr import MambaOCR
from ml.text.charset import Charset
from ml.utils.device import autocast_dtype
from ml.utils.paths import relative_to_root
from ml.utils.timing import stopwatch

__all__ = ["CTCTrainer", "TrainerCallbacks"]

logger = logging.getLogger(__name__)

#: A CTC loss above this is treated as a divergence and the batch is skipped.
_LOSS_SANITY_CEILING = 1e4


@dataclass(slots=True)
class TrainerCallbacks:
    """Hooks used by the API's training-job service.

    Attributes:
        on_epoch_end: called with ``(epoch, metrics)`` after every epoch.
        on_batch_end: called with ``(epoch, step, total_steps, loss)``.
        should_stop: polled once per batch; returning True cancels the run
            cleanly, writing ``last.pt`` and the history first.
    """

    on_epoch_end: Callable[[int, dict[str, float]], None] | None = None
    on_batch_end: Callable[[int, int, int, float], None] | None = None
    should_stop: Callable[[], bool] | None = None


@dataclass(slots=True)
class ValidationOutcome:
    """Validation metrics plus a few example predictions."""

    metrics: dict[str, float]
    examples: list[dict[str, Any]] = field(default_factory=list)


class CTCTrainer:
    """Trains a CRNN+Attention recogniser with the CTC objective.

    Args:
        model: the recogniser.
        charset: must be the charset the model's head was sized for.
        device: target device.
        settings: optimisation settings.
        run_name: used for the history file and log messages.
        geometry: input geometry, recorded in the checkpoint so inference can
            reproduce it without the config file.
        freeze_encoder_epochs: hold the encoder fixed for this many epochs.
        callbacks: progress/cancellation hooks.
        config_snapshot: the full resolved config, stored in the checkpoint.

    Raises:
        ValueError: the model's output size does not match *charset*.
    """

    def __init__(
        self,
        model: CRNNAttentionOCR | CNNTransformerOCR | MambaOCR,
        charset: Charset,
        *,
        device: torch.device,
        settings: TrainSettings,
        run_name: str = "crnn",
        geometry: ImageGeometry | None = None,
        freeze_encoder_epochs: int = 0,
        callbacks: TrainerCallbacks | None = None,
        config_snapshot: Mapping[str, Any] | None = None,
    ) -> None:
        if model.num_classes != charset.num_classes:
            raise ValueError(
                f"Model has {model.num_classes} output classes but the charset needs "
                f"{charset.num_classes}. They must match or decoding is meaningless."
            )

        self.model = model.to(device)
        self.charset = charset
        self.device = device
        self.settings = settings
        self.run_name = run_name
        self.geometry = geometry or ImageGeometry()
        self.freeze_encoder_epochs = max(0, int(freeze_encoder_epochs))
        self.callbacks = callbacks or TrainerCallbacks()
        self.config_snapshot = dict(config_snapshot or {})

        # zero_infinity is essential: a single sample whose label is longer
        # than the available time steps would otherwise return `inf` and
        # destroy every gradient in the batch.
        self.criterion = nn.CTCLoss(blank=charset.blank_index, reduction="mean", zero_infinity=True)

        self.optimizer = build_optimizer(self.model.parameters(), settings)
        self.scheduler = build_scheduler(self.optimizer, settings)

        self.amp_enabled = bool(settings.amp) and device.type == "cuda"
        self.amp_dtype = autocast_dtype(device) if self.amp_enabled else torch.float32
        # A gradient scaler is only needed for float16; bfloat16 has the same
        # exponent range as float32 and does not underflow.
        self.scaler = torch.amp.GradScaler(
            device.type, enabled=self.amp_enabled and self.amp_dtype == torch.float16
        )

        self.checkpoints = CheckpointManager(
            Path(settings.checkpoint_dir) / run_name,
            monitor=settings.monitor,
            mode=settings.monitor_mode,
        )
        self.early_stopping = EarlyStopping(
            patience=settings.early_stopping_patience, mode=settings.monitor_mode
        )

        self.history = TrainingHistory(
            run_name=run_name,
            monitor=settings.monitor,
            monitor_mode=settings.monitor_mode,
            device=str(device),
            config=self.config_snapshot,
        )
        self._skipped_batches = 0

    # -- public API ---------------------------------------------------------
    def fit(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader | None = None,
        *,
        epochs: int | None = None,
    ) -> TrainingHistory:
        """Run the training loop.

        Returns:
            The :class:`TrainingHistory`, also written to
            ``<checkpoint_dir>/history.json`` after every epoch.
        """
        total_epochs = int(epochs or self.settings.epochs)
        history_path = self.checkpoints.directory / "history.json"

        self.history.notes.update(
            {
                "train_samples": len(train_loader.dataset),  # type: ignore[arg-type]
                "val_samples": len(val_loader.dataset) if val_loader is not None else 0,  # type: ignore[arg-type]
                "batch_size": self.settings.batch_size,
                "amp": self.amp_enabled,
                "amp_dtype": str(self.amp_dtype).replace("torch.", ""),
                "freeze_encoder_epochs": self.freeze_encoder_epochs,
                "parameters": self.model.parameter_breakdown(),
                "architecture": self.model.describe(),
                "charset": self.charset.to_dict(),
                "geometry": self.geometry.to_dict(),
            }
        )

        logger.info(
            "Training %r for %d epoch(s) on %s | %d train / %d val sample(s) | AMP=%s",
            self.run_name,
            total_epochs,
            self.device,
            self.history.notes["train_samples"],
            self.history.notes["val_samples"],
            f"{self.amp_dtype}".replace("torch.", "") if self.amp_enabled else "off",
        )

        if self.freeze_encoder_epochs > 0:
            self.model.freeze_encoder(True)

        stopped_early = False
        try:
            for epoch in range(1, total_epochs + 1):
                self._maybe_unfreeze(epoch)

                with stopwatch(self.device) as epoch_timer:
                    metrics = self._train_epoch(train_loader, epoch, total_epochs)

                    if val_loader is not None:
                        outcome = self.validate(val_loader)
                        metrics.update(outcome.metrics)
                        self.history.notes["last_examples"] = outcome.examples

                if self.scheduler is not None:
                    self.scheduler.step()

                record = EpochRecord(
                    epoch=epoch,
                    metrics=metrics,
                    learning_rate=current_learning_rate(self.optimizer),
                    duration_s=epoch_timer.elapsed_ms / 1000.0,
                )
                self.history.append(record)

                self._save(epoch, metrics)
                self.history.best_epoch = self.checkpoints.best_epoch
                self.history.best_value = self.checkpoints.best_value
                self.history.save(history_path)

                logger.info(
                    "epoch %d/%d | %s | lr=%.2e | %.1fs",
                    epoch,
                    total_epochs,
                    "  ".join(f"{k}={v:.4f}" for k, v in metrics.items()),
                    record.learning_rate,
                    record.duration_s,
                )

                if self.callbacks.on_epoch_end is not None:
                    self.callbacks.on_epoch_end(epoch, metrics)

                monitored = metrics.get(self.settings.monitor)
                if monitored is not None and self.early_stopping.step(float(monitored)):
                    stopped_early = True
                    break

        except TrainingStopped:
            logger.warning("Training run %r was cancelled; saving progress so far", self.run_name)
            self.history.notes["cancelled"] = True
            stopped_early = True

        self.history.stopped_early = stopped_early
        self.history.finished_at = time.time()
        if self._skipped_batches:
            self.history.notes["skipped_batches"] = self._skipped_batches
            logger.warning(
                "Skipped %d batch(es) with a non-finite or implausible loss", self._skipped_batches
            )
        self.history.save(history_path)

        logger.info(
            "Finished %r in %.1fs | best %s=%s at epoch %s | checkpoints in %s",
            self.run_name,
            self.history.total_duration_s,
            self.settings.monitor,
            f"{self.checkpoints.best_value:.6f}" if self.checkpoints.best_value is not None else "n/a",
            self.checkpoints.best_epoch,
            relative_to_root(self.checkpoints.directory),
        )
        return self.history

    # -- training -----------------------------------------------------------
    def _maybe_unfreeze(self, epoch: int) -> None:
        if self.freeze_encoder_epochs and epoch == self.freeze_encoder_epochs + 1:
            self.model.freeze_encoder(False)
            # The optimiser was built over the then-trainable parameters only,
            # so it must be rebuilt to pick up the encoder. The schedule is
            # rebuilt with it and fast-forwarded to keep the planned decay.
            self.optimizer = build_optimizer(self.model.parameters(), self.settings)
            self.scheduler = build_scheduler(self.optimizer, self.settings)
            if self.scheduler is not None:
                for _ in range(epoch - 1):
                    self.scheduler.step()
            logger.info("Encoder unfrozen at epoch %d; optimiser rebuilt", epoch)

    def _train_epoch(self, loader: DataLoader, epoch: int, total_epochs: int) -> dict[str, float]:
        self.model.train()
        loss_meter = AverageMeter()
        total_steps = len(loader)

        for step, batch in enumerate(loader, start=1):
            if self.callbacks.should_stop is not None and self.callbacks.should_stop():
                raise TrainingStopped

            loss_value = self._train_step(batch)
            if loss_value is not None:
                loss_meter.update(loss_value, len(batch))

            if self.callbacks.on_batch_end is not None:
                self.callbacks.on_batch_end(epoch, step, total_steps, loss_value or float("nan"))

            if self.settings.log_every and step % self.settings.log_every == 0:
                logger.debug(
                    "  epoch %d/%d step %d/%d loss=%.4f",
                    epoch,
                    total_epochs,
                    step,
                    total_steps,
                    loss_meter.average,
                )

        return {"train_loss": loss_meter.average}

    def _train_step(self, batch: OCRBatch) -> float | None:
        """One optimisation step.  Returns the loss, or ``None`` if skipped."""
        batch = batch.to(self.device, non_blocking=True)
        self.optimizer.zero_grad(set_to_none=True)

        with torch.autocast(
            device_type=self.device.type, dtype=self.amp_dtype, enabled=self.amp_enabled
        ):
            if isinstance(self.model, (MambaOCR, CNNTransformerOCR)):
                time_steps = self.model.encoder.output_length(batch.images.size(-1))
                model_lengths = batch.input_lengths(time_steps)
                output = self.model(batch.images, input_lengths=model_lengths)
            else:
                output = self.model(batch.images)

        # The loss is computed outside autocast: `log_probs` is already float32
        # (the model casts it) and CTCLoss in half precision is unstable.
        input_lengths = batch.input_lengths(output.log_probs.size(1))
        loss = self.criterion(
            output.log_probs.permute(1, 0, 2),  # CTCLoss wants (T, B, C)
            batch.targets,
            input_lengths,
            batch.target_lengths,
        )

        loss_value = float(loss.detach())
        if not math.isfinite(loss_value) or loss_value > _LOSS_SANITY_CEILING:
            self._skipped_batches += 1
            logger.debug("Skipping a batch with loss=%s", loss_value)
            return None

        self.scaler.scale(loss).backward()

        if self.settings.grad_clip and self.settings.grad_clip > 0:
            # Unscale first, otherwise the clip threshold applies to scaled
            # gradients and effectively does nothing.
            self.scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.settings.grad_clip)

        self.scaler.step(self.optimizer)
        self.scaler.update()
        return loss_value

    # -- validation ---------------------------------------------------------
    @torch.no_grad()
    def validate(self, loader: DataLoader, *, max_examples: int = 12) -> ValidationOutcome:
        """Compute validation loss, CER, WER and exact-match accuracy."""
        self.model.eval()
        loss_meter = AverageMeter()
        references: list[str] = []
        predictions: list[str] = []
        confidences: list[float] = []
        examples: list[dict[str, Any]] = []

        for batch in loader:
            batch = batch.to(self.device, non_blocking=True)

            with torch.autocast(
                device_type=self.device.type, dtype=self.amp_dtype, enabled=self.amp_enabled
            ):
                if isinstance(self.model, (MambaOCR, CNNTransformerOCR)):
                    time_steps = self.model.encoder.output_length(batch.images.size(-1))
                    model_lengths = batch.input_lengths(time_steps)
                    output = self.model(batch.images, input_lengths=model_lengths)
                else:
                    output = self.model(batch.images)

            input_lengths = batch.input_lengths(output.log_probs.size(1))
            loss = self.criterion(
                output.log_probs.permute(1, 0, 2),
                batch.targets,
                input_lengths,
                batch.target_lengths,
            )
            loss_meter.update(float(loss.detach()), len(batch))

            decoded = greedy_decode(output.log_probs, self.charset, input_lengths=input_lengths)
            for index, result in enumerate(decoded):
                references.append(batch.texts[index])
                predictions.append(result.text)
                confidences.append(result.confidence)

                if len(examples) < max_examples:
                    examples.append(
                        {
                            "id": batch.sample_ids[index],
                            "reference": batch.texts[index],
                            "prediction": result.text,
                            "confidence": round(result.confidence, 4),
                        }
                    )

        aggregate, _ = evaluate_predictions(references, predictions, return_samples=False)

        metrics = {
            "val_loss": loss_meter.average,
            "val_cer": aggregate.cer,
            "val_wer": aggregate.wer,
            "val_accuracy": aggregate.sequence_accuracy,
            "val_confidence": sum(confidences) / len(confidences) if confidences else 0.0,
        }
        return ValidationOutcome(metrics=metrics, examples=examples)

    # -- checkpointing ------------------------------------------------------
    def _save(self, epoch: int, metrics: Mapping[str, float]) -> bool:
        payload = {
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "charset": self.charset.to_dict(),
            "geometry": self.geometry.to_dict(),
            "architecture": self.model.describe(),
            "config": self.config_snapshot,
            "run_name": self.run_name,
            "settings": self.settings.to_dict(),
        }
        return self.checkpoints.save(payload, epoch=epoch, metrics=dict(metrics))

    @property
    def best_checkpoint(self) -> Path:
        return self.checkpoints.best_path
