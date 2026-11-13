"""Self-supervised SimCLR pretrainer.

Stage 1 of the research experiment.  Trains the CNN encoder on the *unlabeled*
image pool with the NT-Xent contrastive objective, then writes a standalone
``encoder_best.pt`` that the supervised trainer loads with
:meth:`~ml.models.crnn_attention.CRNNAttentionOCR.load_encoder_weights`.

Reading the logs
----------------
The loss alone is hard to interpret because its scale depends on the batch
size (``log(2B - 1)`` is chance level).  Watch ``train_acc`` instead -- the
fraction of anchors whose nearest neighbour is its true positive.  It starts
near ``1 / (2B - 1)`` and should climb steadily; if it jumps to ~1.0 within an
epoch the views are too easy to match and the representation will not transfer.
"""

from __future__ import annotations

import logging
import math
import time
from pathlib import Path
from typing import Any, Mapping

import torch
from torch.utils.data import DataLoader

from ml.datasets.ocr_dataset import ImageGeometry, SimCLRBatch
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
from ml.engine.ctc_trainer import TrainerCallbacks
from ml.models.simclr import NTXentLoss, SimCLRModel
from ml.utils.device import autocast_dtype
from ml.utils.paths import relative_to_root
from ml.utils.timing import stopwatch

__all__ = ["SimCLRTrainer"]

logger = logging.getLogger(__name__)

_LOSS_SANITY_CEILING = 1e4


class SimCLRTrainer:
    """Contrastive pretraining loop.

    Args:
        model: the SimCLR wrapper (encoder + projection head).
        device: target device.
        settings: optimisation settings.
        temperature: NT-Xent temperature.
        run_name: used for the history file and logs.
        geometry: input geometry, recorded in the checkpoint.
        callbacks: progress/cancellation hooks.
        config_snapshot: resolved config, stored in the checkpoint.
    """

    def __init__(
        self,
        model: SimCLRModel,
        *,
        device: torch.device,
        settings: TrainSettings,
        temperature: float = 0.1,
        run_name: str = "simclr_pretrain",
        geometry: ImageGeometry | None = None,
        callbacks: TrainerCallbacks | None = None,
        config_snapshot: Mapping[str, Any] | None = None,
    ) -> None:
        self.model = model.to(device)
        self.device = device
        self.settings = settings
        self.run_name = run_name
        self.geometry = geometry or ImageGeometry()
        self.callbacks = callbacks or TrainerCallbacks()
        self.config_snapshot = dict(config_snapshot or {})

        self.criterion = NTXentLoss(temperature=temperature)
        self.optimizer = build_optimizer(self.model.parameters(), settings)
        self.scheduler = build_scheduler(self.optimizer, settings)

        self.amp_enabled = bool(settings.amp) and device.type == "cuda"
        self.amp_dtype = autocast_dtype(device) if self.amp_enabled else torch.float32
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
        """Run contrastive pretraining.

        Raises:
            ValueError: the loader's batch size is below 2, which leaves
                NT-Xent with no negatives.
        """
        total_epochs = int(epochs or self.settings.epochs)
        history_path = self.checkpoints.directory / "history.json"

        batch_size = int(train_loader.batch_size or self.settings.batch_size)
        if batch_size < 2:
            raise ValueError(
                f"SimCLR needs a batch size of at least 2 to have negatives, got {batch_size}"
            )

        chance = 1.0 / max(1, 2 * batch_size - 1)
        self.history.notes.update(
            {
                "unlabeled_samples": len(train_loader.dataset),  # type: ignore[arg-type]
                "batch_size": batch_size,
                "temperature": self.criterion.temperature,
                "chance_accuracy": round(chance, 6),
                "amp": self.amp_enabled,
                "amp_dtype": str(self.amp_dtype).replace("torch.", ""),
                "architecture": self.model.describe(),
                "geometry": self.geometry.to_dict(),
            }
        )

        logger.info(
            "SimCLR pretraining %r: %d unlabeled image(s), %d epoch(s), batch=%d, "
            "temperature=%.2f, chance accuracy=%.4f",
            self.run_name,
            self.history.notes["unlabeled_samples"],
            total_epochs,
            batch_size,
            self.criterion.temperature,
            chance,
        )

        stopped_early = False
        try:
            for epoch in range(1, total_epochs + 1):
                with stopwatch(self.device) as epoch_timer:
                    metrics = self._train_epoch(train_loader, epoch, total_epochs)
                    if val_loader is not None:
                        metrics.update(self._validate(val_loader))

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
            logger.warning("Pretraining run %r was cancelled; saving progress so far", self.run_name)
            self.history.notes["cancelled"] = True
            stopped_early = True

        self.history.stopped_early = stopped_early
        self.history.finished_at = time.time()
        if self._skipped_batches:
            self.history.notes["skipped_batches"] = self._skipped_batches
        self.history.save(history_path)

        logger.info(
            "Finished %r in %.1fs | encoder written to %s",
            self.run_name,
            self.history.total_duration_s,
            relative_to_root(self.encoder_checkpoint),
        )
        return self.history

    # -- training -----------------------------------------------------------
    def _train_epoch(self, loader: DataLoader, epoch: int, total_epochs: int) -> dict[str, float]:
        self.model.train()
        loss_meter = AverageMeter()
        accuracy_meter = AverageMeter()
        positive_meter = AverageMeter()
        negative_meter = AverageMeter()
        total_steps = len(loader)

        for step, batch in enumerate(loader, start=1):
            if self.callbacks.should_stop is not None and self.callbacks.should_stop():
                raise TrainingStopped

            result = self._train_step(batch)
            if result is not None:
                loss_value, accuracy, positive, negative = result
                size = len(batch)
                loss_meter.update(loss_value, size)
                accuracy_meter.update(accuracy, size)
                positive_meter.update(positive, size)
                negative_meter.update(negative, size)

            if self.callbacks.on_batch_end is not None:
                self.callbacks.on_batch_end(
                    epoch, step, total_steps, result[0] if result else float("nan")
                )

            if self.settings.log_every and step % self.settings.log_every == 0:
                logger.debug(
                    "  epoch %d/%d step %d/%d loss=%.4f acc=%.4f",
                    epoch,
                    total_epochs,
                    step,
                    total_steps,
                    loss_meter.average,
                    accuracy_meter.average,
                )

        return {
            "train_loss": loss_meter.average,
            "train_acc": accuracy_meter.average,
            "positive_similarity": positive_meter.average,
            "negative_similarity": negative_meter.average,
        }

    def _train_step(self, batch: SimCLRBatch) -> tuple[float, float, float, float] | None:
        batch = batch.to(self.device, non_blocking=True)
        self.optimizer.zero_grad(set_to_none=True)

        with torch.autocast(
            device_type=self.device.type, dtype=self.amp_dtype, enabled=self.amp_enabled
        ):
            z_i, z_j = self.model(batch.view_i, batch.view_j)

        # NTXentLoss casts to float32 internally: the similarity matrix divided
        # by a small temperature overflows in half precision.
        outcome = self.criterion(z_i, z_j)
        loss_value = float(outcome.loss.detach())

        if not math.isfinite(loss_value) or loss_value > _LOSS_SANITY_CEILING:
            self._skipped_batches += 1
            logger.debug("Skipping a batch with loss=%s", loss_value)
            return None

        self.scaler.scale(outcome.loss).backward()

        if self.settings.grad_clip and self.settings.grad_clip > 0:
            self.scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.settings.grad_clip)

        self.scaler.step(self.optimizer)
        self.scaler.update()

        return (
            loss_value,
            outcome.accuracy,
            outcome.positive_similarity,
            outcome.negative_similarity,
        )

    @torch.no_grad()
    def _validate(self, loader: DataLoader) -> dict[str, float]:
        """Contrastive loss on a held-out slice of the unlabeled pool.

        Not a downstream metric -- it only says whether the encoder still
        separates pairs it has never seen, which is a useful check that the
        model is not simply memorising the training crops.
        """
        self.model.eval()
        loss_meter = AverageMeter()
        accuracy_meter = AverageMeter()

        for batch in loader:
            batch = batch.to(self.device, non_blocking=True)
            if len(batch) < 2:
                continue

            with torch.autocast(
                device_type=self.device.type, dtype=self.amp_dtype, enabled=self.amp_enabled
            ):
                z_i, z_j = self.model(batch.view_i, batch.view_j)

            outcome = self.criterion(z_i, z_j)
            loss_meter.update(float(outcome.loss.detach()), len(batch))
            accuracy_meter.update(outcome.accuracy, len(batch))

        return {"val_loss": loss_meter.average, "val_acc": accuracy_meter.average}

    # -- checkpointing ------------------------------------------------------
    def _save(self, epoch: int, metrics: Mapping[str, float]) -> bool:
        payload = {
            "model_state_dict": self.model.state_dict(),
            "encoder_state_dict": self.model.encoder_state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "architecture": self.model.describe(),
            "geometry": self.geometry.to_dict(),
            "config": self.config_snapshot,
            "run_name": self.run_name,
            "settings": self.settings.to_dict(),
            "temperature": self.criterion.temperature,
        }

        # The encoder is also written on its own, with no projection head and
        # no optimiser state: that is the artefact the supervised stage loads,
        # and keeping it small and self-contained makes the transfer explicit.
        encoder_only = {
            "encoder_state_dict": self.model.encoder_state_dict(),
            "architecture": self.model.describe(),
            "geometry": self.geometry.to_dict(),
            "epoch": epoch,
            "metrics": dict(metrics),
            "run_name": self.run_name,
            "source": "simclr_self_supervised_pretraining",
        }

        return self.checkpoints.save(
            payload,
            epoch=epoch,
            metrics=dict(metrics),
            extra_best_files={"encoder_best.pt": encoder_only},
        )

    @property
    def encoder_checkpoint(self) -> Path:
        """Path of the standalone encoder written for the supervised stage."""
        return self.checkpoints.directory / "encoder_best.pt"
