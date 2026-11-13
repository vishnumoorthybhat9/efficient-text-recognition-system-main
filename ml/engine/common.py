"""Shared training infrastructure: settings, optimisers, schedules, history.

Used by both :mod:`ml.engine.ctc_trainer` and :mod:`ml.engine.simclr_trainer`
so the two experiment arms are optimised identically -- anything that differed
here would confound the self-supervised comparison.
"""

from __future__ import annotations

import json
import logging
import math
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterable, Literal, Mapping

import torch
import torch.nn as nn
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler

from ml.utils.paths import ensure_dir, relative_to_root

__all__ = [
    "TrainSettings",
    "AverageMeter",
    "EpochRecord",
    "TrainingHistory",
    "EarlyStopping",
    "CheckpointManager",
    "build_optimizer",
    "build_scheduler",
    "TrainingStopped",
]

logger = logging.getLogger(__name__)

MonitorMode = Literal["min", "max"]


class TrainingStopped(Exception):
    """Raised internally when a run is cancelled through the API."""


@dataclass(slots=True)
class TrainSettings:
    """Optimisation settings, mirroring the ``train`` config block."""

    epochs: int = 20
    batch_size: int = 32
    optimizer: str = "adamw"
    learning_rate: float = 8e-4
    weight_decay: float = 1e-4
    scheduler: str = "cosine"
    warmup_epochs: int = 1
    grad_clip: float = 5.0
    amp: bool = True
    seed: int = 42
    num_workers: int = 0
    log_every: int = 20
    early_stopping_patience: int = 6
    checkpoint_dir: str = "artifacts/checkpoints"
    monitor: str = "val_cer"
    monitor_mode: MonitorMode = "min"

    def __post_init__(self) -> None:
        if self.epochs < 1:
            raise ValueError(f"train.epochs must be >= 1, got {self.epochs}")
        if self.batch_size < 1:
            raise ValueError(f"train.batch_size must be >= 1, got {self.batch_size}")
        if self.learning_rate <= 0:
            raise ValueError(f"train.learning_rate must be > 0, got {self.learning_rate}")
        if self.monitor_mode not in ("min", "max"):
            raise ValueError(f"train.monitor_mode must be 'min' or 'max', got {self.monitor_mode!r}")
        self.warmup_epochs = max(0, min(int(self.warmup_epochs), int(self.epochs)))

    @classmethod
    def from_config(cls, cfg: Any) -> "TrainSettings":
        """Build from a loaded :class:`ml.config.Config`."""
        block = cfg.get("train", {}) if hasattr(cfg, "get") else cfg
        data = dict(block) if isinstance(block, Mapping) else dict(getattr(block, "to_dict", dict)())
        known = {f for f in cls.__dataclass_fields__}  # type: ignore[attr-defined]
        unknown = set(data) - known
        if unknown:
            logger.debug("Ignoring unknown train keys: %s", sorted(unknown))
        return cls(**{k: v for k, v in data.items() if k in known})

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class AverageMeter:
    """Running mean of a scalar, weighted by batch size.

    Weighting matters for the final partial batch: an unweighted mean would
    give its samples more influence than the others.
    """

    __slots__ = ("total", "count")

    def __init__(self) -> None:
        self.total = 0.0
        self.count = 0

    def update(self, value: float, n: int = 1) -> None:
        if not math.isfinite(value):
            # Skip rather than poison the average; the caller logs the event.
            return
        self.total += float(value) * int(n)
        self.count += int(n)

    @property
    def average(self) -> float:
        return self.total / self.count if self.count else 0.0

    def reset(self) -> None:
        self.total = 0.0
        self.count = 0


@dataclass(slots=True)
class EpochRecord:
    """Metrics for one epoch."""

    epoch: int
    metrics: dict[str, float]
    learning_rate: float
    duration_s: float
    timestamp: float = field(default_factory=time.time)

    def to_dict(self) -> dict[str, Any]:
        return {
            "epoch": self.epoch,
            "learning_rate": self.learning_rate,
            "duration_s": round(self.duration_s, 3),
            "timestamp": self.timestamp,
            **{k: (round(v, 6) if isinstance(v, float) else v) for k, v in self.metrics.items()},
        }


@dataclass(slots=True)
class TrainingHistory:
    """Full record of a run, serialised to ``history.json``.

    This file is what the evaluation dashboard reads to draw its loss and CER
    curves, so it is written after *every* epoch rather than at the end -- an
    interrupted run still yields a usable curve.
    """

    run_name: str
    config: dict[str, Any] = field(default_factory=dict)
    records: list[EpochRecord] = field(default_factory=list)
    best_epoch: int | None = None
    best_value: float | None = None
    monitor: str = "val_loss"
    monitor_mode: MonitorMode = "min"
    device: str = "cpu"
    started_at: float = field(default_factory=time.time)
    finished_at: float | None = None
    stopped_early: bool = False
    notes: dict[str, Any] = field(default_factory=dict)

    def append(self, record: EpochRecord) -> None:
        self.records.append(record)

    @property
    def total_duration_s(self) -> float:
        end = self.finished_at if self.finished_at is not None else time.time()
        return max(0.0, end - self.started_at)

    def series(self, key: str) -> list[float | None]:
        """Values of *key* across epochs, with ``None`` where it is absent."""
        return [record.metrics.get(key) for record in self.records]

    def to_dict(self) -> dict[str, Any]:
        return {
            "run_name": self.run_name,
            "monitor": self.monitor,
            "monitor_mode": self.monitor_mode,
            "best_epoch": self.best_epoch,
            "best_value": round(self.best_value, 6) if self.best_value is not None else None,
            "device": self.device,
            "started_at": self.started_at,
            "finished_at": self.finished_at,
            "total_duration_s": round(self.total_duration_s, 3),
            "stopped_early": self.stopped_early,
            "epochs_completed": len(self.records),
            "config": self.config,
            "notes": self.notes,
            "epochs": [record.to_dict() for record in self.records],
        }

    def save(self, path: str | Path) -> Path:
        file = Path(path)
        file.parent.mkdir(parents=True, exist_ok=True)
        with file.open("w", encoding="utf-8") as handle:
            json.dump(self.to_dict(), handle, indent=2, ensure_ascii=False)
        return file

    @classmethod
    def load(cls, path: str | Path) -> dict[str, Any]:
        """Read a saved history back as a plain dict (for the API)."""
        with Path(path).open("r", encoding="utf-8") as handle:
            return json.load(handle)


class EarlyStopping:
    """Stop when the monitored metric has not improved for *patience* epochs.

    Args:
        patience: epochs to wait.  Values below 1 disable early stopping.
        mode: ``"min"`` for losses and error rates, ``"max"`` for accuracies.
        min_delta: improvement smaller than this does not count, which stops
            noise from resetting the counter indefinitely.
    """

    def __init__(self, *, patience: int = 6, mode: MonitorMode = "min", min_delta: float = 1e-5) -> None:
        self.patience = int(patience)
        self.mode: MonitorMode = mode
        self.min_delta = float(min_delta)
        self.best: float | None = None
        self.counter = 0
        self.enabled = self.patience >= 1

    def is_improvement(self, value: float) -> bool:
        if self.best is None:
            return True
        if self.mode == "min":
            return value < self.best - self.min_delta
        return value > self.best + self.min_delta

    def step(self, value: float) -> bool:
        """Record *value*; return True when training should stop."""
        if not math.isfinite(value):
            logger.warning("Monitored metric is %s; not counting it towards early stopping", value)
            return False

        if self.is_improvement(value):
            self.best = value
            self.counter = 0
            return False

        if not self.enabled:
            return False

        self.counter += 1
        if self.counter >= self.patience:
            logger.info(
                "Early stopping: no improvement in %d epoch(s) (best %.6f)", self.counter, self.best
            )
            return True
        return False


class CheckpointManager:
    """Writes ``last.pt`` and ``best.pt`` into a run directory.

    Checkpoints are saved atomically (to a temporary file, then renamed): a
    long training run interrupted mid-write would otherwise leave a corrupt
    ``best.pt`` and lose everything.
    """

    def __init__(
        self,
        directory: str | Path,
        *,
        monitor: str = "val_loss",
        mode: MonitorMode = "min",
    ) -> None:
        self.directory = ensure_dir(directory)
        self.monitor = monitor
        self.mode: MonitorMode = mode
        self.best_value: float | None = None
        self.best_epoch: int | None = None

    @property
    def best_path(self) -> Path:
        return self.directory / "best.pt"

    @property
    def last_path(self) -> Path:
        return self.directory / "last.pt"

    def _is_better(self, value: float) -> bool:
        if self.best_value is None:
            return True
        return value < self.best_value if self.mode == "min" else value > self.best_value

    def _atomic_save(self, payload: Mapping[str, Any], path: Path) -> None:
        temporary = path.with_suffix(path.suffix + ".tmp")
        torch.save(payload, temporary)
        temporary.replace(path)

    def save(
        self,
        payload: Mapping[str, Any],
        *,
        epoch: int,
        metrics: Mapping[str, float],
        extra_best_files: Mapping[str, Mapping[str, Any]] | None = None,
    ) -> bool:
        """Save ``last.pt`` and, on improvement, ``best.pt``.

        Args:
            payload: the checkpoint contents.
            epoch: current epoch number.
            metrics: this epoch's metrics; ``self.monitor`` is looked up here.
            extra_best_files: additional ``{filename: payload}`` written only
                when this epoch is the best.  Used to emit a standalone
                ``encoder_best.pt`` from the SimCLR run.

        Returns:
            True when this epoch became the new best.
        """
        full = {**dict(payload), "epoch": epoch, "metrics": dict(metrics)}
        self._atomic_save(full, self.last_path)

        value = metrics.get(self.monitor)
        if value is None:
            logger.debug(
                "Monitor %r absent from this epoch's metrics (%s); not updating best.pt",
                self.monitor,
                sorted(metrics),
            )
            return False

        if not self._is_better(float(value)):
            return False

        self.best_value = float(value)
        self.best_epoch = int(epoch)
        self._atomic_save(full, self.best_path)

        for filename, extra_payload in (extra_best_files or {}).items():
            self._atomic_save(extra_payload, self.directory / filename)

        logger.info(
            "New best: %s=%.6f at epoch %d -> %s",
            self.monitor,
            self.best_value,
            epoch,
            relative_to_root(self.best_path),
        )
        return True


def build_optimizer(
    parameters: Iterable[nn.Parameter],
    settings: TrainSettings,
) -> Optimizer:
    """Create the optimiser named by *settings*.

    Raises:
        ValueError: the optimiser name is unknown, or every parameter is frozen.
    """
    trainable = [p for p in parameters if p.requires_grad]
    if not trainable:
        raise ValueError(
            "No trainable parameters were passed to the optimiser. If the encoder is "
            "frozen, the optimiser must still receive the head's parameters."
        )

    name = settings.optimizer.strip().lower()
    if name == "adamw":
        return torch.optim.AdamW(
            trainable,
            lr=settings.learning_rate,
            weight_decay=settings.weight_decay,
            betas=(0.9, 0.999),
        )
    if name == "adam":
        return torch.optim.Adam(trainable, lr=settings.learning_rate, weight_decay=settings.weight_decay)
    if name == "sgd":
        return torch.optim.SGD(
            trainable,
            lr=settings.learning_rate,
            momentum=0.9,
            weight_decay=settings.weight_decay,
            nesterov=True,
        )
    raise ValueError(f"Unknown optimizer {settings.optimizer!r}; expected adamw, adam or sgd")


def build_scheduler(optimizer: Optimizer, settings: TrainSettings) -> LRScheduler | None:
    """Create the learning-rate schedule, stepped once per epoch.

    ``cosine`` includes a linear warm-up over ``warmup_epochs``.  The warm-up
    is what keeps the first epochs of CTC training stable: before the model
    emits anything but blanks, the gradients are large and poorly conditioned.

    Returns:
        A scheduler, or ``None`` when ``scheduler: none``.
    """
    name = (settings.scheduler or "none").strip().lower()
    if name in ("none", "", "constant"):
        return None

    if name == "cosine":
        warmup = settings.warmup_epochs
        total = settings.epochs

        def factor(epoch: int) -> float:
            if warmup > 0 and epoch < warmup:
                # Start at 10% rather than 0 so the first epoch makes progress.
                return 0.1 + 0.9 * (epoch + 1) / warmup
            progress = (epoch - warmup) / max(1, total - warmup)
            progress = min(1.0, max(0.0, progress))
            # Decay to 1% of the base rate, not 0: a zero rate wastes the
            # final epoch entirely.
            return 0.01 + 0.99 * 0.5 * (1.0 + math.cos(math.pi * progress))

        return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=factor)

    if name == "step":
        step_size = max(1, settings.epochs // 3)
        return torch.optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=0.3)

    raise ValueError(f"Unknown scheduler {settings.scheduler!r}; expected cosine, step or none")


def current_learning_rate(optimizer: Optimizer) -> float:
    """The first parameter group's learning rate (what gets logged)."""
    for group in optimizer.param_groups:
        return float(group["lr"])
    return 0.0
