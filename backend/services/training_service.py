"""Training jobs started from the API.

Design decisions
----------------
**One job at a time.** The development GPU has 4 GB; a second concurrent run
would simply fail with an out-of-memory error partway through, after wasting
minutes. The manager refuses to start a second job and says why.

**A thread, not a subprocess.** The trainers already accept a ``should_stop``
callback, so cancellation is cooperative and clean -- the run saves its
checkpoint and history before returning. PyTorch releases the GIL inside its
kernels, so the API stays responsive. A subprocess would need IPC for progress
and would still not be killable safely mid-checkpoint-write.

**Training never starts by itself.** Nothing here runs at application
start-up; a job exists only because someone asked for one.
"""

from __future__ import annotations

import logging
import threading
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal

from sqlalchemy import select
from sqlalchemy.orm import Session

from backend.core.errors import BadRequest, ConflictError, DependencyMissing, NotFound
from backend.db.models import TrainingJob
from backend.db.session import session_scope
from backend.services.runtime import MLRuntime
from ml.config import ConfigError, load_config
from ml.utils.paths import ensure_dir, relative_to_root, resolve_path

__all__ = ["TrainingJobManager", "get_job_manager", "JobKind", "KNOWN_JOBS"]

logger = logging.getLogger(__name__)

JobKind = Literal["simclr_pretrain", "crnn_train"]


def resolve_encoder_checkpoint(source: str) -> Path:
    """Resolve the current encoder path, including runs saved by older configs."""
    path = resolve_path(source)
    if path is None:
        raise FileNotFoundError(f"Invalid SimCLR encoder path: {source}")
    if path.is_file():
        return path

    legacy_path = path.parent / path.parent.name / path.name
    if legacy_path.is_file():
        logger.warning(
            "Using legacy nested SimCLR encoder checkpoint at %s",
            relative_to_root(legacy_path),
        )
        return legacy_path
    return path


#: The jobs the UI offers, with the config each one uses.
KNOWN_JOBS: dict[str, dict[str, str]] = {
    "simclr_pretrain": {
        "kind": "simclr_pretrain",
        "config": "experiments/simclr_pretrain.yaml",
        "label": "SimCLR self-supervised pretraining",
        "description": (
            "Stage 1. Trains the CNN encoder on the unlabeled image pool with the "
            "contrastive objective. No transcriptions are used. Produces encoder_best.pt."
        ),
    },
    "baseline_scratch": {
        "kind": "crnn_train",
        "config": "experiments/baseline_scratch.yaml",
        "label": "Train recogniser from scratch (control arm)",
        "description": (
            "Stage 2a. CNN + BiLSTM + attention + CTC from random initialisation on the "
            "reduced labeled subset."
        ),
    },
    "simclr_finetune": {
        "kind": "crnn_train",
        "config": "experiments/simclr_finetune.yaml",
        "label": "Train recogniser from SimCLR encoder (treatment arm)",
        "description": (
            "Stage 2b. Identical to the control arm except the encoder starts from the "
            "SimCLR checkpoint. Requires stage 1 to have completed."
        ),
    },
    "mamba_train": {
        "kind": "crnn_train",
        "config": "mamba.yaml",
        "label": "Train Mamba OCR recogniser",
        "description": (
            "CNN visual encoder + bidirectional selective state-space layers + CTC, "
            "trained from scratch on the labeled OCR data."
        ),
    },
    "cnn_transformer_train": {
        "kind": "crnn_train",
        "config": "cnn_transformer.yaml",
        "label": "Train CNN + Transformer OCR recogniser",
        "description": (
            "CNN visual encoder + Transformer sequence encoder + CTC, trained from scratch "
            "on the labeled OCR data."
        ),
    },
}


class TrainingJobManager:
    """Owns the single background training thread."""

    def __init__(self) -> None:
        self._thread: threading.Thread | None = None
        self._stop_event = threading.Event()
        self._active_job_id: int | None = None
        self._lock = threading.Lock()

    # -- state --------------------------------------------------------------
    @property
    def is_running(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    @property
    def active_job_id(self) -> int | None:
        return self._active_job_id if self.is_running else None

    # -- lifecycle ----------------------------------------------------------
    def start(
        self,
        session: Session,
        runtime: MLRuntime,
        *,
        job_key: str,
        epochs: int | None = None,
        batch_size: int | None = None,
        max_samples: int | None = None,
    ) -> TrainingJob:
        """Queue and start a training job.

        Raises:
            BadRequest: unknown job key, or its config is broken.
            ConflictError: a job is already running.
            DependencyMissing: the dataset (or the SimCLR encoder) is absent.
        """
        if job_key not in KNOWN_JOBS:
            raise BadRequest(
                f"Unknown training job {job_key!r}. Available: {', '.join(KNOWN_JOBS)}.",
                detail={"available": list(KNOWN_JOBS)},
            )

        with self._lock:
            if self.is_running:
                raise ConflictError(
                    "A training job is already running. Only one runs at a time, because a "
                    "second would compete for the same GPU memory and fail. Wait for it to "
                    "finish, or cancel it first.",
                    detail={"active_job_id": self._active_job_id},
                )

            spec = KNOWN_JOBS[job_key]
            config_name = spec["config"]

            try:
                config = load_config(config_name)
            except ConfigError as exc:
                raise BadRequest(f"Could not load {config_name}: {exc}") from exc

            self._check_prerequisites(job_key, config, runtime)

            overrides: dict[str, Any] = {}
            if epochs is not None:
                overrides.setdefault("train", {})["epochs"] = int(epochs)
            if batch_size is not None:
                overrides.setdefault("train", {})["batch_size"] = int(batch_size)
            if max_samples is not None:
                overrides.setdefault("data", {})["max_train_samples"] = int(max_samples)

            if overrides:
                config = config.merged_with(overrides)

            log_path = ensure_dir("artifacts/logs") / f"job_{job_key}.log"

            job = TrainingJob(
                kind=spec["kind"],
                config_name=config_name,
                run_name=str(config.get("name", job_key)),
                status="queued",
                epochs_total=int(config.get("train.epochs", 0)),
                checkpoint_dir=str(
                    Path(str(config.get("train.checkpoint_dir", "")))
                    / str(config.get("name", job_key))
                ),
                log_path=relative_to_root(log_path),
                device=str(runtime.device),
            )
            session.add(job)
            session.commit()
            session.refresh(job)

            self._stop_event.clear()
            self._active_job_id = job.id
            self._thread = threading.Thread(
                target=self._run,
                args=(job.id, job_key, config, runtime),
                name=f"training-{job_key}",
                daemon=True,
            )
            self._thread.start()

            logger.info("Started training job %d (%s)", job.id, job_key)
            return job

    def _check_prerequisites(self, job_key: str, config: Any, runtime: MLRuntime) -> None:
        """Fail fast when the data or the stage-1 encoder is missing."""
        train_manifest = resolve_path(config.get("data.train_manifest"))
        unlabeled_manifest = resolve_path(config.get("data.unlabeled_manifest"))

        needed = unlabeled_manifest if job_key == "simclr_pretrain" else train_manifest
        if needed is None or not needed.is_file():
            raise DependencyMissing(
                "The dataset is not prepared yet. Generate it first:\n"
                "  python scripts/make_synthetic_data.py",
                detail={"missing": relative_to_root(needed) if needed else "manifest"},
            )

        encoder_source = config.get("model.init_encoder_from")
        if encoder_source:
            encoder_path = resolve_encoder_checkpoint(str(encoder_source))
            if not encoder_path.is_file():
                raise DependencyMissing(
                    "This job starts from a self-supervised encoder, but "
                    f"{relative_to_root(encoder_path)} "
                    "does not exist. Run the SimCLR pretraining job first.",
                    detail={"required_job": "simclr_pretrain"},
                )

    def cancel(self, session: Session, job_id: int) -> TrainingJob:
        """Request cancellation of a running job.

        Returns immediately; the trainer notices at its next batch boundary and
        saves its progress before stopping.

        Raises:
            NotFound: no such job.
            ConflictError: the job is not running.
        """
        job = session.get(TrainingJob, job_id)
        if job is None:
            raise NotFound(f"No training job with id {job_id}.", detail={"id": job_id})

        if not job.is_active:
            raise ConflictError(
                f"Job {job_id} is {job.status!r} and cannot be cancelled.",
                detail={"id": job_id, "status": job.status},
            )

        self._stop_event.set()
        logger.info("Cancellation requested for training job %d", job_id)
        return job

    # -- worker -------------------------------------------------------------
    def _run(self, job_id: int, job_key: str, config: Any, runtime: MLRuntime) -> None:
        """Thread body.  Must never raise: it owns the job's final status."""
        try:
            self._update(job_id, status="running", started_at=datetime.now(timezone.utc))

            if job_key == "simclr_pretrain":
                history = self._run_simclr(job_id, config, runtime)
            else:
                history = self._run_crnn(job_id, config, runtime)

            cancelled = self._stop_event.is_set()
            self._update(
                job_id,
                status="cancelled" if cancelled else "completed",
                finished_at=datetime.now(timezone.utc),
                progress=1.0 if not cancelled else None,
                latest_metrics=history.records[-1].metrics if history.records else {},
            )

            # A new checkpoint means a new selectable model.
            runtime.refresh_models()
            logger.info("Training job %d finished (%s)", job_id, "cancelled" if cancelled else "completed")

        except Exception as exc:
            logger.exception("Training job %d failed", job_id)
            self._update(
                job_id,
                status="failed",
                finished_at=datetime.now(timezone.utc),
                error=f"{type(exc).__name__}: {exc}\n\n{traceback.format_exc()[-2000:]}",
            )
        finally:
            with self._lock:
                self._active_job_id = None

    def _callbacks(self, job_id: int, total_epochs: int) -> Any:
        """Build trainer callbacks that write progress into the job row."""
        from ml.engine.ctc_trainer import TrainerCallbacks

        def on_epoch_end(epoch: int, metrics: dict[str, float]) -> None:
            self._update(
                job_id,
                current_epoch=epoch,
                epochs_completed=epoch,
                progress=min(1.0, epoch / max(1, total_epochs)),
                latest_metrics={k: round(float(v), 6) for k, v in metrics.items()},
                append_history={"epoch": epoch, **{k: round(float(v), 6) for k, v in metrics.items()}},
            )

        return TrainerCallbacks(
            on_epoch_end=on_epoch_end,
            should_stop=self._stop_event.is_set,
        )

    def _run_simclr(self, job_id: int, config: Any, runtime: MLRuntime) -> Any:
        from ml.datasets.ocr_dataset import (
            ImageGeometry,
            UnlabeledLineDataset,
            build_simclr_dataloader,
        )
        from ml.engine.common import TrainSettings
        from ml.engine.simclr_trainer import SimCLRTrainer
        from ml.models.simclr import SimCLRModel
        from ml.preprocessing.augment import SimCLRViewOptions
        from ml.preprocessing.pipeline import PreprocessOptions

        settings = TrainSettings.from_config(config)
        geometry = ImageGeometry.from_config(config.image)

        dataset = UnlabeledLineDataset(
            manifest=resolve_path(config.get("data.unlabeled_manifest")),
            geometry=geometry,
            preprocess=PreprocessOptions.from_config(config.get("preprocess")),
            views=SimCLRViewOptions.from_config(config.get("simclr.views")),
            seed=settings.seed,
        )
        loader = build_simclr_dataloader(
            dataset,
            batch_size=settings.batch_size,
            num_workers=settings.num_workers,
            device=runtime.device,
        )

        trainer = SimCLRTrainer(
            SimCLRModel.from_config(config),
            device=runtime.device,
            settings=settings,
            temperature=float(config.get("simclr.temperature", 0.1)),
            run_name=str(config.get("name", "simclr_pretrain")),
            geometry=geometry,
            callbacks=self._callbacks(job_id, settings.epochs),
            config_snapshot=config.to_dict(),
        )
        return trainer.fit(loader)

    def _run_crnn(self, job_id: int, config: Any, runtime: MLRuntime) -> Any:
        from ml.datasets.ocr_dataset import (
            ImageGeometry,
            OCRLineDataset,
            build_ocr_dataloader,
        )
        from ml.engine.common import TrainSettings
        from ml.engine.ctc_trainer import CTCTrainer
        from ml.models.cnn_transformer_ocr import CNNTransformerOCR
        from ml.models.crnn_attention import CRNNAttentionOCR
        from ml.models.mamba_ocr import MambaOCR
        from ml.preprocessing.augment import AugmentOptions
        from ml.preprocessing.pipeline import PreprocessOptions
        from ml.text.charset import Charset

        settings = TrainSettings.from_config(config)
        geometry = ImageGeometry.from_config(config.image)
        charset = Charset.from_config(config.charset)
        preprocess = PreprocessOptions.from_config(config.get("preprocess"))

        max_train = config.get("data.max_train_samples")
        max_train = int(max_train) if max_train and int(max_train) > 0 else None

        train_dataset = OCRLineDataset(
            charset=charset,
            manifest=resolve_path(config.get("data.train_manifest")),
            geometry=geometry,
            preprocess=preprocess,
            augment=AugmentOptions.from_config(config.get("augment")),
            train=True,
            max_samples=max_train,
            seed=settings.seed,
        )
        val_dataset = OCRLineDataset(
            charset=charset,
            manifest=resolve_path(config.get("data.val_manifest")),
            geometry=geometry,
            preprocess=preprocess,
            train=False,
            seed=settings.seed,
        )

        model_family = str(config.get("model.family", "crnn_attention"))
        if model_family == "mamba_ocr":
            model = MambaOCR.from_config(config, charset)
        elif model_family == "cnn_transformer":
            model = CNNTransformerOCR.from_config(config, charset)
        elif model_family == "crnn_attention":
            model = CRNNAttentionOCR.from_config(config, charset)
        else:
            raise ValueError(f"Unsupported OCR model family {model_family!r}")

        transfer = None
        encoder_source = config.get("model.init_encoder_from")
        if encoder_source:
            if isinstance(model, MambaOCR):
                raise ValueError("Mamba OCR does not support SimCLR encoder transfer.")
            encoder_path = resolve_encoder_checkpoint(str(encoder_source))
            if not encoder_path.is_file():
                raise FileNotFoundError(
                    f"SimCLR encoder not found at {encoder_source}. Run the pretraining job first."
                )
            transfer = model.load_encoder_weights(encoder_path)

        trainer = CTCTrainer(
            model,
            charset,
            device=runtime.device,
            settings=settings,
            run_name=str(config.get("name", "crnn")),
            geometry=geometry,
            freeze_encoder_epochs=int(config.get("model.freeze_encoder_epochs", 0) or 0),
            callbacks=self._callbacks(job_id, settings.epochs),
            config_snapshot=config.to_dict(),
        )
        trainer.history.notes["encoder_transfer"] = transfer.to_dict() if transfer else None
        trainer.history.notes["experiment_arm"] = (
            f"not an experiment arm ({model.describe()['architecture']} from scratch)"
            if isinstance(model, (MambaOCR, CNNTransformerOCR))
            else (
                "treatment (SimCLR pretrained)" if encoder_source else "control (from scratch)"
            )
        )

        return trainer.fit(
            build_ocr_dataloader(
                train_dataset,
                batch_size=settings.batch_size,
                num_workers=settings.num_workers,
                device=runtime.device,
            ),
            build_ocr_dataloader(
                val_dataset,
                batch_size=settings.batch_size,
                num_workers=settings.num_workers,
                device=runtime.device,
            ),
        )

    # -- persistence --------------------------------------------------------
    def _update(self, job_id: int, *, append_history: dict[str, Any] | None = None, **fields: Any) -> None:
        """Write progress fields onto the job row from the worker thread.

        Uses its own session: the request-scoped session that created the job
        belongs to a different thread and is long gone by now.
        """
        try:
            with session_scope() as session:
                job = session.get(TrainingJob, job_id)
                if job is None:
                    logger.warning("Training job %d vanished while updating it", job_id)
                    return

                for key, value in fields.items():
                    if value is not None:
                        setattr(job, key, value)

                if append_history is not None:
                    # Reassign rather than mutate: SQLAlchemy does not track
                    # in-place changes to a JSON column.
                    job.history = [*(job.history or []), append_history]
        except Exception:
            # Never let a bookkeeping failure kill a training run.
            logger.exception("Could not update training job %d", job_id)


_manager: TrainingJobManager | None = None
_manager_lock = threading.Lock()


def get_job_manager() -> TrainingJobManager:
    """The process-wide job manager."""
    global _manager
    with _manager_lock:
        if _manager is None:
            _manager = TrainingJobManager()
        return _manager


def list_jobs(session: Session, *, limit: int = 50) -> list[TrainingJob]:
    """Recent training jobs, newest first."""
    return list(
        session.scalars(
            select(TrainingJob).order_by(TrainingJob.created_at.desc()).limit(limit)
        )
    )


def get_job(session: Session, job_id: int) -> TrainingJob:
    """Fetch one job.

    Raises:
        NotFound: no such job.
    """
    job = session.get(TrainingJob, job_id)
    if job is None:
        raise NotFound(f"No training job with id {job_id}.", detail={"id": job_id})
    return job


def read_job_log(job: TrainingJob, *, lines: int = 200) -> str:
    """Return the tail of a job's log file."""
    if not job.log_path:
        return ""
    path = resolve_path(job.log_path)
    if path is None or not path.is_file():
        return ""
    try:
        content = path.read_text(encoding="utf-8", errors="replace").splitlines()
        return "\n".join(content[-lines:])
    except OSError as exc:
        logger.warning("Could not read job log %s: %s", path, exc)
        return ""
