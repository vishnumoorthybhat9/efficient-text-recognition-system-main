"""Model evaluation and the self-supervised comparison.

Two responsibilities:

* run a benchmark (:mod:`ml.evaluation.benchmark`) and persist it;
* assemble the comparison table, including the headline
  "did self-supervised pretraining help?" answer.

The comparison is deliberately conservative. It reports an improvement only
when both arms were evaluated on the **same dataset and split**, and it says
so explicitly when they were not, because a CER difference measured on two
different test sets means nothing.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Callable, Sequence

from sqlalchemy import func, select
from sqlalchemy.orm import Session

from backend.core.errors import BadRequest, DependencyMissing, ModelUnavailable, NotFound
from backend.db.models import EvaluationRun
from backend.services.runtime import MLRuntime
from ml.datasets.manifest import ManifestEntry, read_manifest
from ml.evaluation.benchmark import BenchmarkResult, benchmark_predictor
from ml.inference.registry import ModelLoadError, ModelNotFound
from ml.utils.paths import ensure_dir, relative_to_root

__all__ = [
    "load_split_entries",
    "run_evaluation",
    "list_evaluations",
    "get_evaluation",
    "latest_per_model",
    "build_comparison",
]

logger = logging.getLogger(__name__)


def load_split_entries(
    runtime: MLRuntime,
    *,
    dataset: str = "synthetic",
    split: str = "test",
    max_samples: int | None = None,
) -> list[ManifestEntry]:
    """Read a dataset split's manifest.

    Raises:
        DependencyMissing: the dataset has not been prepared.
    """
    manifest = runtime.dataset_dir / dataset / f"{split}.tsv"
    if not manifest.is_file():
        raise DependencyMissing(
            f"The {split!r} split of dataset {dataset!r} is not prepared "
            f"(expected {relative_to_root(manifest)}). Generate it with:\n"
            "  python scripts/make_synthetic_data.py",
            detail={"dataset": dataset, "split": split, "manifest": relative_to_root(manifest)},
        )

    entries = read_manifest(manifest, skip_missing=True)
    if max_samples is not None and 0 < max_samples < len(entries):
        # A prefix, not a sample: evaluation must be repeatable, and the
        # manifest order is fixed.
        entries = entries[:max_samples]
    return entries


def run_evaluation(
    *,
    session: Session,
    runtime: MLRuntime,
    model_id: str,
    dataset: str = "synthetic",
    split: str = "test",
    max_samples: int | None = None,
    batch_size: int = 32,
    decode_strategy: str = "greedy",
    beam_width: int = 10,
    progress: Callable[[int, int], None] | None = None,
    should_stop: Callable[[], bool] | None = None,
    persist: bool = True,
) -> tuple[BenchmarkResult, EvaluationRun | None]:
    """Benchmark one model on one split and store the result.

    Raises:
        BadRequest: unknown model id.
        ModelUnavailable: the model exists but cannot be loaded.
        DependencyMissing: the dataset split is absent.
    """
    try:
        entry = runtime.registry.entry(model_id)
        predictor = runtime.registry.get(model_id)
    except ModelNotFound as exc:
        raise BadRequest(str(exc), detail={"model_id": model_id}) from exc
    except ModelLoadError as exc:
        raise ModelUnavailable(str(exc), detail={"model_id": model_id}) from exc

    entries = load_split_entries(runtime, dataset=dataset, split=split, max_samples=max_samples)

    result = benchmark_predictor(
        predictor,
        entries,
        dataset=dataset,
        split=split,
        batch_size=batch_size,
        decode_strategy=decode_strategy,
        beam_width=beam_width,
        progress=progress,
        should_stop=should_stop,
    )

    report_path = _write_report(result, entry.provenance)

    row: EvaluationRun | None = None
    if persist:
        row = EvaluationRun(
            model_id=result.model_id,
            model_name=result.model_name,
            provenance=entry.provenance,
            experiment_arm=str(entry.extra.get("experiment_arm", "other")),
            dataset=result.dataset,
            split=result.split,
            num_samples=result.num_samples,
            cer=result.metrics.cer,
            wer=result.metrics.wer,
            sequence_accuracy=result.metrics.sequence_accuracy,
            character_accuracy=result.metrics.character_accuracy,
            cer_normalized=result.metrics.cer_normalized,
            avg_inference_ms=result.avg_inference_ms,
            median_inference_ms=result.median_inference_ms,
            throughput_images_per_s=result.throughput_images_per_s,
            peak_memory_mb=result.peak_memory_mb,
            parameter_count=result.parameter_count,
            model_size_mb=result.model_size_mb,
            device=result.device,
            decode_strategy=result.decode_strategy,
            batch_size=result.batch_size,
            metrics=result.to_dict(),
            worst_samples=_worst(result),
            report_path=relative_to_root(report_path) if report_path else None,
            notes=result.notes,
        )
        session.add(row)
        session.commit()
        session.refresh(row)
        logger.info("Stored evaluation run %d for %s", row.id, model_id)

    return result, row


def _worst(result: BenchmarkResult, limit: int = 20) -> list[dict[str, Any]]:
    """The *limit* worst-scoring samples, for error analysis in the UI."""
    from ml.evaluation.metrics import worst_samples

    return [sample.to_dict() for sample in worst_samples(result.samples, limit)]


def _write_report(result: BenchmarkResult, provenance: str) -> Path | None:
    """Write the full benchmark to ``artifacts/reports`` as JSON."""
    try:
        directory = ensure_dir("artifacts/reports")
        name = f"eval_{result.model_id}_{result.dataset}_{result.split}.json"
        path = directory / name

        payload = result.to_dict()
        payload["provenance"] = provenance
        payload["_disclaimer"] = (
            "Numbers produced by running the model on the stated split. "
            "Models with provenance 'pretrained_third_party' were not trained by this "
            "project and are reported as external baselines."
        )
        path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
        return path
    except OSError as exc:
        logger.warning("Could not write the evaluation report: %s", exc)
        return None


def list_evaluations(
    session: Session,
    *,
    model_id: str | None = None,
    dataset: str | None = None,
    split: str | None = None,
    limit: int = 100,
    offset: int = 0,
) -> tuple[list[EvaluationRun], int]:
    """List stored evaluation runs, newest first."""
    filters = []
    if model_id:
        filters.append(EvaluationRun.model_id == model_id)
    if dataset:
        filters.append(EvaluationRun.dataset == dataset)
    if split:
        filters.append(EvaluationRun.split == split)

    total = session.scalar(select(func.count()).select_from(EvaluationRun).where(*filters)) or 0
    rows = list(
        session.scalars(
            select(EvaluationRun)
            .where(*filters)
            .order_by(EvaluationRun.created_at.desc(), EvaluationRun.id.desc())
            .limit(limit)
            .offset(offset)
        )
    )
    return rows, int(total)


def get_evaluation(session: Session, run_id: int) -> EvaluationRun:
    """Fetch one evaluation run.

    Raises:
        NotFound: no such run.
    """
    row = session.get(EvaluationRun, run_id)
    if row is None:
        raise NotFound(f"No evaluation run with id {run_id}.", detail={"id": run_id})
    return row


def latest_per_model(
    session: Session,
    *,
    dataset: str | None = None,
    split: str | None = None,
) -> list[EvaluationRun]:
    """The most recent evaluation of each model.

    Re-running an evaluation should replace the previous figure in the table,
    not add a second row for the same model.
    """
    filters = []
    if dataset:
        filters.append(EvaluationRun.dataset == dataset)
    if split:
        filters.append(EvaluationRun.split == split)

    rows = session.scalars(
        select(EvaluationRun).where(*filters).order_by(EvaluationRun.created_at.desc(), EvaluationRun.id.desc())
    )

    seen: dict[str, EvaluationRun] = {}
    for row in rows:
        seen.setdefault(row.model_id, row)
    return list(seen.values())


def _row_to_comparison(row: EvaluationRun, runtime: MLRuntime) -> dict[str, Any]:
    architecture = ""
    trained_on = "unknown"
    if row.model_id in runtime.registry:
        entry = runtime.registry.entry(row.model_id)
        architecture = {
            "crnn_attention": "CNN + BiLSTM + Self-Attention + CTC",
            "mamba_ocr": "CNN + bidirectional Mamba + CTC",
            "trocr": "Transformer encoder-decoder (TrOCR)",
        }.get(entry.family, entry.family)
        trained_on = str(entry.extra.get("trained_on", "unknown"))

    # Normalised exact-match lives in the stored metrics blob rather than in
    # its own column; fall back to None when an older run lacks it.
    normalized_exact = None
    try:
        normalized_exact = (row.metrics or {}).get("accuracy", {}).get("sequence_accuracy_normalized")
    except AttributeError:
        normalized_exact = None

    return {
        "model_id": row.model_id,
        "model_name": row.model_name,
        "provenance": row.provenance,
        "experiment_arm": row.experiment_arm,
        "architecture": architecture,
        "trained_on": trained_on,
        "cer": row.cer,
        "wer": row.wer,
        "sequence_accuracy": row.sequence_accuracy,
        "character_accuracy": row.character_accuracy,
        "cer_normalized": row.cer_normalized,
        "sequence_accuracy_normalized": normalized_exact,
        "avg_inference_ms": row.avg_inference_ms,
        "throughput_images_per_s": row.throughput_images_per_s,
        "parameter_count": row.parameter_count,
        "model_size_mb": row.model_size_mb,
        "num_samples": row.num_samples,
        "evaluated_at": row.created_at,
        "evaluation_id": row.id,
    }


def _pick_experiment_pair(
    rows: Sequence[EvaluationRun],
    *,
    preferred_dataset: str | None = None,
) -> tuple[EvaluationRun | None, EvaluationRun | None]:
    """Choose a control/treatment pair evaluated under the same conditions.

    The project now runs the same experiment on more than one dataset
    (synthetic and real handwriting), so several models carry the ``control``
    and ``treatment`` arms at once. Simply taking the first of each would
    happily pair the synthetic control with the handwriting treatment and then
    report them as incomparable -- technically correct, but useless.

    Pairs are therefore grouped by ``(dataset, split)`` and only a group
    holding both arms is returned.
    """
    groups: dict[tuple[str, str], dict[str, EvaluationRun]] = {}
    for row in rows:
        if row.experiment_arm not in ("control", "treatment"):
            continue
        key = (row.dataset, row.split)
        # `rows` is newest-first per model, so never overwrite an earlier pick.
        groups.setdefault(key, {}).setdefault(row.experiment_arm, row)

    complete = {key: arms for key, arms in groups.items() if {"control", "treatment"} <= set(arms)}

    if preferred_dataset:
        for (dataset_name, _split), arms in complete.items():
            if dataset_name == preferred_dataset:
                return arms["control"], arms["treatment"]

    if complete:
        # Prefer the pair with the most evaluated samples: the most solid one.
        _, arms = max(
            complete.items(),
            key=lambda item: min(item[1]["control"].num_samples, item[1]["treatment"].num_samples),
        )
        return arms["control"], arms["treatment"]

    # No complete pair: return whatever single arms exist so the caller can
    # say precisely which one is missing.
    any_control = next((r for r in rows if r.experiment_arm == "control"), None)
    any_treatment = next((r for r in rows if r.experiment_arm == "treatment"), None)
    return any_control, any_treatment


def build_comparison(
    session: Session,
    runtime: MLRuntime,
    *,
    dataset: str | None = None,
    split: str = "test",
) -> dict[str, Any]:
    """Assemble the comparison table and the self-supervised verdict.

    Returns a payload matching
    :class:`backend.schemas.evaluation.ExperimentComparisonOut`.
    """
    rows = latest_per_model(session, dataset=dataset, split=split)

    # Project-trained models first, then external baselines; best CER first.
    def sort_key(row: EvaluationRun) -> tuple[int, float]:
        rank = 0 if row.provenance == "trained_in_project" else 1
        return rank, row.cer

    rows = sorted(rows, key=sort_key)
    table = [_row_to_comparison(row, runtime) for row in rows]

    control, treatment = _pick_experiment_pair(rows, preferred_dataset=dataset)

    payload: dict[str, Any] = {
        "rows": table,
        "control": _row_to_comparison(control, runtime) if control else None,
        "treatment": _row_to_comparison(treatment, runtime) if treatment else None,
        "dataset": dataset,
        "split": split,
        "comparable": False,
        "cer_improvement_absolute": None,
        "cer_improvement_relative": None,
        "accuracy_improvement_absolute": None,
        "summary": "",
    }

    if control is None or treatment is None:
        missing = []
        if control is None:
            missing.append("the from-scratch baseline (baseline_scratch)")
        if treatment is None:
            missing.append("the SimCLR-pretrained model (simclr_finetune)")
        payload["summary"] = (
            "The self-supervised comparison is not available yet: "
            + " and ".join(missing)
            + " has not been trained and evaluated. Run scripts/pretrain_simclr.py, then "
            "scripts/train_crnn.py for both experiment configs, then evaluate both on the "
            "same split."
        )
        return payload

    same_setting = control.dataset == treatment.dataset and control.split == treatment.split
    if not same_setting:
        payload["summary"] = (
            f"Both arms have been evaluated, but on different data "
            f"({control.dataset}/{control.split} vs {treatment.dataset}/{treatment.split}), "
            "so the difference between them is not meaningful. Re-evaluate both on the "
            "same dataset and split."
        )
        return payload

    absolute = control.cer - treatment.cer
    relative = absolute / control.cer if control.cer > 0 else 0.0
    accuracy_delta = treatment.sequence_accuracy - control.sequence_accuracy

    payload.update(
        comparable=True,
        dataset=control.dataset,
        split=control.split,
        cer_improvement_absolute=absolute,
        cer_improvement_relative=relative,
        accuracy_improvement_absolute=accuracy_delta,
    )

    direction = "lower" if absolute > 0 else "higher"
    verdict = (
        "self-supervised pretraining improved recognition"
        if absolute > 0
        else "self-supervised pretraining did not improve recognition in this run"
    )

    payload["summary"] = (
        f"On {control.num_samples} held-out {control.dataset}/{control.split} samples, the "
        f"SimCLR-pretrained model reached CER {treatment.cer:.4f} and exact-match accuracy "
        f"{treatment.sequence_accuracy:.4f}, against CER {control.cer:.4f} and "
        f"{control.sequence_accuracy:.4f} for the identical model trained from scratch. "
        f"That is {abs(absolute):.4f} CER {direction} ({abs(relative) * 100:.1f}% relative) "
        f"-- {verdict}. Both arms share the same architecture, labeled subset, seed, "
        "augmentation and schedule; they differ only in whether the CNN encoder was "
        "initialised from self-supervised pretraining."
    )
    return payload
