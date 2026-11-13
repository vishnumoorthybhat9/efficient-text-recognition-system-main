"""Plotly figures for the evaluation dashboard.

Figures are built server-side and sent as Plotly JSON (``{data, layout}``) so
the frontend only has to render them.  That keeps the metric definitions in
one place -- the same Python that computed CER also decides how it is drawn --
instead of duplicating formatting rules in JavaScript.

Every builder returns a payload matching
:class:`backend.schemas.evaluation.ChartResponse` and sets ``empty`` with a
reason rather than returning an empty plot, so the UI can show "train a model
first" instead of blank axes.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Sequence

import plotly.graph_objects as go
from sqlalchemy import select
from sqlalchemy.orm import Session

from backend.db.models import EvaluationRun, OCRRequest
from backend.services.evaluation_service import latest_per_model
from backend.services.runtime import MLRuntime
from ml.engine.common import TrainingHistory
from ml.utils.paths import resolve_path

__all__ = [
    "model_comparison_chart",
    "accuracy_vs_speed_chart",
    "training_curve_chart",
    "simclr_progress_chart",
    "history_activity_chart",
    "available_charts",
    "build_chart",
]

logger = logging.getLogger(__name__)

#: Colour-blind-safe palette (Okabe-Ito), used consistently across figures.
_PALETTE = ["#0072B2", "#E69F00", "#009E73", "#CC79A7", "#56B4E9", "#D55E00", "#F0E442"]

#: The two experiment arms always get the same two colours, everywhere, so a
#: reader can follow them across figures without re-reading the legend.
_ARM_COLOURS = {
    "control": "#D55E00",     # orange-red: trained from scratch
    "treatment": "#0072B2",   # blue: SimCLR pretrained
    "reference": "#009E73",
    "other": "#999999",
}

_LAYOUT_BASE: dict[str, Any] = {
    "template": "plotly_white",
    "margin": {"l": 60, "r": 24, "t": 16, "b": 72},
    "font": {"family": "Inter, Segoe UI, system-ui, sans-serif", "size": 12},
    "hovermode": "closest",
    # Legend below the plot area. The card header already carries the title,
    # so figures render title-less and a top legend would sit over the data.
    "legend": {"orientation": "h", "yanchor": "top", "y": -0.18, "xanchor": "center", "x": 0.5},
}


def _figure(fig: go.Figure, **layout: Any) -> dict[str, Any]:
    """Apply the shared layout and return the figure as plain JSON."""
    fig.update_layout(**{**_LAYOUT_BASE, **layout})
    return json.loads(fig.to_json())


def _empty(chart_id: str, title: str, reason: str, description: str = "") -> dict[str, Any]:
    return {
        "id": chart_id,
        "title": title,
        "description": description,
        "figure": {"data": [], "layout": {**_LAYOUT_BASE, "title": {"text": title}}},
        "empty": True,
        "empty_reason": reason,
    }


def _arm_colour(arm: str, fallback_index: int = 0) -> str:
    return _ARM_COLOURS.get(arm) or _PALETTE[fallback_index % len(_PALETTE)]


# ---------------------------------------------------------------------------
# Accuracy comparison
# ---------------------------------------------------------------------------
def model_comparison_chart(session: Session, *, split: str = "test") -> dict[str, Any]:
    """Grouped bars: CER, WER and exact-match accuracy per model.

    CER and WER are errors (lower is better) while accuracy is the opposite,
    so accuracy is drawn on its own right-hand axis and the title says which
    direction is good -- plotting them on one axis invites misreading.
    """
    chart_id, title = "model_comparison", "Recognition accuracy by model"
    rows = latest_per_model(session, split=split)

    if not rows:
        return _empty(
            chart_id,
            title,
            f"No model has been evaluated on the {split!r} split yet. "
            "Run an evaluation from the Evaluation page, or `python scripts/evaluate.py`.",
        )

    rows = sorted(rows, key=lambda r: r.cer)
    names = [r.model_name for r in rows]

    fig = go.Figure()
    fig.add_bar(
        name="CER (lower is better)",
        x=names,
        y=[r.cer for r in rows],
        marker_color="#D55E00",
        hovertemplate="%{x}<br>CER %{y:.4f}<extra></extra>",
    )
    fig.add_bar(
        name="WER (lower is better)",
        x=names,
        y=[r.wer for r in rows],
        marker_color="#E69F00",
        hovertemplate="%{x}<br>WER %{y:.4f}<extra></extra>",
    )
    fig.add_bar(
        name="Exact match (higher is better)",
        x=names,
        y=[r.sequence_accuracy for r in rows],
        marker_color="#0072B2",
        yaxis="y2",
        hovertemplate="%{x}<br>Exact match %{y:.4f}<extra></extra>",
    )

    sample_counts = {r.num_samples for r in rows}
    subtitle = (
        f"{split} split, {sample_counts.pop()} samples"
        if len(sample_counts) == 1
        else f"{split} split (sample counts differ between models)"
    )

    return {
        "id": chart_id,
        "title": title,
        "description": f"Latest evaluation of each model on the {subtitle}.",
        "figure": _figure(
            fig,
            barmode="group",
            yaxis={"title": "Error rate", "rangemode": "tozero"},
            yaxis2={
                "title": "Exact-match accuracy",
                "overlaying": "y",
                "side": "right",
                "range": [0, 1],
                "showgrid": False,
            },
            xaxis={"title": ""},
        ),
        "empty": False,
        "empty_reason": "",
    }


# ---------------------------------------------------------------------------
# Accuracy vs cost
# ---------------------------------------------------------------------------
def accuracy_vs_speed_chart(session: Session, *, split: str = "test") -> dict[str, Any]:
    """Scatter of character accuracy against latency, sized by parameter count.

    This is the figure that answers the synopsis objective about *lower
    computational complexity*: the best model is up and to the left.
    """
    chart_id, title = "accuracy_vs_speed", "Accuracy against inference cost"
    rows = [r for r in latest_per_model(session, split=split) if r.avg_inference_ms > 0]

    if not rows:
        return _empty(
            chart_id,
            title,
            "No evaluation with usable timing data yet. Evaluate at least one model on "
            "enough samples that some batches are timed after warm-up.",
        )

    fig = go.Figure()
    for index, row in enumerate(rows):
        fig.add_scatter(
            name=row.model_name,
            x=[row.avg_inference_ms],
            y=[row.character_accuracy],
            # Markers only: an on-marker text label overflows the plot area
            # for a model near the right-hand edge, and the legend already
            # names every point.
            mode="markers",
            marker={
                "size": max(14, min(52, (row.parameter_count / 1e6) * 3 + 14)),
                "color": _arm_colour(row.experiment_arm, index),
                "opacity": 0.85,
                "line": {"width": 1, "color": "#ffffff"},
            },
            hovertemplate=(
                f"<b>{row.model_name}</b><br>"
                "latency %{x:.2f} ms/image<br>"
                "character accuracy %{y:.4f}<br>"
                f"CER {row.cer:.4f}<br>"
                f"{row.parameter_count / 1e6:.2f} M parameters<br>"
                f"{row.throughput_images_per_s:.0f} images/s<extra></extra>"
            ),
        )

    device = rows[0].device
    return {
        "id": chart_id,
        "title": title,
        "description": (
            f"{split} split, measured on {device}. Marker area is proportional to parameter "
            "count. Up and to the left is better: more accurate for less time."
        ),
        "figure": _figure(
            fig,
            xaxis={"title": "Mean inference time per image (ms)", "rangemode": "tozero"},
            yaxis={"title": "Character accuracy (1 - CER)"},
        ),
        "empty": False,
        "empty_reason": "",
    }


# ---------------------------------------------------------------------------
# Training curves
# ---------------------------------------------------------------------------
def _load_history(run_name: str) -> dict[str, Any] | None:
    path = resolve_path(f"artifacts/checkpoints/{run_name}/history.json")
    if path is None or not path.is_file():
        return None
    try:
        return TrainingHistory.load(path)
    except (OSError, json.JSONDecodeError) as exc:
        logger.warning("Could not read the training history for %s: %s", run_name, exc)
        return None


def training_curve_chart(run_names: Sequence[str] | None = None) -> dict[str, Any]:
    """Validation CER per epoch for the two experiment arms, overlaid.

    Overlaying them is the point: the gap between the curves *is* the result,
    and it is far more convincing than two numbers in a table.
    """
    chart_id, title = "training_curves", "Validation CER during training"
    names = list(run_names) if run_names else ["baseline_scratch", "simclr_finetune"]

    histories = [(name, _load_history(name)) for name in names]
    found = [(name, h) for name, h in histories if h]

    if not found:
        return _empty(
            chart_id,
            title,
            "No training history found. Train a model with "
            "`python scripts/train_crnn.py --config configs/experiments/baseline_scratch.yaml`.",
        )

    arm_of = {"baseline_scratch": "control", "simclr_finetune": "treatment"}

    fig = go.Figure()
    plotted = 0
    for index, (name, history) in enumerate(found):
        epochs = [record["epoch"] for record in history.get("epochs", [])]
        values = [record.get("val_cer") for record in history.get("epochs", [])]
        if not epochs or all(v is None for v in values):
            continue

        label = (history.get("notes") or {}).get("experiment_arm") or name
        fig.add_scatter(
            name=f"{name} ({label})" if label != name else name,
            x=epochs,
            y=values,
            mode="lines+markers",
            line={"width": 2.5, "color": _arm_colour(arm_of.get(name, "other"), index)},
            marker={"size": 6},
            hovertemplate="epoch %{x}<br>val CER %{y:.4f}<extra></extra>",
        )
        plotted += 1

        best = history.get("best_epoch")
        if best is not None and best in epochs:
            best_value = values[epochs.index(best)]
            if best_value is not None:
                fig.add_scatter(
                    x=[best],
                    y=[best_value],
                    mode="markers",
                    marker={"size": 13, "symbol": "star", "color": _arm_colour(arm_of.get(name, "other"), index)},
                    name=f"{name} best",
                    showlegend=False,
                    hovertemplate=f"best epoch {best}<br>val CER %{{y:.4f}}<extra></extra>",
                )

    if not plotted:
        return _empty(
            chart_id, title, "Training histories were found but contain no validation CER values."
        )

    return {
        "id": chart_id,
        "title": title,
        "description": (
            "Lower is better. The gap between the two arms is the effect of self-supervised "
            "pretraining; stars mark each run's best epoch."
        ),
        "figure": _figure(
            fig,
            xaxis={"title": "Epoch", "dtick": 1},
            yaxis={"title": "Validation CER", "rangemode": "tozero"},
        ),
        "empty": False,
        "empty_reason": "",
    }


def simclr_progress_chart(run_name: str = "simclr_pretrain") -> dict[str, Any]:
    """NT-Xent loss and contrastive accuracy during self-supervised pretraining.

    The chance line is drawn explicitly: contrastive accuracy is meaningless
    without it, since chance depends on the batch size.
    """
    chart_id, title = "simclr_progress", "Self-supervised pretraining (SimCLR)"
    history = _load_history(run_name)

    if not history:
        return _empty(
            chart_id,
            title,
            "SimCLR pretraining has not run yet. Start it with "
            "`python scripts/pretrain_simclr.py --config configs/experiments/simclr_pretrain.yaml`.",
        )

    records = history.get("epochs", [])
    epochs = [r["epoch"] for r in records]
    losses = [r.get("train_loss") for r in records]
    accuracies = [r.get("train_acc") for r in records]

    if not epochs:
        return _empty(chart_id, title, "The SimCLR history file contains no epochs.")

    fig = go.Figure()
    fig.add_scatter(
        name="NT-Xent loss",
        x=epochs,
        y=losses,
        mode="lines+markers",
        line={"width": 2.5, "color": "#D55E00"},
        hovertemplate="epoch %{x}<br>loss %{y:.4f}<extra></extra>",
    )
    fig.add_scatter(
        name="Contrastive accuracy",
        x=epochs,
        y=accuracies,
        mode="lines+markers",
        line={"width": 2.5, "color": "#0072B2"},
        yaxis="y2",
        hovertemplate="epoch %{x}<br>accuracy %{y:.4f}<extra></extra>",
    )

    notes = history.get("notes") or {}
    chance = notes.get("chance_accuracy")
    if chance:
        fig.add_hline(
            y=chance,
            line={"dash": "dot", "width": 1.5, "color": "#999999"},
            annotation_text=f"chance ({chance:.4f})",
            annotation_position="bottom right",
            yref="y2",
        )

    images = notes.get("unlabeled_samples", "?")
    return {
        "id": chart_id,
        "title": title,
        "description": (
            f"Trained on {images} unlabeled images with no transcriptions at all. Contrastive "
            "accuracy is the fraction of views matched to their correct pair; it must rise "
            "clearly above the chance line for the pretraining to be worth transferring."
        ),
        "figure": _figure(
            fig,
            xaxis={"title": "Epoch", "dtick": 1},
            yaxis={"title": "NT-Xent loss", "rangemode": "tozero"},
            yaxis2={
                "title": "Contrastive accuracy",
                "overlaying": "y",
                "side": "right",
                "range": [0, 1],
                "showgrid": False,
            },
        ),
        "empty": False,
        "empty_reason": "",
    }


# ---------------------------------------------------------------------------
# Usage
# ---------------------------------------------------------------------------
def history_activity_chart(session: Session, *, limit: int = 300) -> dict[str, Any]:
    """Confidence against processing time for recent recognitions."""
    chart_id, title = "history_activity", "Recent recognitions"

    rows = list(
        session.scalars(
            select(OCRRequest).order_by(OCRRequest.created_at.desc()).limit(limit)
        )
    )
    if not rows:
        return _empty(
            chart_id, title, "No recognitions yet. Upload an image on the Recognise page."
        )

    by_model: dict[str, list[OCRRequest]] = {}
    for row in rows:
        by_model.setdefault(row.model_name or row.model_id, []).append(row)

    fig = go.Figure()
    for index, (name, items) in enumerate(sorted(by_model.items())):
        fig.add_scatter(
            name=name,
            x=[r.total_ms for r in items],
            y=[r.confidence for r in items],
            mode="markers",
            marker={"size": 9, "color": _PALETTE[index % len(_PALETTE)], "opacity": 0.75},
            customdata=[[r.character_count, r.original_filename] for r in items],
            hovertemplate=(
                "%{customdata[1]}<br>%{x:.0f} ms<br>confidence %{y:.3f}"
                "<br>%{customdata[0]} characters<extra></extra>"
            ),
        )

    return {
        "id": chart_id,
        "title": title,
        "description": f"The {len(rows)} most recent requests, grouped by model.",
        "figure": _figure(
            fig,
            xaxis={"title": "Total processing time (ms)", "rangemode": "tozero"},
            yaxis={"title": "Mean confidence", "range": [0, 1.02]},
        ),
        "empty": False,
        "empty_reason": "",
    }


# ---------------------------------------------------------------------------
# Dispatch
# ---------------------------------------------------------------------------
def available_charts() -> list[dict[str, str]]:
    """Chart ids and titles, so the UI can enumerate them from the API."""
    return [
        {"id": "model_comparison", "title": "Recognition accuracy by model"},
        {"id": "accuracy_vs_speed", "title": "Accuracy against inference cost"},
        {"id": "training_curves", "title": "Validation CER during training"},
        {"id": "simclr_progress", "title": "Self-supervised pretraining (SimCLR)"},
        {"id": "history_activity", "title": "Recent recognitions"},
    ]


def build_chart(
    chart_id: str,
    session: Session,
    runtime: MLRuntime,
    *,
    split: str = "test",
) -> dict[str, Any]:
    """Build one chart by id.

    Raises:
        KeyError: unknown chart id.
    """
    if chart_id == "model_comparison":
        return model_comparison_chart(session, split=split)
    if chart_id == "accuracy_vs_speed":
        return accuracy_vs_speed_chart(session, split=split)
    if chart_id == "training_curves":
        return training_curve_chart()
    if chart_id == "simclr_progress":
        return simclr_progress_chart()
    if chart_id == "history_activity":
        return history_activity_chart(session)

    known = ", ".join(c["id"] for c in available_charts())
    raise KeyError(f"Unknown chart {chart_id!r}. Available: {known}")
