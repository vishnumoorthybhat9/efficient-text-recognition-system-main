"""Build the model-comparison report.

Reads the evaluation runs stored in the database and the training histories on
disk, then writes:

* ``artifacts/reports/comparison.md``   -- tables and a written verdict
* ``artifacts/reports/comparison.json`` -- the same data, machine-readable
* ``artifacts/figures/*.png``           -- figures sized for a project report

The figures are matplotlib rather than Plotly: these are for pasting into a
document, where a static 200-dpi PNG is what is wanted. The interactive
Plotly versions live in the web dashboard.

Usage
-----
::

    python scripts/compare_models.py
    python scripts/compare_models.py --dataset handwriting --split test
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any, Sequence

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib

# A non-interactive backend: this script must run headless, and importing the
# default backend can fail outright on a machine with no display.
matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402

from backend.core.config import get_settings  # noqa: E402
from backend.db.session import init_db, session_scope  # noqa: E402
from backend.services.evaluation_service import build_comparison, latest_per_model  # noqa: E402
from backend.services.runtime import init_runtime  # noqa: E402
from ml.engine.common import TrainingHistory  # noqa: E402
from ml.utils.logging import setup_logging  # noqa: E402
from ml.utils.paths import ensure_dir, relative_to_root, resolve_path  # noqa: E402

logger = logging.getLogger("compare_models")

#: One colour per experiment arm, matching the web dashboard.
ARM_COLOURS = {
    "control": "#d55e00",
    "treatment": "#0072b2",
    "reference": "#009e73",
    "other": "#8c8c8c",
}

plt.rcParams.update(
    {
        "figure.dpi": 130,
        "savefig.dpi": 200,
        "savefig.bbox": "tight",
        "font.size": 9,
        "axes.grid": True,
        "grid.alpha": 0.25,
        "axes.spines.top": False,
        "axes.spines.right": False,
    }
)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate the model-comparison report and figures.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--dataset", default=None, help="restrict to one dataset")
    parser.add_argument("--split", default="test")
    parser.add_argument("--out", default="artifacts/reports", help="report directory")
    parser.add_argument("--figures", default="artifacts/figures", help="figure directory")
    parser.add_argument("--log-level", default="INFO")
    return parser.parse_args(argv)


# ---------------------------------------------------------------------------
# figures
# ---------------------------------------------------------------------------
def figure_accuracy(rows: list[dict[str, Any]], path: Path) -> Path | None:
    """Grouped bars: CER, WER and exact match per model."""
    if not rows:
        return None

    import numpy as np

    rows = sorted(rows, key=lambda r: r["cer"] if r["cer"] is not None else 9.9)
    names = [_short(r["model_name"]) for r in rows]
    positions = np.arange(len(rows))
    width = 0.26

    fig, ax = plt.subplots(figsize=(max(6.5, 1.9 * len(rows)), 4.0))
    ax.bar(positions - width, [r["cer"] or 0 for r in rows], width, label="CER", color="#d55e00")
    ax.bar(positions, [r["wer"] or 0 for r in rows], width, label="WER", color="#e69f00")
    ax.bar(
        positions + width,
        [r["sequence_accuracy"] or 0 for r in rows],
        width,
        label="Exact match",
        color="#0072b2",
    )

    ax.set_xticks(positions)
    ax.set_xticklabels(names, rotation=18, ha="right")
    ax.set_ylabel("Rate")
    ax.set_title("Recognition accuracy by model (CER and WER lower is better)")
    ax.legend(frameon=False, ncol=3, loc="upper center", bbox_to_anchor=(0.5, -0.28))

    for index, row in enumerate(rows):
        if row["cer"] is not None:
            ax.text(index - width, row["cer"] + 0.02, f"{row['cer']:.3f}", ha="center", fontsize=7)

    fig.savefig(path)
    plt.close(fig)
    return path


def figure_arms(control: dict[str, Any] | None, treatment: dict[str, Any] | None, path: Path) -> Path | None:
    """The headline figure: the two experiment arms side by side."""
    if not control or not treatment:
        return None

    import numpy as np

    metrics = [
        ("CER", control["cer"], treatment["cer"], True),
        ("WER", control["wer"], treatment["wer"], True),
        ("Exact match", control["sequence_accuracy"], treatment["sequence_accuracy"], False),
        ("Character accuracy", control["character_accuracy"], treatment["character_accuracy"], False),
    ]

    positions = np.arange(len(metrics))
    width = 0.36

    fig, ax = plt.subplots(figsize=(7.2, 4.0))
    ax.bar(
        positions - width / 2,
        [m[1] or 0 for m in metrics],
        width,
        label="From scratch (control)",
        color=ARM_COLOURS["control"],
    )
    ax.bar(
        positions + width / 2,
        [m[2] or 0 for m in metrics],
        width,
        label="SimCLR pretrained (treatment)",
        color=ARM_COLOURS["treatment"],
    )

    for index, (_, control_value, treatment_value, lower_better) in enumerate(metrics):
        for offset, value in ((-width / 2, control_value), (width / 2, treatment_value)):
            if value is not None:
                ax.text(index + offset, value + 0.015, f"{value:.3f}", ha="center", fontsize=7)

    ax.set_xticks(positions)
    ax.set_xticklabels([m[0] for m in metrics])
    ax.set_ylabel("Value")
    # Headroom so the value labels printed above each bar are not clipped.
    ax.set_ylim(0, 1.12)
    ax.set_title(
        "Effect of self-supervised pretraining\n"
        "(identical architecture, labeled subset, seed and schedule)"
    )
    # Below the axes: inside the plot the legend covered the value label on
    # the tallest bar.
    ax.legend(frameon=False, ncol=2, loc="upper center", bbox_to_anchor=(0.5, -0.09))

    fig.savefig(path)
    plt.close(fig)
    return path


def figure_cost(rows: list[dict[str, Any]], path: Path) -> Path | None:
    """Accuracy against latency, with marker area showing parameter count."""
    usable = [r for r in rows if r.get("avg_inference_ms")]
    if not usable:
        return None

    fig, ax = plt.subplots(figsize=(7.0, 4.4))
    for row in usable:
        size = max(60, min(900, (row.get("parameter_count") or 0) / 1e6 * 26))
        ax.scatter(
            row["avg_inference_ms"],
            row["character_accuracy"] or 0,
            s=size,
            alpha=0.8,
            color=ARM_COLOURS.get(row.get("experiment_arm", "other"), ARM_COLOURS["other"]),
            edgecolors="white",
            linewidths=1.2,
            label=_short(row["model_name"]),
        )

    ax.set_xlabel("Mean inference time per image (ms)")
    ax.set_ylabel("Character accuracy (1 - CER)")
    ax.set_title("Accuracy against inference cost\n(marker area ∝ parameter count)")
    ax.set_xlim(left=0)
    ax.legend(frameon=False, fontsize=7, loc="lower right")

    fig.savefig(path)
    plt.close(fig)
    return path


def figure_training_curves(run_names: Sequence[str], path: Path) -> Path | None:
    """Validation CER per epoch for every run that has a history."""
    series: list[tuple[str, list[int], list[float], str]] = []

    arm_of = {
        "baseline_scratch": "control",
        "hw_baseline_scratch": "control",
        "simclr_finetune": "treatment",
        "hw_simclr_finetune": "treatment",
    }

    for name in run_names:
        history_path = resolve_path(f"artifacts/checkpoints/{name}/history.json")
        if history_path is None or not history_path.is_file():
            continue
        try:
            history = TrainingHistory.load(history_path)
        except (OSError, json.JSONDecodeError):
            continue

        epochs, values = [], []
        for record in history.get("epochs", []):
            if record.get("val_cer") is not None:
                epochs.append(record["epoch"])
                values.append(record["val_cer"])
        if epochs:
            series.append((name, epochs, values, ARM_COLOURS.get(arm_of.get(name, "other"), "#8c8c8c")))

    if not series:
        return None

    fig, ax = plt.subplots(figsize=(7.2, 4.0))
    for name, epochs, values, colour in series:
        # Colour encodes the experiment arm and is consistent across every
        # figure; line style separates the two datasets, which would otherwise
        # be two indistinguishable lines of the same colour.
        style = "--" if name.startswith("hw_") else "-"
        ax.plot(
            epochs,
            values,
            marker="o",
            markersize=3.5,
            linewidth=2,
            linestyle=style,
            label=name,
            color=colour,
        )
        best = min(range(len(values)), key=lambda i: values[i])
        ax.plot(epochs[best], values[best], marker="*", markersize=13, color=colour)

    ax.set_xlabel("Epoch")
    ax.set_ylabel("Validation CER")
    ax.set_ylim(bottom=0)
    ax.set_title("Validation CER during training (stars mark the best epoch)")
    ax.legend(frameon=False, fontsize=8)

    fig.savefig(path)
    plt.close(fig)
    return path


def figure_simclr(run_names: Sequence[str], path: Path) -> Path | None:
    """SimCLR loss and contrastive accuracy during pretraining."""
    histories = []
    for name in run_names:
        history_path = resolve_path(f"artifacts/checkpoints/{name}/history.json")
        if history_path is None or not history_path.is_file():
            continue
        try:
            histories.append((name, TrainingHistory.load(history_path)))
        except (OSError, json.JSONDecodeError):
            continue

    if not histories:
        return None

    fig, (left, right) = plt.subplots(1, 2, figsize=(10.0, 3.8))

    for name, history in histories:
        records = history.get("epochs", [])
        epochs = [r["epoch"] for r in records]
        left.plot(epochs, [r.get("train_loss") for r in records], marker="o", markersize=3, label=name)
        right.plot(epochs, [r.get("train_acc") for r in records], marker="o", markersize=3, label=name)

        chance = (history.get("notes") or {}).get("chance_accuracy")
        if chance:
            right.axhline(chance, linestyle=":", linewidth=1.2, color="#999999")
            right.text(epochs[0] if epochs else 0, chance * 1.4, f"chance {chance:.4f}", fontsize=7, color="#777777")

    left.set_xlabel("Epoch")
    left.set_ylabel("NT-Xent loss")
    left.set_title("Contrastive loss")
    left.legend(frameon=False, fontsize=7)

    right.set_xlabel("Epoch")
    right.set_ylabel("Contrastive accuracy")
    right.set_ylim(0, 1.05)
    right.set_title("View-matching accuracy (no labels used)")
    right.legend(frameon=False, fontsize=7)

    fig.suptitle("Self-supervised pretraining (SimCLR)", y=1.02)
    fig.savefig(path)
    plt.close(fig)
    return path


def _short(name: str, limit: int = 26) -> str:
    return name if len(name) <= limit else name[: limit - 1] + "…"


# ---------------------------------------------------------------------------
# report
# ---------------------------------------------------------------------------
def _table(rows: list[dict[str, Any]]) -> str:
    header = (
        "| Model | Trained on | Provenance | Arm | CER | CER (case-ins.) | WER | Exact | Char acc. "
        "| ms/img | img/s | Params | n |\n"
        "|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|\n"
    )

    def cell(value: Any, digits: int = 4) -> str:
        return "—" if value is None else f"{value:.{digits}f}"

    lines = []
    for row in sorted(rows, key=lambda r: r["cer"] if r["cer"] is not None else 9.9):
        provenance = "ours" if row["provenance"] == "trained_in_project" else "external"
        parameters = f"{(row.get('parameter_count') or 0) / 1e6:.2f}M"
        lines.append(
            f"| {row['model_name']} | {row.get('trained_on', '—')} | {provenance} "
            f"| {row.get('experiment_arm', '—')} "
            f"| {cell(row['cer'])} | {cell(row.get('cer_normalized'))} | {cell(row['wer'])} "
            f"| {cell(row['sequence_accuracy'])} | {cell(row['character_accuracy'])} "
            f"| {cell(row.get('avg_inference_ms'), 2)} "
            f"| {cell(row.get('throughput_images_per_s'), 0)} | {parameters} "
            f"| {row.get('num_samples') or '—'} |"
        )
    return header + "\n".join(lines)


def build_report(payload: dict[str, Any], figures: dict[str, Path | None], split: str) -> str:
    """Render the markdown report."""
    import datetime as _dt

    parts: list[str] = [
        "# Model comparison report",
        "",
        "**Efficient Text Recognition from Images Using Deep Learning and Self-Supervised Learning**",
        "",
        f"Generated {_dt.datetime.now().strftime('%Y-%m-%d %H:%M')} from the evaluation runs stored "
        f"in `artifacts/ocr.db`. Every figure below is measured, not estimated.",
        "",
        "## How to read this",
        "",
        "- **CER / WER** are micro-averaged: total edit distance divided by total reference length. "
        "Lower is better. They can exceed 1.0 because insertions are not bounded by the reference.",
        "- **CER (case-ins.)** ignores letter case and punctuation. It is the fair column for a model "
        "trained on a different casing convention.",
        "- **Exact** is the fraction of samples transcribed perfectly.",
        "- **Provenance** `ours` = trained by this project; `external` = downloaded third-party "
        "weights, included as a baseline only and never as a result of this work.",
        "- Latency excludes image loading and discards warm-up batches; CUDA work is synchronised "
        "before each measurement.",
        "",
    ]

    if payload.get("comparable"):
        control = payload["control"]
        treatment = payload["treatment"]
        parts += [
            "## Headline result: does self-supervised pretraining help?",
            "",
            payload["summary"],
            "",
            "| | From scratch (control) | SimCLR pretrained (treatment) | Change |",
            "|---|---:|---:|---:|",
            f"| CER | {control['cer']:.4f} | {treatment['cer']:.4f} | "
            f"**{payload['cer_improvement_relative'] * 100:+.1f}%** relative |",
            f"| WER | {control['wer']:.4f} | {treatment['wer']:.4f} | — |",
            f"| Exact match | {control['sequence_accuracy']:.4f} | {treatment['sequence_accuracy']:.4f} | "
            f"{payload['accuracy_improvement_absolute'] * 100:+.1f} pp |",
            f"| Parameters | {(control.get('parameter_count') or 0) / 1e6:.2f}M | "
            f"{(treatment.get('parameter_count') or 0) / 1e6:.2f}M | identical |",
            "",
            "Both arms share the same architecture, the same labeled subset (same seed, same "
            "permutation), the same augmentation, optimiser, schedule and epoch count. The only "
            "difference is whether the CNN encoder was initialised from self-supervised "
            "pretraining, so the difference above is attributable to that.",
            "",
        ]
    else:
        parts += ["## Headline result", "", payload.get("summary", "Not available yet."), ""]

    rows = payload.get("rows", [])
    parts += [f"## All models ({split} split)", "", _table(rows), ""]

    cross_domain = [
        r for r in rows
        if r.get("trained_on") not in ("unknown", None) and r.get("trained_on") != payload.get("dataset")
    ]
    if cross_domain:
        names = ", ".join(f"*{r['model_name']}*" for r in cross_domain)
        parts += [
            "> **Cross-domain rows.** " + names + " were trained on a different dataset and are "
            "evaluated here unchanged. Their scores measure how well a model transfers to an "
            "unseen domain, not how well it performs on the data it was built for, and they "
            "should not be read as a like-for-like comparison against the rows above them.",
            "",
        ]

    written = {name: path for name, path in figures.items() if path}
    if written:
        parts += ["## Figures", ""]
        for name, path in written.items():
            parts.append(f"![{name}]({relative_to_root(path)})")
            parts.append("")

    parts += [
        "## Reproducing these numbers",
        "",
        "```bash",
        "python scripts/make_synthetic_data.py",
        "python scripts/pretrain_simclr.py --config configs/experiments/simclr_pretrain.yaml",
        "python scripts/train_crnn.py --config configs/experiments/baseline_scratch.yaml",
        "python scripts/train_crnn.py --config configs/experiments/simclr_finetune.yaml",
        "python scripts/evaluate.py --all --split test",
        "python scripts/compare_models.py",
        "```",
        "",
    ]
    return "\n".join(parts)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    setup_logging(args.log_level, log_file=ensure_dir("artifacts/logs") / "compare_models.log")

    settings = get_settings()
    init_db(settings)
    runtime = init_runtime(settings)

    with session_scope() as session:
        payload = build_comparison(session, runtime, dataset=args.dataset, split=args.split)
        stored = latest_per_model(session, dataset=args.dataset, split=args.split)

    if not payload.get("rows"):
        logger.error(
            "No evaluation runs found for split %r. Run `python scripts/evaluate.py --all` first.",
            args.split,
        )
        return 1

    report_dir = ensure_dir(args.out)
    figure_dir = ensure_dir(args.figures)

    suffix = f"_{args.dataset}" if args.dataset else ""
    figures = {
        "Accuracy by model": figure_accuracy(payload["rows"], figure_dir / f"accuracy{suffix}.png"),
        "Self-supervised comparison": figure_arms(
            payload.get("control"), payload.get("treatment"), figure_dir / f"simclr_effect{suffix}.png"
        ),
        "Accuracy against cost": figure_cost(payload["rows"], figure_dir / f"accuracy_vs_cost{suffix}.png"),
        "Training curves": figure_training_curves(
            ["baseline_scratch", "simclr_finetune", "hw_baseline_scratch", "hw_simclr_finetune"],
            figure_dir / "training_curves.png",
        ),
        "SimCLR pretraining": figure_simclr(
            ["simclr_pretrain", "hw_simclr_pretrain"], figure_dir / "simclr_pretraining.png"
        ),
    }

    report_path = report_dir / f"comparison{suffix}.md"
    report_path.write_text(build_report(payload, figures, args.split), encoding="utf-8")

    json_path = report_dir / f"comparison{suffix}.json"
    json_path.write_text(
        json.dumps(
            {
                "comparison": payload,
                "runs": [
                    {
                        "model_id": run.model_id,
                        "dataset": run.dataset,
                        "split": run.split,
                        "cer": run.cer,
                        "wer": run.wer,
                        "sequence_accuracy": run.sequence_accuracy,
                        "cer_normalized": run.cer_normalized,
                        "avg_inference_ms": run.avg_inference_ms,
                        "parameter_count": run.parameter_count,
                        "num_samples": run.num_samples,
                        "device": run.device,
                        "evaluated_at": run.created_at.isoformat() if run.created_at else None,
                    }
                    for run in stored
                ],
            },
            indent=2,
            default=str,
        ),
        encoding="utf-8",
    )

    logger.info("-" * 72)
    logger.info("Report written to %s", relative_to_root(report_path))
    logger.info("Machine-readable copy: %s", relative_to_root(json_path))
    for name, path in figures.items():
        if path:
            logger.info("  figure: %-28s %s", name, relative_to_root(path))
    logger.info("-" * 72)

    if payload.get("comparable"):
        logger.info("%s", payload["summary"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
