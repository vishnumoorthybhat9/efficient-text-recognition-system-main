"""Dataset discovery, status and provenance.

Serves two needs:

* tell the UI what data is on this machine and whether it is usable;
* carry the official download links and preparation steps for the public
  benchmarks the synopsis names, so the information lives next to the code
  that consumes it rather than only in the README.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from backend.core.errors import NotFound
from backend.services.runtime import MLRuntime
from ml.datasets.manifest import read_manifest
from ml.text.charset import Charset
from ml.utils.paths import relative_to_root

__all__ = ["ExternalDataset", "EXTERNAL_DATASETS", "list_datasets", "dataset_detail", "list_demo_images"]

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ExternalDataset:
    """A public benchmark that must be downloaded by the user.

    None of these are redistributed with the project: IAM requires
    registration, and the synthetic corpora are tens of gigabytes.
    """

    key: str
    name: str
    url: str
    size: str
    content: str
    licence: str
    registration_required: bool
    preparation: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "key": self.key,
            "name": self.name,
            "url": self.url,
            "size": self.size,
            "content": self.content,
            "licence": self.licence,
            "registration_required": self.registration_required,
            "preparation": self.preparation,
        }


#: The four datasets named in the synopsis, with official sources.
EXTERNAL_DATASETS: tuple[ExternalDataset, ...] = (
    ExternalDataset(
        key="iam",
        name="IAM Handwriting Database",
        url="https://fki.tic.heia-fr.ch/databases/iam-handwriting-database",
        size="~1.5 GB (lines) / ~9 GB (forms)",
        content="Handwritten English text by 657 writers; 13,353 labeled text lines.",
        licence="Free for non-commercial research; registration required.",
        registration_required=True,
        preparation=(
            "Register, download `lines.tgz` and `ascii.tgz`, extract both into "
            "data/raw/iam/ so you have data/raw/iam/lines/ and data/raw/iam/ascii/lines.txt, "
            "then run:\n"
            "  python scripts/prepare_dataset.py --dataset iam --source data/raw/iam\n"
            "Note: this importer is implemented but has not been exercised against real IAM "
            "data, because no copy was available on the development machine."
        ),
    ),
    ExternalDataset(
        key="iiit5k",
        name="IIIT 5K-Word",
        url="https://cvit.iiit.ac.in/research/projects/cvit-projects/the-iiit-5k-word-dataset",
        size="~100 MB",
        content="5,000 cropped scene-text word images (2,000 train / 3,000 test).",
        licence="Free for research use.",
        registration_required=False,
        preparation=(
            "Download and extract IIIT5K-Word_V3.0.tar.gz into data/raw/iiit5k/ so you have "
            "data/raw/iiit5k/train/, test/ and the .mat annotation files, then run:\n"
            "  pip install scipy    # needed to read the .mat annotations\n"
            "  python scripts/prepare_dataset.py --dataset iiit5k --source data/raw/iiit5k\n"
            "Note: implemented but not exercised against real data on this machine."
        ),
    ),
    ExternalDataset(
        key="mjsynth",
        name="MJSynth (Synth90k)",
        url="https://www.robots.ox.ac.uk/~vgg/data/text/",
        size="~10 GB",
        content="9 million synthetic cropped word images over a 90k-word lexicon.",
        licence="Free for research use.",
        registration_required=False,
        preparation=(
            "NOT YET IMPLEMENTED in this project. The synthetic generator and the Kaggle "
            "handwriting set already cover the printed and handwritten cases the synopsis "
            "requires, and a 10 GB download was not justified.\n"
            "To add it: write a loader in ml/datasets/external.py yielding (image_path, "
            "label) pairs -- MJSynth encodes the label in the filename, as in "
            "'1_Avenge_4111.jpg' -> 'Avenge' -- then register it in "
            "scripts/prepare_dataset.py. Everything downstream is unchanged."
        ),
    ),
    ExternalDataset(
        key="synthtext",
        name="SynthText in the Wild",
        url="https://www.robots.ox.ac.uk/~vgg/data/scenetext/",
        size="~41 GB",
        content="800k synthetic scene images with word-level boxes and transcriptions.",
        licence="Free for research use.",
        registration_required=False,
        preparation=(
            "NOT YET IMPLEMENTED in this project. At ~41 GB it is far larger than this "
            "project needs.\n"
            "To add it: write a loader in ml/datasets/external.py that reads gt.mat and cuts "
            "word crops from the scene images using the supplied word boxes, then register "
            "it in scripts/prepare_dataset.py."
        ),
    ),
)


def _split_info(directory: Path, split: str) -> dict[str, Any]:
    """Describe one split of a prepared dataset."""
    manifest = directory / f"{split}.tsv"
    if not manifest.is_file():
        return {"split": split, "available": False, "count": 0, "manifest": None}

    try:
        count = sum(
            1
            for line in manifest.read_text(encoding="utf-8").splitlines()
            if line.strip() and not line.lstrip().startswith("#")
        )
    except OSError as exc:
        logger.warning("Could not count %s: %s", manifest, exc)
        count = 0

    return {
        "split": split,
        "available": True,
        "count": count,
        "manifest": relative_to_root(manifest),
    }


def _unlabeled_info(directory: Path) -> dict[str, Any]:
    manifest = directory / "unlabeled.txt"
    if not manifest.is_file():
        return {"split": "unlabeled", "available": False, "count": 0, "manifest": None}
    try:
        count = sum(
            1
            for line in manifest.read_text(encoding="utf-8").splitlines()
            if line.strip() and not line.lstrip().startswith("#")
        )
    except OSError:
        count = 0
    return {
        "split": "unlabeled",
        "available": True,
        "count": count,
        "manifest": relative_to_root(manifest),
    }


def _document_benchmark_entries(path: Path) -> list[dict[str, Any]]:
    """Read the rendered document benchmark's images and exact transcriptions."""
    payload = json.loads(path.read_text(encoding="utf-8"))
    entries = payload.get("entries") if isinstance(payload, dict) else None
    if not isinstance(entries, list):
        raise ValueError(f"Document benchmark must contain an 'entries' list: {path}")

    validated: list[dict[str, Any]] = []
    for index, entry in enumerate(entries):
        if (
            not isinstance(entry, dict)
            or not isinstance(entry.get("file"), str)
            or not entry["file"].strip()
            or not isinstance(entry.get("text"), str)
            or not entry["text"].strip()
        ):
            raise ValueError(f"Invalid document benchmark entry {index + 1} in {path}")
        validated.append(entry)
    return validated


def list_datasets(runtime: MLRuntime) -> dict[str, Any]:
    """Prepared datasets on this machine, plus the external ones to download."""
    root = runtime.dataset_dir
    prepared: list[dict[str, Any]] = []

    if root.is_dir():
        for directory in sorted(p for p in root.iterdir() if p.is_dir()):
            splits = [_split_info(directory, s) for s in ("train", "val", "test")]
            unlabeled = _unlabeled_info(directory)
            benchmark_path = directory / "benchmark.json"
            benchmark_only = benchmark_path.is_file()
            benchmark_notes = ""
            if benchmark_only:
                try:
                    entries = _document_benchmark_entries(benchmark_path)
                except (OSError, json.JSONDecodeError, ValueError) as exc:
                    logger.warning("Could not read document benchmark %s: %s", benchmark_path, exc)
                else:
                    splits[2] = {
                        "split": "test",
                        "available": True,
                        "count": len(entries),
                        "manifest": relative_to_root(benchmark_path),
                    }
                    benchmark_notes = (
                        "Evaluation-only benchmark with exact reference text. "
                        "Do not use these test pages for training."
                    )

            info_path = directory / "dataset_info.json"
            info: dict[str, Any] = {}
            if info_path.is_file():
                try:
                    info = json.loads(info_path.read_text(encoding="utf-8"))
                except (OSError, json.JSONDecodeError) as exc:
                    logger.warning("Could not read %s: %s", info_path, exc)

            split_by_name = {split["split"]: split for split in splits}
            prepared.append(
                {
                    "name": directory.name,
                    "path": relative_to_root(directory),
                    "splits": splits,
                    "unlabeled": unlabeled,
                    "total_labeled": sum(s["count"] for s in splits),
                    "ready": any(s["available"] for s in splits),
                    "trainable": all(
                        split_by_name[required]["available"]
                        and split_by_name[required]["count"] > 0
                        for required in ("train", "val")
                    ),
                    "benchmark_only": benchmark_only,
                    "generated_at": info.get("generated_at"),
                    "source": (
                        "benchmark"
                        if benchmark_only
                        else "generated" if info.get("generator") else "imported"
                    ),
                    "generator": info.get("generator"),
                    "seed": info.get("seed"),
                    "fonts": (info.get("fonts") or {}).get("printed_count"),
                    "notes": info.get("notes") or benchmark_notes,
                }
            )

    return {
        "prepared": prepared,
        "external": [d.to_dict() for d in EXTERNAL_DATASETS],
        "data_root": relative_to_root(root),
        "any_ready": any(d["ready"] for d in prepared),
    }


def dataset_detail(
    runtime: MLRuntime,
    name: str,
    *,
    split: str = "train",
    limit: int = 20,
) -> dict[str, Any]:
    """Sample rows and label statistics for one prepared dataset split.

    Raises:
        NotFound: the dataset or split is absent.
    """
    directory = runtime.dataset_dir / name
    manifest = directory / f"{split}.tsv"

    benchmark_path = directory / "benchmark.json"
    if benchmark_path.is_file():
        if split != "test":
            raise NotFound(
                f"Dataset {name!r} is an evaluation-only benchmark with no {split!r} split. "
                "Choose the 'test' split; benchmark pages must not be used for training.",
                detail={"dataset": name, "split": split, "benchmark_only": True},
            )

        entries = _document_benchmark_entries(benchmark_path)
        charset = Charset.from_config(runtime.config.charset)
        lengths = [len(charset.encode(charset.normalize(entry["text"]))) for entry in entries]
        unsupported: dict[str, int] = {}
        for entry in entries[:2000]:
            for character in charset.unsupported(entry["text"]):
                unsupported[character] = unsupported.get(character, 0) + 1

        for entry in entries:
            image_path = (directory / entry["file"]).resolve()
            try:
                image_path.relative_to(directory.resolve())
            except ValueError:
                raise NotFound(
                    f"Benchmark image path escapes dataset {name!r}.",
                    detail={"dataset": name, "file": entry["file"]},
                ) from None
            if not image_path.is_file():
                raise NotFound(
                    f"Benchmark image {entry['file']!r} is missing from dataset {name!r}.",
                    detail={"dataset": name, "file": entry["file"]},
                )

        return {
            "dataset": name,
            "split": split,
            "count": len(entries),
            "manifest": relative_to_root(benchmark_path),
            "label_length": {
                "min": min(lengths) if lengths else 0,
                "max": max(lengths) if lengths else 0,
                "mean": round(sum(lengths) / len(lengths), 2) if lengths else 0.0,
            },
            "unsupported_characters": dict(
                sorted(unsupported.items(), key=lambda kv: -kv[1])[:20]
            ),
            "samples": [
                {"id": entry["file"], "label": entry["text"]}
                for entry in entries[: max(1, min(limit, 200))]
            ],
        }

    if not manifest.is_file():
        raise NotFound(
            f"Dataset {name!r} has no {split!r} split "
            f"(looked for {relative_to_root(manifest)}).",
            detail={"dataset": name, "split": split},
        )

    entries = read_manifest(manifest, skip_missing=False)
    charset = Charset.from_config(runtime.config.charset)

    lengths = [len(charset.encode(entry.label)) for entry in entries]
    unsupported: dict[str, int] = {}
    for entry in entries[:2000]:  # sampling is enough to flag a charset mismatch
        for character in charset.unsupported(entry.label):
            unsupported[character] = unsupported.get(character, 0) + 1

    return {
        "dataset": name,
        "split": split,
        "count": len(entries),
        "manifest": relative_to_root(manifest),
        "label_length": {
            "min": min(lengths) if lengths else 0,
            "max": max(lengths) if lengths else 0,
            "mean": round(sum(lengths) / len(lengths), 2) if lengths else 0.0,
        },
        "unsupported_characters": dict(sorted(unsupported.items(), key=lambda kv: -kv[1])[:20]),
        "samples": [
            {"id": entry.relative_path, "label": entry.label}
            for entry in entries[: max(1, min(limit, 200))]
        ],
    }


def list_demo_images(runtime: MLRuntime) -> list[dict[str, Any]]:
    """Bundled demo images, so the app is usable with no upload.

    Returns an empty list when the demo set has not been generated; the UI
    simply hides the section.
    """
    directory = runtime.demo_dir()
    if not directory.is_dir():
        return []

    index_path = directory / "demo_index.json"
    labels: dict[str, dict[str, Any]] = {}
    if index_path.is_file():
        try:
            for item in json.loads(index_path.read_text(encoding="utf-8")):
                labels[str(item.get("file"))] = item
        except (OSError, json.JSONDecodeError) as exc:
            logger.warning("Could not read the demo index: %s", exc)

    images: list[dict[str, Any]] = []
    for path in sorted(directory.glob("*.png")):
        meta = labels.get(path.name, {})
        images.append(
            {
                "file": path.name,
                "text": meta.get("text", ""),
                "kind": meta.get("kind", "printed"),
                "size_bytes": path.stat().st_size,
                "url": f"/api/datasets/demo/{path.name}",
            }
        )
    return images


def resolve_demo_image(runtime: MLRuntime, filename: str) -> Path:
    """Resolve a demo image filename to a path inside the demo directory.

    Raises:
        NotFound: the file does not exist or escapes the demo directory.
    """
    directory = runtime.demo_dir().resolve()
    candidate = (directory / filename).resolve()

    try:
        candidate.relative_to(directory)
    except ValueError:
        raise NotFound(f"No demo image named {filename!r}.", detail={"file": filename}) from None

    if not candidate.is_file():
        raise NotFound(f"No demo image named {filename!r}.", detail={"file": filename})
    return candidate
