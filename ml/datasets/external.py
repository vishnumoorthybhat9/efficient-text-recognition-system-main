"""Importers for real, externally sourced OCR datasets.

Each importer converts a public dataset into the project's manifest format
(:mod:`ml.datasets.manifest`), so everything downstream -- datasets, training,
evaluation -- is identical regardless of where the images came from.

Supported sources
-----------------
``handwriting``
    The Kaggle *Handwriting Recognition* set (``written_name_*_v2.csv`` plus
    ``train_v2/``, ``validation_v2/``, ``test_v2/``): 413,701 photographed
    handwritten name fields with transcriptions. Read directly from the zip.
``iam``
    IAM Handwriting Database line images plus ``ascii/lines.txt``.
``iiit5k``
    IIIT 5K-Word cropped scene-text images with ``.mat`` annotations.

Data hygiene
------------
Real datasets are messy, and silently training on their defects is how
unexplainable results happen. Every importer reports exactly what it dropped
and why: unreadable labels, labels outside the charset, missing or corrupt
images, and labels too long for the model's CTC time-step budget.
"""

from __future__ import annotations

import csv
import io
import logging
import re
import zipfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterable, Sequence

import numpy as np

from ml.datasets.manifest import write_manifest, write_unlabeled_manifest
from ml.text.charset import Charset
from ml.utils.image_io import ImageDecodeError, decode_image_bytes, write_image

__all__ = [
    "PreparationReport",
    "prepare_kaggle_handwriting",
    "prepare_iam",
    "prepare_iiit5k",
    "SplitCounts",
]

logger = logging.getLogger(__name__)

#: Labels that mean "this sample has no usable transcription".
_SENTINEL_LABELS = {"unreadable", "nan", "none", "null", "", "#", "?"}

#: Minimum usable image dimensions, in pixels.
_MIN_SIDE = 6


@dataclass(slots=True)
class SplitCounts:
    """How many samples to take from each split."""

    train: int | None = None
    val: int | None = None
    test: int | None = None
    unlabeled: int = 0

    def limit_for(self, split: str) -> int | None:
        return {"train": self.train, "val": self.val, "test": self.test}.get(split)


@dataclass(slots=True)
class PreparationReport:
    """What an importer produced, and what it had to discard."""

    dataset: str
    output_dir: str
    written: dict[str, int] = field(default_factory=dict)
    unlabeled_written: int = 0
    skipped_unreadable: int = 0
    skipped_empty: int = 0
    skipped_unsupported_chars: int = 0
    skipped_too_long: int = 0
    skipped_bad_image: int = 0
    dropped_characters: dict[str, int] = field(default_factory=dict)
    label_length_min: int = 0
    label_length_max: int = 0
    label_length_mean: float = 0.0
    observed_characters: str = ""
    notes: str = ""

    @property
    def total_written(self) -> int:
        return sum(self.written.values())

    @property
    def total_skipped(self) -> int:
        return (
            self.skipped_unreadable
            + self.skipped_empty
            + self.skipped_unsupported_chars
            + self.skipped_too_long
            + self.skipped_bad_image
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "dataset": self.dataset,
            "output_dir": self.output_dir,
            "written": self.written,
            "unlabeled_written": self.unlabeled_written,
            "total_written": self.total_written,
            "skipped": {
                "unreadable": self.skipped_unreadable,
                "empty": self.skipped_empty,
                "unsupported_characters": self.skipped_unsupported_chars,
                "too_long": self.skipped_too_long,
                "bad_image": self.skipped_bad_image,
                "total": self.total_skipped,
            },
            "dropped_characters": self.dropped_characters,
            "label_length": {
                "min": self.label_length_min,
                "max": self.label_length_max,
                "mean": round(self.label_length_mean, 2),
            },
            "observed_characters": self.observed_characters,
            "notes": self.notes,
        }

    def summary(self) -> str:
        parts = [f"{self.dataset}: wrote {self.total_written} labeled sample(s)"]
        if self.unlabeled_written:
            parts.append(f"+ {self.unlabeled_written} unlabeled")
        if self.total_skipped:
            detail = []
            if self.skipped_unreadable:
                detail.append(f"{self.skipped_unreadable} unreadable")
            if self.skipped_empty:
                detail.append(f"{self.skipped_empty} empty")
            if self.skipped_unsupported_chars:
                detail.append(f"{self.skipped_unsupported_chars} outside charset")
            if self.skipped_too_long:
                detail.append(f"{self.skipped_too_long} too long")
            if self.skipped_bad_image:
                detail.append(f"{self.skipped_bad_image} unreadable image")
            parts.append(f"(skipped {self.total_skipped}: {', '.join(detail)})")
        return " ".join(parts)


class _LabelFilter:
    """Applies the charset and length rules, and counts what it rejects."""

    def __init__(self, charset: Charset, max_label_length: int, report: PreparationReport) -> None:
        self.charset = charset
        self.max_label_length = max_label_length
        self.report = report
        self.lengths: list[int] = []
        self.observed: set[str] = set()

    def accept(self, raw_label: str | None) -> str | None:
        """Return the cleaned label, or ``None`` if the sample must be dropped."""
        if raw_label is None:
            self.report.skipped_empty += 1
            return None

        label = " ".join(str(raw_label).split())

        if label.strip().lower() in _SENTINEL_LABELS:
            # The Kaggle handwriting set marks illegible fields "UNREADABLE".
            # Training on those teaches the model to produce that literal word.
            self.report.skipped_unreadable += 1
            return None

        if not label:
            self.report.skipped_empty += 1
            return None

        unsupported = self.charset.unsupported(label)
        if unsupported:
            for character in unsupported:
                self.report.dropped_characters[character] = (
                    self.report.dropped_characters.get(character, 0) + 1
                )
            self.report.skipped_unsupported_chars += 1
            return None

        encoded_length = len(self.charset.encode(label))
        if encoded_length > self.max_label_length:
            self.report.skipped_too_long += 1
            return None
        if encoded_length == 0:
            self.report.skipped_empty += 1
            return None

        self.lengths.append(encoded_length)
        self.observed.update(label)
        return label

    def finalise(self) -> None:
        report = self.report
        if self.lengths:
            report.label_length_min = min(self.lengths)
            report.label_length_max = max(self.lengths)
            report.label_length_mean = sum(self.lengths) / len(self.lengths)
        report.observed_characters = "".join(sorted(self.observed))


def _usable_image(data: bytes) -> np.ndarray | None:
    """Decode *data*, returning ``None`` if it is unusable."""
    try:
        image = decode_image_bytes(data, grayscale=True)
    except ImageDecodeError:
        return None
    if image.shape[0] < _MIN_SIDE or image.shape[1] < _MIN_SIDE:
        return None
    # A uniform image carries no text; these appear in scanned corpora as
    # blank or failed captures.
    if int(image.max()) - int(image.min()) < 8:
        return None
    return image


# ---------------------------------------------------------------------------
# Kaggle "Handwriting Recognition"
# ---------------------------------------------------------------------------
_KAGGLE_SPLITS = {
    "train": ("written_name_train_v2.csv", "train_v2/train/"),
    "val": ("written_name_validation_v2.csv", "validation_v2/validation/"),
    "test": ("written_name_test_v2.csv", "test_v2/test/"),
}


def prepare_kaggle_handwriting(
    source: str | Path,
    output_dir: str | Path,
    *,
    charset: Charset,
    counts: SplitCounts,
    max_label_length: int = 32,
    seed: int = 1234,
    progress: Callable[[str, int, int], None] | None = None,
) -> PreparationReport:
    """Import the Kaggle *Handwriting Recognition* dataset from its zip.

    The archive is read in place -- no 1.3 GB extraction -- and only the
    requested subset is written out.

    Args:
        source: path to ``archive.zip`` (the one containing
            ``written_name_train_v2.csv``).
        output_dir: destination, e.g. ``data/processed/handwriting``.
        charset: labels outside it are dropped, not silently truncated.
        counts: how many samples to take per split, plus an unlabeled pool.
        max_label_length: CTC time-step budget.
        seed: controls which subset is taken.
        progress: called with ``(split, done, total)``.

    Returns:
        A :class:`PreparationReport`.

    Raises:
        FileNotFoundError: the zip does not exist.
        ValueError: the zip is not this dataset.
    """
    archive_path = Path(source)
    if not archive_path.is_file():
        raise FileNotFoundError(f"Dataset archive not found: {archive_path}")

    out = Path(output_dir)
    report = PreparationReport(
        dataset="handwriting",
        output_dir=str(out),
        notes=(
            "Kaggle 'Handwriting Recognition' dataset: photographed handwritten name "
            "fields with transcriptions. Labels are uppercase Latin with spaces, hyphens "
            "and apostrophes. Rows marked UNREADABLE by the annotators are excluded."
        ),
    )
    label_filter = _LabelFilter(charset, max_label_length, report)
    rng = np.random.default_rng(seed)

    with zipfile.ZipFile(archive_path) as archive:
        available = set(archive.namelist())
        missing = [csv_name for csv_name, _ in _KAGGLE_SPLITS.values() if csv_name not in available]
        if missing:
            raise ValueError(
                f"{archive_path.name} does not look like the Kaggle Handwriting Recognition "
                f"dataset: missing {missing}. Expected the archive containing "
                "written_name_train_v2.csv."
            )

        # Collect candidates per split, then subsample, then extract. Doing the
        # filtering first means the sample cap counts *usable* rows rather than
        # rows that will later be thrown away.
        selected: dict[str, list[tuple[str, str]]] = {}

        for split, (csv_name, image_prefix) in _KAGGLE_SPLITS.items():
            limit = counts.limit_for(split)
            if limit is not None and limit <= 0:
                continue

            text = archive.read(csv_name).decode("utf-8", errors="replace")
            rows = list(csv.DictReader(io.StringIO(text)))

            candidates: list[tuple[str, str]] = []
            for row in rows:
                filename = (row.get("FILENAME") or "").strip()
                label = label_filter.accept(row.get("IDENTITY"))
                if not filename or label is None:
                    continue
                member = f"{image_prefix}{filename}"
                if member not in available:
                    report.skipped_bad_image += 1
                    continue
                candidates.append((member, label))

            if limit is not None and limit < len(candidates):
                picked = rng.permutation(len(candidates))[:limit]
                candidates = [candidates[int(i)] for i in sorted(picked)]

            selected[split] = candidates
            logger.info("%s: %d usable sample(s) selected", split, len(candidates))

        # --- write the labeled splits ---
        for split, candidates in selected.items():
            image_dir = out / "images" / split
            image_dir.mkdir(parents=True, exist_ok=True)

            rows_out: list[tuple[str, str]] = []
            for index, (member, label) in enumerate(candidates):
                image = _usable_image(archive.read(member))
                if image is None:
                    report.skipped_bad_image += 1
                    continue

                name = f"{split}_{index:06d}.png"
                write_image(image_dir / name, image)
                rows_out.append((f"images/{split}/{name}", label))

                if progress is not None and (index + 1) % 500 == 0:
                    progress(split, index + 1, len(candidates))

            write_manifest(
                out / f"{split}.tsv",
                rows_out,
                header_comment=(
                    f"Kaggle Handwriting Recognition -- {split} split\n"
                    f"source: {archive_path.name} | {len(rows_out)} samples | seed={seed}\n"
                    "format: <relative image path>\\t<label>"
                ),
            )
            report.written[split] = len(rows_out)

        # --- unlabeled pool for SimCLR ---
        if counts.unlabeled > 0:
            report.unlabeled_written = _write_unlabeled_pool(
                archive=archive,
                out=out,
                used_members={m for rows in selected.values() for m, _ in rows},
                prefix=_KAGGLE_SPLITS["train"][1],
                available=available,
                count=counts.unlabeled,
                rng=rng,
                report=report,
                source_name=archive_path.name,
            )

    label_filter.finalise()
    logger.info("%s", report.summary())
    return report


def _write_unlabeled_pool(
    *,
    archive: zipfile.ZipFile,
    out: Path,
    used_members: set[str],
    prefix: str,
    available: set[str],
    count: int,
    rng: np.random.Generator,
    report: PreparationReport,
    source_name: str,
) -> int:
    """Write an unlabeled pool drawn from *train* images not already used.

    Disjointness from the labeled splits is the point: it keeps the
    self-supervised stage honestly "extra, unlabeled data" rather than a second
    pass over the same images, and guarantees pretraining never saw a
    validation or test image.
    """
    pool_candidates = [
        name
        for name in available
        if name.startswith(prefix) and name.endswith(".jpg") and name not in used_members
    ]
    if not pool_candidates:
        logger.warning("No images left for the unlabeled pool after the labeled splits were taken")
        return 0

    pool_candidates.sort()
    if count < len(pool_candidates):
        picked = rng.permutation(len(pool_candidates))[:count]
        pool_candidates = [pool_candidates[int(i)] for i in sorted(picked)]

    directory = out / "unlabeled"
    directory.mkdir(parents=True, exist_ok=True)

    written: list[str] = []
    for index, member in enumerate(pool_candidates):
        image = _usable_image(archive.read(member))
        if image is None:
            report.skipped_bad_image += 1
            continue
        name = f"unlab_{index:06d}.png"
        write_image(directory / name, image)
        written.append(f"unlabeled/{name}")

    write_unlabeled_manifest(
        out / "unlabeled.txt",
        written,
        header_comment=(
            "Unlabeled pool for SimCLR self-supervised pretraining.\n"
            f"source: {source_name} -- training images NOT used in any labeled split, so "
            "pretraining cannot have seen a validation or test image.\n"
            "Transcriptions exist for these images but are deliberately not read."
        ),
    )
    return len(written)


# ---------------------------------------------------------------------------
# IAM Handwriting Database
# ---------------------------------------------------------------------------
def prepare_iam(
    root: str | Path,
    output_dir: str | Path,
    *,
    charset: Charset,
    counts: SplitCounts,
    max_label_length: int = 32,
    seed: int = 1234,
) -> PreparationReport:
    """Import IAM line images.

    Expects the standard layout after extracting ``lines.tgz`` and
    ``ascii.tgz``::

        <root>/lines/a01/a01-000u/a01-000u-00.png
        <root>/ascii/lines.txt

    ``lines.txt`` is whitespace-delimited; the transcription is the last field,
    with ``|`` as the word separator. Lines whose segmentation the IAM authors
    marked ``err`` are skipped -- their images are known-bad crops.

    Raises:
        FileNotFoundError: the expected files are not present.
    """
    source = Path(root)
    lines_dir = source / "lines"
    index_file = source / "ascii" / "lines.txt"

    if not lines_dir.is_dir() or not index_file.is_file():
        raise FileNotFoundError(
            f"IAM layout not found under {source}. Expected {lines_dir} and {index_file}. "
            "Download lines.tgz and ascii.tgz from "
            "https://fki.tic.heia-fr.ch/databases/iam-handwriting-database and extract both there."
        )

    out = Path(output_dir)
    report = PreparationReport(
        dataset="iam",
        output_dir=str(out),
        notes=(
            "IAM Handwriting Database, line level. Segmentation-error lines (status 'err') "
            "are excluded, as is standard practice."
        ),
    )
    label_filter = _LabelFilter(charset, max_label_length, report)
    rng = np.random.default_rng(seed)

    records: list[tuple[Path, str]] = []
    skipped_err = 0

    for raw in index_file.read_text(encoding="utf-8", errors="replace").splitlines():
        if not raw or raw.startswith("#"):
            continue
        fields = raw.split()
        if len(fields) < 9:
            continue

        line_id, status = fields[0], fields[1]
        if status == "err":
            skipped_err += 1
            continue

        transcription = fields[-1].replace("|", " ")
        label = label_filter.accept(transcription)
        if label is None:
            continue

        # a01-000u-00 -> lines/a01/a01-000u/a01-000u-00.png
        form = line_id.rsplit("-", 1)[0]
        prefix = form.split("-")[0]
        image_path = lines_dir / prefix / form / f"{line_id}.png"
        if not image_path.is_file():
            report.skipped_bad_image += 1
            continue

        records.append((image_path, label))

    if not records:
        raise FileNotFoundError(
            f"Parsed {index_file} but found no usable line images under {lines_dir}."
        )

    logger.info("IAM: %d usable line(s) (%d skipped as segmentation errors)", len(records), skipped_err)
    report.notes += f" {skipped_err} line(s) skipped as segmentation errors."

    _write_split_records(records, out, report, counts, rng, dataset_label="IAM", seed=seed)
    label_filter.finalise()
    logger.info("%s", report.summary())
    return report


# ---------------------------------------------------------------------------
# IIIT 5K-Word
# ---------------------------------------------------------------------------
def prepare_iiit5k(
    root: str | Path,
    output_dir: str | Path,
    *,
    charset: Charset,
    counts: SplitCounts,
    max_label_length: int = 32,
    seed: int = 1234,
) -> PreparationReport:
    """Import IIIT 5K-Word cropped word images.

    Expects ``<root>/traindata.mat`` and ``<root>/testdata.mat`` alongside the
    ``train/`` and ``test/`` image directories.

    Raises:
        FileNotFoundError: the ``.mat`` annotations are missing.
        ImportError: SciPy is needed to read them and is not installed.
    """
    source = Path(root)
    train_mat = source / "traindata.mat"
    test_mat = source / "testdata.mat"

    if not train_mat.is_file() and not test_mat.is_file():
        raise FileNotFoundError(
            f"IIIT5K annotations not found under {source} (expected traindata.mat / testdata.mat). "
            "Download from https://cvit.iiit.ac.in/research/projects/cvit-projects/"
            "the-iiit-5k-word-dataset and extract there."
        )

    try:
        from scipy.io import loadmat
    except ImportError as exc:
        raise ImportError(
            "Reading IIIT5K's .mat annotations needs SciPy. Install it with:  pip install scipy"
        ) from exc

    out = Path(output_dir)
    report = PreparationReport(
        dataset="iiit5k",
        output_dir=str(out),
        notes="IIIT 5K-Word cropped scene-text images, official train/test split.",
    )
    label_filter = _LabelFilter(charset, max_label_length, report)
    rng = np.random.default_rng(seed)

    def read_mat(path: Path, key: str) -> list[tuple[Path, str]]:
        if not path.is_file():
            return []
        payload = loadmat(str(path), squeeze_me=True, struct_as_record=False)
        entries = payload.get(key)
        if entries is None:
            logger.warning("%s has no %r field; skipping", path.name, key)
            return []

        collected: list[tuple[Path, str]] = []
        for item in np.atleast_1d(entries):
            relative = str(getattr(item, "ImgName", "")).replace("\\", "/")
            label = label_filter.accept(str(getattr(item, "GroundTruth", "")))
            if not relative or label is None:
                continue
            image_path = source / relative
            if not image_path.is_file():
                report.skipped_bad_image += 1
                continue
            collected.append((image_path, label))
        return collected

    train_records = read_mat(train_mat, "traindata")
    test_records = read_mat(test_mat, "testdata")

    if not train_records and not test_records:
        raise FileNotFoundError(f"No usable IIIT5K samples found under {source}.")

    # IIIT5K ships only train and test; carve a validation split off train so
    # the trainer has something to monitor and early-stop on.
    if train_records:
        rng.shuffle(train_records)  # type: ignore[arg-type]
        split_at = max(1, int(len(train_records) * 0.1))
        val_records, train_records = train_records[:split_at], train_records[split_at:]
    else:
        val_records = []

    for split, records in (("train", train_records), ("val", val_records), ("test", test_records)):
        limit = counts.limit_for(split)
        chosen = records[:limit] if limit is not None and limit < len(records) else records
        report.written[split] = _copy_records(chosen, out, split)

    if counts.unlabeled > 0:
        leftover = train_records[sum(report.written.get(s, 0) for s in ("train",)) :]
        report.unlabeled_written = _copy_unlabeled(leftover[: counts.unlabeled], out, "IIIT5K")

    label_filter.finalise()
    logger.info("%s", report.summary())
    return report


# ---------------------------------------------------------------------------
# Shared writers
# ---------------------------------------------------------------------------
def _copy_records(records: Sequence[tuple[Path, str]], out: Path, split: str) -> int:
    """Copy images into ``<out>/images/<split>`` and write the manifest."""
    if not records:
        return 0

    directory = out / "images" / split
    directory.mkdir(parents=True, exist_ok=True)

    rows: list[tuple[str, str]] = []
    for index, (source_path, label) in enumerate(records):
        try:
            image = _usable_image(source_path.read_bytes())
        except OSError:
            image = None
        if image is None:
            continue
        name = f"{split}_{index:06d}.png"
        write_image(directory / name, image)
        rows.append((f"images/{split}/{name}", label))

    write_manifest(
        out / f"{split}.tsv",
        rows,
        header_comment=f"{split} split | {len(rows)} samples\nformat: <relative image path>\\t<label>",
    )
    return len(rows)


def _copy_unlabeled(records: Sequence[tuple[Path, str]], out: Path, source_name: str) -> int:
    """Write an unlabeled pool, discarding the labels."""
    if not records:
        return 0

    directory = out / "unlabeled"
    directory.mkdir(parents=True, exist_ok=True)

    written: list[str] = []
    for index, (source_path, _label) in enumerate(records):
        try:
            image = _usable_image(source_path.read_bytes())
        except OSError:
            image = None
        if image is None:
            continue
        name = f"unlab_{index:06d}.png"
        write_image(directory / name, image)
        written.append(f"unlabeled/{name}")

    write_unlabeled_manifest(
        out / "unlabeled.txt",
        written,
        header_comment=(
            f"Unlabeled pool for SimCLR pretraining, from {source_name}.\n"
            "Images not used in any labeled split; their transcriptions are not read."
        ),
    )
    return len(written)


def _write_split_records(
    records: list[tuple[Path, str]],
    out: Path,
    report: PreparationReport,
    counts: SplitCounts,
    rng: np.random.Generator,
    *,
    dataset_label: str,
    seed: int,
    ratios: tuple[float, float, float] = (0.8, 0.1, 0.1),
) -> None:
    """Shuffle *records* and split them into train/val/test by *ratios*.

    Used by datasets that ship without an official split.
    """
    order = rng.permutation(len(records))
    shuffled = [records[int(i)] for i in order]

    n_train = int(len(shuffled) * ratios[0])
    n_val = int(len(shuffled) * ratios[1])

    partitions = {
        "train": shuffled[:n_train],
        "val": shuffled[n_train : n_train + n_val],
        "test": shuffled[n_train + n_val :],
    }

    used: set[Path] = set()
    for split, records_for_split in partitions.items():
        limit = counts.limit_for(split)
        chosen = (
            records_for_split[:limit]
            if limit is not None and limit < len(records_for_split)
            else records_for_split
        )
        report.written[split] = _copy_records(chosen, out, split)
        used.update(path for path, _ in chosen)

    if counts.unlabeled > 0:
        leftover = [item for item in shuffled if item[0] not in used]
        report.unlabeled_written = _copy_unlabeled(leftover[: counts.unlabeled], out, dataset_label)
