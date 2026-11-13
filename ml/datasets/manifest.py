"""Manifest files: the project's single dataset interchange format.

Every dataset -- synthetic, IAM, IIIT5K, MJSynth -- is converted to the same
two-column TSV so that exactly one ``Dataset`` implementation is needed::

    images/000001.png<TAB>Invoice total
    images/000002.png<TAB>INV-4471

Rules
-----
* Image paths are **relative to the directory containing the manifest**, which
  keeps a prepared dataset movable between machines.
* Labels may contain spaces (only the *first* tab is the separator) but not
  tabs or newlines.
* Lines starting with ``#`` are comments; blank lines are ignored.

An unlabeled manifest (for self-supervised pretraining) is the same thing with
the label column omitted.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Iterator, Sequence

from ml.text.charset import Charset

__all__ = [
    "ManifestEntry",
    "ManifestValidation",
    "read_manifest",
    "write_manifest",
    "read_unlabeled_manifest",
    "write_unlabeled_manifest",
    "validate_manifest",
]

logger = logging.getLogger(__name__)

COMMENT_PREFIX = "#"


@dataclass(slots=True)
class ManifestEntry:
    """One labeled image.

    Attributes:
        path: absolute path to the image file.
        label: the ground-truth transcription.
        relative_path: the path exactly as written in the manifest, used as a
            stable sample id in reports and in the API.
    """

    path: Path
    label: str
    relative_path: str = ""

    def __post_init__(self) -> None:
        if not self.relative_path:
            self.relative_path = self.path.name

    def to_dict(self) -> dict[str, Any]:
        return {"path": str(self.path), "label": self.label, "id": self.relative_path}


def _sanitise_label(label: str) -> str:
    """Make *label* safe to store in one TSV field.

    Raises:
        ValueError: the label is empty after cleaning.
    """
    cleaned = str(label).replace("\t", " ").replace("\r", " ").replace("\n", " ")
    cleaned = " ".join(cleaned.split())
    if not cleaned:
        raise ValueError("Manifest labels must not be empty")
    return cleaned


def write_manifest(
    path: str | Path,
    entries: Iterable[tuple[str, str] | ManifestEntry],
    *,
    header_comment: str | None = None,
) -> Path:
    """Write a labeled manifest.

    Args:
        path: output ``.tsv`` file; parent directories are created.
        entries: ``(relative_path, label)`` tuples, or :class:`ManifestEntry`
            objects (whose ``relative_path`` is used).
        header_comment: optional provenance note written as a ``#`` comment.

    Returns:
        The manifest path.
    """
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)

    written = 0
    with target.open("w", encoding="utf-8", newline="\n") as handle:
        if header_comment:
            for line in str(header_comment).splitlines():
                handle.write(f"{COMMENT_PREFIX} {line}\n")

        for entry in entries:
            if isinstance(entry, ManifestEntry):
                relative, label = entry.relative_path, entry.label
            else:
                relative, label = entry
            handle.write(f"{Path(relative).as_posix()}\t{_sanitise_label(label)}\n")
            written += 1

    logger.info("Wrote %d entries to %s", written, target)
    return target


def read_manifest(path: str | Path, *, skip_missing: bool = False) -> list[ManifestEntry]:
    """Read a labeled manifest.

    Args:
        path: the ``.tsv`` file.
        skip_missing: drop entries whose image file does not exist instead of
            keeping them (a missing file would fail later, in a DataLoader
            worker, where the error is much harder to read).

    Returns:
        The entries, in file order.

    Raises:
        FileNotFoundError: the manifest itself is missing.
        ValueError: a line has no tab separator.
    """
    manifest_path = Path(path)
    if not manifest_path.is_file():
        raise FileNotFoundError(
            f"Manifest not found: {manifest_path}\n"
            "Generate the synthetic dataset first:\n"
            "  python scripts/make_synthetic_data.py"
        )

    root = manifest_path.parent
    entries: list[ManifestEntry] = []
    missing = 0
    malformed: list[int] = []

    with manifest_path.open("r", encoding="utf-8") as handle:
        for line_number, raw in enumerate(handle, start=1):
            line = raw.rstrip("\n").rstrip("\r")
            if not line.strip() or line.lstrip().startswith(COMMENT_PREFIX):
                continue

            if "\t" not in line:
                malformed.append(line_number)
                continue

            relative, label = line.split("\t", 1)
            relative, label = relative.strip(), label.strip()
            if not relative or not label:
                malformed.append(line_number)
                continue

            image_path = (root / relative).resolve()
            if skip_missing and not image_path.is_file():
                missing += 1
                continue

            entries.append(ManifestEntry(path=image_path, label=label, relative_path=relative))

    if malformed:
        preview = ", ".join(str(n) for n in malformed[:5])
        raise ValueError(
            f"{manifest_path} has {len(malformed)} malformed line(s) (first few: {preview}). "
            "Each line must be '<relative/path.png><TAB><label>'."
        )
    if missing:
        logger.warning("Skipped %d entry(ies) in %s whose image file is missing", missing, manifest_path)
    if not entries:
        raise ValueError(f"{manifest_path} contains no usable entries")

    logger.info("Loaded %d entries from %s", len(entries), manifest_path)
    return entries


def write_unlabeled_manifest(
    path: str | Path,
    relative_paths: Iterable[str],
    *,
    header_comment: str | None = None,
) -> Path:
    """Write a one-path-per-line manifest for self-supervised pretraining."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)

    written = 0
    with target.open("w", encoding="utf-8", newline="\n") as handle:
        if header_comment:
            for line in str(header_comment).splitlines():
                handle.write(f"{COMMENT_PREFIX} {line}\n")
        for relative in relative_paths:
            handle.write(f"{Path(relative).as_posix()}\n")
            written += 1

    logger.info("Wrote %d unlabeled paths to %s", written, target)
    return target


def read_unlabeled_manifest(path: str | Path, *, skip_missing: bool = True) -> list[Path]:
    """Read an unlabeled manifest into absolute image paths.

    Raises:
        FileNotFoundError: the manifest is missing.
        ValueError: no usable paths remain.
    """
    manifest_path = Path(path)
    if not manifest_path.is_file():
        raise FileNotFoundError(
            f"Unlabeled manifest not found: {manifest_path}\n"
            "Generate the synthetic dataset first:\n"
            "  python scripts/make_synthetic_data.py"
        )

    root = manifest_path.parent
    paths: list[Path] = []
    missing = 0

    with manifest_path.open("r", encoding="utf-8") as handle:
        for raw in handle:
            line = raw.strip()
            if not line or line.startswith(COMMENT_PREFIX):
                continue
            image_path = (root / line).resolve()
            if skip_missing and not image_path.is_file():
                missing += 1
                continue
            paths.append(image_path)

    if missing:
        logger.warning("Skipped %d missing image(s) listed in %s", missing, manifest_path)
    if not paths:
        raise ValueError(f"{manifest_path} contains no usable image paths")

    logger.info("Loaded %d unlabeled image paths from %s", len(paths), manifest_path)
    return paths


@dataclass(slots=True)
class ManifestValidation:
    """Result of :func:`validate_manifest`."""

    manifest: str
    total: int
    usable: int
    missing_files: list[str] = field(default_factory=list)
    empty_labels: list[str] = field(default_factory=list)
    too_long: list[tuple[str, int]] = field(default_factory=list)
    unsupported_characters: dict[str, int] = field(default_factory=dict)
    label_length_min: int = 0
    label_length_max: int = 0
    label_length_mean: float = 0.0

    @property
    def ok(self) -> bool:
        """True when every entry is usable as-is."""
        return not (self.missing_files or self.empty_labels or self.too_long or self.unsupported_characters)

    def to_dict(self) -> dict[str, Any]:
        return {
            "manifest": self.manifest,
            "total": self.total,
            "usable": self.usable,
            "ok": self.ok,
            "missing_files": self.missing_files[:50],
            "missing_file_count": len(self.missing_files),
            "empty_labels": self.empty_labels[:50],
            "too_long": [{"id": i, "length": n} for i, n in self.too_long[:50]],
            "too_long_count": len(self.too_long),
            "unsupported_characters": self.unsupported_characters,
            "label_length_min": self.label_length_min,
            "label_length_max": self.label_length_max,
            "label_length_mean": round(self.label_length_mean, 2),
        }

    def summary(self) -> str:
        if self.ok:
            return (
                f"{self.manifest}: {self.usable}/{self.total} entries OK "
                f"(label length {self.label_length_min}-{self.label_length_max}, "
                f"mean {self.label_length_mean:.1f})"
            )
        problems: list[str] = []
        if self.missing_files:
            problems.append(f"{len(self.missing_files)} missing file(s)")
        if self.empty_labels:
            problems.append(f"{len(self.empty_labels)} empty label(s)")
        if self.too_long:
            problems.append(f"{len(self.too_long)} label(s) too long for the time steps available")
        if self.unsupported_characters:
            problems.append(f"{len(self.unsupported_characters)} character(s) outside the charset")
        return f"{self.manifest}: {self.usable}/{self.total} usable -- " + "; ".join(problems)


def validate_manifest(
    path: str | Path,
    charset: Charset,
    *,
    max_label_length: int | None = None,
    check_files: bool = True,
) -> ManifestValidation:
    """Check a manifest against the charset and the model's time-step budget.

    Catching these problems up front matters: an over-long label makes the CTC
    loss ``inf`` (there are not enough frames to emit it), which otherwise
    surfaces as a mysteriously diverging training run.

    Args:
        path: the manifest to check.
        charset: the charset the model will be trained with.
        max_label_length: time steps available to CTC.  Labels longer than
            this are reported.
        check_files: stat every image path.

    Returns:
        A :class:`ManifestValidation` report (never raises for bad *content*).
    """
    entries = read_manifest(path, skip_missing=False)

    missing: list[str] = []
    empty: list[str] = []
    too_long: list[tuple[str, int]] = []
    unsupported: dict[str, int] = {}
    lengths: list[int] = []

    for entry in entries:
        if check_files and not entry.path.is_file():
            missing.append(entry.relative_path)
            continue

        normalized = charset.normalize(entry.label)
        if not normalized:
            empty.append(entry.relative_path)
            continue

        for character in charset.unsupported(entry.label):
            unsupported[character] = unsupported.get(character, 0) + 1

        encoded_length = len(charset.encode(entry.label))
        lengths.append(encoded_length)
        if max_label_length is not None and encoded_length > max_label_length:
            too_long.append((entry.relative_path, encoded_length))

    usable = len(entries) - len(missing) - len(empty) - len(too_long)

    return ManifestValidation(
        manifest=str(path),
        total=len(entries),
        usable=max(0, usable),
        missing_files=missing,
        empty_labels=empty,
        too_long=too_long,
        unsupported_characters=dict(sorted(unsupported.items(), key=lambda kv: -kv[1])),
        label_length_min=min(lengths) if lengths else 0,
        label_length_max=max(lengths) if lengths else 0,
        label_length_mean=sum(lengths) / len(lengths) if lengths else 0.0,
    )


def iter_labels(entries: Sequence[ManifestEntry]) -> Iterator[str]:
    """Yield just the labels (used to derive a charset from a corpus)."""
    for entry in entries:
        yield entry.label
