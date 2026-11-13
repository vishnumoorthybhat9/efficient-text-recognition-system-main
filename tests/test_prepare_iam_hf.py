from __future__ import annotations

from pathlib import Path

from scripts.prepare_iam_hf import resolve_base_dataset


def _make_dataset(path: Path) -> Path:
    path.mkdir(parents=True)
    for split in ("train.tsv", "val.tsv", "test.tsv"):
        (path / split).touch()
    return path


def test_base_dataset_prefers_combined_over_synthetic(tmp_path: Path) -> None:
    combined = _make_dataset(tmp_path / "combined")
    _make_dataset(tmp_path / "synthetic")

    assert resolve_base_dataset(None, tmp_path) == combined


def test_base_dataset_falls_back_to_synthetic_when_combined_is_missing(
    tmp_path: Path,
) -> None:
    synthetic = _make_dataset(tmp_path / "synthetic")

    assert resolve_base_dataset(None, tmp_path) == synthetic


def test_explicit_base_does_not_fall_back(tmp_path: Path) -> None:
    synthetic = _make_dataset(tmp_path / "synthetic")
    invalid = tmp_path / "custom"

    assert resolve_base_dataset(invalid, tmp_path) is None
    assert synthetic.is_dir()


def test_base_dataset_requires_all_splits(tmp_path: Path) -> None:
    incomplete = tmp_path / "incomplete"
    incomplete.mkdir()
    (incomplete / "train.tsv").touch()
    (incomplete / "val.tsv").touch()

    assert resolve_base_dataset(None, tmp_path) is None
