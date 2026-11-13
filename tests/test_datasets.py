"""Tests for manifests, the synthetic renderer and the PyTorch datasets."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from backend.core.errors import NotFound
from backend.services.dataset_service import dataset_detail, list_datasets
from ml.datasets.lexicon import TextSampler
from ml.datasets.manifest import (
    read_manifest,
    read_unlabeled_manifest,
    validate_manifest,
    write_manifest,
    write_unlabeled_manifest,
)
from ml.datasets.synthetic import SyntheticRenderer, discover_fonts
from ml.text.charset import Charset
from ml.utils.image_io import ImageDecodeError, decode_image_bytes, encode_image, read_image, write_image


# --------------------------------------------------------------------------
# image io
# --------------------------------------------------------------------------
def test_image_round_trips_through_disk(tmp_path: Path, text_image: np.ndarray) -> None:
    path = write_image(tmp_path / "sample.png", text_image)
    assert np.array_equal(read_image(path), text_image)


def test_encode_decode_round_trip(text_image: np.ndarray) -> None:
    decoded = decode_image_bytes(encode_image(text_image, extension=".png"), grayscale=True)
    assert np.array_equal(decoded, text_image)


def test_reading_a_missing_file_raises(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        read_image(tmp_path / "absent.png")


def test_decoding_garbage_raises() -> None:
    with pytest.raises(ImageDecodeError):
        decode_image_bytes(b"this is not an image")


def test_decoding_nothing_raises() -> None:
    with pytest.raises(ImageDecodeError, match="0 bytes"):
        decode_image_bytes(b"")


# --------------------------------------------------------------------------
# manifests
# --------------------------------------------------------------------------
def _tiny_dataset(tmp_path: Path, count: int = 5) -> Path:
    images = tmp_path / "images"
    images.mkdir()
    rows = []
    for index in range(count):
        image = np.full((40, 120), 240, dtype=np.uint8)
        image[12:28, 10:110] = 40
        write_image(images / f"{index:03d}.png", image)
        rows.append((f"images/{index:03d}.png", f"label {index}"))
    return write_manifest(tmp_path / "train.tsv", rows)


def test_manifest_round_trip(tmp_path: Path) -> None:
    manifest = _tiny_dataset(tmp_path, 3)
    entries = read_manifest(manifest)

    assert len(entries) == 3
    assert entries[0].label == "label 0"
    assert entries[0].path.is_file()
    # Paths are relative to the manifest, so a prepared dataset stays movable.
    assert entries[0].relative_path == "images/000.png"


def test_labels_may_contain_spaces(tmp_path: Path) -> None:
    (tmp_path / "images").mkdir()
    write_image(tmp_path / "images" / "a.png", np.full((20, 60), 200, dtype=np.uint8))
    manifest = write_manifest(tmp_path / "m.tsv", [("images/a.png", "two  words here")])
    assert read_manifest(manifest)[0].label == "two words here"


def test_comments_and_blank_lines_are_ignored(tmp_path: Path) -> None:
    (tmp_path / "images").mkdir()
    write_image(tmp_path / "images" / "a.png", np.full((20, 60), 200, dtype=np.uint8))
    manifest = tmp_path / "m.tsv"
    manifest.write_text("# a comment\n\nimages/a.png\tlabel\n", encoding="utf-8")
    assert len(read_manifest(manifest)) == 1


def test_malformed_line_is_rejected(tmp_path: Path) -> None:
    manifest = tmp_path / "bad.tsv"
    manifest.write_text("images/a.png no tab here\n", encoding="utf-8")
    with pytest.raises(ValueError, match="malformed"):
        read_manifest(manifest)


def test_missing_manifest_says_how_to_make_one(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="make_synthetic_data"):
        read_manifest(tmp_path / "absent.tsv")


def test_skip_missing_drops_absent_images(tmp_path: Path) -> None:
    manifest = _tiny_dataset(tmp_path, 3)
    (tmp_path / "images" / "001.png").unlink()
    assert len(read_manifest(manifest, skip_missing=True)) == 2


def test_empty_labels_are_rejected_on_write(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="must not be empty"):
        write_manifest(tmp_path / "m.tsv", [("a.png", "   ")])


def test_unlabeled_manifest_round_trip(tmp_path: Path) -> None:
    (tmp_path / "images").mkdir()
    for index in range(3):
        write_image(tmp_path / "images" / f"{index}.png", np.full((20, 60), 200, dtype=np.uint8))

    manifest = write_unlabeled_manifest(
        tmp_path / "unlabeled.txt", [f"images/{i}.png" for i in range(3)]
    )
    assert len(read_unlabeled_manifest(manifest)) == 3


def test_validation_reports_over_long_labels(tmp_path: Path, charset: Charset) -> None:
    (tmp_path / "images").mkdir()
    write_image(tmp_path / "images" / "a.png", np.full((20, 60), 200, dtype=np.uint8))
    manifest = write_manifest(tmp_path / "m.tsv", [("images/a.png", "x" * 50)])

    report = validate_manifest(manifest, charset, max_label_length=10)
    assert not report.ok
    assert len(report.too_long) == 1


def test_validation_reports_unsupported_characters(tmp_path: Path) -> None:
    (tmp_path / "images").mkdir()
    write_image(tmp_path / "images" / "a.png", np.full((20, 60), 200, dtype=np.uint8))
    manifest = write_manifest(tmp_path / "m.tsv", [("images/a.png", "abc")])

    report = validate_manifest(manifest, Charset("xyz"), max_label_length=10)
    assert set(report.unsupported_characters) == {"a", "b", "c"}


def test_validation_passes_a_clean_manifest(tmp_path: Path, charset: Charset) -> None:
    manifest = _tiny_dataset(tmp_path, 4)
    report = validate_manifest(manifest, charset, max_label_length=30)
    assert report.ok and report.usable == 4


# --------------------------------------------------------------------------
# synthetic rendering
# --------------------------------------------------------------------------
def test_text_sampler_respects_the_charset_and_length() -> None:
    # Lowercase letters plus a space: enough of the lexicon survives filtering,
    # while digits, capitals and punctuation must never appear.
    allowed = "abcdefghijklmnopqrstuvwxyz "
    sampler = TextSampler(charset_characters=allowed, max_length=6, seed=0)

    for _ in range(60):
        text = sampler.sample()
        assert text and len(text) <= 6
        assert set(text) <= set(allowed), f"{text!r} escaped the charset"


def test_text_sampler_is_reproducible() -> None:
    first = TextSampler(seed=11).sample_many(15)
    second = TextSampler(seed=11).sample_many(15)
    assert first == second


def test_text_sampler_rejects_an_impossible_charset() -> None:
    with pytest.raises(ValueError, match="No lexicon word"):
        TextSampler(charset_characters="0123456789")


def test_renderer_produces_a_legible_line() -> None:
    renderer = SyntheticRenderer(height=64, seed=0)
    image, style = renderer.render("Hello World", kind="printed")

    assert image.ndim == 2 and image.dtype == np.uint8
    assert image.shape[0] == 64
    # There must be real contrast, or nothing was drawn.
    assert int(image.max()) - int(image.min()) > 60
    assert style.kind == "printed"


def test_renderer_handles_the_handwritten_style() -> None:
    image, style = SyntheticRenderer(height=64, seed=1).render("signature", kind="handwritten")
    assert image.shape[0] == 64
    assert style.kind == "handwritten"
    # Handwriting is rendered with per-character baseline jitter.
    assert style.char_jitter > 0


def test_renderer_rejects_empty_text() -> None:
    with pytest.raises(ValueError, match="empty text"):
        SyntheticRenderer(seed=0).render("   ")


def test_renderer_is_reproducible() -> None:
    first, _ = SyntheticRenderer(height=48, seed=5).render("Reproducible", kind="printed")
    second, _ = SyntheticRenderer(height=48, seed=5).render("Reproducible", kind="printed")
    assert np.array_equal(first, second)


def test_font_discovery_never_raises() -> None:
    catalogue = discover_fonts(verify=False)
    assert catalogue.total >= 0  # may be zero in a bare container
    assert isinstance(catalogue.to_dict(), dict)


# --------------------------------------------------------------------------
# torch datasets
# --------------------------------------------------------------------------
def test_ocr_dataset_yields_model_ready_tensors(tmp_path: Path, charset: Charset) -> None:
    pytest.importorskip("torch")
    from ml.datasets.ocr_dataset import ImageGeometry, OCRLineDataset

    manifest = _tiny_dataset(tmp_path, 4)
    dataset = OCRLineDataset(
        charset=charset, manifest=manifest, geometry=ImageGeometry(), train=False
    )
    sample = dataset[0]

    assert sample.image.shape == (1, 32, 256)
    assert sample.target.numel() == len(charset.encode(sample.text))
    assert 0 < sample.width_ratio <= 1


def test_collated_input_lengths_never_fall_below_target_lengths(
    tmp_path: Path, charset: Charset
) -> None:
    """CTC returns inf when the input is shorter than the label."""
    pytest.importorskip("torch")
    from ml.datasets.ocr_dataset import ImageGeometry, OCRLineDataset, build_ocr_dataloader

    images = tmp_path / "images"
    images.mkdir()
    rows = []
    for index in range(4):
        # Deliberately narrow images with long labels: the worst case.
        image = np.full((40, 50), 240, dtype=np.uint8)
        image[10:30, 5:45] = 30
        write_image(images / f"{index}.png", image)
        rows.append((f"images/{index}.png", "a long label for a narrow image"))
    manifest = write_manifest(tmp_path / "m.tsv", rows)

    dataset = OCRLineDataset(charset=charset, manifest=manifest, geometry=ImageGeometry(), train=False)
    batch = next(iter(build_ocr_dataloader(dataset, batch_size=4)))
    lengths = batch.input_lengths(63)

    assert (lengths >= batch.target_lengths).all()
    assert (lengths <= 63).all() and (lengths >= 1).all()


def test_max_samples_is_deterministic_across_instances(tmp_path: Path, charset: Charset) -> None:
    """Both experiment arms must see the identical labeled subset."""
    pytest.importorskip("torch")
    from ml.datasets.ocr_dataset import ImageGeometry, OCRLineDataset

    manifest = _tiny_dataset(tmp_path, 20)
    first = OCRLineDataset(
        charset=charset, manifest=manifest, geometry=ImageGeometry(), train=True, max_samples=7, seed=42
    )
    second = OCRLineDataset(
        charset=charset, manifest=manifest, geometry=ImageGeometry(), train=True, max_samples=7, seed=42
    )

    assert len(first) == len(second) == 7
    assert [e.relative_path for e in first.entries] == [e.relative_path for e in second.entries]


def test_document_benchmark_is_listed_as_evaluation_only(tmp_path: Path) -> None:
    directory = tmp_path / "docbench"
    pages = directory / "pages"
    pages.mkdir(parents=True)
    (pages / "page.png").write_bytes(b"image")
    (directory / "benchmark.json").write_text(
        json.dumps(
            {
                "entries": [
                    {
                        "file": "pages/page.png",
                        "text": "Exact benchmark text",
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    runtime = SimpleNamespace(
        dataset_dir=tmp_path,
        config=SimpleNamespace(charset={"characters": Charset.default().characters}),
    )

    listed = list_datasets(runtime)["prepared"][0]
    detail = dataset_detail(runtime, "docbench", split="test")

    assert listed["benchmark_only"] is True
    assert listed["source"] == "benchmark"
    assert listed["trainable"] is False
    assert listed["splits"][2]["available"] is True
    assert listed["splits"][2]["count"] == 1
    assert detail["samples"] == [
        {"id": "pages/page.png", "label": "Exact benchmark text"}
    ]

    with pytest.raises(NotFound, match="evaluation-only benchmark"):
        dataset_detail(runtime, "docbench", split="train")


def test_unlabeled_dataset_yields_two_views(tmp_path: Path) -> None:
    pytest.importorskip("torch")
    import torch

    from ml.datasets.ocr_dataset import ImageGeometry, UnlabeledLineDataset

    images = tmp_path / "images"
    images.mkdir()
    for index in range(3):
        image = np.full((40, 120), 240, dtype=np.uint8)
        image[12:28, 10:110] = 40
        write_image(images / f"{index}.png", image)
    manifest = write_unlabeled_manifest(
        tmp_path / "unlabeled.txt", [f"images/{i}.png" for i in range(3)]
    )

    dataset = UnlabeledLineDataset(manifest=manifest, geometry=ImageGeometry(), seed=0)
    view_a, view_b, _name = dataset[0]

    assert view_a.shape == view_b.shape == (1, 32, 256)
    assert not torch.equal(view_a, view_b)
