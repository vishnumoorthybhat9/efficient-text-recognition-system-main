"""Tests for the image-enhancement pipeline and augmentation."""

from __future__ import annotations

import numpy as np
import pytest

from ml.preprocessing import ops
from ml.preprocessing.augment import (
    AugmentOptions,
    OCRAugmentor,
    SimCLRViewGenerator,
    SimCLRViewOptions,
)
from ml.preprocessing.pipeline import (
    PreprocessOptions,
    PreprocessPipeline,
    prepare_model_input,
)


# --------------------------------------------------------------------------
# ops
# --------------------------------------------------------------------------
def test_to_grayscale_handles_every_channel_count() -> None:
    for shape in [(20, 30), (20, 30, 1), (20, 30, 3), (20, 30, 4)]:
        image = np.full(shape, 128, dtype=np.uint8)
        gray = ops.to_grayscale(image)
        assert gray.ndim == 2 and gray.shape == (20, 30)


def test_to_grayscale_does_not_mutate_its_input() -> None:
    image = np.full((10, 10, 3), 100, dtype=np.uint8)
    before = image.copy()
    ops.to_grayscale(image)
    assert np.array_equal(image, before)


def test_ensure_uint8_scales_unit_floats() -> None:
    result = ops.ensure_uint8(np.full((4, 4), 0.5, dtype=np.float32))
    assert result.dtype == np.uint8
    assert result[0, 0] == pytest.approx(127, abs=2)


def test_polarity_normalisation_inverts_light_on_dark() -> None:
    inverted = np.full((40, 80), 20, dtype=np.uint8)
    inverted[15:25, 10:70] = 230  # light text on dark paper

    result, was_inverted = ops.normalise_polarity(inverted)
    assert was_inverted
    # Background should now be dark-valued -> high, text low.
    assert result[0, 0] > 200


def test_polarity_leaves_dark_on_light_alone(text_image: np.ndarray) -> None:
    result, was_inverted = ops.normalise_polarity(text_image)
    assert not was_inverted
    assert np.array_equal(result, text_image)


@pytest.mark.parametrize("method", ["otsu", "adaptive", "sauvola"])
def test_binarisation_produces_two_tones(text_image: np.ndarray, method: str) -> None:
    result = ops.binarize(text_image, method=method)
    assert set(np.unique(result)).issubset({0, 255})
    # Some ink and some paper: a method that produced a blank image would pass
    # the two-tone check but be useless.
    assert 0.001 < (result == 0).mean() < 0.9


def test_unknown_binarisation_method_is_rejected(text_image: np.ndarray) -> None:
    with pytest.raises(ValueError, match="Unknown binarisation method"):
        ops.binarize(text_image, method="magic")


def test_deskew_recovers_a_known_rotation(text_image: np.ndarray) -> None:
    rotated, _ = ops.deskew(text_image, angle=-6.0)
    _, recovered = ops.deskew(rotated)
    # The estimator should see roughly the opposite of the rotation applied.
    assert recovered == pytest.approx(6.0, abs=2.5)


def test_deskew_leaves_straight_text_alone(text_image: np.ndarray) -> None:
    _, angle = ops.deskew(text_image)
    assert abs(angle) < 1.5


def test_resize_with_padding_keeps_aspect_ratio() -> None:
    image = np.full((50, 400), 255, dtype=np.uint8)
    canvas, info = ops.resize_with_padding(image, height=32, width=256)

    assert canvas.shape == (32, 256)
    # 400 x (32/50) = 256, so the content fills the canvas exactly.
    assert info.scaled_width == 256
    assert info.content_ratio == pytest.approx(1.0)


def test_resize_pads_a_narrow_image() -> None:
    image = np.full((50, 100), 0, dtype=np.uint8)
    canvas, info = ops.resize_with_padding(image, height=32, width=256, pad_value=255)

    assert info.scaled_width == 64
    assert info.content_ratio == pytest.approx(64 / 256)
    # Everything past the content must be padding.
    assert (canvas[:, 64:] == 255).all()


def test_resize_crops_an_over_wide_image() -> None:
    image = np.full((32, 2000), 0, dtype=np.uint8)
    canvas, info = ops.resize_with_padding(image, height=32, width=256)
    assert canvas.shape == (32, 256)
    assert info.content_ratio == pytest.approx(1.0)


def test_empty_image_is_rejected() -> None:
    with pytest.raises(ValueError):
        ops.resize_with_padding(np.zeros((0, 0), dtype=np.uint8), height=32, width=256)


# --------------------------------------------------------------------------
# pipeline
# --------------------------------------------------------------------------
def test_pipeline_defaults_to_grayscale(text_image: np.ndarray) -> None:
    result = PreprocessPipeline().run(text_image)
    assert result.image.ndim == 2


def test_pipeline_captures_a_stage_per_step(text_image: np.ndarray) -> None:
    # deskew is on by default, so it appears without being asked for.
    options = PreprocessOptions(clahe=True, sharpen=True, binarize="otsu")
    result = PreprocessPipeline(options).run(text_image, capture_stages=True)

    names = [stage.name for stage in result.stages]
    assert names == ["input", "grayscale", "polarity", "deskew", "clahe", "sharpen", "binarize"]
    assert all(stage.image.size > 0 for stage in result.stages)


def test_deskew_is_on_by_default() -> None:
    """Photographed pages are rarely square, and skew collapses line detection."""
    assert PreprocessOptions().deskew is True


def test_deskew_can_be_turned_off(text_image: np.ndarray) -> None:
    options = PreprocessOptions(deskew=False, binarize="otsu")
    result = PreprocessPipeline(options).run(text_image, capture_stages=True)
    assert "deskew" not in [stage.name for stage in result.stages]


def test_pipeline_skips_stage_capture_by_default(text_image: np.ndarray) -> None:
    result = PreprocessPipeline().run(text_image)
    assert result.stages == []


def test_pipeline_metadata_is_json_serialisable(text_image: np.ndarray) -> None:
    import json

    result = PreprocessPipeline(PreprocessOptions(deskew=True)).run(text_image)
    json.dumps(result.metadata())  # must not raise


def test_pipeline_rejects_an_empty_image() -> None:
    with pytest.raises(ValueError, match="empty image"):
        PreprocessPipeline().run(np.zeros((0, 0), dtype=np.uint8))


def test_invalid_binarise_option_is_rejected() -> None:
    with pytest.raises(ValueError, match="binarize must be one of"):
        PreprocessOptions(binarize="nope")


def test_prepare_model_input_shape_and_range(text_image: np.ndarray) -> None:
    array, info = prepare_model_input(text_image, height=32, width=256, channels=1)

    assert array.shape == (1, 32, 256)
    assert array.dtype == np.float32
    # Default normalisation is (x/255 - 0.5)/0.5, i.e. [-1, 1].
    assert array.min() >= -1.0001 and array.max() <= 1.0001
    assert 0 < info.content_ratio <= 1


def test_prepare_model_input_supports_three_channels(text_image: np.ndarray) -> None:
    array, _ = prepare_model_input(text_image, height=32, width=256, channels=3)
    assert array.shape == (3, 32, 256)


# --------------------------------------------------------------------------
# augmentation
# --------------------------------------------------------------------------
def test_augmentation_preserves_shape_and_dtype(text_image: np.ndarray) -> None:
    augmented = OCRAugmentor(AugmentOptions(), seed=0)(text_image)
    assert augmented.shape == text_image.shape
    assert augmented.dtype == np.uint8


def test_augmentation_is_reproducible_with_a_seed(text_image: np.ndarray) -> None:
    first = OCRAugmentor(AugmentOptions(), seed=7)(text_image)
    second = OCRAugmentor(AugmentOptions(), seed=7)(text_image)
    assert np.array_equal(first, second)


def test_disabled_augmentation_is_a_no_op(text_image: np.ndarray) -> None:
    augmented = OCRAugmentor(AugmentOptions(enabled=False), seed=0)(text_image)
    assert np.array_equal(augmented, text_image)


def test_horizontal_flip_is_refused() -> None:
    # Mirroring an image invalidates its transcription; the option exists only
    # so that asking for it fails loudly.
    with pytest.raises(ValueError, match="horizontal_flip"):
        AugmentOptions(horizontal_flip=True)


def test_simclr_generates_two_different_views(text_image: np.ndarray) -> None:
    view_a, view_b = SimCLRViewGenerator(SimCLRViewOptions(), seed=0)(text_image)

    assert view_a.dtype == np.uint8 and view_b.dtype == np.uint8
    # Different crops may give different widths; compare the overlap.
    width = min(view_a.shape[1], view_b.shape[1])
    assert not np.array_equal(view_a[:, :width], view_b[:, :width])


def test_simclr_views_are_reproducible(text_image: np.ndarray) -> None:
    first = SimCLRViewGenerator(SimCLRViewOptions(), seed=3)(text_image)
    second = SimCLRViewGenerator(SimCLRViewOptions(), seed=3)(text_image)
    assert np.array_equal(first[0], second[0])
    assert np.array_equal(first[1], second[1])
