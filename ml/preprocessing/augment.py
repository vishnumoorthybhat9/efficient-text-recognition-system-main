"""Augmentation for supervised OCR training and for SimCLR view generation.

Implemented directly on top of OpenCV rather than a third-party augmentation
library, for two reasons:

1. text augmentation has a hard constraint -- the transform must preserve the
   transcription -- so the useful set of operations is small and specific;
2. one less fast-moving dependency to pin.

**Never horizontally flip text.**  It is the single most common augmentation in
generic vision pipelines and it destroys the label, which is why the option is
absent rather than merely defaulted off.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Mapping

import cv2
import numpy as np

from ml.preprocessing import ops

__all__ = [
    "AugmentOptions",
    "SimCLRViewOptions",
    "OCRAugmentor",
    "SimCLRViewGenerator",
    "random_affine",
    "photometric_jitter",
]

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Primitive transforms
# ---------------------------------------------------------------------------
def random_affine(
    image: np.ndarray,
    rng: np.random.Generator,
    *,
    rotate_deg: float = 0.0,
    shear_deg: float = 0.0,
    scale_jitter: float = 0.0,
    translate_px: int = 0,
    border_value: int = 255,
) -> np.ndarray:
    """Apply a small random rotation / shear / scale / translation.

    Shear in particular mimics the natural slant variation of handwriting,
    which is why it is applied more aggressively than rotation.
    """
    if max(abs(rotate_deg), abs(shear_deg), abs(scale_jitter), abs(translate_px)) <= 0:
        return image

    height, width = image.shape[:2]
    angle = float(rng.uniform(-rotate_deg, rotate_deg)) if rotate_deg else 0.0
    scale = 1.0 + float(rng.uniform(-scale_jitter, scale_jitter)) if scale_jitter else 1.0

    matrix = cv2.getRotationMatrix2D((width / 2.0, height / 2.0), angle, scale)

    if shear_deg:
        shear = np.tan(np.deg2rad(float(rng.uniform(-shear_deg, shear_deg))))
        # Horizontal shear about the vertical centre, so the line stays put.
        shear_matrix = np.array([[1.0, shear, -shear * height / 2.0], [0.0, 1.0, 0.0]], dtype=np.float64)
        combined = np.vstack([shear_matrix, [0.0, 0.0, 1.0]]) @ np.vstack([matrix, [0.0, 0.0, 1.0]])
        matrix = combined[:2, :]

    if translate_px:
        matrix[0, 2] += float(rng.integers(-translate_px, translate_px + 1))
        matrix[1, 2] += float(rng.integers(-translate_px, translate_px + 1))

    fill: float | tuple[float, ...] = (
        float(border_value) if image.ndim == 2 else (float(border_value),) * image.shape[2]
    )
    return cv2.warpAffine(
        image,
        matrix,
        (width, height),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=fill,
    )


def photometric_jitter(
    image: np.ndarray,
    rng: np.random.Generator,
    *,
    brightness: float = 0.0,
    contrast: float = 0.0,
) -> np.ndarray:
    """Randomly scale contrast and shift brightness.

    ``out = (in - 128) * (1 + c) + 128 + b * 255``
    """
    if brightness <= 0.0 and contrast <= 0.0:
        return image

    gain = 1.0 + float(rng.uniform(-contrast, contrast)) if contrast else 1.0
    bias = float(rng.uniform(-brightness, brightness)) * 255.0 if brightness else 0.0

    adjusted = (image.astype(np.float32) - 128.0) * gain + 128.0 + bias
    return np.clip(adjusted, 0, 255).astype(np.uint8)


def add_gaussian_noise(image: np.ndarray, rng: np.random.Generator, std: float) -> np.ndarray:
    """Add zero-mean Gaussian noise; *std* is a fraction of full scale."""
    if std <= 0.0:
        return image
    sigma = float(rng.uniform(0.0, std)) * 255.0
    if sigma <= 0.0:
        return image
    noise = rng.normal(0.0, sigma, size=image.shape).astype(np.float32)
    return np.clip(image.astype(np.float32) + noise, 0, 255).astype(np.uint8)


def random_blur(image: np.ndarray, rng: np.random.Generator, max_sigma: float) -> np.ndarray:
    """Gaussian blur with a random radius, simulating focus/scan softness."""
    if max_sigma <= 0.0:
        return image
    sigma = float(rng.uniform(0.1, max_sigma))
    return cv2.GaussianBlur(image, (0, 0), sigmaX=sigma, sigmaY=sigma)


def random_jpeg_artifacts(image: np.ndarray, rng: np.random.Generator, min_quality: int) -> np.ndarray:
    """Re-encode as low-quality JPEG to introduce blocking artefacts.

    Real-world uploads are overwhelmingly recompressed photographs, so this
    closes a genuine train/test gap.
    """
    if min_quality <= 0 or min_quality >= 100:
        return image
    quality = int(rng.integers(min_quality, 96))
    success, buffer = cv2.imencode(".jpg", image, [int(cv2.IMWRITE_JPEG_QUALITY), quality])
    if not success:
        logger.debug("JPEG augmentation failed to encode; returning the input unchanged")
        return image
    decoded = cv2.imdecode(buffer, cv2.IMREAD_GRAYSCALE if image.ndim == 2 else cv2.IMREAD_COLOR)
    return decoded if decoded is not None else image


def random_erode_dilate(image: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Thicken or thin the strokes (ink bleed vs. a drying-out pen).

    Images are dark-text-on-light, so ``erode`` (which shrinks bright regions)
    thickens the text and ``dilate`` thins it.
    """
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2, 2))
    return cv2.erode(image, kernel) if rng.random() < 0.5 else cv2.dilate(image, kernel)


def random_width_crop(image: np.ndarray, rng: np.random.Generator, min_ratio: float) -> np.ndarray:
    """Crop a horizontal window of at least *min_ratio* of the width.

    Used only for SimCLR views: it changes the margins and the amount of
    context without changing which text is visible, provided *min_ratio*
    stays high.  It is **not** used for supervised training, where even a
    partial character loss would corrupt the label.
    """
    if min_ratio >= 1.0:
        return image
    width = image.shape[1]
    keep = int(round(width * float(rng.uniform(min_ratio, 1.0))))
    keep = max(8, min(width, keep))
    if keep >= width:
        return image
    start = int(rng.integers(0, width - keep + 1))
    return image[:, start : start + keep]


# ---------------------------------------------------------------------------
# Supervised OCR augmentation
# ---------------------------------------------------------------------------
@dataclass(slots=True)
class AugmentOptions:
    """Augmentation strength for supervised training.

    Mirrors the ``augment`` block of ``configs/default.yaml``.
    """

    enabled: bool = True
    probability: float = 0.8
    rotate_deg: float = 2.0
    shear_deg: float = 4.0
    scale_jitter: float = 0.08
    translate_px: int = 2
    brightness: float = 0.25
    contrast: float = 0.25
    gaussian_noise_std: float = 0.035
    blur_max_sigma: float = 0.9
    jpeg_quality_min: int = 45
    erode_dilate: bool = True
    horizontal_flip: bool = False  # present only to reject it loudly

    def __post_init__(self) -> None:
        if self.horizontal_flip:
            raise ValueError(
                "horizontal_flip is not supported for text recognition: mirroring an "
                "image invalidates its transcription."
            )
        self.probability = float(max(0.0, min(1.0, self.probability)))

    @classmethod
    def from_config(cls, cfg: Mapping[str, Any] | Any) -> "AugmentOptions":
        if cfg is None:
            return cls()
        data = dict(cfg) if isinstance(cfg, Mapping) else dict(getattr(cfg, "to_dict", dict)())
        known = {f for f in cls.__dataclass_fields__}  # type: ignore[attr-defined]
        return cls(**{k: v for k, v in data.items() if k in known})

    def to_dict(self) -> dict[str, Any]:
        return {f: getattr(self, f) for f in self.__dataclass_fields__}  # type: ignore[attr-defined]


class OCRAugmentor:
    """Label-preserving augmentation for supervised OCR training.

    >>> aug = OCRAugmentor(AugmentOptions(), seed=0)
    >>> out = aug(gray_image)       # same shape, same text
    """

    def __init__(self, options: AugmentOptions | None = None, *, seed: int | None = None) -> None:
        self.options = options or AugmentOptions()
        self._rng = np.random.default_rng(seed)

    def reseed(self, seed: int) -> None:
        """Reset the internal RNG (used by DataLoader worker init)."""
        self._rng = np.random.default_rng(seed)

    def __call__(self, image: np.ndarray) -> np.ndarray:
        opts = self.options
        if not opts.enabled:
            return image

        rng = self._rng
        if rng.random() > opts.probability:
            return image

        out = ops.ensure_uint8(image)

        out = random_affine(
            out,
            rng,
            rotate_deg=opts.rotate_deg,
            shear_deg=opts.shear_deg,
            scale_jitter=opts.scale_jitter,
            translate_px=opts.translate_px,
        )
        out = photometric_jitter(out, rng, brightness=opts.brightness, contrast=opts.contrast)

        if opts.erode_dilate and rng.random() < 0.3:
            out = random_erode_dilate(out, rng)
        if rng.random() < 0.4:
            out = random_blur(out, rng, opts.blur_max_sigma)
        if rng.random() < 0.4:
            out = add_gaussian_noise(out, rng, opts.gaussian_noise_std)
        if rng.random() < 0.25:
            out = random_jpeg_artifacts(out, rng, opts.jpeg_quality_min)

        return out


# ---------------------------------------------------------------------------
# SimCLR view generation
# ---------------------------------------------------------------------------
@dataclass(slots=True)
class SimCLRViewOptions:
    """Strength of the two stochastic views used by the contrastive loss.

    SimCLR needs *stronger* augmentation than supervised training: the whole
    learning signal is "these two views came from the same crop".  If the views
    are too similar the task is trivial and the representation is weak.  The
    ceiling is that both views must still be legible as the same text, which is
    what rules out the colour-jitter-plus-random-resized-crop recipe used for
    natural images.
    """

    brightness: float = 0.4
    contrast: float = 0.4
    gaussian_noise_std: float = 0.06
    blur_max_sigma: float = 1.4
    rotate_deg: float = 3.0
    shear_deg: float = 6.0
    scale_jitter: float = 0.12
    translate_px: int = 3
    erode_dilate: bool = True
    jpeg_quality_min: int = 35
    width_crop_ratio: float = 0.9

    @classmethod
    def from_config(cls, cfg: Mapping[str, Any] | Any) -> "SimCLRViewOptions":
        if cfg is None:
            return cls()
        data = dict(cfg) if isinstance(cfg, Mapping) else dict(getattr(cfg, "to_dict", dict)())
        known = {f for f in cls.__dataclass_fields__}  # type: ignore[attr-defined]
        return cls(**{k: v for k, v in data.items() if k in known})

    def to_dict(self) -> dict[str, Any]:
        return {f: getattr(self, f) for f in self.__dataclass_fields__}  # type: ignore[attr-defined]


class SimCLRViewGenerator:
    """Produce two independently augmented views of one unlabeled image.

    >>> gen = SimCLRViewGenerator(SimCLRViewOptions(), seed=0)
    >>> view_a, view_b = gen(gray_image)
    """

    def __init__(self, options: SimCLRViewOptions | None = None, *, seed: int | None = None) -> None:
        self.options = options or SimCLRViewOptions()
        self._rng = np.random.default_rng(seed)

    def reseed(self, seed: int) -> None:
        self._rng = np.random.default_rng(seed)

    def _one_view(self, image: np.ndarray) -> np.ndarray:
        opts = self.options
        rng = self._rng
        out = image

        out = random_width_crop(out, rng, opts.width_crop_ratio)
        out = random_affine(
            out,
            rng,
            rotate_deg=opts.rotate_deg,
            shear_deg=opts.shear_deg,
            scale_jitter=opts.scale_jitter,
            translate_px=opts.translate_px,
        )
        out = photometric_jitter(out, rng, brightness=opts.brightness, contrast=opts.contrast)

        # Applied with high probability: these are the transforms that force
        # the encoder to become invariant to scan quality.
        if rng.random() < 0.7:
            out = random_blur(out, rng, opts.blur_max_sigma)
        if rng.random() < 0.6:
            out = add_gaussian_noise(out, rng, opts.gaussian_noise_std)
        if opts.erode_dilate and rng.random() < 0.5:
            out = random_erode_dilate(out, rng)
        if rng.random() < 0.4:
            out = random_jpeg_artifacts(out, rng, opts.jpeg_quality_min)

        return out

    def __call__(self, image: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        base = ops.ensure_uint8(image)
        return self._one_view(base), self._one_view(base)
