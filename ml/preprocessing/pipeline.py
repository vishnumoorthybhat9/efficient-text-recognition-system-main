"""Configurable image-enhancement pipeline.

The pipeline covers *enhancement* only -- it returns an image of arbitrary
size.  Shaping the result into the fixed tensor the network expects is a
separate concern handled by :func:`ml.preprocessing.pipeline.prepare_model_input`,
because training (random crops, augmentation) and inference (line
segmentation) shape it differently.

Stage capture exists so the frontend's "Preprocessing Lab" can display what
each toggle actually did, instead of describing it in prose.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field, replace
from typing import Any, Mapping, Sequence

import cv2
import numpy as np

from ml.preprocessing import ops

__all__ = [
    "PreprocessOptions",
    "PreprocessStage",
    "PreprocessResult",
    "PreprocessPipeline",
    "prepare_model_input",
]

logger = logging.getLogger(__name__)

BINARIZE_METHODS = ("none", "otsu", "adaptive", "sauvola")
DENOISE_METHODS = ("nlmeans", "bilateral", "median")


@dataclass(slots=True)
class PreprocessOptions:
    """Which enhancement steps to run, and with what parameters.

    Defaults match ``configs/default.yaml`` and are deliberately conservative:
    every optional step can hurt as well as help, so the pipeline only does
    what it is asked to do.
    """

    grayscale: bool = True
    # On by default: a photographed page is almost never perfectly square,
    # and even 3-4 degrees of rotation makes neighbouring text lines overlap
    # vertically, which collapses a whole page into one unreadable row.
    # `ops.deskew` is a no-op below 0.1 degrees, so a clean scan is untouched
    # for the ~13 ms the estimate costs.
    deskew: bool = True
    # Off by default: it only helps a photographed page with a visible
    # border, and a wrong warp is far more damaging than none. The step is
    # a no-op when no convincing page outline is found.
    perspective: bool = False
    denoise: bool = False
    denoise_strength: int = 7
    denoise_method: str = "nlmeans"
    clahe: bool = False
    clahe_clip_limit: float = 2.0
    sharpen: bool = False
    sharpen_amount: float = 1.0
    binarize: str = "none"
    binarize_block_size: int = 31
    binarize_offset: int = 10
    invert_if_dark: bool = True
    normalize: bool = True

    def __post_init__(self) -> None:
        method = str(self.binarize or "none").strip().lower()
        if method in ("", "false", "off"):
            method = "none"
        if method not in BINARIZE_METHODS:
            raise ValueError(
                f"binarize must be one of {BINARIZE_METHODS}, got {self.binarize!r}"
            )
        self.binarize = method

        denoise_method = str(self.denoise_method or "nlmeans").strip().lower()
        if denoise_method not in DENOISE_METHODS:
            raise ValueError(
                f"denoise_method must be one of {DENOISE_METHODS}, got {self.denoise_method!r}"
            )
        self.denoise_method = denoise_method

        self.denoise_strength = int(max(1, min(30, self.denoise_strength)))
        self.clahe_clip_limit = float(max(0.1, min(10.0, self.clahe_clip_limit)))
        self.sharpen_amount = float(max(0.0, min(3.0, self.sharpen_amount)))
        self.binarize_block_size = int(max(3, min(199, self.binarize_block_size)))

    @classmethod
    def from_config(cls, cfg: Mapping[str, Any] | Any) -> "PreprocessOptions":
        """Build from the ``preprocess`` block of a config (or any mapping)."""
        if cfg is None:
            return cls()
        data = dict(cfg) if isinstance(cfg, Mapping) else dict(getattr(cfg, "to_dict", dict)())
        known = {f for f in cls.__dataclass_fields__}  # type: ignore[attr-defined]
        unknown = set(data) - known
        if unknown:
            logger.debug("Ignoring unknown preprocess keys: %s", sorted(unknown))
        return cls(**{k: v for k, v in data.items() if k in known})

    def with_overrides(self, **changes: Any) -> "PreprocessOptions":
        """Return a copy with *changes* applied, dropping ``None`` values."""
        clean = {k: v for k, v in changes.items() if v is not None}
        return replace(self, **clean) if clean else self

    def to_dict(self) -> dict[str, Any]:
        return {f: getattr(self, f) for f in self.__dataclass_fields__}  # type: ignore[attr-defined]

    @property
    def active_steps(self) -> list[str]:
        """Names of the steps that will actually run, in order."""
        steps = ["grayscale"] if self.grayscale else []
        if self.invert_if_dark:
            steps.append("polarity")
        if self.perspective:
            steps.append("perspective")
        if self.deskew:
            steps.append("deskew")
        if self.denoise:
            steps.append("denoise")
        if self.clahe:
            steps.append("clahe")
        if self.sharpen:
            steps.append("sharpen")
        if self.binarize != "none":
            steps.append(f"binarize:{self.binarize}")
        return steps


@dataclass(slots=True)
class PreprocessStage:
    """One captured intermediate image."""

    name: str
    label: str
    image: np.ndarray
    duration_ms: float
    detail: str = ""


@dataclass(slots=True)
class PreprocessResult:
    """Output of :meth:`PreprocessPipeline.run`."""

    image: np.ndarray
    stages: list[PreprocessStage] = field(default_factory=list)
    skew_angle: float = 0.0
    inverted: bool = False
    perspective_corrected: bool = False
    total_ms: float = 0.0
    options: PreprocessOptions = field(default_factory=PreprocessOptions)

    def metadata(self) -> dict[str, Any]:
        """JSON-serialisable summary (no image data)."""
        return {
            "skew_angle_deg": round(self.skew_angle, 3),
            "inverted": self.inverted,
            "perspective_corrected": self.perspective_corrected,
            "total_ms": round(self.total_ms, 3),
            "height": int(self.image.shape[0]),
            "width": int(self.image.shape[1]),
            "channels": 1 if self.image.ndim == 2 else int(self.image.shape[2]),
            "steps": self.options.active_steps,
            "stage_timings_ms": {s.name: round(s.duration_ms, 3) for s in self.stages},
        }


class PreprocessPipeline:
    """Applies :class:`PreprocessOptions` to an image.

    The instance is stateless apart from its options, so one pipeline can be
    shared across requests.

    >>> pipeline = PreprocessPipeline(PreprocessOptions(clahe=True))
    >>> result = pipeline.run(image, capture_stages=True)
    >>> [stage.name for stage in result.stages]
    ['input', 'grayscale', 'polarity', 'clahe']
    """

    def __init__(self, options: PreprocessOptions | None = None) -> None:
        self.options = options or PreprocessOptions()

    # -- public API ---------------------------------------------------------
    def run(
        self,
        image: np.ndarray,
        *,
        capture_stages: bool = False,
        options: PreprocessOptions | None = None,
    ) -> PreprocessResult:
        """Enhance *image*.

        Args:
            image: gray, BGR or BGRA ``uint8`` image.
            capture_stages: keep a copy of the image after every step.  Costs
                memory, so it is opt-in and used only by the preview endpoint.
            options: override this pipeline's options for a single call.

        Raises:
            ValueError: the input is empty or has an unsupported shape.
        """
        from ml.utils.timing import stopwatch  # local import: avoids a cycle

        opts = options or self.options
        if image is None or getattr(image, "size", 0) == 0:
            raise ValueError("PreprocessPipeline.run received an empty image")

        current = ops.ensure_uint8(image)
        stages: list[PreprocessStage] = []
        skew_angle = 0.0
        inverted = False
        perspective_corrected = False

        def record(name: str, label: str, img: np.ndarray, ms: float, detail: str = "") -> None:
            if capture_stages:
                stages.append(PreprocessStage(name, label, img.copy(), ms, detail))

        record("input", "Original", current, 0.0, f"{current.shape[1]}x{current.shape[0]} px")

        with stopwatch() as total:
            if opts.grayscale:
                with stopwatch() as sw:
                    current = ops.to_grayscale(current)
                record("grayscale", "Grayscale", current, sw.elapsed_ms, "single channel")

            if opts.invert_if_dark:
                with stopwatch() as sw:
                    current, inverted = ops.normalise_polarity(current)
                record(
                    "polarity",
                    "Polarity normalised",
                    current,
                    sw.elapsed_ms,
                    "inverted (light text on dark paper)" if inverted else "already dark-on-light",
                )

            if opts.perspective:
                with stopwatch() as sw:
                    current, perspective_corrected = ops.perspective_correct(current)
                record(
                    "perspective",
                    "Perspective corrected",
                    current,
                    sw.elapsed_ms,
                    "flattened to a head-on view" if perspective_corrected
                    else "no page outline found; left unchanged",
                )

            if opts.deskew:
                with stopwatch() as sw:
                    current, skew_angle = ops.deskew(current)
                record("deskew", "Deskewed", current, sw.elapsed_ms, f"rotated {skew_angle:+.2f} deg")

            if opts.denoise:
                with stopwatch() as sw:
                    current = ops.denoise(
                        current, strength=opts.denoise_strength, method=opts.denoise_method
                    )
                record(
                    "denoise",
                    "Denoised",
                    current,
                    sw.elapsed_ms,
                    f"{opts.denoise_method}, h={opts.denoise_strength}",
                )

            if opts.clahe:
                with stopwatch() as sw:
                    current = ops.apply_clahe(current, clip_limit=opts.clahe_clip_limit)
                record("clahe", "Contrast enhanced", current, sw.elapsed_ms, f"CLAHE clip={opts.clahe_clip_limit}")

            if opts.sharpen:
                with stopwatch() as sw:
                    current = ops.unsharp_mask(current, amount=opts.sharpen_amount)
                record("sharpen", "Sharpened", current, sw.elapsed_ms, f"unsharp amount={opts.sharpen_amount}")

            if opts.binarize != "none":
                with stopwatch() as sw:
                    current = ops.binarize(
                        current,
                        method=opts.binarize,
                        block_size=opts.binarize_block_size,
                        offset=opts.binarize_offset,
                    )
                record("binarize", f"Binarised ({opts.binarize})", current, sw.elapsed_ms, opts.binarize)

        return PreprocessResult(
            image=current,
            stages=stages,
            skew_angle=skew_angle,
            inverted=inverted,
            perspective_corrected=perspective_corrected,
            total_ms=total.elapsed_ms,
            options=opts,
        )

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return f"PreprocessPipeline(steps={self.options.active_steps})"


def prepare_model_input(
    image: np.ndarray,
    *,
    height: int,
    width: int,
    channels: int = 1,
    keep_aspect_ratio: bool = True,
    pad_value: int = 255,
    normalize: bool = True,
) -> tuple[np.ndarray, ops.ResizeInfo]:
    """Shape an enhanced image into the CHW float array the network expects.

    Returns:
        ``(chw_float32_array, resize_info)``.
    """
    canvas, info = ops.resize_with_padding(
        image,
        height=height,
        width=width,
        keep_aspect_ratio=keep_aspect_ratio,
        pad_value=pad_value,
    )
    tensor = ops.to_model_tensor_array(canvas, channels=channels, normalize=normalize)
    return tensor, info


#: Smallest inference canvas. Below this the encoder emits too few time steps
#: to decode even a short word.
MIN_INFERENCE_WIDTH = 128

#: Largest inference canvas. The attention stack's positional encoding holds
#: 512 positions and the encoder emits ``W/4 - 1`` of them, so 2048 px is the
#: ceiling the architecture supports (511 time steps, ~255 characters).
MAX_INFERENCE_WIDTH = 2048


def prepare_variable_width_batch(
    images: Sequence[np.ndarray],
    *,
    height: int = 32,
    channels: int = 1,
    pad_value: int = 255,
    normalize: bool = True,
    min_width: int = MIN_INFERENCE_WIDTH,
    max_width: int = MAX_INFERENCE_WIDTH,
) -> tuple[np.ndarray, list[float], list[ops.ResizeInfo]]:
    """Shape a batch of line crops onto a canvas sized to fit the widest one.

    Why this exists
    ---------------
    Training uses a fixed 32x256 canvas, and anything wider than that is
    cropped -- which silently throws away the right-hand end of a long line.
    A full sentence from a document can lose more than half its characters
    that way, and the result looks like a complete transcription.

    The recogniser is fully convolutional and recurrent along the width axis,
    so it accepts any width at inference regardless of what it trained on.
    This function therefore scales each crop by *height* and sizes the shared
    canvas to the widest crop in the batch, instead of forcing 256.

    A crop whose natural width exceeds *max_width* is compressed horizontally
    to fit rather than cropped: squeezed glyphs cost some accuracy, but losing
    the text entirely costs all of it.

    Args:
        images: line crops, gray or BGR.
        height: model input height.
        channels: 1 for grayscale, 3 for colour.
        pad_value: fill for the padded region.
        normalize: standardise to roughly [-1, 1].
        min_width / max_width: canvas bounds, rounded to a multiple of 4
            because the encoder downsamples width by exactly that.

    Returns:
        ``(batch, width_ratios, infos)`` -- an ``(N, C, H, W)`` float32 array,
        the fraction of the canvas each crop occupies (for ``input_lengths``),
        and the per-crop :class:`~ml.preprocessing.ops.ResizeInfo`.

    Raises:
        ValueError: *images* is empty.
    """
    if not images:
        raise ValueError("prepare_variable_width_batch received no images")

    def round_to_4(value: int) -> int:
        return max(4, int(np.ceil(value / 4.0)) * 4)

    # Natural width each crop would need at the model's height.
    natural: list[int] = []
    for image in images:
        source = ops.ensure_uint8(image)
        src_h, src_w = source.shape[:2]
        natural.append(max(1, int(round(src_w * (height / max(1, src_h))))))

    canvas_width = int(np.clip(round_to_4(max(natural)), min_width, max_width))

    arrays: list[np.ndarray] = []
    ratios: list[float] = []
    infos: list[ops.ResizeInfo] = []

    for image, natural_width in zip(images, natural):
        source = ops.ensure_uint8(image)

        if natural_width > canvas_width:
            # Compress to fit instead of cropping: distorted is recoverable,
            # missing is not.
            squeezed = cv2.resize(
                source, (canvas_width, height), interpolation=cv2.INTER_AREA
            )
            canvas, info = ops.resize_with_padding(
                squeezed,
                height=height,
                width=canvas_width,
                keep_aspect_ratio=False,
                pad_value=pad_value,
            )
            info.natural_width = natural_width
            logger.debug(
                "Line needed %d px but the canvas caps at %d; compressed %.2fx "
                "rather than cropping",
                natural_width, canvas_width, natural_width / canvas_width,
            )
        else:
            canvas, info = ops.resize_with_padding(
                source,
                height=height,
                width=canvas_width,
                keep_aspect_ratio=True,
                pad_value=pad_value,
            )

        arrays.append(ops.to_model_tensor_array(canvas, channels=channels, normalize=normalize))
        ratios.append(float(info.content_ratio))
        infos.append(info)

    return np.stack(arrays, axis=0), ratios, infos


def build_pipeline(cfg: Mapping[str, Any] | Any, **overrides: Any) -> PreprocessPipeline:
    """Convenience factory: config block (plus overrides) -> pipeline."""
    options = PreprocessOptions.from_config(cfg).with_overrides(**overrides)
    return PreprocessPipeline(options)


def stage_names(options: PreprocessOptions) -> Sequence[str]:
    """Names of the stages :meth:`PreprocessPipeline.run` would capture."""
    return ["input", *(s.split(":")[0] for s in options.active_steps)]
