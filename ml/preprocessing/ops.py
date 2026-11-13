"""Primitive image operations used by the preprocessing pipeline.

Conventions
-----------
* Images are ``numpy.ndarray`` of dtype ``uint8``.
* Colour images are BGR (OpenCV order); grayscale images are 2-D.
* Every function returns a **new** array and never mutates its input.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import cv2
import numpy as np

__all__ = [
    "ensure_uint8",
    "to_grayscale",
    "to_bgr",
    "estimate_skew_angle",
    "deskew",
    "denoise",
    "apply_clahe",
    "unsharp_mask",
    "binarize",
    "normalise_polarity",
    "ResizeInfo",
    "resize_with_padding",
    "to_model_tensor_array",
    "encode_png",
]

logger = logging.getLogger(__name__)

#: Skew correction beyond this many degrees is almost always a detection error
#: on a text line, so the estimate is clamped instead of trusted.
MAX_SKEW_DEGREES = 15.0


def ensure_uint8(image: np.ndarray) -> np.ndarray:
    """Return *image* as ``uint8``, scaling float inputs in [0, 1] to [0, 255]."""
    if image.dtype == np.uint8:
        return image
    array = np.asarray(image)
    if np.issubdtype(array.dtype, np.floating):
        peak = float(array.max()) if array.size else 0.0
        if peak <= 1.0 + 1e-6:
            array = array * 255.0
    return np.clip(array, 0, 255).astype(np.uint8)


def to_grayscale(image: np.ndarray) -> np.ndarray:
    """Convert to single-channel grayscale, accepting gray/BGR/BGRA input."""
    array = ensure_uint8(image)
    if array.ndim == 2:
        return array.copy()
    if array.ndim == 3:
        channels = array.shape[2]
        if channels == 1:
            return array[:, :, 0].copy()
        if channels == 3:
            return cv2.cvtColor(array, cv2.COLOR_BGR2GRAY)
        if channels == 4:
            # Composite onto white so transparent regions read as background.
            bgr = array[:, :, :3].astype(np.float32)
            alpha = (array[:, :, 3:4].astype(np.float32)) / 255.0
            flattened = bgr * alpha + 255.0 * (1.0 - alpha)
            return cv2.cvtColor(flattened.astype(np.uint8), cv2.COLOR_BGR2GRAY)
    raise ValueError(f"Unsupported image shape for grayscale conversion: {array.shape}")


def to_bgr(image: np.ndarray) -> np.ndarray:
    """Convert to a 3-channel BGR image (for drawing overlays)."""
    array = ensure_uint8(image)
    if array.ndim == 2:
        return cv2.cvtColor(array, cv2.COLOR_GRAY2BGR)
    if array.ndim == 3 and array.shape[2] == 4:
        return cv2.cvtColor(array, cv2.COLOR_BGRA2BGR)
    if array.ndim == 3 and array.shape[2] == 3:
        return array.copy()
    if array.ndim == 3 and array.shape[2] == 1:
        return cv2.cvtColor(array[:, :, 0], cv2.COLOR_GRAY2BGR)
    raise ValueError(f"Unsupported image shape for BGR conversion: {array.shape}")


#: Longest side used for the skew search. Skew is a global property, so a
#: downscaled copy gives the same answer far more cheaply.
_SKEW_WORK_WIDTH = 600


def _projection_sharpness(mask: np.ndarray) -> float:
    """Score how well *mask*'s text rows line up with the horizontal axis.

    The row-projection profile of straight text is spiky -- dense inside a
    text line, near zero in the gaps between lines. Skewed text smears those
    peaks together. The variance of the profile therefore peaks exactly when
    the text is straight, which is the classic projection-profile criterion.
    """
    projection = mask.sum(axis=1, dtype=np.float64)
    if projection.size < 2:
        return 0.0
    # Squared first difference rewards sharp transitions, which discriminates
    # better than raw variance when the page has only one or two text lines.
    return float(np.diff(projection).astype(np.float64).__pow__(2).sum())


def _rotate_mask(mask: np.ndarray, angle: float) -> np.ndarray:
    height, width = mask.shape[:2]
    matrix = cv2.getRotationMatrix2D((width / 2.0, height / 2.0), angle, 1.0)
    return cv2.warpAffine(
        mask, matrix, (width, height), flags=cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT, borderValue=0
    )


def estimate_skew_angle(gray: np.ndarray, *, max_angle: float = MAX_SKEW_DEGREES) -> float:
    """Estimate the skew of a text image in degrees.

    Searches candidate rotations and keeps the one whose horizontal projection
    profile is sharpest (see :func:`_projection_sharpness`), coarse-to-fine.

    This replaced a ``minAreaRect`` fit over the ink pixels, which was cheaper
    but only ever detected one direction of rotation: the bounding rectangle
    of left-leaning text is reported by OpenCV at the opposite end of its
    ``(0, 90]`` range, and the fit also collapses to axis-aligned once the
    morphological closing used to join characters is wide enough to matter.
    Negative skew silently estimated as 0.0 in every case.

    Args:
        gray: grayscale (or convertible) image.
        max_angle: search bound. An image skewed further than this is almost
            always a detection error rather than a document.

    Returns:
        Degrees to rotate to straighten the text: positive undoes a clockwise
        skew, negative undoes an anticlockwise one. 0.0 when there is too
        little ink to judge.
    """
    if gray.ndim != 2:
        gray = to_grayscale(gray)

    # Work on a downscaled copy: skew is global, and this bounds the cost of
    # the search on a large scan.
    height, width = gray.shape[:2]
    if width > _SKEW_WORK_WIDTH:
        scale = _SKEW_WORK_WIDTH / width
        gray = cv2.resize(gray, (_SKEW_WORK_WIDTH, max(1, int(height * scale))), interpolation=cv2.INTER_AREA)

    # Otsu with INV so ink becomes 255 and paper 0.
    _, mask = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)
    if int(np.count_nonzero(mask)) < 20:
        return 0.0

    # Join characters into words so the profile reflects text lines rather
    # than individual glyphs.
    kernel_width = max(3, mask.shape[1] // 50)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_width, 1)))
    mask = (mask > 0).astype(np.uint8)

    def best_of(candidates: np.ndarray) -> tuple[float, float]:
        best_angle, best_score = 0.0, -1.0
        for candidate in candidates:
            score = _projection_sharpness(_rotate_mask(mask, float(candidate)))
            if score > best_score:
                best_angle, best_score = float(candidate), score
        return best_angle, best_score

    coarse, _ = best_of(np.arange(-max_angle, max_angle + 0.001, 1.0))
    fine, _ = best_of(np.arange(coarse - 1.0, coarse + 1.001, 0.2))

    if abs(fine) > max_angle:
        logger.debug("Discarding implausible skew estimate of %.2f degrees", fine)
        return 0.0
    # Sub-tenth-of-a-degree results are noise, and rotating for them only
    # costs an interpolation pass.
    return float(fine) if abs(fine) >= 0.1 else 0.0


def deskew(
    image: np.ndarray,
    *,
    angle: float | None = None,
    max_angle: float = MAX_SKEW_DEGREES,
    border_value: int = 255,
) -> tuple[np.ndarray, float]:
    """Rotate *image* so its text becomes horizontal.

    Args:
        image: gray or BGR image.
        angle: rotation to apply; estimated with :func:`estimate_skew_angle`
            when omitted.
        max_angle: clamp used by the estimator.
        border_value: fill colour for the exposed corners.

    Returns:
        ``(rotated_image, angle_applied_in_degrees)``.  When the estimated
        angle is negligible the input is returned unchanged.
    """
    array = ensure_uint8(image)
    gray = array if array.ndim == 2 else to_grayscale(array)

    if angle is None:
        angle = estimate_skew_angle(gray, max_angle=max_angle)

    if abs(angle) < 0.1:
        return array.copy(), 0.0

    height, width = array.shape[:2]
    centre = (width / 2.0, height / 2.0)
    matrix = cv2.getRotationMatrix2D(centre, angle, 1.0)

    # Expand the canvas so rotation does not clip the corners of the text.
    cos, sin = abs(matrix[0, 0]), abs(matrix[0, 1])
    new_width = int(height * sin + width * cos)
    new_height = int(height * cos + width * sin)
    matrix[0, 2] += new_width / 2.0 - centre[0]
    matrix[1, 2] += new_height / 2.0 - centre[1]

    fill: float | tuple[float, ...] = (
        float(border_value) if array.ndim == 2 else (float(border_value),) * array.shape[2]
    )
    rotated = cv2.warpAffine(
        array,
        matrix,
        (new_width, new_height),
        flags=cv2.INTER_CUBIC,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=fill,
    )
    return rotated, float(angle)


def find_document_quad(gray: np.ndarray, *, min_area_ratio: float = 0.25) -> np.ndarray | None:
    """Locate a four-cornered document outline in *gray*.

    Returns:
        A ``(4, 2)`` float32 array of corners ordered top-left, top-right,
        bottom-right, bottom-left, or ``None`` when no convincing quadrilateral
        is found.
    """
    height, width = gray.shape[:2]
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blurred, 60, 180)
    # Close small breaks so a dashed or faint border still forms one contour.
    edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))

    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None

    page_area = float(height * width)
    for contour in sorted(contours, key=cv2.contourArea, reverse=True)[:6]:
        area = cv2.contourArea(contour)
        if area < page_area * min_area_ratio:
            break

        perimeter = cv2.arcLength(contour, True)
        approximation = cv2.approxPolyDP(contour, 0.02 * perimeter, True)
        if len(approximation) != 4 or not cv2.isContourConvex(approximation):
            continue

        points = approximation.reshape(4, 2).astype(np.float32)

        # Order: top-left and bottom-right have the extreme coordinate sums;
        # top-right and bottom-left the extreme differences.
        total = points.sum(axis=1)
        diff = np.diff(points, axis=1).ravel()
        ordered = np.array(
            [
                points[np.argmin(total)],
                points[np.argmin(diff)],
                points[np.argmax(total)],
                points[np.argmax(diff)],
            ],
            dtype=np.float32,
        )
        if len({tuple(p) for p in ordered}) != 4:
            continue
        return ordered

    return None


def perspective_correct(
    image: np.ndarray,
    *,
    quad: np.ndarray | None = None,
    min_area_ratio: float = 0.25,
) -> tuple[np.ndarray, bool]:
    """Flatten a photographed page to a head-on view.

    Deskewing only undoes rotation. A page photographed at an angle is also
    *keystoned* -- one edge is nearer the camera, so its text is larger and
    the baselines converge. Warping the detected page outline to a rectangle
    removes that, which both improves recognition and lets line detection work
    on parallel baselines.

    Returns:
        ``(image, corrected)``. When no convincing page outline is found the
        input is returned unchanged rather than guessing, because a wrong warp
        is far more damaging than none.
    """
    array = ensure_uint8(image)
    gray = array if array.ndim == 2 else to_grayscale(array)

    corners = quad if quad is not None else find_document_quad(gray, min_area_ratio=min_area_ratio)
    if corners is None:
        return array.copy(), False

    top_left, top_right, bottom_right, bottom_left = corners
    target_width = int(max(np.linalg.norm(top_right - top_left), np.linalg.norm(bottom_right - bottom_left)))
    target_height = int(max(np.linalg.norm(bottom_left - top_left), np.linalg.norm(bottom_right - top_right)))

    if target_width < 32 or target_height < 32:
        return array.copy(), False

    destination = np.array(
        [[0, 0], [target_width - 1, 0], [target_width - 1, target_height - 1], [0, target_height - 1]],
        dtype=np.float32,
    )
    matrix = cv2.getPerspectiveTransform(corners, destination)
    warped = cv2.warpPerspective(
        array, matrix, (target_width, target_height), flags=cv2.INTER_CUBIC, borderValue=255
    )
    logger.debug("Perspective corrected to %dx%d", target_width, target_height)
    return warped, True


def denoise(
    image: np.ndarray,
    *,
    strength: int = 7,
    method: str = "nlmeans",
) -> np.ndarray:
    """Remove sensor/scan noise while trying to keep stroke edges.

    Args:
        strength: filter strength; higher removes more noise and more detail.
        method: ``"nlmeans"`` (best quality), ``"bilateral"`` (fast, strong
            edge preservation) or ``"median"`` (best for salt-and-pepper).
    """
    array = ensure_uint8(image)
    choice = method.strip().lower()

    if choice == "median":
        kernel = 3 if strength <= 7 else 5
        return cv2.medianBlur(array, kernel)

    if choice == "bilateral":
        return cv2.bilateralFilter(array, d=5, sigmaColor=float(strength) * 6.0, sigmaSpace=7.0)

    if choice != "nlmeans":
        logger.warning("Unknown denoise method %r; using nlmeans", method)

    if array.ndim == 2:
        return cv2.fastNlMeansDenoising(array, None, h=float(strength), templateWindowSize=7, searchWindowSize=21)
    return cv2.fastNlMeansDenoisingColored(
        array, None, h=float(strength), hColor=float(strength), templateWindowSize=7, searchWindowSize=21
    )


def apply_clahe(image: np.ndarray, *, clip_limit: float = 2.0, tile_grid: int = 8) -> np.ndarray:
    """Contrast Limited Adaptive Histogram Equalisation.

    Recovers text from unevenly lit photographs, where a global stretch would
    blow out the bright side of the page.
    """
    array = ensure_uint8(image)
    clahe = cv2.createCLAHE(clipLimit=float(clip_limit), tileGridSize=(int(tile_grid), int(tile_grid)))

    if array.ndim == 2:
        return clahe.apply(array)

    # Equalise luminance only, so colours are not shifted.
    lab = cv2.cvtColor(array, cv2.COLOR_BGR2LAB)
    lab[:, :, 0] = clahe.apply(lab[:, :, 0])
    return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)


def unsharp_mask(
    image: np.ndarray,
    *,
    amount: float = 1.0,
    sigma: float = 1.0,
    threshold: int = 0,
) -> np.ndarray:
    """Sharpen by subtracting a blurred copy (classic unsharp mask).

    Args:
        amount: strength of the sharpening; 0 returns the input.
        sigma: Gaussian radius of the blur that defines "detail".
        threshold: minimum local contrast before a pixel is sharpened, which
            avoids amplifying flat-area noise.
    """
    array = ensure_uint8(image)
    if amount <= 0.0:
        return array.copy()

    blurred = cv2.GaussianBlur(array, (0, 0), sigmaX=float(sigma), sigmaY=float(sigma))
    sharpened = cv2.addWeighted(array, 1.0 + float(amount), blurred, -float(amount), 0.0)

    if threshold > 0:
        low_contrast = np.abs(array.astype(np.int16) - blurred.astype(np.int16)) < threshold
        sharpened = np.where(low_contrast, array, sharpened)

    return np.clip(sharpened, 0, 255).astype(np.uint8)


def _sauvola_threshold(gray: np.ndarray, window: int, k: float, r: float) -> np.ndarray:
    """Per-pixel Sauvola threshold computed with box filters (O(1) per pixel).

    ``T(x, y) = mean * (1 + k * (std / r - 1))``
    """
    window = max(3, int(window) | 1)  # force odd
    source = gray.astype(np.float32)

    mean = cv2.boxFilter(source, ddepth=-1, ksize=(window, window), normalize=True, borderType=cv2.BORDER_REPLICATE)
    mean_sq = cv2.boxFilter(
        source * source, ddepth=-1, ksize=(window, window), normalize=True, borderType=cv2.BORDER_REPLICATE
    )
    variance = np.clip(mean_sq - mean * mean, 0.0, None)
    std = np.sqrt(variance)

    return mean * (1.0 + float(k) * (std / float(r) - 1.0))


def binarize(
    image: np.ndarray,
    *,
    method: str = "otsu",
    block_size: int = 31,
    offset: int = 10,
    sauvola_k: float = 0.2,
    sauvola_r: float = 128.0,
) -> np.ndarray:
    """Threshold to a two-tone image (ink = 0, paper = 255).

    Args:
        method: ``"otsu"`` (global, fast, good for clean scans),
            ``"adaptive"`` (Gaussian local mean) or ``"sauvola"`` (local
            mean + standard deviation, the strongest option for degraded or
            unevenly lit documents).
        block_size: neighbourhood size for the local methods; forced odd.
        offset: constant subtracted from the local mean (``adaptive`` only).
        sauvola_k: Sauvola sensitivity; lower keeps more ink.
        sauvola_r: Sauvola dynamic range of the standard deviation.

    Raises:
        ValueError: *method* is not recognised.
    """
    gray = to_grayscale(image)
    choice = method.strip().lower()
    block = max(3, int(block_size) | 1)

    if choice == "otsu":
        _, result = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
        return result

    if choice == "adaptive":
        return cv2.adaptiveThreshold(
            gray,
            255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY,
            block,
            int(offset),
        )

    if choice == "sauvola":
        threshold = _sauvola_threshold(gray, block, sauvola_k, sauvola_r)
        return np.where(gray.astype(np.float32) > threshold, 255, 0).astype(np.uint8)

    raise ValueError(f"Unknown binarisation method {method!r}; expected otsu, adaptive or sauvola")


def normalise_polarity(image: np.ndarray, *, border_fraction: float = 0.12) -> tuple[np.ndarray, bool]:
    """Ensure the image is dark text on a light background.

    Otsu splits the pixels into an ink class and a paper class. The image
    border is background almost by definition -- a text crop is mostly
    background, and its edges especially so -- so whichever class the border
    sits closer to *is* the background. If that is the darker class, the image
    is light-on-dark and gets inverted.

    An earlier version compared the border median against the image median.
    That can never fire: the background is the majority of the pixels whatever
    the polarity, so both medians are the background value and the test is
    always false.

    Returns:
        ``(image, inverted)`` where ``inverted`` says whether the input was
        light-on-dark and therefore flipped.
    """
    array = ensure_uint8(image)
    gray = array if array.ndim == 2 else to_grayscale(array)

    height, width = gray.shape[:2]
    margin_y = max(1, int(height * border_fraction))
    margin_x = max(1, int(width * border_fraction))

    border = np.concatenate(
        [
            gray[:margin_y, :].ravel(),
            gray[-margin_y:, :].ravel(),
            gray[:, :margin_x].ravel(),
            gray[:, -margin_x:].ravel(),
        ]
    )
    background = float(np.median(border))

    threshold, _ = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    dark = gray[gray <= threshold]
    light = gray[gray > threshold]

    if dark.size == 0 or light.size == 0:
        # A single-tone image has no text to orient; leave it alone.
        return array.copy(), False

    if abs(background - float(dark.mean())) < abs(background - float(light.mean())):
        return (255 - array), True
    return array.copy(), False


@dataclass(slots=True)
class ResizeInfo:
    """How :func:`resize_with_padding` mapped the source into the canvas.

    Needed to project model-space coordinates (e.g. attention positions) back
    onto the original image, and to tell whether anything was lost.
    """

    original_height: int
    original_width: int
    scaled_height: int
    scaled_width: int
    target_height: int
    target_width: int
    scale: float
    pad_left: int
    pad_top: int
    #: Width the content *would* have needed at this height. Larger than
    #: `target_width` means the right-hand side was cut off.
    natural_width: int = 0
    #: True when the content was too wide for the canvas and was cropped.
    truncated: bool = False

    @property
    def content_ratio(self) -> float:
        """Fraction of the canvas width occupied by real content."""
        return self.scaled_width / self.target_width if self.target_width else 0.0

    @property
    def lost_fraction(self) -> float:
        """Fraction of the line that was cut off, 0.0 when nothing was lost.

        Reported because the alternative -- silently returning a partial
        transcription that looks complete -- is the worst possible failure for
        an OCR system.
        """
        if not self.truncated or not self.natural_width:
            return 0.0
        return max(0.0, 1.0 - self.target_width / self.natural_width)


def resize_with_padding(
    image: np.ndarray,
    *,
    height: int,
    width: int,
    keep_aspect_ratio: bool = True,
    pad_value: int = 255,
) -> tuple[np.ndarray, ResizeInfo]:
    """Fit *image* into a fixed ``height x width`` canvas.

    With ``keep_aspect_ratio`` the image is scaled by height and then padded
    (or centre-cropped, if it is too wide) horizontally, which keeps glyph
    aspect ratio intact -- stretching characters measurably hurts CER.

    Returns:
        ``(canvas, resize_info)``.
    """
    array = ensure_uint8(image)
    if array.size == 0:
        raise ValueError("Cannot resize an empty image")

    src_h, src_w = array.shape[:2]
    channels = 1 if array.ndim == 2 else array.shape[2]
    height, width = int(height), int(width)

    if not keep_aspect_ratio:
        resized = cv2.resize(array, (width, height), interpolation=cv2.INTER_AREA)
        info = ResizeInfo(
            src_h, src_w, height, width, height, width, width / src_w, 0, 0,
            natural_width=width, truncated=False,
        )
        return resized, info

    scale = height / float(src_h)
    natural_w = max(1, int(round(src_w * scale)))
    scaled_w = natural_w

    interpolation = cv2.INTER_AREA if scale < 1.0 else cv2.INTER_CUBIC
    resized = cv2.resize(array, (scaled_w, height), interpolation=interpolation)

    truncated = False
    if scaled_w > width:
        # Too wide for the canvas: keep the left-hand side, which is where
        # reading starts, rather than squashing the glyphs.
        #
        # This loses text, so it is recorded. Callers that can afford a wider
        # canvas (inference does -- the network is width-agnostic) should size
        # it from `natural_width` instead of cropping here; see
        # `prepare_variable_width_batch`.
        resized = resized[:, :width]
        scaled_w = width
        truncated = True

    canvas_shape = (height, width) if channels == 1 else (height, width, channels)
    canvas = np.full(canvas_shape, int(pad_value), dtype=np.uint8)
    canvas[:, :scaled_w] = resized

    info = ResizeInfo(
        original_height=src_h,
        original_width=src_w,
        scaled_height=height,
        scaled_width=scaled_w,
        target_height=height,
        target_width=width,
        scale=scale,
        pad_left=0,
        pad_top=0,
        natural_width=natural_w,
        truncated=truncated,
    )
    if truncated:
        logger.debug(
            "Line cropped: needed %d px at height %d but the canvas is %d px; "
            "%.0f%% of the line was discarded",
            natural_w, height, width, info.lost_fraction * 100,
        )
    return canvas, info


def to_model_tensor_array(
    image: np.ndarray,
    *,
    channels: int = 1,
    normalize: bool = True,
    mean: float = 0.5,
    std: float = 0.5,
) -> np.ndarray:
    """Convert a uint8 image to a CHW float32 array ready for ``torch.from_numpy``.

    Args:
        channels: 1 for grayscale, 3 for BGR->RGB.
        normalize: standardise with ``(x - mean) / std`` after scaling to [0, 1].
    """
    array = ensure_uint8(image)

    if channels == 1:
        array = array if array.ndim == 2 else to_grayscale(array)
        planes = array[np.newaxis, :, :]
    elif channels == 3:
        bgr = to_bgr(array)
        planes = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB).transpose(2, 0, 1)
    else:
        raise ValueError(f"channels must be 1 or 3, got {channels}")

    tensor = planes.astype(np.float32) / 255.0
    if normalize:
        tensor = (tensor - float(mean)) / float(std)
    return np.ascontiguousarray(tensor)


def encode_png(image: np.ndarray) -> bytes:
    """Encode *image* as PNG bytes (used to return preview stages over HTTP)."""
    success, buffer = cv2.imencode(".png", ensure_uint8(image))
    if not success:
        raise RuntimeError("cv2.imencode failed to produce a PNG")
    return buffer.tobytes()
