"""Handwritten-document support: page cleaning and line detection.

Photographed handwritten pages -- notebook notes, letters, prescriptions --
break the standard pipeline in three ways:

1. **Ruled lines and margins.** A notebook's printed rules run the full width
   of the page and touch every handwritten word that sits on them. The ink
   mask then becomes one giant component and the projection profile sees ink
   in every row, so the line detector cannot find the gaps between lines.
2. **Uneven lighting.** A phone photo has shadows and a brightness gradient,
   so one global threshold either erases faint strokes or turns the shadow
   into "ink".
3. **Irregular lines.** Handwriting drifts up and down and its ascenders and
   descenders overlap the neighbouring line, so the gaps a projection profile
   relies on are often not there.

This module fixes (1) and (2) with plain OpenCV, and offers two line
detectors for (3):

``projection``
    The project's own detector (:func:`ml.inference.segmentation.detect_text_lines`)
    run on the cleaned page. No extra dependencies.
``doctr``
    The DBNet text detector from the `python-doctr` library. It finds *word*
    boxes with a neural network trained on real documents, which copes with
    drifting handwriting far better; the words are then chained into lines
    here. Optional: used when the package is installed.

``auto`` picks ``doctr`` when it is installed and ``projection`` otherwise.

Recognition itself is unchanged: the line crops go to whichever recogniser the
user selected (the project's CNN model or TrOCR), so the CNN-vs-Transformer
comparison remains fair on handwritten documents too.
"""

from __future__ import annotations

import logging
import threading
from dataclasses import dataclass
from typing import Any, Sequence

import cv2
import numpy as np

from ml.inference.segmentation import SegmentationResult, TextRegion, detect_text_lines
from ml.preprocessing import ops

__all__ = [
    "CleanedPage",
    "LINE_DETECTORS",
    "clean_handwritten_page",
    "doctr_available",
    "detect_handwritten_lines",
    "group_words_into_lines",
    "flatten_background",
    "remove_ruled_lines",
    "remove_thin_strokes",
    "split_tall_regions",
    "whiten_dark_borders",
]

logger = logging.getLogger(__name__)

#: Accepted values for the ``line_detector`` option.
LINE_DETECTORS = ("auto", "projection", "doctr")


# ---------------------------------------------------------------------------
# Page cleaning
# ---------------------------------------------------------------------------
@dataclass(slots=True)
class CleanedPage:
    """A handwritten page prepared for line detection and recognition."""

    #: Grayscale ``uint8``, dark ink on a white background, rules removed.
    image: np.ndarray
    horizontal_rules: int = 0
    vertical_rules: int = 0
    background_flattened: bool = False
    hairlines_removed: int = 0

    def metadata(self) -> dict[str, Any]:
        return {
            "horizontal_rules_removed": self.horizontal_rules,
            "vertical_rules_removed": self.vertical_rules,
            "background_flattened": self.background_flattened,
            "hairlines_removed": self.hairlines_removed,
        }


def flatten_background(gray: np.ndarray) -> np.ndarray:
    """Remove shadows and lighting gradients from a photographed page.

    The page background is estimated by closing (dilating away the thin ink
    strokes) and a large median blur; dividing by it leaves the ink on an
    evenly white page. This is what lets one threshold work on a photo whose
    right edge is in shadow.
    """
    if gray.ndim != 2:
        gray = ops.to_grayscale(gray)
    height, width = gray.shape[:2]
    # Kernel sizes scale with the page so a 4000 px photo and an 800 px scan
    # are treated alike. The dilation must be wider than a pen stroke.
    stroke = max(3, int(round(min(height, width) / 150)) | 1)
    blur = max(21, int(round(min(height, width) / 25)) | 1)

    background = cv2.dilate(gray, np.ones((stroke, stroke), np.uint8))
    background = cv2.medianBlur(background, min(blur, 255))
    background = np.maximum(background.astype(np.float32), 1.0)

    flattened = gray.astype(np.float32) / background * 255.0
    flattened = np.clip(flattened, 0, 255)
    # Stretch so the faintest real ink still reaches a useful contrast.
    low = float(np.percentile(flattened, 0.5))
    if low < 250:
        flattened = (flattened - low) / (255.0 - low) * 255.0
    return np.clip(flattened, 0, 255).astype(np.uint8)


def _rule_mask(
    binary: np.ndarray, *, horizontal: bool, min_length: int
) -> tuple[np.ndarray, int]:
    """Pixels belonging to long, thin, straight rules in one direction.

    Photographed rules are never perfectly level: across a page they drift by
    a few pixels, so a one-pixel-high structuring element breaks them into
    pieces. The mask is therefore thickened across the rule direction before
    the opening, the surviving segments are joined along it, and only then is
    the length test applied. The result is intersected with the original
    binary image so only the rule's own pixels are returned.
    """
    segment = max(25, min_length // 4)
    if horizontal:
        thicken = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 3))
        opener = cv2.getStructuringElement(cv2.MORPH_RECT, (segment, 1))
        joiner = cv2.getStructuringElement(cv2.MORPH_RECT, (segment, 1))
    else:
        thicken = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 1))
        opener = cv2.getStructuringElement(cv2.MORPH_RECT, (1, segment))
        joiner = cv2.getStructuringElement(cv2.MORPH_RECT, (1, segment))

    candidates = cv2.morphologyEx(cv2.dilate(binary, thicken), cv2.MORPH_OPEN, opener)
    candidates = cv2.morphologyEx(candidates, cv2.MORPH_CLOSE, joiner)

    count, labels, stats, _ = cv2.connectedComponentsWithStats(candidates, connectivity=8)
    kept = np.zeros_like(binary)
    n = 0
    for index in range(1, count):
        w = int(stats[index, cv2.CC_STAT_WIDTH])
        h = int(stats[index, cv2.CC_STAT_HEIGHT])
        area = int(stats[index, cv2.CC_STAT_AREA])
        length, extent = (w, h) if horizontal else (h, w)
        if length < min_length:
            continue
        # Mean thickness = area / length. A printed rule is a few pixels thick
        # even when it slants (the slant shows up in `extent`, not here).
        # (Up to 14 px: a notebook's double margin line merges into one band.)
        if area / max(1, length) > 14:
            continue
        # A rule is straight: it drifts by a small fraction of its length.
        if extent > max(12, 0.06 * length):
            continue
        kept[labels == index] = 255
        n += 1
    return cv2.bitwise_and(cv2.dilate(kept, thicken), binary), n


def remove_ruled_lines(
    gray: np.ndarray,
    *,
    horizontal: bool = True,
    vertical: bool = True,
) -> tuple[np.ndarray, int, int]:
    """Erase printed rules (notebook lines, margins, table borders).

    Long thin horizontal and vertical runs are found morphologically and then
    inpainted, so a handwritten stroke that crosses a rule is filled back in
    rather than left with a gap.

    Expects a background-flattened page (see :func:`flatten_background`),
    where the paper is close to white.

    Returns:
        ``(cleaned, horizontal_count, vertical_count)``.
    """
    height, width = gray.shape[:2]
    # Printed rules are lighter than pen ink but clearly darker than paper.
    binary = np.where(gray < 215, 255, 0).astype(np.uint8)

    mask = np.zeros_like(binary)
    h_count = v_count = 0
    if horizontal:
        h_mask, h_count = _rule_mask(binary, horizontal=True, min_length=max(60, width // 5))
        mask = cv2.bitwise_or(mask, h_mask)
    if vertical:
        v_mask, v_count = _rule_mask(binary, horizontal=False, min_length=max(60, height // 5))
        mask = cv2.bitwise_or(mask, v_mask)

    if not (h_count or v_count):
        return gray, 0, 0

    # Widen by a pixel so the anti-aliased edge of the rule goes too.
    mask = cv2.dilate(mask, np.ones((3, 3), np.uint8))
    cleaned = cv2.inpaint(gray, mask, 3, cv2.INPAINT_TELEA)
    # Whatever the inpaint left faint is background, not ink.
    cleaned = np.where((mask > 0) & (cleaned > 150), 255, cleaned).astype(np.uint8)
    return cleaned, h_count, v_count


def remove_thin_strokes(gray: np.ndarray, *, ink: int = 170) -> tuple[np.ndarray, int]:
    """Delete long hairline marks that are not writing.

    Pieces of rules too short to be caught as rules, page edges of the
    sheets underneath, curved binding shadows: all are long and only a pixel
    or two thick. Pen strokes long enough to matter are joined into letters,
    so their components are thicker. Without this, a hairline lying in the
    gap between two text rows fills the gap in the projection profile and the
    two lines merge.
    """
    binary = np.where(gray < ink, 255, 0).astype(np.uint8)
    count, labels, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
    out = gray.copy()
    removed = 0
    for index in range(1, count):
        w = int(stats[index, cv2.CC_STAT_WIDTH])
        h = int(stats[index, cv2.CC_STAT_HEIGHT])
        area = int(stats[index, cv2.CC_STAT_AREA])
        length = max(w, h)
        if length < 40:
            continue
        # mean thickness along the longer side
        if area / length <= 2.6 and min(w, h) <= 0.6 * length:
            out[labels == index] = 255
            removed += 1
    return out, removed


def whiten_dark_borders(gray: np.ndarray, *, dark: int = 100, min_area: float = 0.01) -> np.ndarray:
    """Turn the table or desk visible around a photographed page white.

    A large dark region touching the image edge is not paper. Left in, it
    reads as one enormous "character" that every rule and line connects to,
    and line detection collapses into a single box. Checking connected
    regions rather than whole edge columns keeps this working after deskew,
    which pads the corners with white.
    """
    mask = np.where(gray < dark, 255, 0).astype(np.uint8)
    height, width = gray.shape[:2]
    # Opening with a kernel much wider than a pen stroke deletes all writing
    # and keeps only solid dark areas, so ink touching the desk edge cannot
    # merge with it.
    solid = max(15, min(height, width) // 40)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((solid, solid), np.uint8))
    count, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    limit = min_area * height * width
    out = gray.copy()
    for index in range(1, count):
        x, y, w, h, area = (int(v) for v in stats[index])
        touches = x <= 1 or y <= 1 or x + w >= width - 1 or y + h >= height - 1
        # Large, solid and on the edge: handwriting is none of the three.
        if touches and area >= limit and area / max(1, w * h) > 0.5:
            region = cv2.dilate((labels == index).astype(np.uint8), np.ones((7, 7), np.uint8))
            out[region > 0] = 255
    return out


def clean_handwritten_page(
    image: np.ndarray,
    *,
    flatten: bool = True,
    remove_rules: bool = True,
) -> CleanedPage:
    """Prepare a photographed handwritten page for line detection.

    Args:
        image: gray or BGR ``uint8`` image.
        flatten: remove shadows / lighting gradients first.
        remove_rules: erase printed notebook rules and margin lines.
    """
    gray = ops.to_grayscale(image)
    gray, _ = ops.normalise_polarity(gray)

    flattened = False
    if flatten and min(gray.shape[:2]) >= 64:
        # Borders first: flattening would turn a dark desk edge mid-gray and
        # hide it from the darkness test.
        gray = flatten_background(whiten_dark_borders(gray))
        flattened = True

    h_count = v_count = 0
    if remove_rules and min(gray.shape[:2]) >= 64:
        gray, h_count, v_count = remove_ruled_lines(gray)

    hairlines = 0
    if remove_rules and min(gray.shape[:2]) >= 64:
        gray, hairlines = remove_thin_strokes(gray)

    if h_count or v_count:
        logger.debug("Removed %d horizontal and %d vertical rule(s)", h_count, v_count)
    return CleanedPage(gray, h_count, v_count, flattened, hairlines)


# ---------------------------------------------------------------------------
# Splitting merged lines
# ---------------------------------------------------------------------------
def split_tall_regions(
    gray: np.ndarray,
    regions: Sequence[TextRegion],
    *,
    tall_factor: float = 1.6,
    ink: int = 170,
) -> list[TextRegion]:
    """Split boxes that hold two or more handwritten lines.

    Handwritten descenders often touch the line below, so a projection-based
    detector returns the two as one box. A box much taller than the typical
    line is cut at the emptiest rows of its ink profile, near where the line
    breaks are expected.
    """
    if len(regions) < 3:
        return list(regions)

    heights = np.array([r.height for r in regions], dtype=np.float32)
    # The typical line, ignoring tiny fragments (dots, stray marks).
    substantial = heights[heights >= np.percentile(heights, 30)]
    typical = float(np.median(substantial)) if substantial.size else float(np.median(heights))
    if typical <= 0:
        return list(regions)

    out: list[TextRegion] = []
    for region in regions:
        parts = int(round(region.height / typical))
        if region.height < tall_factor * typical or parts < 2:
            out.append(region)
            continue

        crop = gray[region.y : region.y + region.height, region.x : region.x + region.width]
        profile = (crop < ink).sum(axis=1).astype(np.float32)
        window = max(3, int(typical * 0.15))
        profile = np.convolve(profile, np.ones(window) / window, mode="same")

        cuts: list[int] = []
        step = region.height / parts
        for k in range(1, parts):
            centre = int(k * step)
            lo = max(1, centre - int(step * 0.35))
            hi = min(region.height - 1, centre + int(step * 0.35))
            if hi <= lo:
                continue
            cuts.append(lo + int(np.argmin(profile[lo:hi])))

        bounds = [0, *cuts, region.height]
        for top, bottom in zip(bounds[:-1], bounds[1:]):
            if bottom - top < typical * 0.4:
                continue
            out.append(TextRegion(region.x, region.y + top, region.width, bottom - top))

    for index, region in enumerate(out):
        region.index = index
    return out


# ---------------------------------------------------------------------------
# Word grouping (shared by every detector that returns word boxes)
# ---------------------------------------------------------------------------
def group_words_into_lines(
    boxes: Sequence[tuple[int, int, int, int]],
    *,
    min_overlap: float = 0.4,
    max_gap_ratio: float = 4.0,
) -> list[list[int]]:
    """Chain word boxes into text lines, left to right.

    Each word joins the line whose *rightmost* word overlaps it vertically by
    at least ``min_overlap`` of the smaller height. Comparing with the
    rightmost word rather than the line's overall extent is what lets a line
    that drifts upward across the page stay one line.

    Args:
        boxes: ``(x, y, width, height)`` per word.
        min_overlap: required vertical overlap, as a fraction of the smaller box.
        max_gap_ratio: words further apart than this many line-heights
            horizontally start a new line (two columns, a margin note).

    Returns:
        Lines as lists of indices into *boxes*, top to bottom.
    """
    if not boxes:
        return []

    order = sorted(range(len(boxes)), key=lambda i: boxes[i][0])
    lines: list[list[int]] = []

    for index in order:
        x, y, w, h = boxes[index]
        best, best_score = None, 0.0
        for line_number, line in enumerate(lines):
            lx, ly, lw, lh = boxes[line[-1]]
            overlap = min(y + h, ly + lh) - max(y, ly)
            smaller = max(1, min(h, lh))
            ratio = overlap / smaller
            gap = x - (lx + lw)
            if ratio >= min_overlap and gap <= max_gap_ratio * max(h, lh) and ratio > best_score:
                best, best_score = line_number, ratio
        if best is None:
            lines.append([index])
        else:
            lines[best].append(index)

    def centre(line: list[int]) -> float:
        return float(np.median([boxes[i][1] + boxes[i][3] / 2 for i in line]))

    lines.sort(key=centre)
    return lines


def _lines_to_regions(
    boxes: Sequence[tuple[int, int, int, int]],
    lines: Sequence[Sequence[int]],
    *,
    image_shape: tuple[int, int],
    padding_ratio: float = 0.15,
) -> list[TextRegion]:
    height, width = image_shape
    regions: list[TextRegion] = []
    for line in lines:
        xs = [boxes[i][0] for i in line]
        ys = [boxes[i][1] for i in line]
        x2s = [boxes[i][0] + boxes[i][2] for i in line]
        y2s = [boxes[i][1] + boxes[i][3] for i in line]
        x1, y1, x2, y2 = min(xs), min(ys), max(x2s), max(y2s)
        line_height = float(np.median([boxes[i][3] for i in line]))
        pad = int(round(line_height * padding_ratio))
        x1, y1 = max(0, x1 - pad), max(0, y1 - pad)
        x2, y2 = min(width, x2 + pad), min(height, y2 + pad)
        regions.append(TextRegion(x1, y1, x2 - x1, y2 - y1, index=len(regions)))
    return regions


# ---------------------------------------------------------------------------
# docTR detector (optional dependency)
# ---------------------------------------------------------------------------
_DOCTR_LOCK = threading.Lock()
_DOCTR_PREDICTOR: Any = None
_DOCTR_ERROR: str | None = None


def doctr_available() -> bool:
    """True when the `python-doctr` package can be imported."""
    try:
        import doctr  # noqa: F401
    except Exception:
        return False
    return True


def _doctr_predictor(arch: str = "db_resnet50") -> Any:
    """Load the docTR detector once and reuse it (it is ~100 MB)."""
    global _DOCTR_PREDICTOR, _DOCTR_ERROR
    if _DOCTR_PREDICTOR is not None:
        return _DOCTR_PREDICTOR
    with _DOCTR_LOCK:
        if _DOCTR_PREDICTOR is not None:
            return _DOCTR_PREDICTOR
        try:
            import torch
            from doctr.models import detection_predictor

            predictor = detection_predictor(
                arch=arch, pretrained=True, assume_straight_pages=True
            )
            if torch.cuda.is_available():
                predictor = predictor.cuda()
            if hasattr(predictor, "eval"):
                predictor.eval()
            _DOCTR_PREDICTOR = predictor
            logger.info("Loaded docTR text detector (%s)", arch)
        except Exception as exc:  # missing package, no network on first use...
            _DOCTR_ERROR = f"{type(exc).__name__}: {exc}"
            raise
    return _DOCTR_PREDICTOR


def _doctr_word_boxes(gray: np.ndarray, *, min_score: float = 0.3) -> list[tuple[int, int, int, int]]:
    """Run docTR's DBNet detector and return word boxes in pixels."""
    import torch

    predictor = _doctr_predictor()
    rgb = cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB) if gray.ndim == 2 else cv2.cvtColor(gray, cv2.COLOR_BGR2RGB)
    with torch.inference_mode():
        output = predictor([rgb])

    page = output[0]
    # docTR >= 0.7 returns {class_name: array}; older releases a bare array.
    if isinstance(page, dict):
        arrays = [np.asarray(v) for v in page.values() if np.asarray(v).size]
        detections = np.concatenate(arrays, axis=0) if arrays else np.zeros((0, 5))
    else:
        detections = np.asarray(page)

    height, width = gray.shape[:2]
    boxes: list[tuple[int, int, int, int]] = []
    for row in detections:
        if row.shape[-1] >= 5 and float(row[4]) < min_score:
            continue
        x1, y1, x2, y2 = (float(v) for v in row[:4])
        # docTR returns coordinates relative to the page.
        if max(x1, y1, x2, y2) <= 1.5:
            x1, x2 = x1 * width, x2 * width
            y1, y2 = y1 * height, y2 * height
        x1, y1 = max(0, int(x1)), max(0, int(y1))
        x2, y2 = min(width, int(round(x2))), min(height, int(round(y2)))
        if x2 - x1 >= 3 and y2 - y1 >= 3:
            boxes.append((x1, y1, x2 - x1, y2 - y1))
    return boxes


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------
def _resolve_detector(detector: str) -> str:
    detector = (detector or "auto").lower()
    if detector not in LINE_DETECTORS:
        raise ValueError(f"Unknown line detector {detector!r}; choose from {', '.join(LINE_DETECTORS)}")
    if detector == "auto":
        return "doctr" if doctr_available() else "projection"
    return detector


def detect_handwritten_lines(
    page: np.ndarray,
    *,
    detector: str = "auto",
    max_lines: int = 200,
) -> tuple[SegmentationResult, str]:
    """Find text lines on a cleaned handwritten page.

    Args:
        page: output of :func:`clean_handwritten_page` (``CleanedPage.image``)
            or any gray/BGR image.
        detector: ``auto`` | ``projection`` | ``doctr``.
        max_lines: cap on returned lines.

    Returns:
        ``(segmentation, detector_used)``. If docTR was asked for but fails
        (not installed, weights not downloadable), the projection detector is
        used and ``detector_used`` says so -- a degraded result beats an error.
    """
    chosen = _resolve_detector(detector)
    gray = ops.to_grayscale(page)
    fell_back = False

    if chosen == "doctr":
        try:
            boxes = _doctr_word_boxes(gray)
        except Exception as exc:
            logger.warning("docTR detection unavailable (%s); using the projection detector", exc)
            boxes = None
        if boxes:
            lines = group_words_into_lines(boxes)[:max_lines]
            regions = _lines_to_regions(boxes, lines, image_shape=gray.shape[:2])
            heights = [b[3] for b in boxes]
            return (
                SegmentationResult(
                    regions=regions,
                    used_whole_image=False,
                    median_char_height=float(np.median(heights)) if heights else 0.0,
                ),
                "doctr",
            )
        if boxes is None:
            fell_back = True
        else:
            logger.info("docTR found no words; falling back to the projection detector")

    segmentation = detect_text_lines(gray, max_lines=max_lines)
    if not segmentation.used_whole_image:
        regions = split_tall_regions(gray, segmentation.regions)
        # Slivers hugging the left/right edge are the edges of the sheets
        # underneath or the binding, not writing.
        width = gray.shape[1]
        regions = [
            r for r in regions
            if not ((r.x <= 15 or r.x + r.width >= width - 15) and r.width < 0.07 * width)
        ] or regions
        for index, region in enumerate(regions):
            region.index = index
        segmentation.regions = regions[:max_lines]
    return segmentation, "projection" if not fell_back else "projection (docTR unavailable)"
