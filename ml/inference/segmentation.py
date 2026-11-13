"""Text-line detection.

The recogniser consumes one text line at a time.  Real uploads are whole
photographs or scans, so something has to find the lines first.  This module
does that with classical morphology rather than a learned detector, which is a
deliberate scope decision: the synopsis is about *recognition*, and a
morphological line finder is reliable on documents, needs no training data,
and costs a few milliseconds.

The pipeline is:

1. binarise so ink is foreground;
2. estimate the typical character height from connected components, which
   makes every later kernel size relative to the text rather than to the
   image resolution -- this is what lets one set of parameters work on a
   200 px crop and a 3000 px scan;
3. dilate horizontally to merge characters into words and words into lines;
4. take connected components as candidate lines, filter and merge them;
5. sort top-to-bottom.

If no plausible line is found, the whole image is returned as a single region,
so the caller always gets something to recognise.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Sequence

import cv2
import numpy as np

from ml.preprocessing import ops

__all__ = ["TextRegion", "SegmentationResult", "detect_text_lines", "crop_region"]

logger = logging.getLogger(__name__)

#: Components smaller than this fraction of the median character height are
#: treated as noise (dust, JPEG speckle, punctuation fragments).
_MIN_HEIGHT_RATIO = 0.35
#: Components taller than this multiple of the median are structural lines or
#: borders, not text.
_MAX_HEIGHT_RATIO = 6.0

#: An image shorter than this multiple of its own median character height is
#: treated as a single, already-cropped text line. Two lines of text plus the
#: gap between them occupy at least ~2.5 character heights, so anything below
#: that cannot be multi-line.
SINGLE_LINE_HEIGHT_RATIO = 2.3

#: Character height below which detection runs on an upscaled copy.
_SMALL_TEXT_HEIGHT = 16.0


@dataclass(slots=True)
class TextRegion:
    """A detected text line, in original-image pixel coordinates."""

    x: int
    y: int
    width: int
    height: int
    index: int = 0
    ink_ratio: float = 0.0

    @property
    def box(self) -> tuple[int, int, int, int]:
        return self.x, self.y, self.width, self.height

    @property
    def area(self) -> int:
        return self.width * self.height

    @property
    def centre_y(self) -> float:
        return self.y + self.height / 2.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "index": self.index,
            "x": self.x,
            "y": self.y,
            "width": self.width,
            "height": self.height,
            "ink_ratio": round(self.ink_ratio, 4),
        }


@dataclass(slots=True)
class SegmentationResult:
    """Outcome of :func:`detect_text_lines`."""

    regions: list[TextRegion]
    #: True when detection found nothing and the whole image is being returned.
    used_whole_image: bool = False
    median_char_height: float = 0.0
    #: Vertical bands that separate genuine text columns. Boxes either side of
    #: one belong to different columns and must not be joined into one line.
    column_gutters: list[tuple[int, int]] = field(default_factory=list)

    def __len__(self) -> int:
        return len(self.regions)

    def to_dict(self) -> dict[str, Any]:
        return {
            "line_count": len(self.regions),
            "used_whole_image": self.used_whole_image,
            "median_char_height": round(self.median_char_height, 2),
            "column_gutters": [list(g) for g in self.column_gutters],
            "regions": [region.to_dict() for region in self.regions],
        }


def group_into_rows(
    regions: Sequence[TextRegion],
    *,
    gutters: Sequence[tuple[int, int]] = (),
) -> list[list[int]]:
    """Group region indices into visual rows, left to right.

    Segmentation deliberately keeps a table's columns as separate boxes, so
    that each is a tight crop the recogniser handles well. This puts the row
    back together for the output: ``Laptop stand`` + ``2`` + ``51.25`` become
    one line again, joined with spaces, instead of three.

    Two boxes share a row when their vertical extents overlap by more than
    half the smaller box's height and no column gutter lies between them.

    Returns:
        A list of rows, each a list of indices into *regions*, ordered top to
        bottom and then left to right.
    """
    if not regions:
        return []

    order = sorted(range(len(regions)), key=lambda i: (regions[i].y, regions[i].x))
    rows: list[list[int]] = []

    def crosses_gutter(a: TextRegion, b: TextRegion) -> bool:
        left, right = (a, b) if a.x <= b.x else (b, a)
        gap_start, gap_end = left.x + left.width, right.x
        return any(start < gap_end and end > gap_start for start, end in gutters)

    for index in order:
        region = regions[index]
        placed = False

        for row in rows:
            member = regions[row[-1]]
            top = max(member.y, region.y)
            bottom = min(member.y + member.height, region.y + region.height)
            overlap = bottom - top
            smaller = min(member.height, region.height)

            if overlap > 0 and overlap >= 0.5 * smaller and not crosses_gutter(member, region):
                row.append(index)
                placed = True
                break

        if not placed:
            rows.append([index])

    for row in rows:
        row.sort(key=lambda i: regions[i].x)
    rows.sort(key=lambda row: min(regions[i].y for i in row))
    return rows


def _ink_mask(gray: np.ndarray) -> np.ndarray:
    """Binarise so text pixels are 255 and background is 0.

    Otsu is tried first; when the image has very low global contrast (a faint
    pencil scan, say) its single threshold collapses, so an adaptive threshold
    is used instead.
    """
    blurred = cv2.GaussianBlur(gray, (3, 3), 0)
    _, mask = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)

    ink_fraction = float(np.count_nonzero(mask)) / mask.size
    # A sane page is a few percent ink. Far outside that range means Otsu
    # split noise rather than text.
    if ink_fraction < 0.002 or ink_fraction > 0.55:
        logger.debug("Otsu produced an implausible ink ratio (%.4f); using adaptive", ink_fraction)
        mask = cv2.adaptiveThreshold(
            blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 31, 12
        )

    # Remove isolated speckle without eroding thin strokes.
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((2, 2), np.uint8))
    return _suppress_page_furniture(mask)


def _suppress_page_furniture(mask: np.ndarray) -> np.ndarray:
    """Delete borders, frames and long rules from the ink mask.

    A scanned page or a screenshot very often has a printed frame around the
    text. Its vertical sides put ink in *every* row, which fuses the whole
    page into a single band in the projection profile -- so the line detector
    concludes there is one enormous line and hands the entire page to the
    recogniser as one crop. That is exactly how a full paragraph came back as
    ``"0 0"``.

    No glyph spans most of a page, so any component that does is furniture:
    a border, a table rule, or an underline.
    """
    height, width = mask.shape[:2]
    count, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    if count <= 1:
        return mask

    cleaned = mask.copy()
    removed = 0

    for index in range(1, count):
        component_width = int(stats[index, cv2.CC_STAT_WIDTH])
        component_height = int(stats[index, cv2.CC_STAT_HEIGHT])
        area = int(stats[index, cv2.CC_STAT_AREA])

        spans_width = component_width >= width * 0.80
        spans_height = component_height >= height * 0.80

        if not (spans_width or spans_height):
            continue

        # A frame is thin relative to its extent: its filled area is a small
        # fraction of its bounding box. A genuine full-width text line is not,
        # and must survive.
        box_area = max(1, component_width * component_height)
        if area / box_area < 0.28:
            cleaned[labels == index] = 0
            removed += 1

    if removed:
        logger.debug("Removed %d page border/rule component(s) from the ink mask", removed)
    return cleaned


def _median_char_height(mask: np.ndarray) -> float:
    """Median height of plausible character components, in pixels."""
    count, _, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    if count <= 1:
        return 0.0

    heights: list[int] = []
    image_height = mask.shape[0]
    for index in range(1, count):
        height = int(stats[index, cv2.CC_STAT_HEIGHT])
        width = int(stats[index, cv2.CC_STAT_WIDTH])
        area = int(stats[index, cv2.CC_STAT_AREA])
        # Skip dust and skip anything spanning most of the page (borders,
        # table rules, the edge of a photograph).
        if area < 6 or height < 3 or height > image_height * 0.6:
            continue
        if width > mask.shape[1] * 0.9:
            continue
        heights.append(height)

    return float(np.median(heights)) if heights else 0.0


def _row_bands(mask: np.ndarray) -> list[tuple[int, int]]:
    """Find horizontal bands of ink via the row projection profile.

    More robust than connected components for deciding *how many lines* an
    image holds: a local threshold such as Sauvola can shatter glyphs into
    stroke fragments, which collapses the median component height and makes a
    single line look like many small ones. A projection profile is unaffected
    -- a line of text is one contiguous run of inked rows however fragmented
    its strokes are.

    Returns:
        ``(start_row, end_row_exclusive)`` pairs, top to bottom.
    """
    height, width = mask.shape[:2]
    # Ignore rows holding only a speck of ink (dust, a stray mark).
    row_ink = (mask > 0).sum(axis=1)
    threshold = max(1, int(width * 0.005))
    inked = row_ink >= threshold

    bands: list[tuple[int, int]] = []
    start: int | None = None
    for row in range(height):
        if inked[row] and start is None:
            start = row
        elif not inked[row] and start is not None:
            bands.append((start, row))
            start = None
    if start is not None:
        bands.append((start, height))

    if not bands:
        return []

    # Join a diacritic band to the body it belongs to -- the dot of an 'i',
    # an accent, the top of a capital.
    #
    # The test is deliberately narrow. Merging on "gap smaller than half the
    # typical band height" swallowed the normal spacing between lines of a
    # paragraph, fusing an entire page into one band: body text is often 20 px
    # tall with a 6 px leading, so every real line gap qualified. A diacritic
    # band is instead *thin* and sits *close* above a much thicker band, so
    # both conditions are required.
    # Compare against the *median* band height rather than the neighbour's.
    # A pairwise comparison mis-fires when a page yields only two bands of very
    # different sizes: a 134 px block counted as "thin" beside a 381 px one and
    # the two were fused.
    typical = float(np.median([end - begin for begin, end in bands]))
    max_gap = max(1.0, typical * 0.22)

    merged: list[tuple[int, int]] = [bands[0]]
    for begin, end in bands[1:]:
        previous_begin, previous_end = merged[-1]
        gap = begin - previous_end
        upper_height = previous_end - previous_begin

        # A diacritic band is a small fraction of a text line, in absolute
        # terms, and sits right on top of it.
        upper_is_diacritic = upper_height <= max(3.0, typical * 0.45)
        if gap <= max_gap and upper_is_diacritic:
            merged[-1] = (previous_begin, end)
        else:
            merged.append((begin, end))

    # Drop slivers that cannot be text.
    return [(begin, end) for begin, end in merged if end - begin >= 3]


def _detect_column_gutters(
    regions: list[TextRegion], *, char_height: float, page_width: int
) -> list[tuple[int, int]]:
    """Find vertical gutters that separate genuine text columns.

    A table row and a two-column layout look identical on any single line --
    two boxes sharing a baseline with a wide gap between them. What tells them
    apart is *repetition*: a column gutter is an empty vertical band running
    down most of the page, whereas a table's inter-column gaps sit at
    different x positions on different rows.

    Returns:
        ``(start_x, end_x)`` bands that should never be merged across.
    """
    if len(regions) < 6:
        # Too few lines to establish that a gap repeats.
        return []

    # Group boxes into text rows by their vertical centre.
    row_height = max(1.0, char_height * 1.5)
    rows: dict[int, list[TextRegion]] = {}
    for region in regions:
        rows.setdefault(int(region.centre_y // row_height), []).append(region)

    if len(rows) < 4:
        return []

    # For each x, in how many rows is that column covered by some box?
    coverage = np.zeros(page_width + 1, dtype=np.int32)
    for row_regions in rows.values():
        covered = np.zeros(page_width + 1, dtype=bool)
        for region in row_regions:
            covered[region.x : min(region.x + region.width, page_width + 1)] = True
        coverage += covered

    row_count = len(rows)
    # A true column gutter is crossed by almost nothing. The threshold is not
    # zero because a heading or a full-width rule legitimately spans the page
    # in a two-column document.
    quiet = coverage <= max(1, int(row_count * 0.2))
    min_gutter = max(3, int(char_height * 1.5))

    bands: list[tuple[int, int]] = []
    start: int | None = None
    for x in range(page_width + 1):
        if quiet[x] and start is None:
            start = x
        elif not quiet[x] and start is not None:
            if x - start >= min_gutter:
                bands.append((start, x))
            start = None
    if start is not None and page_width + 1 - start >= min_gutter:
        bands.append((start, page_width + 1))

    # Only interior bands separate columns; the page margins are not gutters.
    margin = int(page_width * 0.12)
    interior = [(a, b) for a, b in bands if a > margin and b < page_width - margin]

    # Finally, a gutter must have substantial text on both sides in most rows
    # -- otherwise it is just the ragged right edge of a single column.
    confirmed: list[tuple[int, int]] = []
    for a, b in interior:
        both_sides = sum(
            1
            for row_regions in rows.values()
            if any(r.x + r.width <= a + 1 for r in row_regions)
            and any(r.x >= b - 1 for r in row_regions)
        )
        if both_sides >= row_count * 0.6:
            confirmed.append((a, b))

    return confirmed


def _split_dropcap_bands(
    mask: np.ndarray, bands: list[tuple[int, int]]
) -> list[tuple[int, int]]:
    """Split bands fused by an oversized initial capital.

    A drop cap spans two or three text lines, so its ink fills the rows
    between them and the projection profile sees one tall band. The lines it
    overlaps are then handed to the recogniser as a single crop and come back
    as nonsense -- a book page lost two whole lines to a single ``M``.

    The fix is to re-project the band with its over-tall components removed,
    which reveals the ordinary lines underneath.
    """
    if len(bands) < 2:
        return bands

    heights = [end - begin for begin, end in bands]
    typical = float(np.median(heights))
    if typical <= 0:
        return bands

    result: list[tuple[int, int]] = []
    for begin, end in bands:
        height = end - begin
        # Only a band well over one-and-a-half lines tall can hide another.
        if height < typical * 1.6:
            result.append((begin, end))
            continue

        strip = mask[begin:end, :]
        count, labels, stats, _ = cv2.connectedComponentsWithStats(strip, connectivity=8)
        if count <= 1:
            result.append((begin, end))
            continue

        without_dropcap = strip.copy()
        removed = False
        for index in range(1, count):
            component_height = int(stats[index, cv2.CC_STAT_HEIGHT])
            if component_height >= typical * 1.4:
                without_dropcap[labels == index] = 0
                removed = True

        if not removed:
            result.append((begin, end))
            continue

        inner = _row_bands(without_dropcap)
        # Only accept the split if it genuinely found more than one line.
        if len(inner) >= 2:
            logger.debug(
                "Split a %d px band into %d lines after removing an oversized initial",
                height,
                len(inner),
            )
            result.extend((begin + a, begin + b) for a, b in inner)
        else:
            result.append((begin, end))

    return sorted(result)


def _merge_overlapping(
    regions: list[TextRegion],
    *,
    vertical_tolerance: float,
    # Joins words the dilation failed to bridge, but deliberately leaves a
    # table's column gaps alone. A single box spanning a wide blank stretch
    # recognises badly -- the model never saw that much internal whitespace in
    # training -- so wide-apart fragments are recognised separately and
    # rejoined afterwards by `group_into_rows`.
    max_gap_ratio: float = 3.0,
    gutters: Sequence[tuple[int, int]] = (),
) -> list[TextRegion]:
    """Merge boxes that sit on the same text line.

    Two boxes merge when their vertical extents overlap by more than half the
    smaller box's height -- a word and the dot of an 'i' above it, or two
    words the dilation failed to join -- and they are horizontally close.

    "Close" is ``max_gap_ratio`` times the character height. The previous 2.5
    was too tight for tabular text: an invoice row such as
    ``Laptop stand   2   51.25`` has column gaps around 3x the character
    height, so every row was split into separate "lines" and the recognised
    text came out fragmented.

    Merging repeats to a fixed point, because joining A to B can bring the
    grown box within reach of C.
    """
    if len(regions) <= 1:
        return regions

    def crosses_gutter(left: TextRegion, right: TextRegion) -> bool:
        gap_start = left.x + left.width
        gap_end = right.x
        return any(a < gap_end and b > gap_start for a, b in gutters)

    current = sorted(regions, key=lambda r: (r.y, r.x))

    for _ in range(4):  # a fixed point is reached well within this
        merged: list[TextRegion] = []
        changed = False

        for region in current:
            placed = False
            for index, existing in enumerate(merged):
                top = max(existing.y, region.y)
                bottom = min(existing.y + existing.height, region.y + region.height)
                overlap = bottom - top
                smaller = min(existing.height, region.height)

                if overlap <= 0 or overlap < 0.5 * smaller:
                    continue

                left, right = (existing, region) if existing.x <= region.x else (region, existing)
                gap = right.x - (left.x + left.width)

                if gap > vertical_tolerance * max_gap_ratio:
                    continue
                if crosses_gutter(left, right):
                    # A repeated vertical band: these are separate columns.
                    continue

                x1 = min(existing.x, region.x)
                y1 = min(existing.y, region.y)
                x2 = max(existing.x + existing.width, region.x + region.width)
                y2 = max(existing.y + existing.height, region.y + region.height)
                merged[index] = TextRegion(x1, y1, x2 - x1, y2 - y1)
                placed = True
                changed = True
                break

            if not placed:
                merged.append(region)

        current = sorted(merged, key=lambda r: (r.y, r.x))
        if not changed:
            break

    return current


def detect_text_lines(
    image: np.ndarray,
    *,
    min_line_height: int = 8,
    max_lines: int = 200,
    padding_ratio: float = 0.18,
    merge_lines: bool = True,
) -> SegmentationResult:
    """Find text lines in *image*.

    Args:
        image: gray or BGR ``uint8`` image of any size.
        min_line_height: absolute floor on a line's height in pixels.
        max_lines: cap on returned regions; the tallest are kept.  Guards
            against a textured photograph yielding hundreds of fragments.
        padding_ratio: fraction of the line height added around each box, so
            ascenders and descenders are not clipped.
        merge_lines: join boxes that share a baseline.

    Returns:
        A :class:`SegmentationResult`, ordered top-to-bottom.  Never empty:
        it falls back to the whole image.
    """
    gray = ops.to_grayscale(image)
    gray, _ = ops.normalise_polarity(gray)
    height, width = gray.shape[:2]

    whole = SegmentationResult(
        regions=[TextRegion(0, 0, width, height, index=0)],
        used_whole_image=True,
    )

    if height < min_line_height or width < min_line_height:
        return whole

    # Small text defeats the pixel-based thresholds below: an invoice whose
    # glyphs are 7 px tall produced rows with too little ink to register, and
    # ~30 lines collapsed to 3. Detection runs on an upscaled copy and the
    # boxes are mapped back, so the caller still crops from the original.
    scale = 1.0
    probe = _median_char_height(_ink_mask(gray))
    if 0 < probe < _SMALL_TEXT_HEIGHT:
        scale = min(4.0, _SMALL_TEXT_HEIGHT / probe)
        gray = cv2.resize(gray, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
        height, width = gray.shape[:2]
        logger.debug("Text is only %.1f px tall; detecting at %.1fx", probe, scale)

    mask = _ink_mask(gray)
    char_height = _median_char_height(mask)
    bands = _split_dropcap_bands(mask, _row_bands(mask))

    if char_height <= 0 and not bands:
        logger.debug("No character-sized component found; treating the image as one line")
        return whole

    # The projection profile is the authority on *how many* lines there are,
    # and its band height is a floor on the character height: binarisation can
    # shatter glyphs into fragments, which would otherwise make one line of
    # text look like a page of tiny ones.
    if bands:
        band_height = float(np.median([end - begin for begin, end in bands]))
        char_height = max(char_height, band_height * 0.55)

    if char_height <= 0:
        return whole

    # One ink band, or an image barely taller than a single character, means
    # the input is an already-cropped line. Without this guard such a crop is
    # split into word-sized fragments whenever the horizontal dilation fails
    # to bridge a wide inter-word gap, and each fragment is then recognised
    # separately -- breaking the text into pieces for no reason.
    if len(bands) <= 1 or height < char_height * SINGLE_LINE_HEIGHT_RATIO:
        logger.debug(
            "Single text line detected (%d ink band(s), %d px tall, character height %.1f px)",
            len(bands),
            height,
            char_height,
        )
        return SegmentationResult(
            regions=[TextRegion(0, 0, width, height, index=0)],
            used_whole_image=True,
            median_char_height=char_height,
        )

    # Each row band *is* a text line: take the vertical extent from the band
    # and the horizontal extent from the ink inside it.
    #
    # This replaced a morphological approach that dilated the mask and took
    # connected components. The dilation kernel's height was a fraction of the
    # character height -- about 3 px for body text -- which is exactly the
    # leading between lines of a paragraph, so it welded the lines back
    # together and a nine-line paragraph came out as three regions. The
    # projection profile already separates the lines correctly; this uses it
    # rather than fighting it.
    candidates: list[TextRegion] = []

    for band_start, band_end in bands:
        band_height = band_end - band_start
        if band_height < max(min_line_height, char_height * _MIN_HEIGHT_RATIO):
            continue

        strip = mask[band_start:band_end, :]
        columns = (strip > 0).sum(axis=0)
        inked = columns > 0
        if not inked.any():
            continue

        # Bridge inter-character and inter-word gaps, but not column gaps.
        bridge = max(2, int(char_height * 1.2))

        segments: list[tuple[int, int]] = []
        start: int | None = None
        last_seen = -10**9
        for x in range(width):
            if inked[x]:
                if start is None:
                    start = x
                elif x - last_seen > bridge:
                    segments.append((start, last_seen + 1))
                    start = x
                last_seen = x
        if start is not None:
            segments.append((start, last_seen + 1))

        for left, right in segments:
            segment_width = right - left
            if segment_width < max(2, char_height * 0.35):
                continue

            patch = strip[:, left:right]
            ink_ratio = float(np.count_nonzero(patch)) / max(1, patch.size)
            # A solid blob (logo, photo, filled cell) is not text.
            if ink_ratio < 0.02 or ink_ratio > 0.92:
                continue

            candidates.append(
                TextRegion(left, band_start, segment_width, band_height, ink_ratio=ink_ratio)
            )

    if not candidates:
        logger.debug("Segmentation found no plausible line; using the whole image")
        return whole

    gutters: list[tuple[int, int]] = []
    if merge_lines:
        gutters = _detect_column_gutters(candidates, char_height=char_height, page_width=width)
        if gutters:
            logger.debug("Detected %d column gutter(s): %s", len(gutters), gutters)
        candidates = _merge_overlapping(
            candidates, vertical_tolerance=char_height, gutters=gutters
        )

    if len(candidates) > max_lines:
        logger.debug("Keeping the %d tallest of %d candidate lines", max_lines, len(candidates))
        candidates = sorted(candidates, key=lambda r: -r.area)[:max_lines]

    # Pad, clamp to the image, and order for reading.
    regions: list[TextRegion] = []
    for region in sorted(candidates, key=lambda r: (r.y, r.x)):
        pad = int(region.height * padding_ratio)
        x1 = max(0, region.x - pad)
        y1 = max(0, region.y - pad)
        x2 = min(width, region.x + region.width + pad)
        y2 = min(height, region.y + region.height + pad)
        if x2 - x1 < 4 or y2 - y1 < 4:
            continue
        regions.append(
            TextRegion(x1, y1, x2 - x1, y2 - y1, index=len(regions), ink_ratio=region.ink_ratio)
        )

    if not regions:
        return whole

    if scale != 1.0:
        regions = [
            TextRegion(
                int(r.x / scale), int(r.y / scale),
                max(1, int(r.width / scale)), max(1, int(r.height / scale)),
                index=r.index, ink_ratio=r.ink_ratio,
            )
            for r in regions
        ]
        gutters = [(int(a / scale), int(b / scale)) for a, b in gutters]
        char_height /= scale

    # A single region covering nearly everything is the same as no detection,
    # but reporting it as a real detection is more honest for the caller.
    return SegmentationResult(
        regions=regions,
        used_whole_image=False,
        median_char_height=char_height,
        column_gutters=gutters,
    )


def crop_region(image: np.ndarray, region: TextRegion) -> np.ndarray:
    """Extract *region* from *image*, clamped to the image bounds."""
    height, width = image.shape[:2]
    x1 = max(0, min(region.x, width - 1))
    y1 = max(0, min(region.y, height - 1))
    x2 = max(x1 + 1, min(region.x + region.width, width))
    y2 = max(y1 + 1, min(region.y + region.height, height))
    return image[y1:y2, x1:x2].copy()


def draw_regions(
    image: np.ndarray,
    regions: Sequence[TextRegion],
    *,
    colour: tuple[int, int, int] = (0, 170, 255),
    thickness: int = 2,
) -> np.ndarray:
    """Return a BGR copy of *image* with the detected boxes drawn on it."""
    canvas = ops.to_bgr(image)
    for region in regions:
        cv2.rectangle(
            canvas,
            (region.x, region.y),
            (region.x + region.width, region.y + region.height),
            colour,
            thickness,
        )
        cv2.putText(
            canvas,
            str(region.index + 1),
            (region.x + 3, max(14, region.y + 16)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            colour,
            1,
            cv2.LINE_AA,
        )
    return canvas
