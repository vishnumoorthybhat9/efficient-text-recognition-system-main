"""Synthetic text-line renderer.

Why this exists
---------------
IAM, MJSynth and SynthText are the right datasets for a published result, but
they are 1-40 GB downloads behind registration walls.  The project still has
to be *runnable and trainable* out of the box -- on a marker's machine, during
a viva, with no dataset in hand -- so this module renders its own labeled
corpus from the fonts already installed on the system.

What it is and is not
---------------------
It **is** genuine training data: real glyph shapes from real typefaces, varied
backgrounds, and labels that are correct by construction.  Models trained on
it reach honest, reproducible numbers, and the self-supervised comparison is
valid because both arms see exactly the same data.

It is **not** a substitute for IAM when reporting handwriting results.  The
"handwritten" styles here are script *typefaces* with per-character jitter,
which is more regular than real cursive.  :mod:`ml.datasets.iam` loads the
real thing when it is available, and the README says which numbers came from
which source.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Literal, Sequence

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

__all__ = [
    "FontCatalogue",
    "RenderStyle",
    "SyntheticRenderer",
    "TextKind",
    "discover_fonts",
]

logger = logging.getLogger(__name__)

TextKind = Literal["printed", "handwritten"]

#: Windows font filenames that read as ordinary print.
_PRINTED_FONT_FILES: tuple[str, ...] = (
    "arial.ttf", "arialbd.ttf", "ariali.ttf", "arialn.ttf",
    "times.ttf", "timesbd.ttf", "timesi.ttf",
    "cour.ttf", "courbd.ttf",
    "calibri.ttf", "calibrib.ttf", "calibrii.ttf",
    "verdana.ttf", "verdanab.ttf",
    "georgia.ttf", "georgiab.ttf",
    "tahoma.ttf", "tahomabd.ttf",
    "consola.ttf", "consolab.ttf",
    "segoeui.ttf", "segoeuib.ttf", "segoeuil.ttf",
    "trebuc.ttf", "trebucbd.ttf",
    "candara.ttf", "corbel.ttf", "constan.ttf", "framd.ttf",
    "ebrima.ttf", "lucon.ttf", "impact.ttf",
    "pala.ttf", "palab.ttf", "bookos.ttf", "gara.ttf", "garabd.ttf",
    "LeelawUI.ttf", "micross.ttf",
)

#: Windows font filenames that read as handwriting or script.
_HANDWRITTEN_FONT_FILES: tuple[str, ...] = (
    "inkfree.ttf",          # Ink Free
    "segoesc.ttf", "segoescb.ttf",  # Segoe Script
    "gabriola.ttf",
    "comic.ttf", "comicbd.ttf",     # Comic Sans MS
    "mistral.ttf",
    "BRADHITC.TTF",         # Bradley Hand ITC
    "FREESCPT.TTF",         # Freestyle Script
    "ITCKRIST.TTF",         # Kristen ITC
    "LHANDW.TTF",           # Lucida Handwriting
    "PAPYRUS.TTF",
    "PRISTINA.TTF",
    "BRUSHSCI.TTF",         # Brush Script MT
    "VLADIMIR.TTF",         # Vladimir Script
    "MTCORSVA.TTF",         # Monotype Corsiva
    "JUICE___.TTF",
    "SCRIPTBL.TTF",
)

#: Directories searched for fonts, most specific first.
def _font_directories() -> list[Path]:
    candidates: list[Path] = []
    windir = os.environ.get("WINDIR") or os.environ.get("SystemRoot")
    if windir:
        candidates.append(Path(windir) / "Fonts")
    local = os.environ.get("LOCALAPPDATA")
    if local:
        candidates.append(Path(local) / "Microsoft" / "Windows" / "Fonts")
    # Linux / macOS, so the generator also works in CI or on a lab machine.
    candidates += [
        Path("/usr/share/fonts"),
        Path("/usr/local/share/fonts"),
        Path.home() / ".fonts",
        Path("/Library/Fonts"),
        Path("/System/Library/Fonts"),
        Path.home() / "Library" / "Fonts",
    ]
    return [path for path in candidates if path.is_dir()]


@dataclass(slots=True)
class FontCatalogue:
    """Usable font files on this machine, split by visual style."""

    printed: list[Path] = field(default_factory=list)
    handwritten: list[Path] = field(default_factory=list)
    used_fallback: bool = False

    def for_kind(self, kind: TextKind) -> list[Path]:
        return self.handwritten if kind == "handwritten" else self.printed

    @property
    def total(self) -> int:
        return len(self.printed) + len(self.handwritten)

    def to_dict(self) -> dict[str, object]:
        return {
            "printed_count": len(self.printed),
            "handwritten_count": len(self.handwritten),
            "printed": [p.name for p in self.printed],
            "handwritten": [p.name for p in self.handwritten],
            "used_fallback": self.used_fallback,
        }


def _is_loadable(path: Path) -> bool:
    """Check a font really opens; a few system files are stubs or restricted."""
    try:
        ImageFont.truetype(str(path), 24)
        return True
    except Exception:  # pragma: no cover - depends on the installed fonts
        logger.debug("Font %s could not be opened; skipping", path.name)
        return False


def discover_fonts(
    *,
    extra_dirs: Iterable[str | Path] = (),
    verify: bool = True,
) -> FontCatalogue:
    """Find usable printed and handwriting fonts on this machine.

    Unknown font files found in the search directories are added to the
    *printed* pool, so a Linux box with DejaVu/Liberation installed still gets
    a varied corpus even though none of the Windows filenames match.

    Args:
        extra_dirs: additional directories to search.
        verify: open each candidate to confirm it loads.

    Returns:
        A :class:`FontCatalogue`.  Never raises: if nothing is found, the
        catalogue is marked ``used_fallback`` and the renderer uses Pillow's
        built-in font.
    """
    directories = _font_directories() + [Path(d) for d in extra_dirs if Path(d).is_dir()]

    printed_targets = {name.lower() for name in _PRINTED_FONT_FILES}
    handwritten_targets = {name.lower() for name in _HANDWRITTEN_FONT_FILES}

    printed: list[Path] = []
    handwritten: list[Path] = []
    seen: set[str] = set()
    others: list[Path] = []

    for directory in directories:
        try:
            entries = sorted(directory.iterdir())
        except OSError:  # pragma: no cover - permission denied
            logger.debug("Could not list font directory %s", directory)
            continue

        for entry in entries:
            if not entry.is_file() or entry.suffix.lower() not in (".ttf", ".otf"):
                continue
            key = entry.name.lower()
            if key in seen:
                continue
            seen.add(key)

            if key in handwritten_targets:
                handwritten.append(entry)
            elif key in printed_targets:
                printed.append(entry)
            else:
                others.append(entry)

    # Top up the printed pool from whatever else is installed, skipping the
    # symbol and icon fonts that would render text as unreadable glyphs.
    if len(printed) < 6:
        skip_markers = (
            "webding", "wingding", "symbol", "marlett", "emoji", "mticon",
            "holomdl", "segmdl", "segoeicons", "opensymbol", "dingbat",
        )
        for entry in others:
            if len(printed) >= 12:
                break
            if any(marker in entry.name.lower() for marker in skip_markers):
                continue
            printed.append(entry)

    if verify:
        printed = [p for p in printed if _is_loadable(p)]
        handwritten = [p for p in handwritten if _is_loadable(p)]

    catalogue = FontCatalogue(printed=printed, handwritten=handwritten)

    if not catalogue.printed and not catalogue.handwritten:
        catalogue.used_fallback = True
        logger.warning(
            "No usable TrueType font was found in %d directory(ies); falling back to "
            "Pillow's built-in bitmap font. The synthetic corpus will have only one "
            "typeface, which limits how well a model trained on it generalises.",
            len(directories),
        )
    else:
        logger.info(
            "Font discovery: %d printed, %d handwriting typeface(s)",
            len(catalogue.printed),
            len(catalogue.handwritten),
        )
        if not catalogue.handwritten:
            logger.warning(
                "No handwriting typeface found; 'handwritten' samples will be rendered "
                "with printed fonts and stronger per-character jitter."
            )

    return catalogue


@dataclass(slots=True)
class RenderStyle:
    """The randomised choices made for one rendered sample.

    Recorded alongside each image so a surprising sample can be traced back to
    the exact style that produced it.
    """

    font_name: str
    font_size: int
    text_gray: int
    background_gray: int
    background_kind: str
    padding_x: int
    padding_y: int
    char_jitter: float
    slant_deg: float
    stroke_width: int
    kind: str

    def to_dict(self) -> dict[str, object]:
        return {
            "font": self.font_name,
            "font_size": self.font_size,
            "text_gray": self.text_gray,
            "background_gray": self.background_gray,
            "background": self.background_kind,
            "char_jitter": round(self.char_jitter, 3),
            "slant_deg": round(self.slant_deg, 2),
            "stroke_width": self.stroke_width,
            "kind": self.kind,
        }


class SyntheticRenderer:
    """Renders a text string into a grayscale text-line image.

    >>> renderer = SyntheticRenderer(seed=0)
    >>> image, style = renderer.render("hello world", kind="printed")
    >>> image.ndim, image.dtype
    (2, dtype('uint8'))

    Args:
        catalogue: fonts to use; discovered automatically when omitted.
        height: target image height in pixels.  Rendered at 2x this and
            downsampled, which anti-aliases the strokes the way a real scanner
            would rather than leaving hard pixel edges.
        seed: RNG seed.
    """

    #: Pillow's built-in font is rendered at this size when no TrueType exists.
    _FALLBACK_SIZE = 16

    def __init__(
        self,
        catalogue: FontCatalogue | None = None,
        *,
        height: int = 64,
        seed: int | None = None,
    ) -> None:
        self.catalogue = catalogue if catalogue is not None else discover_fonts()
        self.height = int(height)
        self._rng = np.random.default_rng(seed)
        self._font_cache: dict[tuple[str, int], ImageFont.FreeTypeFont] = {}

    def reseed(self, seed: int) -> None:
        self._rng = np.random.default_rng(seed)

    # -- font handling ------------------------------------------------------
    def _load_font(self, path: Path | None, size: int) -> ImageFont.ImageFont | ImageFont.FreeTypeFont:
        if path is None:
            try:
                return ImageFont.load_default(size=size)
            except TypeError:  # pragma: no cover - Pillow < 10.1
                return ImageFont.load_default()

        key = (str(path), size)
        cached = self._font_cache.get(key)
        if cached is not None:
            return cached
        try:
            font = ImageFont.truetype(str(path), size)
        except Exception:
            logger.debug("Falling back to the default font: %s failed to load", path.name)
            return self._load_font(None, size)
        self._font_cache[key] = font
        return font

    def _pick_font(self, kind: TextKind) -> Path | None:
        options = self.catalogue.for_kind(kind)
        if not options:
            # No handwriting font installed: fall back to printed faces, which
            # the caller compensates for with extra jitter.
            options = self.catalogue.printed
        if not options:
            return None
        return Path(str(self._rng.choice([str(p) for p in options])))

    # -- backgrounds --------------------------------------------------------
    def _make_background(self, width: int, height: int, base_gray: int) -> tuple[np.ndarray, str]:
        """Build a page-like background of the requested size."""
        rng = self._rng
        kind = str(rng.choice(["plain", "gradient", "noise", "paper"], p=[0.40, 0.22, 0.20, 0.18]))

        canvas = np.full((height, width), float(base_gray), dtype=np.float32)

        if kind in ("gradient", "paper"):
            # Uneven illumination, as from a photograph or a flatbed edge.
            amplitude = float(rng.uniform(8.0, 28.0))
            if rng.random() < 0.5:
                ramp = np.linspace(-amplitude, amplitude, width, dtype=np.float32)[np.newaxis, :]
            else:
                ramp = np.linspace(-amplitude, amplitude, height, dtype=np.float32)[:, np.newaxis]
            canvas = canvas + ramp

        if kind == "paper":
            # Low-frequency blotches: a coarse random field, blurred up.
            coarse = rng.normal(0.0, 1.0, size=(max(2, height // 8), max(2, width // 8))).astype(np.float32)
            blotches = cv2.resize(coarse, (width, height), interpolation=cv2.INTER_CUBIC)
            canvas = canvas + cv2.GaussianBlur(blotches, (0, 0), 3.0) * float(rng.uniform(4.0, 12.0))

        if kind in ("noise", "paper"):
            canvas = canvas + rng.normal(0.0, float(rng.uniform(1.5, 5.0)), size=canvas.shape).astype(np.float32)

        return np.clip(canvas, 0, 255).astype(np.uint8), kind

    # -- rendering ----------------------------------------------------------
    def render(
        self,
        text: str,
        *,
        kind: TextKind = "printed",
    ) -> tuple[np.ndarray, RenderStyle]:
        """Render *text* as a grayscale line image.

        Args:
            text: the transcription to draw.  Must be non-empty.
            kind: ``"printed"`` or ``"handwritten"``.

        Returns:
            ``(image, style)`` where *image* is a 2-D ``uint8`` array of height
            ``self.height`` and *style* records the random choices.

        Raises:
            ValueError: *text* is empty or whitespace only.
        """
        if not text or not text.strip():
            raise ValueError("Cannot render empty text")

        rng = self._rng
        supersample = 2
        is_hand = kind == "handwritten"

        font_path = self._pick_font(kind)
        has_handwriting_font = bool(self.catalogue.handwritten) and is_hand

        # Render large, then downsample: gives smooth, scanner-like strokes.
        render_height = self.height * supersample
        font_size = int(render_height * float(rng.uniform(0.52, 0.72)))
        font = self._load_font(font_path, font_size)

        text_gray = int(rng.integers(0, 85))
        background_gray = int(rng.integers(205, 256))
        # Script faces look wrong when emboldened, so keep the stroke thin.
        stroke_width = int(rng.integers(0, 2)) if not is_hand else 0

        # Handwriting jitter: a real hand does not sit characters on a perfect
        # baseline.  When no script typeface is installed, raise the jitter so
        # the "handwritten" class is still visually distinct from "printed".
        if is_hand:
            char_jitter = float(rng.uniform(0.02, 0.06)) if has_handwriting_font else float(
                rng.uniform(0.05, 0.10)
            )
            slant_deg = float(rng.uniform(-6.0, 10.0))
        else:
            char_jitter = 0.0
            slant_deg = float(rng.uniform(-1.0, 1.0))

        padding_x = int(rng.integers(4, 18)) * supersample
        padding_y = int(rng.integers(2, 9)) * supersample

        image, background_kind = self._draw(
            text,
            font=font,
            render_height=render_height,
            padding_x=padding_x,
            padding_y=padding_y,
            text_gray=text_gray,
            background_gray=background_gray,
            stroke_width=stroke_width,
            char_jitter=char_jitter,
        )

        if abs(slant_deg) > 0.2:
            image = self._apply_slant(image, slant_deg, background_gray)

        # Downsample to the requested height with area averaging.
        target_width = max(8, int(round(image.shape[1] / supersample)))
        image = cv2.resize(image, (target_width, self.height), interpolation=cv2.INTER_AREA)

        style = RenderStyle(
            font_name=font_path.name if font_path else f"PIL-default-{self._FALLBACK_SIZE}",
            font_size=font_size // supersample,
            text_gray=text_gray,
            background_gray=background_gray,
            background_kind=background_kind,
            padding_x=padding_x // supersample,
            padding_y=padding_y // supersample,
            char_jitter=char_jitter,
            slant_deg=slant_deg,
            stroke_width=stroke_width,
            kind=kind,
        )
        return image, style

    def _draw(
        self,
        text: str,
        *,
        font: ImageFont.ImageFont | ImageFont.FreeTypeFont,
        render_height: int,
        padding_x: int,
        padding_y: int,
        text_gray: int,
        background_gray: int,
        stroke_width: int,
        char_jitter: float,
    ) -> tuple[np.ndarray, str]:
        """Draw *text* onto a generated background.

        Returns:
            ``(array, background_kind)``.
        """
        # Measure first so the canvas is exactly big enough.
        probe = Image.new("L", (8, 8), background_gray)
        probe_draw = ImageDraw.Draw(probe)
        try:
            left, top, right, bottom = probe_draw.textbbox(
                (0, 0), text, font=font, stroke_width=stroke_width
            )
        except Exception:  # pragma: no cover - bitmap fallback font
            right, bottom, left, top = probe_draw.textlength(text, font=font), render_height, 0, 0

        text_width = max(1, int(right - left))
        text_height = max(1, int(bottom - top))

        canvas_width = text_width + 2 * padding_x
        canvas_height = max(render_height, text_height + 2 * padding_y)

        background, background_kind = self._make_background(canvas_width, canvas_height, background_gray)
        image = Image.fromarray(background, mode="L")
        draw = ImageDraw.Draw(image)

        origin_x = padding_x - int(left)
        origin_y = (canvas_height - text_height) // 2 - int(top)

        if char_jitter <= 0.0:
            draw.text(
                (origin_x, origin_y),
                text,
                font=font,
                fill=text_gray,
                stroke_width=stroke_width,
                stroke_fill=text_gray,
            )
        else:
            self._draw_jittered(
                draw,
                text,
                font=font,
                origin=(origin_x, origin_y),
                text_gray=text_gray,
                jitter=char_jitter * canvas_height,
            )

        return np.asarray(image, dtype=np.uint8), background_kind

    def _draw_jittered(
        self,
        draw: ImageDraw.ImageDraw,
        text: str,
        *,
        font: ImageFont.ImageFont | ImageFont.FreeTypeFont,
        origin: tuple[int, int],
        text_gray: int,
        jitter: float,
    ) -> None:
        """Draw character by character with a wobbling baseline.

        Advances are taken from the cumulative string width rather than
        per-character widths, so kerning is preserved and the text does not
        drift apart.
        """
        rng = self._rng
        origin_x, origin_y = origin

        for index, character in enumerate(text):
            if character == " ":
                continue
            try:
                advance = float(font.getlength(text[:index]))  # type: ignore[union-attr]
            except Exception:  # pragma: no cover - bitmap fallback font
                advance = float(index) * jitter

            offset_y = float(rng.normal(0.0, jitter))
            offset_x = float(rng.normal(0.0, jitter * 0.25))
            # Vary ink density slightly, like pen pressure.
            ink = int(np.clip(text_gray + rng.normal(0.0, 12.0), 0, 140))

            draw.text(
                (origin_x + advance + offset_x, origin_y + offset_y),
                character,
                font=font,
                fill=ink,
            )

    def _apply_slant(self, image: np.ndarray, degrees: float, fill: int) -> np.ndarray:
        """Shear the image horizontally to imitate a consistent writing slant."""
        height, width = image.shape[:2]
        shear = float(np.tan(np.deg2rad(degrees)))
        extra = int(abs(shear) * height) + 1

        matrix = np.array(
            [[1.0, shear, 0.0 if shear > 0 else abs(shear) * height], [0.0, 1.0, 0.0]],
            dtype=np.float64,
        )
        return cv2.warpAffine(
            image,
            matrix,
            (width + extra, height),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=float(fill),
        )

    def render_many(
        self,
        texts: Sequence[str],
        *,
        kinds: Sequence[TextKind] | None = None,
    ) -> list[tuple[np.ndarray, RenderStyle]]:
        """Render several strings, optionally with a per-string *kind*."""
        results: list[tuple[np.ndarray, RenderStyle]] = []
        for index, text in enumerate(texts):
            kind: TextKind = kinds[index] if kinds is not None else "printed"
            results.append(self.render(text, kind=kind))
        return results
