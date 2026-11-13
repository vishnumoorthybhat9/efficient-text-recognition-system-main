"""Text post-processing for recognised output.

Only *systematic, reversible* corrections belong here -- transformations that
undo a known convention of the model's training transcriptions. Anything that
guesses at content (spell-checking, dictionary snapping) is deliberately
excluded: it would make the output look better while hiding real recognition
errors, which is the opposite of what an OCR evaluation needs.

The one convention corrected here is IAM-style tokenisation. The IAM
transcriptions that TrOCR's document models were fine-tuned on separate
punctuation with spaces, so the model emits::

    42 Industrial Estate , Mangalore 575001
    Invoice No . INV-44718

which is faithful to its training data but is not what the page says.
"""

from __future__ import annotations

import logging
import re

__all__ = ["detokenize", "normalize_recognised_text", "POSTPROCESS_MODES"]

logger = logging.getLogger(__name__)

POSTPROCESS_MODES = ("none", "detokenize")

#: Punctuation that attaches to the preceding word.
_ATTACH_LEFT = r",;:!?%"

# " word ," -> " word,"   (also handles . but not between digits, see below)
_SPACE_BEFORE_PUNCT = re.compile(rf"\s+([{_ATTACH_LEFT}])")
# A full stop only closes up when it is not a decimal point: "51 . 25" must
# become "51.25", and "Estate ." must become "Estate.", but "1 . 5" is
# ambiguous, so the digit case is handled separately and first.
_SPACED_DECIMAL = re.compile(r"(\d)\s+\.\s+(\d)")
_SPACE_BEFORE_PERIOD = re.compile(r"\s+\.")
#: Opening brackets attach to what follows.
_SPACE_AFTER_OPEN = re.compile(r"([(\[{$])\s+")
#: Closing brackets attach to what precedes.
_SPACE_BEFORE_CLOSE = re.compile(r"\s+([)\]}])")
#: "don ' t" -> "don't", "John ' s" -> "John's"
_SPACED_APOSTROPHE = re.compile(r"(\w)\s*'\s*(\w)")
#: "INV - 44718" -> "INV-44718" and "USB - C" -> "USB-C"
_SPACED_HYPHEN = re.compile(r"(\w)\s+-\s+(\w)")
#: "1 , 250" -> "1,250"
_SPACED_THOUSANDS = re.compile(r"(\d),\s+(\d)")
_MULTISPACE = re.compile(r"[ \t]{2,}")


def detokenize(text: str) -> str:
    """Undo IAM-style punctuation spacing.

    Applied line by line so a newline is never absorbed into a match.

    >>> detokenize("42 Industrial Estate , Mangalore")
    '42 Industrial Estate, Mangalore'
    >>> detokenize("Laptop stand 51 . 25")
    'Laptop stand 51.25'
    >>> detokenize("USB - C cable")
    'USB-C cable'
    """
    lines = []
    for line in str(text).split("\n"):
        out = _SPACED_DECIMAL.sub(r"\1.\2", line)
        out = _SPACE_BEFORE_PUNCT.sub(r"\1", out)
        out = _SPACE_BEFORE_PERIOD.sub(".", out)
        out = _SPACE_AFTER_OPEN.sub(r"\1", out)
        out = _SPACE_BEFORE_CLOSE.sub(r"\1", out)
        out = _SPACED_APOSTROPHE.sub(r"\1'\2", out)
        out = _SPACED_HYPHEN.sub(r"\1-\2", out)
        out = _SPACED_THOUSANDS.sub(r"\1,\2", out)
        out = _MULTISPACE.sub(" ", out)
        lines.append(out.strip())
    return "\n".join(lines)


def normalize_recognised_text(text: str, *, mode: str = "none") -> str:
    """Apply the named post-processing mode.

    Args:
        text: recognised text.
        mode: ``"none"`` leaves it exactly as the model produced it;
            ``"detokenize"`` undoes IAM-style punctuation spacing.

    Raises:
        ValueError: *mode* is not recognised.
    """
    if mode == "none":
        return text
    if mode == "detokenize":
        return detokenize(text)
    raise ValueError(f"Unknown post-processing mode {mode!r}; expected one of {POSTPROCESS_MODES}")
