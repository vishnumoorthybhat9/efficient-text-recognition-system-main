"""Text source for the synthetic dataset generator.

The synthetic corpus has to exercise the whole charset, not just common
letters, so the sampler mixes four kinds of string:

* **words** -- ordinary English vocabulary (the bulk of the data);
* **phrases** -- two to four words, which is what teaches the model to handle
  inter-word spacing;
* **alphanumeric codes** -- plate numbers, invoice ids, part numbers.  Scene
  text is full of these and they are the hardest case, because there is no
  language model to fall back on;
* **numeric / punctuated** -- dates, amounts, percentages, times.

A custom corpus can be supplied instead, see :func:`load_lexicon`.
"""

from __future__ import annotations

import logging
import string
from pathlib import Path
from typing import Sequence

import numpy as np

__all__ = ["WORDS", "TextSampler", "load_lexicon"]

logger = logging.getLogger(__name__)

#: Common English vocabulary, chosen to cover every letter and a range of
#: word lengths. Kept inline so the generator works with no downloads at all.
WORDS: tuple[str, ...] = (
    # --- high frequency function words ---
    "the", "and", "for", "are", "but", "not", "you", "all", "any", "can",
    "her", "was", "one", "our", "out", "day", "get", "has", "him", "his",
    "how", "its", "new", "now", "old", "see", "two", "way", "who", "boy",
    "did", "may", "she", "use", "that", "with", "have", "this", "will",
    "from", "they", "know", "want", "been", "good", "much", "some", "time",
    # --- nouns ---
    "report", "invoice", "receipt", "document", "address", "signature",
    "university", "college", "student", "project", "synopsis", "chapter",
    "figure", "table", "result", "method", "dataset", "sample", "image",
    "text", "word", "letter", "number", "digit", "page", "line", "column",
    "model", "network", "layer", "encoder", "decoder", "feature", "vector",
    "machine", "learning", "training", "accuracy", "error", "metric",
    "camera", "scanner", "printer", "monitor", "keyboard", "laptop",
    "library", "archive", "manuscript", "notebook", "journal", "article",
    "payment", "account", "balance", "amount", "total", "subtotal",
    "customer", "supplier", "product", "quantity", "price", "discount",
    "delivery", "shipment", "warehouse", "inventory", "catalogue",
    "department", "engineering", "science", "information", "technology",
    "computer", "software", "hardware", "system", "platform", "server",
    "doctor", "patient", "hospital", "clinic", "medicine", "treatment",
    "teacher", "lecture", "seminar", "workshop", "assignment", "exam",
    "railway", "airport", "station", "terminal", "platform", "ticket",
    "passport", "visa", "license", "permit", "certificate", "diploma",
    "january", "february", "march", "april", "may", "june", "july",
    "august", "september", "october", "november", "december",
    "monday", "tuesday", "wednesday", "thursday", "friday", "saturday",
    "mumbai", "delhi", "bangalore", "chennai", "kolkata", "mangalore",
    "london", "paris", "berlin", "tokyo", "sydney", "toronto", "boston",
    "quartz", "jigsaw", "zephyr", "vortex", "oxygen", "jacket", "xerox",
    "quiver", "banjo", "waltz", "sphinx", "jukebox", "zigzag", "quick",
    # --- verbs ---
    "recognise", "recognize", "extract", "detect", "predict", "classify",
    "evaluate", "compare", "improve", "optimise", "reduce", "increase",
    "process", "analyse", "convert", "transform", "generate", "validate",
    "upload", "download", "submit", "approve", "reject", "cancel",
    "deliver", "receive", "confirm", "register", "verify", "update",
    # --- adjectives / adverbs ---
    "efficient", "accurate", "robust", "modern", "digital", "optical",
    "automatic", "manual", "printed", "handwritten", "supervised",
    "unlabeled", "pretrained", "complex", "simple", "clear", "noisy",
    "quickly", "slowly", "carefully", "exactly", "nearly", "finally",
    "annual", "monthly", "weekly", "daily", "urgent", "pending",
    # --- proper nouns / names ---
    "Pranava", "Tejaswi", "Simran", "Vishnumoorthy", "Rashmi", "Kumara",
    "Shankar", "Bhat", "Sayed", "Naveen", "Raghavendra", "Kasim",
    "Nitte", "Karnataka", "India",
)

#: Short connective tokens used when stitching phrases together.
_CONNECTORS: tuple[str, ...] = ("of", "in", "on", "to", "by", "for", "and", "the", "with", "at")

#: Punctuation that genuinely appears mid-text in documents.
_INLINE_PUNCTUATION: tuple[str, ...] = (",", ".", ":", ";", "-", "'", "?", "!")

_MONTH_ABBR: tuple[str, ...] = (
    "Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec",
)

_CODE_PREFIXES: tuple[str, ...] = (
    "INV", "REF", "PO", "ID", "NNM", "KA", "MH", "DL", "SKU", "ORD", "TXN", "ACC",
)


def load_lexicon(path: str | Path | None = None) -> tuple[str, ...]:
    """Return the word list, optionally read from a file.

    Args:
        path: a UTF-8 text file with one word per line.  Blank lines and lines
            starting with ``#`` are ignored.  ``None`` returns :data:`WORDS`.

    Raises:
        FileNotFoundError: *path* was given but does not exist.
        ValueError: the file contains no usable words.
    """
    if path is None:
        return WORDS

    file = Path(path)
    if not file.is_file():
        raise FileNotFoundError(f"Lexicon file not found: {file}")

    words = tuple(
        line.strip()
        for line in file.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    )
    if not words:
        raise ValueError(f"Lexicon file {file} contained no usable words")

    logger.info("Loaded %d words from %s", len(words), file)
    return words


class TextSampler:
    """Draws random strings for the synthetic renderer.

    Args:
        words: vocabulary to draw from.
        charset_characters: when given, every emitted string is filtered to
            these characters, so the generator can never produce a label the
            model's CTC head is unable to represent.
        max_length: hard cap on string length.  Must stay below the number of
            CTC time steps, otherwise the label is undecodable and the loss
            becomes infinite -- see ``image.width`` in the config.
        seed: RNG seed.

    >>> sampler = TextSampler(seed=0)
    >>> isinstance(sampler.sample(), str)
    True
    """

    #: Relative weights of the string kinds.
    #:
    #: `sentence` was added after the document benchmark showed the model
    #: failing specifically on long lines: it had only ever trained on strings
    #: up to ~37 characters, so a 50-60 character document line came out
    #: garbled even though short lines were near-perfect.
    KIND_WEIGHTS: dict[str, float] = {
        "word": 0.22,
        "phrase": 0.22,
        "sentence": 0.34,
        "code": 0.10,
        "numeric": 0.12,
    }

    def __init__(
        self,
        words: Sequence[str] | None = None,
        *,
        charset_characters: str | None = None,
        max_length: int = 28,
        min_length: int = 1,
        seed: int | None = None,
    ) -> None:
        self.words = tuple(words) if words else WORDS
        if not self.words:
            raise ValueError("TextSampler needs a non-empty word list")

        self.allowed: set[str] | None = set(charset_characters) if charset_characters else None
        self.max_length = int(max_length)
        self.min_length = max(1, int(min_length))
        self._rng = np.random.default_rng(seed)

        kinds = list(self.KIND_WEIGHTS)
        weights = np.array([self.KIND_WEIGHTS[k] for k in kinds], dtype=np.float64)
        self._kinds = kinds
        self._weights = weights / weights.sum()

        if self.allowed is not None:
            usable = [w for w in self.words if set(w) <= self.allowed]
            dropped = len(self.words) - len(usable)
            if dropped:
                logger.debug("Dropped %d lexicon words not covered by the charset", dropped)
            if not usable:
                raise ValueError(
                    "No lexicon word is representable in the configured charset; "
                    "check charset.characters."
                )
            self.words = tuple(usable)

    # -- individual kinds ---------------------------------------------------
    def _a_word(self) -> str:
        word = str(self._rng.choice(self.words))
        roll = self._rng.random()
        if roll < 0.12:
            return word.upper()
        if roll < 0.40:
            return word.capitalize()
        return word

    def _a_phrase(self) -> str:
        count = int(self._rng.integers(2, 5))
        parts: list[str] = []
        for position in range(count):
            # A connector in the middle makes the phrase read like real text.
            if position > 0 and self._rng.random() < 0.25:
                parts.append(str(self._rng.choice(_CONNECTORS)))
            else:
                parts.append(self._a_word())

        phrase = " ".join(parts)
        if self._rng.random() < 0.22:
            phrase += str(self._rng.choice(_INLINE_PUNCTUATION))
        return phrase

    def _a_sentence(self) -> str:
        """A long, document-like line.

        Real documents are mostly sentences, often 50-70 characters wide and
        carrying currency, percentages and punctuation. Training only on short
        words and phrases left the model unable to read them.
        """
        rng = self._rng
        target = int(rng.integers(max(18, self.min_length), max(20, self.max_length)))

        parts: list[str] = []
        length = 0
        while length < target:
            roll = rng.random()
            if roll < 0.10:
                token = self._a_number()
            elif roll < 0.16:
                token = self._a_code()
            elif roll < 0.34:
                token = str(rng.choice(_CONNECTORS))
            else:
                token = self._a_word()

            # Occasional inline punctuation, as in prose.
            if parts and rng.random() < 0.10:
                parts[-1] = parts[-1] + str(rng.choice((",", ";", ":")))

            parts.append(token)
            length += len(token) + 1

        text = " ".join(parts)
        if rng.random() < 0.35:
            text += str(rng.choice((".", ".", ".", "!", "?")))
        return text

    def _a_code(self) -> str:
        prefix = str(self._rng.choice(_CODE_PREFIXES))
        digits = "".join(str(d) for d in self._rng.integers(0, 10, size=int(self._rng.integers(2, 7))))
        style = self._rng.random()
        if style < 0.3:
            return f"{prefix}-{digits}"
        if style < 0.5:
            letters = "".join(
                str(c) for c in self._rng.choice(list(string.ascii_uppercase), size=2)
            )
            return f"{prefix}{digits}{letters}"
        if style < 0.7:
            return f"{prefix}/{digits}"
        return f"{prefix}{digits}"

    def _a_number(self) -> str:
        style = self._rng.random()
        if style < 0.22:  # date, numeric
            day, month, year = self._rng.integers(1, 29), self._rng.integers(1, 13), self._rng.integers(1990, 2031)
            separator = str(self._rng.choice(["/", "-", "."]))
            return f"{day:02d}{separator}{month:02d}{separator}{year}"
        if style < 0.40:  # date, written month
            day = self._rng.integers(1, 29)
            month = str(self._rng.choice(_MONTH_ABBR))
            year = self._rng.integers(1990, 2031)
            return f"{day:02d} {month} {year}"
        if style < 0.58:  # amount, sometimes with a currency symbol
            whole = self._rng.integers(1, 100000)
            cents = self._rng.integers(0, 100)
            amount = f"{whole:,}.{cents:02d}"
            # Documents are full of these, and the old charset could not even
            # represent '$'.
            marker = self._rng.random()
            if marker < 0.22:
                return f"${amount}"
            if marker < 0.30:
                return f"Rs.{amount}"
            return amount
        if style < 0.72:  # percentage
            return f"{self._rng.integers(0, 101)}.{self._rng.integers(0, 10)}%"
        if style < 0.86:  # time
            return f"{self._rng.integers(0, 24):02d}:{self._rng.integers(0, 60):02d}"
        plain_digits = int(self._rng.integers(3, 11))
        return "".join(str(d) for d in self._rng.integers(0, 10, size=plain_digits))

    # -- public API ---------------------------------------------------------
    def _filter(self, text: str) -> str:
        """Drop characters outside the charset and trim to *max_length*.

        Trimming happens at a word boundary when possible, so a truncated
        phrase stays a plausible piece of text rather than a cut-off word.
        """
        if self.allowed is not None:
            text = "".join(c for c in text if c in self.allowed)
        text = " ".join(text.split())

        if len(text) > self.max_length:
            clipped = text[: self.max_length]
            boundary = clipped.rfind(" ")
            text = clipped[:boundary] if boundary >= self.min_length else clipped
        return text.strip()

    def sample(self) -> str:
        """Return one random string, guaranteed non-empty and within limits."""
        for _ in range(12):
            kind = str(self._rng.choice(self._kinds, p=self._weights))
            if kind == "word":
                raw = self._a_word()
            elif kind == "phrase":
                raw = self._a_phrase()
            elif kind == "sentence":
                raw = self._a_sentence()
            elif kind == "code":
                raw = self._a_code()
            else:
                raw = self._a_number()

            text = self._filter(raw)
            if self.min_length <= len(text) <= self.max_length:
                return text

        # Fallback: a short word is always representable.
        return self._filter(self._a_word()) or "text"

    def sample_many(self, count: int, *, unique: bool = False) -> list[str]:
        """Return *count* strings.

        Args:
            unique: try to avoid duplicates.  Deduplication gives up after a
                bounded number of attempts rather than looping forever on a
                small lexicon, so the result may still contain repeats.
        """
        if count <= 0:
            return []

        if not unique:
            return [self.sample() for _ in range(count)]

        seen: dict[str, None] = {}
        attempts = 0
        limit = count * 25
        while len(seen) < count and attempts < limit:
            seen.setdefault(self.sample(), None)
            attempts += 1

        results = list(seen)
        if len(results) < count:
            logger.debug(
                "Could only produce %d unique strings out of %d requested; padding with repeats",
                len(results),
                count,
            )
            results.extend(self.sample() for _ in range(count - len(results)))
        return results[:count]
