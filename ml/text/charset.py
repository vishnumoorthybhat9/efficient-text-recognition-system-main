"""Character-set handling for the CTC head.

Index 0 is permanently reserved for the CTC blank, so a label index ``i``
corresponds to ``characters[i - 1]``.  A :class:`Charset` is serialised into
every checkpoint: loading a model with a mismatched charset silently produces
garbage, so the mapping travels with the weights.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

__all__ = ["Charset", "CharsetError", "DEFAULT_CHARACTERS"]

logger = logging.getLogger(__name__)

BLANK_INDEX = 0
BLANK_TOKEN = "<blank>"

#: Digits + lowercase + uppercase + common punctuation + space.
DEFAULT_CHARACTERS = (
    "0123456789"
    "abcdefghijklmnopqrstuvwxyz"
    "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    "!\"#&'()*+,-./:;?@[]_ "
)


class CharsetError(Exception):
    """Raised for a malformed character set or an unencodable label."""


@dataclass(frozen=True)
class Charset:
    """Immutable bidirectional map between characters and CTC class indices.

    Attributes:
        characters: the ordered alphabet, excluding the blank.
        lowercase: when true, :meth:`normalize` lowercases text first, which
            halves the number of classes for a case-insensitive task.

    Note:
        Deliberately *not* ``slots=True``: ``__post_init__`` installs the two
        lookup dictionaries via ``object.__setattr__``, which a slotted class
        would reject.
    """

    characters: str
    lowercase: bool = False

    # -- construction -------------------------------------------------------
    def __post_init__(self) -> None:
        if not self.characters:
            raise CharsetError("Character set must not be empty")

        chars = self.characters.lower() if self.lowercase else self.characters
        if self.lowercase:
            # Deduplicate while preserving order, so "aA" collapses to "a".
            seen: dict[str, None] = {}
            for char in chars:
                seen.setdefault(char, None)
            chars = "".join(seen)
            object.__setattr__(self, "characters", chars)

        duplicates = sorted({c for c in self.characters if self.characters.count(c) > 1})
        if duplicates:
            raise CharsetError(f"Character set contains duplicates: {duplicates!r}")

        # frozen dataclass: build the lookup tables through object.__setattr__
        object.__setattr__(self, "_char_to_index", {c: i + 1 for i, c in enumerate(self.characters)})
        object.__setattr__(self, "_index_to_char", {i + 1: c for i, c in enumerate(self.characters)})

    @classmethod
    def default(cls, *, lowercase: bool = False) -> "Charset":
        """The project's default alphabet."""
        return cls(DEFAULT_CHARACTERS, lowercase=lowercase)

    @classmethod
    def from_config(cls, cfg: object) -> "Charset":
        """Build from the ``charset`` block of a loaded :class:`ml.config.Config`."""
        characters = getattr(cfg, "characters", None)
        if characters is None and isinstance(cfg, dict):
            characters = cfg.get("characters")
        lowercase = getattr(cfg, "lowercase", None)
        if lowercase is None and isinstance(cfg, dict):
            lowercase = cfg.get("lowercase", False)
        if not characters:
            raise CharsetError("charset config block is missing a non-empty `characters` entry")
        return cls(str(characters), lowercase=bool(lowercase))

    @classmethod
    def from_dict(cls, payload: dict[str, object]) -> "Charset":
        """Inverse of :meth:`to_dict` (used when loading a checkpoint)."""
        try:
            return cls(str(payload["characters"]), lowercase=bool(payload.get("lowercase", False)))
        except KeyError as exc:
            raise CharsetError(f"Serialised charset is missing key: {exc}") from exc

    @classmethod
    def from_corpus(cls, texts: Iterable[str], *, lowercase: bool = False) -> "Charset":
        """Derive an alphabet from observed labels, sorted for determinism."""
        observed: set[str] = set()
        for text in texts:
            observed.update(text.lower() if lowercase else text)
        if not observed:
            raise CharsetError("Cannot build a charset from an empty corpus")
        return cls("".join(sorted(observed)), lowercase=lowercase)

    @classmethod
    def load(cls, path: str | Path) -> "Charset":
        """Read a charset from a JSON file."""
        file = Path(path)
        if not file.is_file():
            raise CharsetError(f"Charset file not found: {file}")
        with file.open("r", encoding="utf-8") as handle:
            return cls.from_dict(json.load(handle))

    # -- serialisation ------------------------------------------------------
    def to_dict(self) -> dict[str, object]:
        return {
            "characters": self.characters,
            "lowercase": self.lowercase,
            "num_classes": self.num_classes,
        }

    def save(self, path: str | Path) -> Path:
        """Write this charset to *path* as JSON and return the path."""
        file = Path(path)
        file.parent.mkdir(parents=True, exist_ok=True)
        with file.open("w", encoding="utf-8") as handle:
            json.dump(self.to_dict(), handle, ensure_ascii=False, indent=2)
        return file

    # -- properties ---------------------------------------------------------
    @property
    def num_classes(self) -> int:
        """Number of CTC output units, i.e. alphabet size **plus the blank**."""
        return len(self.characters) + 1

    @property
    def blank_index(self) -> int:
        return BLANK_INDEX

    def __len__(self) -> int:
        return len(self.characters)

    def __contains__(self, char: object) -> bool:
        return isinstance(char, str) and char in self._char_to_index  # type: ignore[attr-defined]

    # -- text handling ------------------------------------------------------
    def normalize(self, text: str) -> str:
        """Apply the case policy and collapse runs of whitespace."""
        cleaned = " ".join(str(text).split())
        return cleaned.lower() if self.lowercase else cleaned

    def unsupported(self, text: str) -> list[str]:
        """Return the distinct characters of *text* that cannot be encoded."""
        table = self._char_to_index  # type: ignore[attr-defined]
        return sorted({c for c in self.normalize(text) if c not in table})

    def is_encodable(self, text: str) -> bool:
        return not self.unsupported(text)

    def encode(self, text: str, *, strict: bool = False) -> list[int]:
        """Encode *text* into CTC target indices (1-based, blank excluded).

        Args:
            text: the transcription.
            strict: raise on an unsupported character instead of dropping it.

        Raises:
            CharsetError: *strict* is set and *text* contains an unsupported
                character.
        """
        normalized = self.normalize(text)
        table = self._char_to_index  # type: ignore[attr-defined]

        indices: list[int] = []
        dropped: list[str] = []
        for char in normalized:
            index = table.get(char)
            if index is None:
                dropped.append(char)
                continue
            indices.append(index)

        if dropped:
            if strict:
                raise CharsetError(
                    f"Cannot encode {sorted(set(dropped))!r} with this charset "
                    f"({self.num_classes - 1} characters)"
                )
            logger.debug("Dropped unsupported characters %r from label %r", sorted(set(dropped)), text)

        return indices

    def encode_batch(self, texts: Sequence[str], *, strict: bool = False) -> list[list[int]]:
        return [self.encode(text, strict=strict) for text in texts]

    def decode(self, indices: Iterable[int]) -> str:
        """Map raw class indices back to text, skipping blanks.

        This does *not* collapse repeats -- that is CTC's job, see
        :mod:`ml.decoding.ctc`.
        """
        table = self._index_to_char  # type: ignore[attr-defined]
        return "".join(table[int(i)] for i in indices if int(i) != BLANK_INDEX and int(i) in table)

    def token(self, index: int) -> str:
        """Human-readable name for a class index (for attention/plot labels)."""
        if index == BLANK_INDEX:
            return BLANK_TOKEN
        table = self._index_to_char  # type: ignore[attr-defined]
        return table.get(int(index), "<unk>")

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return f"Charset(num_classes={self.num_classes}, lowercase={self.lowercase})"
