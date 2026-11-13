"""Recognition metrics: Character Error Rate, Word Error Rate and accuracy.

Definitions used throughout this project
----------------------------------------
Let ``S``, ``D`` and ``I`` be the substitutions, deletions and insertions in
the optimal alignment between a prediction and its reference, and ``N`` the
length of the reference.

* **CER** ``= (S + D + I) / N`` over characters.
* **WER** ``= (S + D + I) / N`` over whitespace-delimited words.
* **Sequence accuracy** -- the fraction of samples recognised *exactly*.  This
  is the strictest of the three and the one most often quoted as "accuracy" in
  scene-text papers.
* **Character accuracy** ``= max(0, 1 - CER)``.

Both error rates can exceed 1.0, because insertions are not bounded by the
reference length.  They are reported unclamped; only *accuracy* is clamped,
and that is stated wherever it is shown.

Aggregation
-----------
Two aggregates are reported for every run, because they answer different
questions and disagreeing figures in the literature usually come down to this:

* **micro** -- ``sum(edits) / sum(reference_lengths)``.  Long lines carry more
  weight.  This is the standard figure and the one to quote.
* **macro** -- the unweighted mean of per-sample rates.  Every sample counts
  equally, so a handful of short, badly-recognised samples stay visible.
"""

from __future__ import annotations

import logging
import re
import unicodedata
from dataclasses import asdict, dataclass, field
from typing import Any, Sequence

__all__ = [
    "edit_distance",
    "character_error_rate",
    "word_error_rate",
    "SampleMetrics",
    "AggregateMetrics",
    "evaluate_predictions",
    "normalize_text",
]

logger = logging.getLogger(__name__)

# Prefer the C implementation; fall back to the local DP version so the
# project never hard-depends on an optional wheel.
try:  # pragma: no cover - depends on the environment
    from Levenshtein import distance as _fast_distance

    _HAS_FAST_LEVENSHTEIN = True
except ImportError:  # pragma: no cover
    _fast_distance = None  # type: ignore[assignment]
    _HAS_FAST_LEVENSHTEIN = False
    logger.debug("python-Levenshtein unavailable; using the pure-Python edit distance")

_NON_ALNUM = re.compile(r"[^0-9a-zA-Z]+")
_WHITESPACE = re.compile(r"\s+")


def _dp_edit_distance(reference: Sequence[Any], hypothesis: Sequence[Any]) -> int:
    """Levenshtein distance via dynamic programming, O(min(n, m)) memory."""
    if len(reference) < len(hypothesis):
        reference, hypothesis = hypothesis, reference

    if not hypothesis:
        return len(reference)

    previous = list(range(len(hypothesis) + 1))
    for i, ref_item in enumerate(reference, start=1):
        current = [i] + [0] * len(hypothesis)
        for j, hyp_item in enumerate(hypothesis, start=1):
            cost = 0 if ref_item == hyp_item else 1
            current[j] = min(
                previous[j] + 1,        # deletion
                current[j - 1] + 1,     # insertion
                previous[j - 1] + cost, # substitution / match
            )
        previous = current

    return previous[-1]


def edit_distance(reference: Sequence[Any], hypothesis: Sequence[Any]) -> int:
    """Levenshtein distance between two sequences.

    Works on strings (character level) and on lists of tokens (word level).

    >>> edit_distance("kitten", "sitting")
    3
    >>> edit_distance(["a", "b"], ["a", "c"])
    1
    """
    if _HAS_FAST_LEVENSHTEIN and isinstance(reference, str) and isinstance(hypothesis, str):
        return int(_fast_distance(reference, hypothesis))  # type: ignore[misc]
    return _dp_edit_distance(reference, hypothesis)


def normalize_text(
    text: str,
    *,
    lowercase: bool = False,
    collapse_whitespace: bool = True,
    alphanumeric_only: bool = False,
    unicode_form: str | None = "NFKC",
) -> str:
    """Canonicalise *text* before scoring.

    Args:
        lowercase: fold case.  Enables the case-insensitive figures that
            scene-text benchmarks such as IIIT5K conventionally report.
        collapse_whitespace: squeeze runs of whitespace and strip the ends.
        alphanumeric_only: drop punctuation and symbols entirely.
        unicode_form: Unicode normal form applied first, so that visually
            identical characters compare equal.
    """
    result = str(text)
    if unicode_form:
        result = unicodedata.normalize(unicode_form, result)
    if collapse_whitespace:
        result = _WHITESPACE.sub(" ", result).strip()
    if alphanumeric_only:
        result = _NON_ALNUM.sub("", result)
    if lowercase:
        result = result.lower()
    return result


def character_error_rate(reference: str, hypothesis: str) -> float:
    """CER for one pair.

    An empty reference is a degenerate case: 0.0 when the prediction is also
    empty, else 1.0 (every predicted character is a spurious insertion, and
    normalising by zero is undefined).
    """
    if not reference:
        return 0.0 if not hypothesis else 1.0
    return edit_distance(reference, hypothesis) / len(reference)


def word_error_rate(reference: str, hypothesis: str) -> float:
    """WER for one pair, splitting on whitespace."""
    ref_words = reference.split()
    hyp_words = hypothesis.split()
    if not ref_words:
        return 0.0 if not hyp_words else 1.0
    return edit_distance(ref_words, hyp_words) / len(ref_words)


@dataclass(slots=True)
class SampleMetrics:
    """Per-sample scores, kept so the UI can show the worst cases."""

    reference: str
    prediction: str
    cer: float
    wer: float
    exact_match: bool
    char_edits: int
    ref_chars: int
    word_edits: int
    ref_words: int
    sample_id: str | None = None

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["cer"] = round(self.cer, 6)
        payload["wer"] = round(self.wer, 6)
        return payload


@dataclass(slots=True)
class AggregateMetrics:
    """Corpus-level results.

    ``cer``/``wer`` are the micro aggregates -- the headline numbers.
    """

    num_samples: int
    cer: float
    wer: float
    sequence_accuracy: float
    character_accuracy: float
    cer_macro: float
    wer_macro: float
    total_char_edits: int
    total_ref_chars: int
    total_word_edits: int
    total_ref_words: int
    # Case- and punctuation-insensitive variants, reported alongside the strict
    # figures because benchmark papers differ on which they quote.
    cer_normalized: float = 0.0
    sequence_accuracy_normalized: float = 0.0
    extra: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "num_samples": self.num_samples,
            "cer": round(self.cer, 6),
            "wer": round(self.wer, 6),
            "sequence_accuracy": round(self.sequence_accuracy, 6),
            "character_accuracy": round(self.character_accuracy, 6),
            "cer_macro": round(self.cer_macro, 6),
            "wer_macro": round(self.wer_macro, 6),
            "cer_normalized": round(self.cer_normalized, 6),
            "sequence_accuracy_normalized": round(self.sequence_accuracy_normalized, 6),
            "total_char_edits": self.total_char_edits,
            "total_ref_chars": self.total_ref_chars,
            "total_word_edits": self.total_word_edits,
            "total_ref_words": self.total_ref_words,
            **self.extra,
        }

    def summary(self) -> str:
        return (
            f"n={self.num_samples}  CER={self.cer:.4f}  WER={self.wer:.4f}  "
            f"exact={self.sequence_accuracy:.4f}  char-acc={self.character_accuracy:.4f}"
        )


def evaluate_predictions(
    references: Sequence[str],
    predictions: Sequence[str],
    *,
    sample_ids: Sequence[str] | None = None,
    collapse_whitespace: bool = True,
    return_samples: bool = True,
) -> tuple[AggregateMetrics, list[SampleMetrics]]:
    """Score a batch of predictions against their references.

    Args:
        references: ground-truth transcriptions.
        predictions: model outputs, same length and order as *references*.
        sample_ids: optional identifiers carried into the per-sample results.
        collapse_whitespace: apply whitespace normalisation before scoring.
            Leading/trailing space is an artefact of line cropping, not a
            recognition error, so this is on by default.
        return_samples: set false to skip building the per-sample list on very
            large test sets.

    Returns:
        ``(aggregate, per_sample)``.

    Raises:
        ValueError: the two sequences have different lengths.
    """
    if len(references) != len(predictions):
        raise ValueError(
            f"Got {len(references)} references but {len(predictions)} predictions; "
            "they must correspond one-to-one."
        )

    if not references:
        empty = AggregateMetrics(0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0, 0, 0, 0)
        return empty, []

    samples: list[SampleMetrics] = []
    total_char_edits = total_ref_chars = 0
    total_word_edits = total_ref_words = 0
    exact_matches = 0
    exact_matches_normalized = 0
    cer_sum = wer_sum = 0.0
    normalized_char_edits = normalized_ref_chars = 0

    for index, (raw_ref, raw_pred) in enumerate(zip(references, predictions)):
        reference = normalize_text(raw_ref, collapse_whitespace=collapse_whitespace)
        prediction = normalize_text(raw_pred, collapse_whitespace=collapse_whitespace)

        char_edits = edit_distance(reference, prediction)
        ref_chars = len(reference)
        word_edits = edit_distance(reference.split(), prediction.split())
        ref_words = len(reference.split())

        sample_cer = character_error_rate(reference, prediction)
        sample_wer = word_error_rate(reference, prediction)
        is_exact = reference == prediction

        total_char_edits += char_edits
        total_ref_chars += ref_chars
        total_word_edits += word_edits
        total_ref_words += ref_words
        cer_sum += sample_cer
        wer_sum += sample_wer
        exact_matches += int(is_exact)

        # Case-insensitive, alphanumeric-only variant.
        soft_ref = normalize_text(raw_ref, lowercase=True, alphanumeric_only=True)
        soft_pred = normalize_text(raw_pred, lowercase=True, alphanumeric_only=True)
        normalized_char_edits += edit_distance(soft_ref, soft_pred)
        normalized_ref_chars += len(soft_ref)
        exact_matches_normalized += int(soft_ref == soft_pred)

        if return_samples:
            samples.append(
                SampleMetrics(
                    reference=reference,
                    prediction=prediction,
                    cer=sample_cer,
                    wer=sample_wer,
                    exact_match=is_exact,
                    char_edits=char_edits,
                    ref_chars=ref_chars,
                    word_edits=word_edits,
                    ref_words=ref_words,
                    sample_id=sample_ids[index] if sample_ids is not None else None,
                )
            )

    count = len(references)
    micro_cer = total_char_edits / total_ref_chars if total_ref_chars else 0.0
    micro_wer = total_word_edits / total_ref_words if total_ref_words else 0.0

    aggregate = AggregateMetrics(
        num_samples=count,
        cer=micro_cer,
        wer=micro_wer,
        sequence_accuracy=exact_matches / count,
        character_accuracy=max(0.0, 1.0 - micro_cer),
        cer_macro=cer_sum / count,
        wer_macro=wer_sum / count,
        total_char_edits=total_char_edits,
        total_ref_chars=total_ref_chars,
        total_word_edits=total_word_edits,
        total_ref_words=total_ref_words,
        cer_normalized=(
            normalized_char_edits / normalized_ref_chars if normalized_ref_chars else 0.0
        ),
        sequence_accuracy_normalized=exact_matches_normalized / count,
    )

    return aggregate, samples


def worst_samples(samples: Sequence[SampleMetrics], limit: int = 20) -> list[SampleMetrics]:
    """The *limit* samples with the highest CER (for error analysis in the UI)."""
    return sorted(samples, key=lambda s: (-s.cer, -s.char_edits))[:limit]
