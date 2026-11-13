"""CTC decoding: greedy (best-path) and prefix beam search.

Both decoders consume the ``log_probs`` tensor produced by
:class:`ml.models.crnn_attention.CRNNAttentionOCR` -- shape ``(B, T, C)`` with
class 0 as the blank -- and return :class:`DecodedText`.

Greedy decoding takes the arg-max per frame then collapses repeats and blanks.
It is what you want at interactive latency.  Prefix beam search sums the
probability of *all* alignments that yield the same string, which is the
mathematically correct objective and typically recovers a little accuracy on
blurred or ambiguous input, at a cost of roughly ``beam_width`` times the
decode time.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from typing import Iterable, Sequence

import numpy as np
import torch

from ml.text.charset import BLANK_INDEX, Charset

__all__ = [
    "DecodedText",
    "greedy_decode",
    "beam_search_decode",
    "decode",
    "collapse_ctc",
]

logger = logging.getLogger(__name__)

NEG_INF = -float("inf")


@dataclass(slots=True)
class DecodedText:
    """One decoded transcription.

    Attributes:
        text: the recognised string.
        confidence: aggregate confidence in ``[0, 1]``.  For greedy decoding
            this is the geometric mean of the per-frame arg-max probabilities
            of the emitting frames; for beam search it is the exponentiated
            normalised score of the winning prefix.
        char_confidences: per-character confidence, aligned with *text*.
        timesteps: for each character, the frame index that emitted it.  Only
            populated by greedy decoding, where the alignment is unambiguous.
        n_timesteps: number of frames the decoder consumed.
    """

    text: str
    confidence: float = 0.0
    char_confidences: list[float] = field(default_factory=list)
    timesteps: list[int] = field(default_factory=list)
    n_timesteps: int = 0

    def to_dict(self) -> dict[str, object]:
        return {
            "text": self.text,
            "confidence": round(self.confidence, 6),
            "char_confidences": [round(c, 6) for c in self.char_confidences],
            "timesteps": self.timesteps,
            "n_timesteps": self.n_timesteps,
        }


def collapse_ctc(indices: Sequence[int], *, blank: int = BLANK_INDEX) -> list[int]:
    """Apply the CTC collapse rule to a frame-level index sequence.

    Consecutive duplicates are merged, then blanks are removed.  The order
    matters: ``[a, a, blank, a]`` must decode to ``"aa"``, not ``"a"`` -- the
    blank is what separates a genuine double letter.

    >>> collapse_ctc([1, 1, 0, 1])
    [1, 1]
    >>> collapse_ctc([1, 1, 1])
    [1]
    """
    collapsed: list[int] = []
    previous: int | None = None
    for raw in indices:
        index = int(raw)
        if index != previous:
            collapsed.append(index)
            previous = index
    return [i for i in collapsed if i != blank]


def _as_numpy_log_probs(log_probs: torch.Tensor | np.ndarray) -> np.ndarray:
    """Normalise the input to a 3-D float32 numpy array ``(B, T, C)``."""
    if isinstance(log_probs, torch.Tensor):
        array = log_probs.detach().to(dtype=torch.float32, device="cpu").numpy()
    else:
        array = np.asarray(log_probs, dtype=np.float32)

    if array.ndim == 2:
        array = array[np.newaxis, ...]
    if array.ndim != 3:
        raise ValueError(f"log_probs must be (B, T, C) or (T, C), got shape {array.shape}")
    return array


def _resolve_lengths(
    input_lengths: torch.Tensor | np.ndarray | Sequence[int] | None,
    batch: int,
    seq_len: int,
) -> np.ndarray:
    if input_lengths is None:
        return np.full(batch, seq_len, dtype=np.int64)

    if isinstance(input_lengths, torch.Tensor):
        lengths = input_lengths.detach().cpu().numpy()
    else:
        lengths = np.asarray(input_lengths)

    lengths = lengths.reshape(-1).astype(np.int64)
    if lengths.size != batch:
        raise ValueError(f"input_lengths has {lengths.size} entries but the batch size is {batch}")
    return np.clip(lengths, 1, seq_len)


def greedy_decode(
    log_probs: torch.Tensor | np.ndarray,
    charset: Charset,
    *,
    input_lengths: torch.Tensor | np.ndarray | Sequence[int] | None = None,
    blank: int = BLANK_INDEX,
) -> list[DecodedText]:
    """Best-path decoding.

    Args:
        log_probs: ``(B, T, C)`` log-probabilities (or ``(T, C)`` for one item).
        charset: maps class indices back to characters.
        input_lengths: valid frame count per sample; defaults to all frames.
        blank: the CTC blank index.

    Returns:
        One :class:`DecodedText` per batch item.
    """
    array = _as_numpy_log_probs(log_probs)
    batch, seq_len, num_classes = array.shape

    if num_classes != charset.num_classes:
        logger.warning(
            "Decoding %d classes with a charset of %d; the mapping is probably wrong.",
            num_classes,
            charset.num_classes,
        )

    lengths = _resolve_lengths(input_lengths, batch, seq_len)
    results: list[DecodedText] = []

    for index in range(batch):
        valid = int(lengths[index])
        frame_log_probs = array[index, :valid, :]

        best_indices = frame_log_probs.argmax(axis=1)
        best_log_probs = frame_log_probs[np.arange(valid), best_indices]

        characters: list[str] = []
        char_confidences: list[float] = []
        timesteps: list[int] = []

        previous: int | None = None
        for frame, class_index in enumerate(best_indices):
            class_index = int(class_index)
            is_repeat = class_index == previous
            previous = class_index

            if class_index == blank or is_repeat:
                continue

            character = charset.token(class_index)
            if len(character) != 1:
                # Not a real glyph (e.g. "<unk>"); skip rather than emit markup.
                continue

            characters.append(character)
            char_confidences.append(float(np.exp(best_log_probs[frame])))
            timesteps.append(frame)

        # Geometric mean in log space: avoids underflow and is the natural
        # aggregate for a product of independent per-frame probabilities.
        confidence = (
            float(np.exp(np.mean(np.log(np.clip(char_confidences, 1e-12, 1.0)))))
            if char_confidences
            else 0.0
        )

        results.append(
            DecodedText(
                text="".join(characters),
                confidence=confidence,
                char_confidences=char_confidences,
                timesteps=timesteps,
                n_timesteps=valid,
            )
        )

    return results


def _log_add(a: float, b: float) -> float:
    """Numerically stable ``log(exp(a) + exp(b))``."""
    if a == NEG_INF:
        return b
    if b == NEG_INF:
        return a
    if a < b:
        a, b = b, a
    return a + math.log1p(math.exp(b - a))


def _beam_search_single(
    frame_log_probs: np.ndarray,
    *,
    beam_width: int,
    blank: int,
    prune_threshold: float,
    max_candidates: int,
) -> tuple[tuple[int, ...], float]:
    """Prefix beam search over one sample's ``(T, C)`` log-probabilities.

    Each beam tracks two probabilities, following Graves' formulation:
    ``p_blank`` (the prefix's last alignment step was a blank) and
    ``p_nonblank`` (it was the prefix's final character).  Keeping them apart
    is what lets the decoder tell a repeated character from a prolonged one.

    Returns:
        ``(best_prefix, total_log_probability)``.
    """
    seq_len, num_classes = frame_log_probs.shape

    # prefix -> (log p_blank, log p_nonblank)
    beams: dict[tuple[int, ...], tuple[float, float]] = {(): (0.0, NEG_INF)}

    # Only the strongest classes per frame are worth expanding; the tail is
    # numerically irrelevant and dominates the runtime otherwise.
    candidate_count = min(num_classes, max(2, max_candidates))

    for frame in range(seq_len):
        step = frame_log_probs[frame]
        candidates = np.argpartition(-step, candidate_count - 1)[:candidate_count]
        candidates = candidates[step[candidates] > prune_threshold]
        if candidates.size == 0:
            candidates = np.array([int(step.argmax())])

        next_beams: dict[tuple[int, ...], tuple[float, float]] = {}

        def accumulate(prefix: tuple[int, ...], blank_lp: float, nonblank_lp: float) -> None:
            current = next_beams.get(prefix)
            if current is None:
                next_beams[prefix] = (blank_lp, nonblank_lp)
            else:
                next_beams[prefix] = (
                    _log_add(current[0], blank_lp),
                    _log_add(current[1], nonblank_lp),
                )

        for prefix, (p_blank, p_nonblank) in beams.items():
            prefix_total = _log_add(p_blank, p_nonblank)

            for raw_class in candidates:
                class_index = int(raw_class)
                class_lp = float(step[class_index])

                if class_index == blank:
                    # Emitting a blank keeps the string and moves it to the
                    # "ended on blank" state.
                    accumulate(prefix, prefix_total + class_lp, NEG_INF)
                    continue

                if prefix and class_index == prefix[-1]:
                    # Same character again. Two distinct outcomes:
                    #  - extend the string, but only from the blank state
                    #    (a blank separated the two occurrences);
                    #  - stay on the same string, prolonging the character.
                    accumulate(prefix + (class_index,), NEG_INF, p_blank + class_lp)
                    accumulate(prefix, NEG_INF, p_nonblank + class_lp)
                else:
                    accumulate(prefix + (class_index,), NEG_INF, prefix_total + class_lp)

        # Keep the best `beam_width` prefixes by total probability.
        beams = dict(
            sorted(
                next_beams.items(),
                key=lambda item: _log_add(item[1][0], item[1][1]),
                reverse=True,
            )[:beam_width]
        )

    best_prefix, (best_blank, best_nonblank) = max(
        beams.items(), key=lambda item: _log_add(item[1][0], item[1][1])
    )
    return best_prefix, _log_add(best_blank, best_nonblank)


def beam_search_decode(
    log_probs: torch.Tensor | np.ndarray,
    charset: Charset,
    *,
    input_lengths: torch.Tensor | np.ndarray | Sequence[int] | None = None,
    beam_width: int = 10,
    blank: int = BLANK_INDEX,
    prune_threshold: float = -12.0,
    max_candidates: int = 12,
) -> list[DecodedText]:
    """CTC prefix beam search.

    Args:
        log_probs: ``(B, T, C)`` log-probabilities.
        charset: index-to-character mapping.
        input_lengths: valid frame count per sample.
        beam_width: number of prefixes kept per frame.  Above ~25 the returns
            are negligible for this model while the cost keeps growing.
        blank: the CTC blank index.
        prune_threshold: per-frame log-probability floor; classes below it are
            not expanded.
        max_candidates: hard cap on classes expanded per frame.

    Returns:
        One :class:`DecodedText` per batch item.  ``timesteps`` is empty
        because a beam aggregates over many alignments, so no single frame
        "owns" a character.
    """
    if beam_width < 1:
        raise ValueError(f"beam_width must be >= 1, got {beam_width}")
    if beam_width == 1:
        # Width 1 is best-path decoding; the dedicated implementation is
        # faster and additionally reports the alignment.
        return greedy_decode(log_probs, charset, input_lengths=input_lengths, blank=blank)

    array = _as_numpy_log_probs(log_probs)
    batch, seq_len, _ = array.shape
    lengths = _resolve_lengths(input_lengths, batch, seq_len)

    results: list[DecodedText] = []
    for index in range(batch):
        valid = int(lengths[index])
        prefix, score = _beam_search_single(
            array[index, :valid, :],
            beam_width=int(beam_width),
            blank=blank,
            prune_threshold=float(prune_threshold),
            max_candidates=int(max_candidates),
        )

        text = charset.decode(prefix)
        # Normalise by length so confidence is comparable across strings of
        # different lengths, then map back out of log space.  An empty prefix
        # gets 0.0 rather than exp(score/1): "the model emitted nothing" is not
        # a confident answer, and greedy decoding reports it the same way.
        if not prefix or score == NEG_INF:
            confidence = 0.0
        else:
            confidence = float(min(1.0, math.exp(score / len(prefix))))

        results.append(
            DecodedText(
                text=text,
                confidence=confidence,
                char_confidences=[],
                timesteps=[],
                n_timesteps=valid,
            )
        )

    return results


def decode(
    log_probs: torch.Tensor | np.ndarray,
    charset: Charset,
    *,
    strategy: str = "greedy",
    input_lengths: torch.Tensor | np.ndarray | Sequence[int] | None = None,
    beam_width: int = 10,
    blank: int = BLANK_INDEX,
) -> list[DecodedText]:
    """Dispatch to :func:`greedy_decode` or :func:`beam_search_decode`.

    Raises:
        ValueError: *strategy* is not ``"greedy"`` or ``"beam"``.
    """
    choice = (strategy or "greedy").strip().lower()
    if choice == "greedy":
        return greedy_decode(log_probs, charset, input_lengths=input_lengths, blank=blank)
    if choice in ("beam", "beam_search"):
        return beam_search_decode(
            log_probs,
            charset,
            input_lengths=input_lengths,
            beam_width=beam_width,
            blank=blank,
        )
    raise ValueError(f"Unknown decode strategy {strategy!r}; expected 'greedy' or 'beam'")


def texts_from(results: Iterable[DecodedText]) -> list[str]:
    """Extract just the strings (handy in evaluation loops)."""
    return [result.text for result in results]
