"""Tests for CTC decoding.

The collapse rule is the part people get wrong: a blank between two identical
characters is what distinguishes "aa" from "a", and losing that silently
halves every double letter in the output.
"""

from __future__ import annotations

import numpy as np
import pytest

from ml.decoding.ctc import beam_search_decode, collapse_ctc, decode, greedy_decode
from ml.text.charset import Charset


def _log_probs_for(charset: Charset, text: str, *, frames_per_char: int = 3, total: int | None = None):
    """Build a confident log-prob tensor that spells *text*.

    Each character occupies one frame followed by blanks, which is the
    alignment a well-trained model produces. The frame budget is derived from
    the text so a longer string cannot silently overrun it.
    """
    indices = charset.encode(text)
    required = len(indices) * frames_per_char
    total = max(required + 4, total or 0)

    array = np.full((1, total, charset.num_classes), -25.0, dtype=np.float32)

    frame = 0
    for index in indices:
        array[0, frame, index] = 0.0
        for offset in range(1, frames_per_char):
            array[0, frame + offset, 0] = 0.0
        frame += frames_per_char

    array[0, frame:, 0] = 0.0
    # Normalise so the values really are log-probabilities.
    return array - np.log(np.exp(array).sum(axis=-1, keepdims=True))


def test_collapse_keeps_a_double_letter_separated_by_a_blank() -> None:
    assert collapse_ctc([1, 1, 0, 1]) == [1, 1]


def test_collapse_merges_a_repeated_frame() -> None:
    assert collapse_ctc([1, 1, 1]) == [1]


def test_collapse_removes_blanks() -> None:
    assert collapse_ctc([0, 1, 0, 2, 0]) == [1, 2]


def test_collapse_of_all_blanks_is_empty() -> None:
    assert collapse_ctc([0, 0, 0]) == []


def test_greedy_recovers_a_planted_string(charset: Charset) -> None:
    log_probs = _log_probs_for(charset, "hello")
    [result] = greedy_decode(log_probs, charset)
    assert result.text == "hello"
    assert result.confidence > 0.99
    assert len(result.timesteps) == 5


def test_greedy_recovers_a_double_letter(charset: Charset) -> None:
    # "aa" needs a blank between the two frames, which _log_probs_for provides.
    log_probs = _log_probs_for(charset, "aa")
    [result] = greedy_decode(log_probs, charset)
    assert result.text == "aa"


def test_beam_matches_greedy_on_confident_input(charset: Charset) -> None:
    log_probs = _log_probs_for(charset, "Recognition 42")
    [greedy] = greedy_decode(log_probs, charset)
    [beam] = beam_search_decode(log_probs, charset, beam_width=8)
    assert greedy.text == beam.text == "Recognition 42"


def test_all_blank_input_decodes_to_empty_with_zero_confidence(charset: Charset) -> None:
    array = np.full((1, 20, charset.num_classes), -25.0, dtype=np.float32)
    array[0, :, 0] = 0.0
    array = array - np.log(np.exp(array).sum(axis=-1, keepdims=True))

    [greedy] = greedy_decode(array, charset)
    [beam] = beam_search_decode(array, charset, beam_width=5)

    assert greedy.text == "" and greedy.confidence == 0.0
    # Both decoders must agree that "nothing" is not a confident answer.
    assert beam.text == "" and beam.confidence == 0.0


def test_input_lengths_truncate_the_decode(charset: Charset) -> None:
    log_probs = _log_probs_for(charset, "abcdef")
    [full] = greedy_decode(log_probs, charset)
    [clipped] = greedy_decode(log_probs, charset, input_lengths=[7])

    assert full.text == "abcdef"
    # Seven frames at three frames per character covers only the first three.
    assert clipped.text == "abc"
    assert clipped.n_timesteps == 7


def test_beam_width_one_falls_back_to_greedy(charset: Charset) -> None:
    log_probs = _log_probs_for(charset, "fallback")
    [result] = beam_search_decode(log_probs, charset, beam_width=1)
    # Greedy reports an alignment; beam search does not. Getting one back
    # proves the fallback happened.
    assert result.text == "fallback"
    assert result.timesteps


def test_batch_decoding_preserves_order(charset: Charset) -> None:
    first = _log_probs_for(charset, "one")
    second = _log_probs_for(charset, "two")
    batch = np.concatenate([first, second], axis=0)

    results = greedy_decode(batch, charset)
    assert [r.text for r in results] == ["one", "two"]


def test_two_dimensional_input_is_accepted(charset: Charset) -> None:
    log_probs = _log_probs_for(charset, "single")[0]
    results = greedy_decode(log_probs, charset)
    assert len(results) == 1 and results[0].text == "single"


def test_unknown_strategy_is_rejected(charset: Charset) -> None:
    log_probs = _log_probs_for(charset, "x")
    with pytest.raises(ValueError, match="Unknown decode strategy"):
        decode(log_probs, charset, strategy="telepathy")


def test_beam_width_must_be_positive(charset: Charset) -> None:
    log_probs = _log_probs_for(charset, "x")
    with pytest.raises(ValueError, match="beam_width"):
        beam_search_decode(log_probs, charset, beam_width=0)
