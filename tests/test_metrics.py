"""Tests for CER, WER and accuracy.

Every number this project reports flows through these functions, so the
expected values below are worked out by hand rather than captured from a run.
"""

from __future__ import annotations

import pytest

from ml.evaluation.metrics import (
    character_error_rate,
    edit_distance,
    evaluate_predictions,
    normalize_text,
    word_error_rate,
    worst_samples,
)


@pytest.mark.parametrize(
    ("reference", "hypothesis", "expected"),
    [
        ("kitten", "sitting", 3),
        ("", "", 0),
        ("abc", "", 3),
        ("", "abc", 3),
        ("same", "same", 0),
        ("ab", "ba", 2),
    ],
)
def test_edit_distance(reference: str, hypothesis: str, expected: int) -> None:
    assert edit_distance(reference, hypothesis) == expected


def test_edit_distance_on_token_lists() -> None:
    assert edit_distance(["a", "b", "c"], ["a", "x", "c"]) == 1


def test_cer_counts_one_substitution() -> None:
    # "hello" -> "hallo": one substitution over five reference characters.
    assert character_error_rate("hello", "hallo") == pytest.approx(1 / 5)


def test_cer_can_exceed_one_through_insertions() -> None:
    # Insertions are not bounded by the reference length, so the rate can
    # exceed 1.0. This is reported unclamped and is not a bug.
    assert character_error_rate("a", "abcde") == pytest.approx(4.0)


def test_cer_with_an_empty_reference() -> None:
    assert character_error_rate("", "") == 0.0
    assert character_error_rate("", "text") == 1.0


def test_wer_operates_on_words() -> None:
    assert word_error_rate("the quick brown fox", "the quick brown dog") == pytest.approx(1 / 4)


def test_wer_with_an_empty_reference() -> None:
    assert word_error_rate("", "") == 0.0
    assert word_error_rate("", "word") == 1.0


def test_micro_aggregate_is_total_edits_over_total_length() -> None:
    references = ["hello world", "abc", "Test"]
    predictions = ["hello wordl", "abc", "test"]

    # Hand-computed: 2 + 0 + 1 = 3 edits over 11 + 3 + 4 = 18 characters.
    aggregate, samples = evaluate_predictions(references, predictions)

    assert aggregate.total_char_edits == 3
    assert aggregate.total_ref_chars == 18
    assert aggregate.cer == pytest.approx(3 / 18)
    assert len(samples) == 3


def test_wer_aggregate_is_hand_checkable() -> None:
    references = ["hello world", "abc", "Test"]
    predictions = ["hello wordl", "abc", "test"]
    aggregate, _ = evaluate_predictions(references, predictions)
    # 1 + 0 + 1 = 2 word edits over 2 + 1 + 1 = 4 words.
    assert aggregate.wer == pytest.approx(2 / 4)


def test_sequence_accuracy_is_exact_match_only() -> None:
    aggregate, _ = evaluate_predictions(["abc", "def"], ["abc", "dXf"])
    assert aggregate.sequence_accuracy == pytest.approx(0.5)


def test_normalized_metrics_ignore_case_and_punctuation() -> None:
    aggregate, _ = evaluate_predictions(["Hello, World!"], ["hello world"])
    assert aggregate.sequence_accuracy == 0.0          # strict comparison fails
    assert aggregate.sequence_accuracy_normalized == 1.0  # case/punct-insensitive passes


def test_character_accuracy_is_clamped_at_zero() -> None:
    # CER above 1.0 must not produce a negative accuracy.
    aggregate, _ = evaluate_predictions(["a"], ["abcdefgh"])
    assert aggregate.cer > 1.0
    assert aggregate.character_accuracy == 0.0


def test_macro_and_micro_differ_when_lengths_differ() -> None:
    # One long perfect sample and one short broken one: micro is dominated by
    # the long sample, macro is not.
    references = ["a" * 100, "xy"]
    predictions = ["a" * 100, "zz"]
    aggregate, _ = evaluate_predictions(references, predictions)

    assert aggregate.cer == pytest.approx(2 / 102)
    assert aggregate.cer_macro == pytest.approx((0.0 + 1.0) / 2)
    assert aggregate.cer_macro > aggregate.cer


def test_mismatched_lengths_are_rejected() -> None:
    with pytest.raises(ValueError, match="one-to-one"):
        evaluate_predictions(["a", "b"], ["a"])


def test_empty_input_yields_zeroed_metrics() -> None:
    aggregate, samples = evaluate_predictions([], [])
    assert aggregate.num_samples == 0 and samples == []


def test_whitespace_is_normalised_before_scoring() -> None:
    aggregate, _ = evaluate_predictions(["  hello   world "], ["hello world"])
    assert aggregate.cer == 0.0


def test_normalize_text_options() -> None:
    assert normalize_text("  A  b ") == "A b"
    assert normalize_text("A-b!", alphanumeric_only=True) == "Ab"
    assert normalize_text("AbC", lowercase=True) == "abc"


def test_worst_samples_are_sorted_by_cer() -> None:
    _, samples = evaluate_predictions(["abcd", "abcd", "abcd"], ["abcd", "abXd", "XXXX"])
    worst = worst_samples(samples, limit=2)
    assert worst[0].cer >= worst[1].cer
    assert worst[0].prediction == "XXXX"


def test_sample_ids_are_carried_through() -> None:
    _, samples = evaluate_predictions(["a"], ["b"], sample_ids=["img_001.png"])
    assert samples[0].sample_id == "img_001.png"
