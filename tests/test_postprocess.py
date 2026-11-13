"""Tests for recognised-text post-processing.

The detokeniser undoes IAM-style punctuation spacing, which TrOCR's document
models reproduce faithfully from their training transcriptions. It must be
strictly a *formatting* correction: it may never alter the characters that
were recognised, only the spaces between them.
"""

from __future__ import annotations

import pytest

from ml.text.postprocess import detokenize, normalize_recognised_text


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("42 Industrial Estate , Mangalore", "42 Industrial Estate, Mangalore"),
        ("Laptop stand 51 . 25", "Laptop stand 51.25"),
        ("USB - C cable 18 . 00", "USB-C cable 18.00"),
        ("Tax at 18 % 20 . 12", "Tax at 18% 20.12"),
        ("Invoice No . INV-44718", "Invoice No. INV-44718"),
        ("Total 1 , 250 . 00", "Total 1,250.00"),
        ("don ' t stop", "don't stop"),
        ("Date : 22 March 2026", "Date: 22 March 2026"),
        ("Hello ; world", "Hello; world"),
        ("really ?", "really?"),
        ("stop !", "stop!"),
        ("( aside )", "(aside)"),
        ("[ ref ]", "[ref]"),
    ],
)
def test_detokenize_cases(raw: str, expected: str) -> None:
    assert detokenize(raw) == expected


def test_already_clean_text_is_untouched() -> None:
    text = "Acme Corporation Limited"
    assert detokenize(text) == text


def test_decimal_numbers_are_joined() -> None:
    # "51 . 25" is a decimal, not a sentence boundary.
    assert detokenize("Subtotal 111 . 75") == "Subtotal 111.75"


def test_newlines_are_preserved() -> None:
    raw = "Line one ,\nLine two ."
    assert detokenize(raw) == "Line one,\nLine two."


def test_line_count_is_unchanged() -> None:
    raw = "a ,\nb ,\nc ."
    assert len(detokenize(raw).split("\n")) == len(raw.split("\n"))


def test_only_whitespace_changes() -> None:
    """The characters themselves must survive exactly.

    A post-processor that could change letters would be hiding recognition
    errors rather than fixing formatting.
    """
    raw = "42 Industrial Estate , Mangalore 575001 ; Tax 18 % = 20 . 12"
    cleaned = detokenize(raw)

    assert "".join(cleaned.split()) == "".join(raw.split())


def test_repeated_spaces_collapse() -> None:
    assert detokenize("too    many     spaces") == "too many spaces"


def test_mode_none_is_identity() -> None:
    raw = "Estate , Mangalore"
    assert normalize_recognised_text(raw, mode="none") == raw


def test_mode_detokenize_applies() -> None:
    assert normalize_recognised_text("Estate , x", mode="detokenize") == "Estate, x"


def test_unknown_mode_is_rejected() -> None:
    with pytest.raises(ValueError, match="Unknown post-processing mode"):
        normalize_recognised_text("text", mode="spellcheck")


def test_empty_input() -> None:
    assert detokenize("") == ""
