"""Tests for character-set encoding.

The charset is the contract between a trained model and its decoder. A silent
change here turns every prediction into plausible-looking nonsense, so these
tests pin the behaviour precisely.
"""

from __future__ import annotations

import pytest

from ml.text.charset import BLANK_INDEX, Charset, CharsetError


def test_blank_is_index_zero(charset: Charset) -> None:
    assert charset.blank_index == BLANK_INDEX == 0
    # Every real character maps to 1..N, never 0.
    assert all(index > 0 for index in charset.encode("abcXYZ123"))


def test_num_classes_includes_the_blank(charset: Charset) -> None:
    assert charset.num_classes == len(charset.characters) + 1


def test_encode_decode_round_trip(charset: Charset) -> None:
    text = "Hello World 123"
    assert charset.decode(charset.encode(text)) == text


def test_decode_skips_blanks(charset: Charset) -> None:
    indices = charset.encode("ab")
    with_blanks = [BLANK_INDEX, indices[0], BLANK_INDEX, indices[1], BLANK_INDEX]
    assert charset.decode(with_blanks) == "ab"


def test_unsupported_characters_are_reported() -> None:
    small = Charset("abc")
    assert small.unsupported("abcd") == ["d"]
    assert small.is_encodable("abc")
    assert not small.is_encodable("abcd")


def test_unsupported_characters_are_dropped_by_default() -> None:
    small = Charset("abc")
    # "d" is dropped rather than raising, so one bad label cannot abort a run.
    assert small.decode(small.encode("abcd")) == "abc"


def test_strict_encoding_raises() -> None:
    small = Charset("abc")
    with pytest.raises(CharsetError, match="Cannot encode"):
        small.encode("abcd", strict=True)


def test_whitespace_is_collapsed(charset: Charset) -> None:
    assert charset.normalize("  hello   world  ") == "hello world"


def test_lowercase_charset_folds_case_and_deduplicates() -> None:
    folded = Charset("aAbB", lowercase=True)
    assert folded.characters == "ab"
    assert folded.decode(folded.encode("AB")) == "ab"


def test_duplicate_characters_are_rejected() -> None:
    with pytest.raises(CharsetError, match="duplicates"):
        Charset("aab")


def test_empty_charset_is_rejected() -> None:
    with pytest.raises(CharsetError):
        Charset("")


def test_round_trips_through_json(tmp_path, charset: Charset) -> None:
    path = charset.save(tmp_path / "charset.json")
    assert Charset.load(path) == charset


def test_from_corpus_is_deterministic() -> None:
    first = Charset.from_corpus(["cab", "bac"])
    second = Charset.from_corpus(["bac", "cab"])
    assert first.characters == second.characters == "abc"


def test_from_dict_matches_to_dict(charset: Charset) -> None:
    assert Charset.from_dict(charset.to_dict()) == charset


def test_token_names_the_blank(charset: Charset) -> None:
    assert charset.token(BLANK_INDEX) == "<blank>"
    assert charset.token(1) == charset.characters[0]


def test_default_charset_covers_the_expected_groups(charset: Charset) -> None:
    for sample in ("0123456789", "abcdefghijklmnopqrstuvwxyz", "ABCDEFGHIJKLMNOPQRSTUVWXYZ", " ", ".,-'"):
        assert charset.is_encodable(sample), f"{sample!r} should be encodable"


def test_default_charset_covers_document_symbols(charset: Charset) -> None:
    """Currency, percent and brackets must be representable.

    The original 83-character set omitted ``$ % < > { } | ~ ^ ` = \\``, so the
    model could not emit them at all -- the output layer had no class for
    them. Any invoice or report was therefore unrecognisable by construction.
    """
    document_text = 'Cost: $1,250.00 (50% off) <tag> {ref} | 5 = 5; "quoted" ~ok ^2 \\ path `x`'
    assert charset.unsupported(document_text) == []


def test_default_charset_is_full_printable_ascii(charset: Charset) -> None:
    printable = "".join(chr(code) for code in range(32, 127))
    assert charset.unsupported(printable) == []
    # 95 printable characters plus the CTC blank.
    assert charset.num_classes == 96
