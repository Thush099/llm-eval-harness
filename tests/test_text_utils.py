"""Unit tests for tokenization, stemming, number extraction, and claim splitting."""

import json
import math

import pytest

from src.utils import (
    compute_idf_map,
    extract_numbers,
    make_bigrams,
    save_results,
    split_claims,
    split_sentences,
    stem_token,
    tokenize,
)


class TestTokenize:
    def test_lowercases_and_drops_contraction_apostrophes(self) -> None:
        assert tokenize("Don't PANIC") == ["dont", "panic"]

    def test_empty_input_returns_empty_list(self) -> None:
        assert tokenize("") == []


class TestStemToken:
    def test_short_or_numeric_tokens_are_unchanged(self) -> None:
        assert stem_token("cat") == "cat"
        assert stem_token("42") == "42"

    def test_plural_s_is_stripped(self) -> None:
        assert stem_token("models") == "model"

    def test_ies_becomes_y(self) -> None:
        assert stem_token("studies") == "study"

    def test_ing_suffix_is_stripped(self) -> None:
        assert stem_token("running") == "runn"

    def test_stemming_is_deterministic(self) -> None:
        assert stem_token("evaluated") == stem_token("evaluated")


class TestExtractNumbers:
    def test_plain_and_decimal_numbers(self) -> None:
        assert extract_numbers("It costs 5 or 2.5 units") == [5.0, 2.5]

    def test_percentages_are_plain_values(self) -> None:
        assert extract_numbers("accuracy 97.5%") == [97.5]

    def test_comma_grouped_numbers(self) -> None:
        assert extract_numbers("population 1,200,000") == [1200000.0]

    def test_magnitude_suffixes_are_scaled(self) -> None:
        assert extract_numbers("raised 3 million and 2k") == [3000000.0, 2000.0]

    def test_empty_text_returns_empty_list(self) -> None:
        assert extract_numbers("") == []

    def test_all_values_finite(self) -> None:
        values = extract_numbers("1.5k, 2.2 percent, 7 billion")
        assert all(math.isfinite(v) for v in values)


class TestSplitSentences:
    def test_basic_sentence_split(self) -> None:
        assert split_sentences("First one. Second two.") == ["First one.", "Second two."]

    def test_decimals_do_not_split(self) -> None:
        assert split_sentences("It is 3.14 exactly.") == ["It is 3.14 exactly."]

    def test_empty_text_returns_empty_list(self) -> None:
        assert split_sentences("   ") == []


class TestSplitClaims:
    def test_simple_sentence_is_kept_whole(self) -> None:
        assert split_claims("The model is accurate.") == ["The model is accurate."]

    def test_empty_text_yields_no_claims(self) -> None:
        assert split_claims("") == []


class TestMakeBigrams:
    def test_adjacent_pairs(self) -> None:
        assert make_bigrams(["x", "y", "z"]) == [("x", "y"), ("y", "z")]


class TestComputeIdfMap:
    def test_empty_corpus_returns_empty_map(self) -> None:
        assert compute_idf_map([]) == {}

    def test_rare_terms_weigh_more(self) -> None:
        idf = compute_idf_map(["common rare", "common", "common"])
        assert idf["rare"] > idf["common"]


class TestSaveResults:
    def test_creates_parent_directories_and_writes_json(self, tmp_path) -> None:
        out = tmp_path / "nested" / "results.json"
        returned = save_results(out, {"score": 0.5})
        assert returned == out
        assert json.loads(out.read_text()) == {"score": 0.5}
