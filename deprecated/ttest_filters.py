"""tests_filters.py.

100% Coverage as of May 19 2024.
"""

import re

import numpy as np
import pytest
import spacy
from spacy.attrs import ENT_TYPE, IS_ALPHA, LOWER, POS, SENT_START, SPACY
from spacy.tokens import Token

from rollingwindows.filters import (Filter, NonStopwordFilter, WordFilter,
                                    filter_doc, get_doc_array,
                                    is_not_roman_numeral)

# Fixtures


@pytest.fixture
def spacy_attrs():
    return [LOWER, POS, ENT_TYPE, IS_ALPHA, SENT_START, SPACY]


@pytest.fixture
def nlp():
    return spacy.load("en_core_web_sm")


@pytest.fixture
def doc(nlp):
    return nlp("This is a test document. It has multiple sentences.")


@pytest.fixture
def doc_with_digits(nlp):
    return nlp("This is a test document. It has 2 sentences.")


@pytest.fixture
def doc_with_roman_numerals(nlp):
    return nlp("This is a test document. It has II sentences.")


@pytest.fixture
def doc_with_custom_attributes(nlp):
    test_text = "test"
    is_test_getter = lambda token: token.text == test_text
    Token.set_extension("is_test", getter=is_test_getter)
    return nlp("This is a test document. It has multiple sentences.")


@pytest.fixture
def test_filter():
    class TestFilter(Filter):
        id = "test_filter"
        doc = "Test document"

    return TestFilter()


# Filter Protocol


def test_filter_protocol(test_filter):
    metadata = test_filter.metadata
    assert metadata["id"] == "test_filter"
    assert "doc" not in metadata


@pytest.mark.xfail(raises=NotImplementedError)
def test_filter_protocol_apply_method(test_filter):
    test_filter.apply()


# Functions


def test_filter_doc_with_list(doc):
    filtered_doc = filter_doc(doc, keep_ids=[0, 2])
    assert isinstance(filtered_doc, spacy.tokens.doc.Doc)
    assert len(filtered_doc) == 2


def test_filter_doc_with_set(doc):
    filtered_doc = filter_doc(doc, keep_ids={0, 2})
    assert isinstance(filtered_doc, spacy.tokens.doc.Doc)
    assert len(filtered_doc) == 2


def test_filter_doc_with_spacy_attrs(doc):
    filtered_doc = filter_doc(
        doc, keep_ids=[0, 1, 2], spacy_attrs=["ORTH", "LOWER", "POS"]
    )
    assert isinstance(filtered_doc, spacy.tokens.doc.Doc)
    assert len(filtered_doc) == 3
    assert filtered_doc[0].pos_ == doc[0].pos_


def test_filter_doc_with_custom_attributes(doc_with_custom_attributes):
    filtered_doc = filter_doc(doc_with_custom_attributes, keep_ids=[0, 1, 2])
    assert len(filtered_doc) == 3
    assert filtered_doc[2]._.is_test == doc_with_custom_attributes[2]._.is_test


def test_get_doc_array_force_ws_true(doc, spacy_attrs):
    np_array = get_doc_array(doc, spacy_attrs, force_ws=True)
    assert isinstance(np_array, np.ndarray)


def test_get_doc_array_force_ws_false(doc, spacy_attrs):
    np_array = get_doc_array(doc, spacy_attrs, force_ws=False)
    assert isinstance(np_array, np.ndarray)


def test_is_not_roman_numeral_with_valid_numeral():
    assert is_not_roman_numeral("MCMXC") == False


def test_is_not_roman_numeral_with_invalid_numeral():
    assert is_not_roman_numeral("MCMXCIIV") == True


def test_is_not_roman_numeral_with_non_numeral():
    assert is_not_roman_numeral("Lexos") == True
    assert is_not_roman_numeral("27") == True


def test_is_not_roman_numeral_with_empty_string():
    assert is_not_roman_numeral("") == True


# WordFilter Class


def test_word_filter_word_ids(doc):
    word_filter = WordFilter(doc)
    assert isinstance(word_filter.word_ids, set)


def test_word_filter_apply(doc):
    word_filter = WordFilter(doc)
    filtered_doc = word_filter.apply()
    assert isinstance(filtered_doc, spacy.tokens.doc.Doc)
    assert len(filtered_doc) == len(word_filter.word_ids)


def test_word_filter_exclude_digits(doc_with_digits):
    word_filter = WordFilter(doc_with_digits, exclude_digits=True)
    filtered_doc = word_filter.apply()
    assert isinstance(filtered_doc, spacy.tokens.doc.Doc)
    has_digits = {t.is_digit for t in filtered_doc}
    assert has_digits == {False}
    assert len(filtered_doc) == len(word_filter.word_ids)


def test_word_filter_exclude_roman_numerals(doc_with_roman_numerals):
    word_filter = WordFilter(doc_with_roman_numerals, exclude_roman_numerals=True)
    filtered_doc = word_filter.apply()
    assert isinstance(filtered_doc, spacy.tokens.doc.Doc)
    has_no_roman_numerals = {is_not_roman_numeral(t.text) for t in filtered_doc}
    assert has_no_roman_numerals == {True}
    assert len(filtered_doc) == len(word_filter.word_ids)


def test_word_filter_exclude_pattern(doc):
    exclude_pattern = ["This", "test"]
    exclude_pat = "|".join(exclude_pattern)
    word_filter = WordFilter(doc, exclude_pattern=exclude_pattern)
    filtered_doc = word_filter.apply()
    print(filtered_doc.text)
    assert isinstance(filtered_doc, spacy.tokens.doc.Doc)
    pattern_matches = {re.search(exclude_pat, t.text) for t in filtered_doc}
    assert pattern_matches == {None}
    assert len(filtered_doc) == len(word_filter.word_ids)


# NonStopwordFilter Class


def test_non_stopword_filter_word_ids(doc):
    non_word_filter = NonStopwordFilter(doc)
    assert isinstance(non_word_filter.word_ids, set)


def test_non_stopword_filter_apply(doc):
    non_word_filter = NonStopwordFilter(doc)
    filtered_doc = non_word_filter.apply()
    assert isinstance(filtered_doc, spacy.tokens.doc.Doc)
    assert len(filtered_doc) == len(non_word_filter.word_ids)


def test_non_stopword_filter_additional_stopwords_case_sensitive_false(doc):
    stopword_ids = [t.i for t in doc if t.is_stop]
    additional_stopwords = ["Test", "document"]
    additional_stopwords = [x.lower() for x in additional_stopwords]
    additional_stopword_ids = [t.i for t in doc if t.lower_ in additional_stopwords]
    all_stopword_ids = stopword_ids + additional_stopword_ids
    num_stopwords = len(all_stopword_ids)
    num_words = len([t for t in doc if not t.is_punct])
    non_word_filter = NonStopwordFilter(doc, additional_stopwords=additional_stopwords)
    filtered_doc = non_word_filter.apply()
    assert isinstance(filtered_doc, spacy.tokens.doc.Doc)
    assert len(filtered_doc) == num_words - num_stopwords


def test_non_stopword_filter_additional_stopwords_case_sensitive_true(doc):
    stopword_ids = [t.i for t in doc if t.is_stop]
    additional_stopwords = ["Test", "document"]
    additional_stopword_ids = [t.i for t in doc if t.text in additional_stopwords]
    all_stopword_ids = stopword_ids + additional_stopword_ids
    num_stopwords = len(all_stopword_ids)
    num_words = len([t for t in doc if not t.is_punct])
    non_word_filter = NonStopwordFilter(
        doc, additional_stopwords=additional_stopwords, case_sensitive=True
    )
    filtered_doc = non_word_filter.apply()
    assert isinstance(filtered_doc, spacy.tokens.doc.Doc)
    assert len(filtered_doc) == num_words - num_stopwords


def test_non_stopword_filter_word_ids(doc):
    non_word_filter = NonStopwordFilter(doc)
    words = sorted([t.text for t in doc if not t.is_punct and not t.is_stop])
    words_from_word_ids = sorted([doc[i].text for i in non_word_filter.word_ids])
    assert words == words_from_word_ids
