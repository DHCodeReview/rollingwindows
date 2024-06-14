"""test_helpers.py.

As of 10 June 2024 this has 98% coverage for helpers.py.
"""

import re

import pytest
import spacy

from rollingwindows.helpers import (ensure_doc, ensure_list, flatten,
                                    regex_escape, spacy_rule_to_lower)

# Fixtures


@pytest.fixture
def nlp():
    return spacy.load("en_core_web_sm")


@pytest.fixture
def text(nlp):
    return nlp("This is a test document. It has multiple sentences.")


# Functions


def test_ensure_doc_from_str(nlp):
    s = "A test string"
    doc = ensure_doc(s, nlp)
    assert isinstance(doc, spacy.tokens.doc.Doc)


def test_ensure_doc_from_doc(text, nlp):
    doc = ensure_doc(text, nlp)
    assert isinstance(doc, spacy.tokens.doc.Doc)


def test_ensure_doc_from_list(nlp):
    s = "A test string"
    doc = ensure_doc(s.split(), nlp)
    assert isinstance(doc, spacy.tokens.doc.Doc)


def test_ensure_list():
    assert isinstance(ensure_list("A test string"), list)


def test_flatten_dict():
    pattern = {"LOWER": "love"}
    for item in flatten(pattern):
        assert isinstance(item, (dict, str))


def test_flatten_list():
    pattern = [[[{"LOWER": "love"}, {"REGEX": {"ORTH": "marriag."}}]]]
    for item in flatten(pattern):
        assert isinstance(item, (dict, str))


def test_flatten_str():
    pattern = "love"
    for item in flatten(pattern):
        assert isinstance(item, (dict, str))


def test_regex_escape():
    regex = re.search("lov.", "loving").group()
    escaped = re.search(regex_escape("lov."), "loving")
    if escaped is not None:
        escape = escape.group()
    assert regex != escaped


def test_spacy_rule_to_lower_with_dict():
    pattern = {"ORTH": "text"}
    result = spacy_rule_to_lower(pattern)
    assert result == {"LOWER": "text"}


def test_spacy_rule_to_lower_with_list():
    pattern = [{"ORTH": "text"}]
    result = spacy_rule_to_lower(pattern)
    assert result == [{"LOWER": "text"}]


def test_spacy_rule_to_lower_with_text():
    pattern = {"TEXT": "text"}
    result = spacy_rule_to_lower(pattern)
    assert result == {"LOWER": "text"}


def test_spacy_rule_to_lower_with_regex():
    pattern = [
        {"TEXT": {"REGEX": "^[Uu](\\.?|nited)$"}},
        {"TEXT": {"REGEX": "^[Ss](\\.?|tates)$"}},
        {"LOWER": "president"},
    ]
    result = spacy_rule_to_lower(pattern)
    assert result == [
        {"LOWER": {"REGEX": "^[Uu](\\.?|nited)$"}},
        {"LOWER": {"REGEX": "^[Ss](\\.?|tates)$"}},
        {"LOWER": "president"},
    ]
