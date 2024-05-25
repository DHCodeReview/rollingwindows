"""tests_helpers.py.

100% Coverage as of May 19 2024.
"""

import pytest
import spacy

from rollingwindows.helpers import ensure_doc, ensure_list

# Fixtures


@pytest.fixture
def nlp():
    return spacy.load("en_core_web_sm")


@pytest.fixture
def doc(nlp):
    return nlp("This is a test.")


# Functions


def test_ensure_doc_with_string_input(nlp):
    doc = ensure_doc("This is a test.", nlp)
    assert isinstance(doc, spacy.tokens.doc.Doc)
    assert [token.text for token in doc] == ["This", "is", "a", "test", "."]


def test_ensure_doc_with_string_input_and_string_model():
    doc = ensure_doc("This is a test.", "en_core_web_sm")
    assert isinstance(doc, spacy.tokens.doc.Doc)
    assert [token.text for token in doc] == ["This", "is", "a", "test", "."]


def test_ensure_doc_with_list_of_strings_input(nlp):
    doc = ensure_doc(["This", "is", "a", "test", "."], nlp)
    assert isinstance(doc, spacy.tokens.doc.Doc)
    assert [token.text for token in doc] == ["This", "is", "a", "test", "."]


def test_ensure_doc_with_doc_input(doc, nlp):
    doc2 = ensure_doc(doc, nlp)
    assert doc2 is doc


def test_ensure_doc_with_invalid_input(nlp):
    with pytest.raises(Exception):
        ensure_doc(123, nlp)


def test_ensure_list_with_string_input():
    result = ensure_list("test")
    assert isinstance(result, list)
    assert result == ["test"]


def test_ensure_list_with_list_of_strings_input():
    result = ensure_list(["test1", "test2"])
    assert isinstance(result, list)
    assert result == ["test1", "test2"]
