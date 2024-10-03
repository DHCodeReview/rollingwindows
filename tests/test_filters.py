"""test_filters.py.

As of 12 June 2024 this has 94% coverage for filters.py.
"""
import numpy as np
import pytest
import spacy
from spacy.attrs import (DEP, ENT_ID, ENT_IOB, ENT_TYPE, IS_ALPHA, IS_ASCII,
                         IS_DIGIT, IS_LOWER, IS_PUNCT, IS_SPACE, IS_STOP,
                         IS_TITLE, IS_UPPER, LEMMA, LENGTH, LIKE_EMAIL,
                         LIKE_NUM, LIKE_URL, LOWER, MORPH, NORM, ORTH, POS,
                         SENT_START, SHAPE, SPACY, TAG)

from rollingwindows.filters import (BaseFilter, NonStopwordFilter, WordFilter,
                                    filter_doc, get_doc_array,
                                    is_not_roman_numeral)

SPACY_ATTRS = [
    "DEP",
    "ENT_ID",
    "ENT_IOB",
    "ENT_TYPE",
    "IS_ALPHA",
    "IS_ASCII",
    "IS_DIGIT",
    "IS_LOWER",
    "IS_PUNCT",
    "IS_SPACE",
    "IS_STOP",
    "IS_TITLE",
    "IS_UPPER",
    "LEMMA",
    "LENGTH",
    "LIKE_EMAIL",
    "LIKE_NUM",
    "LIKE_URL",
    "LOWER",
    "MORPH",
    "NORM",
    "ORTH",
    "POS",
    "SENT_START",
    "SHAPE",
    "SPACY",
    "TAG",
]

# Functions

@pytest.fixture
def sample_doc():
    nlp = spacy.blank("en")
    doc = nlp("This is a sample document. It contains several tokens.")
    return doc

@pytest.fixture
def short_doc():
    nlp = spacy.blank("en")
    doc = nlp("This is a sample document.")
    return doc

def test_filter_doc_keep_all(sample_doc):
    keep_ids = list(range(len(sample_doc)))
    filtered_doc = filter_doc(sample_doc, keep_ids)
    assert len(filtered_doc) == len(sample_doc)
    assert all(token.text == filtered_token.text for token, filtered_token in zip(sample_doc, filtered_doc))

def test_filter_doc_remove_some(sample_doc):
    keep_ids = [0, 1, 2, 3, 4, 5]  # Keep only the first sentence
    filtered_doc = filter_doc(sample_doc, keep_ids)
    assert len(filtered_doc) == 6
    assert filtered_doc.text == "This is a sample document . "

def test_filter_doc_attributes_preserved(sample_doc):
    keep_ids = [0, 1, 2, 3]  # Keep only the first sentence
    spacy_attrs = ["LEMMA", "POS"]
    filtered_doc = filter_doc(sample_doc, keep_ids, spacy_attrs=spacy_attrs)
    # Assuming the function is correctly preserving attributes, this test checks a couple.
    assert all(token.lemma_ == filtered_token.lemma_ for token, filtered_token in zip(sample_doc[:4], filtered_doc))
    assert all(token.pos_ == filtered_token.pos_ for token, filtered_token in zip(sample_doc[:4], filtered_doc))

def test_filter_doc_force_whitespace(sample_doc):
    keep_ids = [0, 1, 2, 3, 4, 5]  # Keep only the first sentence
    filtered_doc = filter_doc(sample_doc, keep_ids, force_ws=True)
    # Check if whitespace is forced after each token except the last
    assert all(filtered_doc[i].whitespace_ == " " for i in range(len(filtered_doc) - 1))

def test_get_doc_array_default_attrs(short_doc):
    np_array = get_doc_array(short_doc)
    # Need to account for the ID as the first element in the array
    assert np_array.shape[1] == len(SPACY_ATTRS) + 1

def test_get_doc_array_force_ws(short_doc):
    np_array = get_doc_array(short_doc, force_ws=True)
    # Check if the last column (assumed to be whitespace) is set correctly
    assert np.all(np_array[:-1, -1] == 1), "All but the last token should have whitespace forced"

def test_get_doc_array_no_force_ws(short_doc):
    custom_attrs = ["LEMMA", "SPACY"]
    orig_array = short_doc.to_array(custom_attrs)
    orig_whitespace = list(orig_array[:, -1])
    np_array = get_doc_array(short_doc, force_ws=False, spacy_attrs=custom_attrs)
    new_whitespace = list(np_array[:, -1])
    assert new_whitespace == orig_whitespace

def test_get_doc_array_custom_attrs(short_doc):
    custom_attrs = ["ORTH", "LEMMA"]
    np_array = get_doc_array(short_doc, spacy_attrs=custom_attrs)
    assert np_array.shape[1] == len(custom_attrs), "Array should only contain columns for custom attributes"

@pytest.mark.parametrize("input_str, expected", [
    ("III", False),  # Valid Roman numeral
    ("IV", False),   # Valid Roman numeral
    ("IX", False),   # Valid Roman numeral
    ("MDCCLXXVI", False),  # Valid Roman numeral
    ("MMXX", False),  # Valid Roman numeral
    ("", True),  # Empty string
    ("123", True),  # Not a Roman numeral
    ("ABC", True),  # Not a Roman numeral
    ("IVIV", True),  # Invalid Roman numeral pattern
    ("MMMM", True),  # Exceeds the maximum count for 'M'
    ("LL", True),  # Invalid repetition of 'L'
    ("IC", True),  # Invalid order
])

def test_is_not_roman_numeral(input_str, expected):
    assert is_not_roman_numeral(input_str) == expected, f"Failed for input: {input_str}"


class TestFilter(BaseFilter):
    # Suppress the warning because the class name begins with "Test"
    __test__ = False

    def __init__(self, id, test_attr):
        self.id = id
        self.test_attr = test_attr
        self.doc = "This should be excluded"

@pytest.mark.parametrize("id, test_attr, expected_metadata", [
    (1, "value1", {"id": 1, "test_attr": "value1"}),
    (2, "value2", {"id": 2, "test_attr": "value2"}),
])

def test_metadata(id, test_attr, expected_metadata):
    test_filter = TestFilter(id, test_attr)
    metadata = test_filter.metadata
    assert metadata == expected_metadata, f"Metadata did not match expected. Got {metadata}, expected {expected_metadata}"
    assert "doc" not in metadata, "Excluded 'doc' attribute found in metadata"

# WordFilter

# Mocking necessary components
class MockWordFilterDoc:
    def __init__(self, text):
        self.text = text
        self.tokens = [MockWordFilterToken(word) for word in text.split()]

    def __iter__(self):
        return iter(self.tokens)

class MockWordFilterToken:
    def __init__(self, text):
        self.text = text
        self.i = hash(text)

@pytest.fixture
def mock_word_filter_doc():
    return MockWordFilterDoc("The quick brown fox jumps over 2 lazy dogs.")

@pytest.fixture
def word_filter_doc():
    nlp = spacy.blank("en")
    return nlp("The quick brown fox jumps over 2 lazy dogs.")

# Test initialization
def test_word_filter_initialization(mock_word_filter_doc):
    wf = WordFilter(doc=mock_word_filter_doc, exclude_digits=True)
    assert wf.id == "word_filter"
    assert wf.exclude_digits == True, "exclude_digits should be True"

# Test word_ids property
@pytest.mark.parametrize("exclude_digits, expected_count", [
    (True, 8),  # Excluding '2' and '.', expecting 8 tokens
    (False, 9)  # Including '2' and '.', expecting 9 tokens
])

def test_word_ids(word_filter_doc, exclude_digits, expected_count):
    wf = WordFilter(doc=word_filter_doc, exclude_digits=exclude_digits)
    assert len(wf.word_ids) == expected_count, f"Expected {expected_count} tokens, got {len(wf.word_ids)}"

# Test apply method
def test_apply_method(word_filter_doc):
    wf = WordFilter(doc=word_filter_doc)
    filtered_doc = wf.apply()
    assert isinstance(filtered_doc, spacy.tokens.doc.Doc), "The apply method should return a spacy.tokens.doc.Doc object"

# NonStopwordFilter

# Mocking necessary components
class MockNonStopwordFilterToken:
    def __init__(self, text, is_punct=False, is_stop=False, lemma_="", pos_="", i=0):
        self.text = text
        self.is_punct = is_punct
        self.is_stop = is_stop
        self.lemma_ = lemma_
        self.pos_ = pos_
        self.lower_ = text.lower()
        self.i = i

class MockNonStopwordFilterDoc:
    def __init__(self, tokens):
        self.tokens = tokens

    def __iter__(self):
        return iter(self.tokens)

@pytest.fixture
def non_stopword_filter_doc():
    nlp = spacy.blank("en")
    return nlp("The quick brown fox jumps over the lazy dog")

@pytest.fixture
def mock_non_stopword_filter_doc():
    tokens = [
        MockNonStopwordFilterToken("The", is_stop=True, i=0),
        MockNonStopwordFilterToken("quick", i=1),
        MockNonStopwordFilterToken("brown", i=2),
        MockNonStopwordFilterToken("fox", i=3),
        MockNonStopwordFilterToken(",", is_punct=True, i=4),
        MockNonStopwordFilterToken("jumps", i=5),
        MockNonStopwordFilterToken("over", is_stop=True, i=6),
        MockNonStopwordFilterToken("the", is_stop=True, i=7),
        MockNonStopwordFilterToken("lazy", i=8),
        MockNonStopwordFilterToken("dog", i=9)
    ]
    return MockNonStopwordFilterDoc(tokens)

# Test initialization
def test_non_stopword_filter_initialization(non_stopword_filter_doc):
    filter = NonStopwordFilter(doc=non_stopword_filter_doc, additional_stopwords=["lazy"], case_sensitive=True)
    assert filter.id == "non_stopword_filter"
    assert filter.case_sensitive == True, "case_sensitive should be True"
    assert "lazy" in filter.additional_stopwords, "'lazy' should be in additional_stopwords"

# Test word_ids property
def test_word_ids(mock_non_stopword_filter_doc):
    nlp = spacy.blank("en")
    doc = nlp("The quick brown fox jumps over the lazy dog")
    filter = NonStopwordFilter(doc=doc, additional_stopwords=["lazy"], case_sensitive=False)
    expected_ids = {1, 2, 3, 4, 8}  # 'quick', 'brown', 'fox', 'jumps', 'dog'
    assert filter.word_ids == expected_ids, f"Expected word ids {expected_ids}, got {filter.word_ids}"

# Test _is_non_stopword method
@pytest.mark.parametrize("text, is_punct, is_stop, additional_stopwords, case_sensitive, expected", [
    ("quick", False, False, [], False, True),
    ("The", False, True, [], False, False),
    ("lazy", False, False, ["lazy"], False, False),
    ("Lazy", False, False, ["lazy"], True, True)  # Case sensitive
])

def test_is_non_stopword(text, is_punct, is_stop, additional_stopwords, case_sensitive, expected):
    token = MockNonStopwordFilterToken(text, is_punct=is_punct, is_stop=is_stop)
    filter = NonStopwordFilter(doc=mock_non_stopword_filter_doc, additional_stopwords=additional_stopwords, case_sensitive=case_sensitive)
    assert filter._is_non_stopword(token) == expected, f"Expected {filter._is_non_stopword(token)} to be {expected}"

# Test apply method
def test_apply_method(non_stopword_filter_doc):
    filter = NonStopwordFilter(doc=non_stopword_filter_doc)
    filtered_doc = filter.apply()
    assert isinstance(filtered_doc, spacy.tokens.doc.Doc), "The apply method should return a spacy.tokens.doc.Doc object"
