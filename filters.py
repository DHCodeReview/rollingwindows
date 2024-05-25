"""filters.py.

Last Update: May 25 2024
"""

import re
from timeit import default_timer as timer
from typing import List, Protocol, Union

import numpy as np
import spacy
from spacy.attrs import (
    DEP,
    ENT_ID,
    ENT_IOB,
    ENT_TYPE,
    IS_ALPHA,
    IS_ASCII,
    IS_DIGIT,
    IS_LOWER,
    IS_PUNCT,
    IS_SPACE,
    IS_STOP,
    IS_TITLE,
    IS_UPPER,
    LEMMA,
    LENGTH,
    LIKE_EMAIL,
    LIKE_NUM,
    LIKE_URL,
    LOWER,
    MORPH,
    NORM,
    ORTH,
    POS,
    SENT_START,
    SHAPE,
    SPACY,
    TAG,
)
from spacy.tokens import Doc

from rollingwindows import helpers

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


def filter_doc(
    doc: spacy.tokens.doc.Doc,
    keep_ids: Union[list, set],
    spacy_attrs: List[str] = SPACY_ATTRS,
    force_ws: bool = True,
) -> spacy.tokens.doc.Doc:
    """Create a filter doc, preserving desired spaCy attributes and whitespace.

    Args:
        doc: A spaCy doc.
        keep_ids: The token ids to keep.
        spacy_attrs: A list of spaCy attributes to preserve.
        force_ws: Force a whitespace at the end of every token except the last.

    Returns:
            A filtered doc.

    Note:
        In spaCy 3.6.1 `Doc.to_array()` seems to preserve custom attributes.
    """
    words = []
    remove_indexes = []
    for i, token in enumerate(doc):
        if i in keep_ids:
            words.append(token.text)
        else:
            remove_indexes.append(i)
    np_array = get_doc_array(doc, spacy_attrs, force_ws)
    np_array = np.delete(np_array, remove_indexes, axis=0)
    doc2 = Doc(doc.vocab, words=words)
    doc2.from_array(spacy_attrs, np_array)
    return doc2


def get_doc_array(
    doc: spacy.tokens.doc.Doc,
    spacy_attrs: List[str] = SPACY_ATTRS,
    force_ws: bool = True,
) -> np.ndarray:
    """Get a numpy array of the doc.

    Args:
        doc: A spaCy doc.
        spacy_attrs: A list of spaCy attributes to preserve.
        force_ws: Force a whitespace at the end of every token except the last.

    Returns:
        A numpy array of the doc.

    Notes:
        1. `force_ws=True` ensures that `token_with_ws` and `whitespace_` attributes
                are preserved, but all tokens will be separated by whitespaces in the
                text of a doc created from the array.
        2. `force_ws=False` with `SPACY` in `spacy_attrs` preserves the `token_with_ws`
                and `whitespace_` attributes and their original values. This may cause
                tokens to be merged if subsequent processing operates on the `doc.text`.
        3. `force_ws=False` without `SPACY` in `spacy_attrs` does not preserve the
                `token_with_ws` and `whitespace_` attributes or their values. By default,
                `doc.text` displays a single space between each token.
    """
    if force_ws:
        if SPACY not in spacy_attrs:
            spacy_attrs.append(SPACY)
        np_array = doc.to_array(spacy_attrs)
        np_array[:-1, spacy_attrs.index(SPACY)] = 1
    else:
        np_array = doc.to_array(spacy_attrs)
    return np_array


def is_not_roman_numeral(s: str) -> bool:
    """Detect Roman numerals (capitals only).

    Args:
        s: A string to match against the pattern.

    Returns:
        A boolean indicated whether or not the numeral is a Roman numeral.
    """
    if s == "":
        return True
    pattern = r"^M{0,3}(CM|CD|D?C{0,3})(XC|XL|L?X{0,3})(IX|IV|V?I{0,3})$"
    return not bool(re.search(pattern, s))


class Filter(Protocol):

    @property
    def metadata(self) -> dict:
        exclude = ["doc"]
        metadata = {"id": self.id}
        return metadata | dict(
            (key, getattr(self, key))
            for key in dir(self)
            if key not in exclude and key not in dir(self.__class__)
        )

    def apply(self): ...


class WordFilter(Filter):
    id: str = "word_filter"

    def __init__(
        self,
        doc: spacy.tokens.doc.Doc,
        *,
        spacy_attrs: List[str] = SPACY_ATTRS,
        exclude: Union[List[str], str] = [" ", "\n"],
        exclude_digits: bool = False,
        exclude_roman_numerals: bool = False,
        exclude_pattern: Union[List[str], str] = None,
    ):
        """Initialise the filter object with configuration.

        Args:
            doc: A spaCy doc.
            spacy_attrs: A list of spaCy token attributes to preserve in the filtered doc.
            exclude: A string/regex or list of strings/regex patterns to exclude.
            exclude_digits: If True, digits will not be treated as words.
            exclude_roman_numerals: Same as above for Roman numerals, but only																																works on capital letters.
            exclude_pattern: Additional patterns to add to the default exclude list.
        """
        self.doc = doc
        self.spacy_attrs = spacy_attrs
        self.exclude = []
        self.exclude_digits = exclude_digits
        self.exclude_roman_numerals = exclude_roman_numerals
        self.exclude_pattern = []
        if exclude:
            self.exclude = helpers.ensure_list(exclude)
        if exclude_pattern:
            self.exclude_pattern = helpers.ensure_list(exclude_pattern)

    @property
    def word_ids(self):
        """Get a list of word_ids to keep after filtering."""
        predicates = []
        if self.exclude_digits:
            predicates.append(lambda t: t.text.isalpha())
        else:
            predicates.append(lambda t: t.text.isalpha() or t.text.isdigit())
        if self.exclude_roman_numerals:
            predicates.append(lambda token: is_not_roman_numeral(token.text))
        if self.exclude_pattern:
            self.exclude += self.exclude_pattern
        if len(self.exclude) > 0:
            exclude_pat = "|".join(self.exclude)
            predicates.append(lambda token: re.search(exclude_pat, token.text) is None)
        return {t.i for t in self.doc if all([f(t) for f in predicates])}

    def apply(self) -> spacy.tokens.doc.Doc:
        """Apply the filter."""
        return filter_doc(self.doc, self.word_ids, self.spacy_attrs)


class NonStopwordFilter(Filter):
    id: str = "non_stopword_filter"

    def __init__(
        self,
        doc: spacy.tokens.doc.Doc,
        *,
        spacy_attrs: List[str] = SPACY_ATTRS,
        additional_stopwords: List[str] = None,
        case_sensitive: bool = False,
    ):
        """Initialise the filter object with configuration.

        Args:
            doc: A spaCy doc
            spacy_attrs: A list of spaCy token attributes to preserve in the filtered doc.
            additional_stopwords: A list of stop words to add to those labelled as stop words by the model.
            case_sensitive: Use only lower case forms if False.

        Note:
            This is a minimal function that strips punctuation and returns words or ids
            not flagged as stop words in the doc or in an additional stop words list.
        """
        self.doc = doc
        self.spacy_attrs = spacy_attrs
        self.additional_stopwords = additional_stopwords
        self.case_sensitive = case_sensitive

    @property
    def word_ids(self):
        """Get a list of word_ids to keep after filtering."""
        if not self.additional_stopwords:
            self.additional_stopwords = set()
        else:
            self.additional_stopwords = set(
                helpers.ensure_list(self.additional_stopwords)
            )
            if not self.case_sensitive:
                self.additional_stopwords = {
                    text.lower() for text in self.additional_stopwords
                }
        return {token.i for token in self.doc if self._is_non_stopword(token)}

    def _is_non_stopword(self, token: spacy.tokens.Token) -> bool:
        """Check if a token should be retained.

        Args:
            token (spacy.tokens.Token): A spaCy token

        Returns:
            True if the token should be retained.
        """
        if self.case_sensitive:
            text = token.text
        else:
            text = token.lower_
        if (
            not token.is_punct
            and not token.is_stop
            and text not in self.additional_stopwords
        ):
            return True
        else:
            return False

    def apply(self) -> spacy.tokens.doc.Doc:
        """Apply the filter.

        Returns:
            The filtered doc.
        """
        return filter_doc(self.doc, self.word_ids, self.spacy_attrs)
