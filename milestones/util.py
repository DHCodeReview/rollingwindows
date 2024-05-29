"""util.py.

Last Update: May 25 2024
"""

from typing import Any, Dict, List, Union

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


def chars_to_tokens(doc: spacy.tokens.doc.Doc) -> Dict[int, int]:
    """Generate a characters to tokens mapping for _match_regex().

    Args:
        doc: A spaCy doc.

    Returns:
        A dict mapping character indexes to token indexes.
    """
    chars_to_tokens = {}
    for token in doc:
        for i in range(token.idx, token.idx + len(token.text)):
            chars_to_tokens[i] = token.i
    return chars_to_tokens


def lowercase_spacy_rules(
    patterns: List[List[Dict[str, Any]]],
    old_key: Union[List[str], str] = ["TEXT", "ORTH"],
    new_key: str = "LOWER",
) -> list:
    """Convert spaCy Rule Matcher patterns to lowercase.

    Args:
        patterns: A list of spacy Rule Matcher patterns.
        old_key: A dictionary key or list of keys to rename.
        new_key: The new key name.

    Returns:
        A list of spaCy Rule Matcher patterns.
    """

    def convert(key):
        if key in old_key:
            return new_key
        else:
            return key

    if isinstance(patterns, dict):
        new_dict = {}
        for key, value in patterns.items():
            value = lowercase_spacy_rules(value)
            key = convert(key)
            new_dict[key] = value
        return new_dict
    if isinstance(patterns, list):
        new_list = []
        for value in patterns:
            new_list.append(lowercase_spacy_rules(value))
        return new_list
    return patterns


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
        doc (spacy.tokens.doc.Doc): A spaCy doc.
        spacy_attrs (List[str]): A list of spaCy attributes to preserve.
        force_ws (bool): Force a whitespace at the end of every token except the last.

    Returns:
        np.ndarray: A numpy array of the doc.

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
