"""util.py."""

from typing import Any, Dict, List, Union

import spacy
from spacy.tokens import Doc
import numpy

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
    doc, filter: Union[list, set], spacy_attrs: List[str] = SPACY_ATTRS
) -> spacy.tokens.doc.Doc:
    """Filter a doc by applying a filter function.

    Args:
        doc: A doc to filter.
        filter: A list of token indexes to keep.
        spacy_attrs: A list of attributes to save to the numpy array.

    Returns:
        A new doc with only the filtered tokens and their original annotations.
    """
    # Handle docs with custom attributes
    if doc.user_data is not {}:
        # Make a character to token index map
        chars_to_tokens_map = chars_to_tokens(doc)
        # In Python 3.11, a single loop seems to be a bit faster than a list
        # comprehension with zip. This code is commented out in case further
        # testing shows otherwise.
        # words = [(t.i, t.text_with_ws) for t in doc if t.i in filter]
        # map_old_to_new, words = zip(*words)
        words, map_old_to_new = [], []
        for t in doc:
            if t.i in filter:
                words.append(t.text_with_ws)
                map_old_to_new.append(t.i)
        new_doc = Doc(doc.vocab, words=words)
        # Replace the new attributes with the old ones
        attrs_array = doc.to_array(spacy_attrs)
        filtered_array = attrs_array[list(filter)]
        new_doc.from_array(spacy_attrs, filtered_array)
        # Handle docs without custom attributes
        # Alternative method at https://gist.github.com/Jacobe2169/5086c7c4f6c56e9d3c7cfb1eb0010fe8
        new_user_data = {}
        for k, v in doc.user_data.items():
            # Get the old token index from the old character index
            token_index = chars_to_tokens_map[k[2]]
            if token_index in filter:
                # Add to the new user_data dict with a new character index
                new_token_index = map_old_to_new.index(token_index)
                new_char_index = new_doc[new_token_index].idx
                new_user_data[(k[0], k[1], new_char_index, k[3])] = v
        setattr(new_doc, "user_data", new_user_data)
    else:
        new_doc = Doc(doc.vocab, words=[t.text_with_ws for t in doc if t.i in filter])
        # Replace the new attributes with the old ones
        attrs_array = doc.to_array(spacy_attrs)
        filtered_array = attrs_array[filter]
        new_doc.from_array(spacy_attrs, filtered_array)
    return new_doc


def remove_tokens_on_match(doc, remove_ids, spacy_attrs: List[str] = SPACY_ATTRS):
    indexes = []
    doc_length = len(doc)
    for index, token in enumerate(doc):
        if token.i in remove_ids:
            indexes.append(index)
        # TODO: Try to account for trailing whitespace(s)
        # elif token.i + 1 < doc_length:
        #     if doc[token.i + 1].is_space:
        #         indexes.append(index)
    np_array = doc.to_array(spacy_attrs)
    np_array = numpy.delete(np_array, indexes, axis=0)
    if doc.user_data is not {}:
        chars_to_tokens_map = chars_to_tokens(doc)
        map_old_to_new = [t.i for t in doc if t.i not in indexes]
        doc2 = Doc(
            doc.vocab, words=[t.text for i, t in enumerate(doc) if i not in indexes]
        )
        doc2.from_array(spacy_attrs, np_array)
        new_user_data = {}
        for k, v in doc.user_data.items():
            # Get the old token index from the old character index
            token_index = chars_to_tokens_map[k[2]]
            if token_index not in remove_ids:
                # Add to the new user_data dict with a new character index
                new_token_index = map_old_to_new.index(token_index)
                new_char_index = doc2[new_token_index].idx
                new_user_data[(k[0], k[1], new_char_index, k[3])] = v
        setattr(doc2, "user_data", new_user_data)
    else:
        doc2 = Doc(
            doc.vocab, words=[t.text for i, t in enumerate(doc) if i not in remove_ids]
        )
        doc2.from_array(spacy_attrs, np_array)
    return doc2
