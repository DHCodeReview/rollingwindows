"""helpers.py.

Helper functions for word_windows4.py.
"""

from typing import Any, Iterable, List, Union
import collections
from itertools import chain, islice
import spacy
from spacy.matcher import Matcher

SpacyDoc = spacy.tokens.doc.Doc


def ensure_doc(
    input: Union[str, List[str], spacy.tokens.doc.Doc], nlp, batch_size: int = 1000
) -> spacy.tokens.doc.Doc:
    """Converts string or list inputs to spaCy docs.

    Args:
        input: A string, list of tokens, or a spaCy doc.
        nlp: The language model to use.
        batch_size: The number of texts to accumulate in an internal buffer.

    Returns:
        A spaCy doc, unannotated if derived from a string or list of tokens.
    """
    if isinstance(input, str):
        return list(nlp.tokenizer.pipe([input], batch_size=batch_size))[0]
    elif isinstance(input, list):
        return list(nlp.tokenizer.pipe([" ".join(input)], batch_size=batch_size))[0]
    else:
        return input


def ensure_list(input: Any) -> list:
    """Ensure that an item is of type list.

    Args:
        input: An input variable.

    Returns:
        The input variable in a list if it is not already a list.
    """
    if not isinstance(input, list):
        input = [input]
    return input


#########################################
## The functions below are deprecated. ##
#########################################


def sliding_window(iterable: Iterable, n: int = 1000) -> Iterable:
    """Return a generator of windows.

    Each window is a tuple of tokens.
    """
    it = iter(iterable)
    window = collections.deque(islice(it, n), maxlen=n)
    if len(window) == n:
        yield tuple(window)
    for x in it:
        window.append(x)
        yield tuple(window)


def get_str_windows(
    doc: Union[SpacyDoc, str], n: int = 1000, alignment_mode: str = "contract"
) -> Iterable:
    """Return a generator of windows.

    Args:
        doc: A spaCy doc or string.
        n: The size of the window.
        alignment_mode: The method to snap windows to token boundaries.

    Returns:
        A generator of window strings.

    Note:
        Window boundaries are snapped to token boundaries in the original doc.
        "Contract" means that the window will contain all tokens completely
        within the boundaries of `i:i + n`. "Expand" means that window will
        contain all tokens partially withn those boundaries. Setting
        `alignment_mode="strict"` in `doc.char_span()` is not advised
        because it returns `None` for any string that cannot be aligned to
        token boundaries. As a result, a slice method is used if you want
        to simply cut the text strictly on `n` characters.
    """
    # TODO: This is actually calculated twice because we need it to get
    # the length of the boundaries.
    if isinstance(doc, str):
        alignment_mode = "strict"
        boundaries = [(i, i + n) for i in range(len(doc))]
    else:
        boundaries = [(i, i + n) for i in range(len(doc.text))]

    if alignment_mode == "strict":
        for start_char, end_char in boundaries:
            yield doc.text[start_char:end_char]
    else:
        for start_char, end_char in boundaries:
            yield doc.char_span(start_char, end_char, alignment_mode=alignment_mode)


def consume(iterator, n=None):
    """Advance *iterable* by *n* steps or consume it entirely.

    Used for the Seekable class below.
    """
    # Use functions that consume iterators at C speed.
    if n is None:
        # feed the entire iterator into a zero-length deque
        collections.deque(iterator, maxlen=0)
    else:
        # advance to the empty slice starting at position n
        next(islice(iterator, n, n), None)


class Seekable:
    """Wrap an iterator to allow for seeking.

    This is a shortened version of more_itertools.seekable:
    see https://more-itertools.readthedocs.io/en/latest/api.html#more_itertools.seekable.
    """

    def __init__(self, iterable: Iterable, maxlen: int = None):
        """Initialise the object.

        Args:
            iterable: An iterable.
            maxlen: The maximum size of the iterable to prevent overlarge or infinite growth.
        """
        self._source = iter(iterable)
        if maxlen is None:
            self._cache = []
        else:
            self._cache = collections.deque([], maxlen)
        self._index = None

    def __iter__(self):
        """Make the object iterable."""
        return self

    def __next__(self):
        """Increment the object."""
        if self._index is not None:
            try:
                item = self._cache[self._index]
            except IndexError:
                self._index = None
            else:
                self._index += 1
                return item

        item = next(self._source)
        self._cache.append(item)
        return item

    def seek(self, index: int):
        """Find a location in the object.

        Args:
            index: The index to seek.
        """
        self._index = index
        remainder = index - len(self._cache)
        if remainder > 0:
            consume(self, remainder)


def is_spacy_pattern(patterns: list, is_spacy: bool = False) -> bool:
    """Flatten a pattern list and check for dicts.

    Does not check whether the dicts contain a valid spacy pattern.
    Args:
        patterns: A list of patterns
        is_spacy: Whether or not the pattern is a spacy pattern.

    Returns:
        A boolean
    """
    if any(isinstance(pat, dict) for pat in list(chain(*patterns))):
        is_spacy = True
    return is_spacy


def get_columns(patterns: list, nlp: spacy.vocab.Vocab) -> List[str]:
    """Get a list of column names from a list of patterns.

    Args:
        patterns: A list of token, regex, or spaCy patterns
        nlp: A spacy.vocab object

    Returns:
        A list of column names
    """
    if is_spacy_pattern(patterns):
        columns = []
        for pattern in patterns:
            # Raises an error if the pattern is not valid
            matcher = Matcher(nlp.vocab, validate=True)
            matcher.add("Test", [pattern])
            columns.append([str(pat) for pat in matcher.get("Test")[1]])
    else:
        columns = patterns
    return [str(col) for col in columns]


def ensure_vocab(nlp: Union[spacy.vocab.Vocab, str]) -> spacy.vocab.Vocab:
    """Return a spacy.vocab object, even if a model name is supplied.

    Args:
        nlp: A spacy.vocab object or the name of one.

    Returns:
        A spacy.vocab object.
    """
    if isinstance(nlp, str):
        return spacy.load(nlp)
    elif isinstance(nlp, spacy.vocab.Vocab):
        return nlp
    else:
        raise Exception(
            "The value of `nlp` is not a valid model name or `spacy.vocab` object."
        )
