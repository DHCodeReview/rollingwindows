"""helpers.py.

Last Update: May 29 2024
"""

from typing import Any, List, Union

import spacy
from spacy.language import Language


def ensure_doc(
    input: Union[str, List[str], spacy.tokens.doc.Doc],
    nlp: Union[Language, str],
    batch_size: int = 1000,
) -> spacy.tokens.doc.Doc:
    """Converts string or list inputs to spaCy docs.

    Args:
        input (Union[str, List[str], spacy.tokens.doc.Doc]): A string, list of tokens, or a spaCy doc.
        nlp (Union[Language, str]): The language model to use.
        batch_size (int): The number of texts to accumulate in an internal buffer.

    Returns:
        spacy.tokens.doc.Doc: A spaCy doc, unannotated if derived from a string or list of tokens.
    """
    if isinstance(input, spacy.tokens.doc.Doc):
        return input
    else:
        if isinstance(nlp, str):
            nlp = spacy.load(nlp)
        if isinstance(input, str):
            return list(nlp.tokenizer.pipe([input], batch_size=batch_size))[0]
        elif isinstance(input, list):
            return list(nlp.tokenizer.pipe([" ".join(input)], batch_size=batch_size))[0]
        else:
            raise Exception(
                "Invalid data type. Input data must be a string, a list of strings, or a spaCy doc."
            )


def ensure_list(input: Any) -> list:
    """Ensure that an item is of type list.

    Args:
        input (Any): An input variable.

    Returns:
        list: The input variable in a list if it is not already a list.
    """
    if not isinstance(input, list):
        input = [input]
    return input
