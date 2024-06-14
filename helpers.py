"""helpers.py.

Last Update: June 9 2024
"""

from typing import Any, Dict, Iterable, List, Union
from dataclasses import dataclass

import re
import spacy
from spacy.language import Language
from spacy.matcher import Matcher

@dataclass
class Windows:
    windows: Iterable
    window_units: str
    n: int
    alignment_mode: str = "strict"

    def __iter__(self):
        return iter(self.windows)


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


def flatten(input: Union[dict, list, str]) -> Iterable:
    """Yield items from any nested iterable.

    Args:
        input (Union[dict, list, str]): A list of lists or dicts.

    Yields:
        d

    Notes:
        See https://stackoverflow.com/a/40857703.
    """
    for x in input:
        if isinstance(x, Iterable) and not isinstance(x, str):
            if isinstance(x, list):
                for sub_x in flatten(x):
                    yield sub_x
            elif isinstance(x, dict):
                yield list(x.values())[0]
        else:
            yield x


def regex_escape(s: str) -> str:
	"""Escape only regex special characters.

	Args:
		s (str): A string.

	Returns:
		An escaped string.

	Note:
		See https://stackoverflow.com/a/78136529/22853742.
	"""
	if type(s) == bytes:
		return re.sub(rb"[][(){}?*+.^$]", lambda m: b"\\" + m.group(), s)
	return re.sub(r"[][(){}?*+.^$]", lambda m: "\\" + m.group(), s)

def spacy_rule_to_lower(
	patterns: Union[Dict, List[Dict]],
	old_key: Union[List[str], str] = ["TEXT", "ORTH"],
	new_key: str = "LOWER",
) -> list:
	"""Convert spacy Rule Matcher patterns to lowercase.

	Args:
		patterns (Union[Dict, List[Dict]]): A list of spacy Rule Matcher patterns.
		old_key (Union[List[str], str]): A dictionary key or list of keys to rename.
		new_key (str): The new key name.

	Returns:
		A list of spacy Rule Matcher patterns
	"""

	def convert(key):
		if key in old_key:
			return new_key
		else:
			return key

	if isinstance(patterns, dict):
		new_dict = {}
		for key, value in patterns.items():
			key = convert(key)
			new_dict[key] = value
		return new_dict

	if isinstance(patterns, list):
		new_list = []
		for value in patterns:
			new_list.append(spacy_rule_to_lower(value))
		return new_list
