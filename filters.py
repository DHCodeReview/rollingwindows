"""filters.py.

Working draft: April 4 2024
"""
from typing import List, Protocol, Union
import re
from rollingwindows2 import helpers
import spacy

from timeit import default_timer as timer


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

def filter_doc(
	doc, keep_ids: Union[list, set], spacy_attrs: List[str] = SPACY_ATTRS
) -> spacy.tokens.doc.Doc:
	"""Filter a doc by applying a filter function.

	Args:
		doc: A doc to filter.
		keep_ids: A list of token indexes to keep.
		spacy_attrs: A list of attributes to save to the numpy array.

	Returns:
		A new doc with only the filtered tokens and their original annotations.
	"""
	# Handle docs with custom attributes
	if doc.user_data is not {}:
		# Make a character to token index map
		chars_to_tokens = map_chars_to_tokens(doc)
		# In Python 3.11, a single loop seems to be a bit faster than a list
		# comprehension with zip. This code is commented out in case further
		# testing shows otherwise.
		# words = [(t.i, t.text_with_ws) for t in doc if t.i in filter]
		# map_old_to_new, words = zip(*words)
		words, map_old_to_new = [], []
		for t in doc:
			if t.i in keep_ids:
				words.append(t.text_with_ws)
				map_old_to_new.append(t.i)
		new_doc = Doc(doc.vocab, words=words)
		# Replace the new attributes with the old ones
		attrs_array = doc.to_array(spacy_attrs)
		filtered_array = attrs_array[list(keep_ids)]
		new_doc.from_array(spacy_attrs, filtered_array)
		# Handle docs without custom attributes
		# Alternative method at https://gist.github.com/Jacobe2169/5086c7c4f6c56e9d3c7cfb1eb0010fe8
		new_user_data = {}
		for k, v in doc.user_data.items():
			# Get the old token index from the old character index
			token_index = chars_to_tokens[k[2]]
			if token_index in keep_ids:
				# Add to the new user_data dict with a new character index
				new_token_index = map_old_to_new.index(token_index)
				new_char_index = new_doc[new_token_index].idx
				new_user_data[(k[0], k[1], new_char_index, k[3])] = v
		setattr(new_doc, "user_data", new_user_data)
	else:
		new_doc = Doc(doc.vocab, words=[t.text_with_ws for t in doc if t.i in keep_ids])
		# Replace the new attributes with the old ones
		attrs_array = doc.to_array(spacy_attrs)
		filtered_array = attrs_array[keep_ids]
		new_doc.from_array(spacy_attrs, filtered_array)
	return new_doc

def group_consecutive(seq: list) -> List[list]:
    """Group a list of integers.

    Args:
        A list of integers.

    Returns:
        A list containing tuples of start and end indexes.
    """
    if not seq:
        return seq
    grouped = [[seq[0]]]
    for x in seq[1:]:
        if x == grouped[-1][-1] + 1:
            grouped[-1].append(x)
        else:
            grouped.append([x])
    # Return tuples instead of sublists
    return [(x[0], x[0] + 1) if len(x) == 1 else (x[0], x[1]) for x in grouped]

def is_roman_numeral(s: str) -> bool:
	"""Detect Roman numerals (capitals only).

	Args:
		s: A string to match against the pattern.

	Returns:
		A boolean indicated whether or not the numeral is a Roman numeral.
	"""
	pattern = r"^M{0,3}(CM|CD|D?C{0,3})(XC|XL|L?X{0,3})(IX|IV|V?I{0,3})$"
	return bool(re.search(pattern, s))

def map_chars_to_tokens(doc: spacy.tokens.doc.Doc) -> dict:
	"""Map a doc's character indexes to token indexes.

	Args:
		A spaCy doc.

	Returns:
		A dict mapping the character indexes to the token indexes.
	"""
	chars_to_tokens = {}
	for token in doc:
		for i in range(token.idx, token.idx + len(token.text)):
			chars_to_tokens[i] = token.i
	return chars_to_tokens

# TODO: This function may fail in some edge cases with multitoken milestones. It needs further testing.
def replace_milestones(
    filtered_doc: spacy.tokens.doc.Doc, original_labels: List[dict]
) -> List[dict]:
    """Create a new milestone labels list on a filtered doc.

    Args:
        filtered_doc: A spaCy doc.
        original_labels: A list of dicts with containing the milestone labels in the unfiltered doc.

    Return:
        A new milestone labels list.
    """
    # Get a list of all milestone ids in the filtered doc
    milestone_ids = [t.i for t in filtered_doc if t._.is_milestone]
    # Convert to a list of start, end tuples
    milestone_ids = group_consecutive(milestone_ids)
    # Iterate through the milestone ids and match them to the original labels
    milestone_labels = [
        {list(original_labels[i].keys())[0]: x} for i, x in enumerate(milestone_ids)
    ]
    return milestone_labels


class Filter(Protocol):
	name: str = "FilterProtocol"

	@property
	def metadata(self) -> dict:
		exclude = ["doc"]
		metadata = {"id": self.id}
		return metadata | dict((key, getattr(self, key)) for key in dir(self) if key not in exclude and key not in dir(self.__class__))

	def apply(self):
		...

class WordFilter(Filter):
	id: str = "word_filter"

	def __init__(self,
			doc: spacy.tokens.doc.Doc,
			*,
			spacy_attrs: List[str] = SPACY_ATTRS,
			exclude: Union[List[str], str] = [" ", "\n"],
			exclude_digits: bool = False,
			exclude_roman_numerals: bool = False,
			exclude_pattern: Union[List[str], str] = None):
		"""Initialise the filter object with configuration.

		Args:
			doc: A spaCy doc.
			spacy_attrs: A list of spaCy token attributes to preserve in the filtered doc.
			exclude: A string/regex or list of strings/regex patterns to exclude.
			exclude_digits: If True, digits will not be treated as words.
			exclude_roman_numerals: Same as above for Roman numerals, but only
				works on capital letters.
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
			predicates.append(lambda token: is_roman_numeral(token.text))
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

	def __init__(self,
			  doc: spacy.tokens.doc.Doc,
			  *,
			  spacy_attrs: List[str] = SPACY_ATTRS,
			  additional_stopwords: List[str] = None,
			  case_sensitive: bool = False
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
		self.additional_stopwords = additional_stopwords
		self.case_sensitive = case_sensitive

	@property
	def word_ids(self):
		"""Get a list of word_ids to keep after filtering."""
		if not self.additional_stopwords:
			self.additional_stopwords = set()
		else:
			self.additional_stopwords = set(helpers.ensure_list(self.additional_stopwords))
			if not self.case_sensitive:
				self.additional_stopwords = {text.lower() for text in self.additional_stopwords}
		return {token.i for token in self.doc if self.is_non_stopword(token)}


	def apply(self) -> spacy.tokens.doc.Doc:
		"""Apply the filter."""
		return filter_doc(self.doc, self.word_ids, self.spacy_attrs)

	def is_non_stopword(self, token: spacy.tokens.Token) -> bool:
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
			and not token._is_stop
			and text not in self.additional_stopwords
		):
			return True
		else:
			return False
