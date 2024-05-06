"""calculators.py.

Working draft: April 20 2024
"""

from typing import Dict, Iterable, List, Protocol, Union
from enum import Enum
import re
import pandas as pd

import spacy
from spacy.matcher import Matcher
from rollingwindows.helpers import ensure_list

def spacy_rule_to_lower(
	patterns: Union[Dict, List[Dict]],
	old_key: Union[List[str], str] = ["TEXT", "ORTH"],
	new_key: str = "LOWER",
) -> list:
	"""Convert spacy Rule Matcher patterns to lowercase.

	Args:
		patterns: A list of spacy Rule Matcher patterns.
		old_key: A dictionary key or list of keys to rename.
		new_key: The new key name.

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

class Calculator(Protocol):
	"""The Calculator class."""

	@property
	def metadata(self) -> dict:
		exclude = ["data", "windows"]
		metadata = {"id": self.id}
		return metadata | dict(
			(key, getattr(self, key))
			for key in dir(self)
			if key not in exclude and key not in dir(self.__class__)
		)

	def run(self, **kwargs):
		"""Perform the calculations."""
		...

	def to_df(self, **kwargs) -> pd.DataFrame:
		"""Convert the data to a pandas dataframe."""
		...

class Averages(Calculator):
	id: str = "averages"

	def __init__(
		self,
		patterns: Union[list, str],
		windows: Iterable,
		*,
		search_method: str = "count",
		model: str = None,
		doc: spacy.tokens.doc.Doc = None,
		alignment_mode: str = "strict",
		regex: bool = False,
		case_sensitive: bool = True,
		use_span_text: bool = False,
	) -> None:
		"""Initialise the calculator.

		Args:
			patterns: A pattern or list of patterns to search in windows.
			windows: A Windows object containing the windows to search.
			search_method: The preliminary search method to use.
			model: The language model to be used for searching spaCy tokens/spans.
			alignment_mode: Whether to snap searches to token boundaries. Values are
				"strict", "contract", and "expand".
			doc: The "re_finditer" method returns character start and end indexes in the window. Access to the doc from which the windows was generated is necessary to map these to the token indexes in order to use `alignment_mode`.
			regex: Whether to use regex for searching.
			case_sensitive: Whether to make searches case-sensitive.
		"""
		if model:
			self.nlp = spacy.load(model)
		self._validate_config(patterns, windows, search_method)
		self.windows = windows
		self.window_units = self.windows.window_units
		self.n = self.windows.n
		self.patterns = ensure_list(patterns)
		self.search_method = search_method
		self.alignment_mode = alignment_mode
		self.regex = regex
		self.case_sensitive = case_sensitive
		self.use_span_text = use_span_text
		self.doc = doc
		self.data = []

	@property
	def regex_flags(self) -> Enum:
		"""Return regex flags based on case_sensitive setting."""
		if not self.case_sensitive:
			return re.IGNORECASE | re.UNICODE
		else:
			return re.UNICODE

	def _configure_search_method(self):
		"""Override the initial search method based on config."""
		# For tokens, convert string patterns to spaCy rules and use spacy_matcher
		if self.window_units in ["lines", "sentences", "tokens"]:
			patterns = []
			for pattern in self.patterns:
				if isinstance(pattern, str):
					if self.regex:
						pattern = [{"TEXT": {"REGEX": pattern}}]
					else:
						pattern = [{"TEXT": pattern}]
					if not self.case_sensitive:
						pattern = spacy_rule_to_lower(pattern)
				patterns.append(pattern)
			self.patterns = patterns
			self.search_method == "spacy_matcher"
		# Change count to re_search if regex is enabled
		if self.search_method == "count" and self.regex:
			self.search_method = "re_search"
		# Change spacy_matcher to re_finditer if use_span_text is enabled
		if self.search_method == "spacy_matcher":
			if self.use_span_text:
				self.search_method = "re_finditer"
			# This might be a redundant check
			matcher = Matcher(self.nlp.vocab)
			for pattern in self.patterns:
				try:
					matcher.add("MatcherRule", [pattern])
				except ValueError:
					raise Exception(f"{pattern} is an invalid spaCy rule.")

	def _validate_config(self, patterns, windows, search_method) -> None:
		"""Check that all required are configurations are present and valid.

		Args:
 			patterns: A pattern or list of patterns to search in windows.
			windows: A Windows object containing the windows to search.
			search_method: Name of the search_method to use.
		"""
		check = [patterns, windows, search_method]
		msg = "An averages calculator must be initialised with `patterns`, `windows`, and `search_method` attributes."
		for arg in check:
			if arg is None:
				raise Exception(msg)
		if search_method in ["count", "regex", "re_finditer"] and not all(isinstance(x, str) for x in patterns):
			raise Exception(f"One or more patterns is not a valid string, which is required for the `{search_method}` search_method.")
		if search_method == "spacy_matcher":
			if windows.window_units == "characters":
				raise Exception("You cannot use the `spacy_matcher` method to search character windows.")
			matcher = Matcher(self.nlp.vocab)
			for pattern in patterns:
				try:
					matcher.add("MatcherRule", [pattern])
				except ValueError:
					raise Exception(f"{pattern} is an invalid spaCy rule.")

	def _count_pattern_matches(self, pattern, window):
		"""Count the matches for a single pattern in a single window."""
		# Count exact strings
		if self.search_method == "count":
			if not self.case_sensitive:
				pattern = pattern.lower()
			return window.count(pattern)

		# Count regex matches
		elif self.search_method == "regex":
			return len(re.findall(pattern, window, flags=self.regex_flags))

		# Count spaCy rule matches (for tokens or spans)
		elif self.search_method == "spacy_matcher":
			if not self.case_sensitive:
				pattern = spacy_rule_to_lower(pattern)
			matcher = Matcher(self.nlp.vocab)
			matcher.add("MatcherRule", [pattern])
			return len(matcher(window))

		# Count token matches over the full text
		else:
			return sum([(1 if self.doc.char_span(match.span()[0], match.span()[1], alignment_mode=self.alignment_mode) else 0)
							for match in re.finditer(pattern, window, flags=self.regex_flags)])

	def run(self) -> None:
		"""Run the calculator."""
		self._configure_search_method()
		self.data = [
			[self._count_pattern_matches(pattern, window) / self.n for pattern in self.patterns]
			for window in self.windows
		]

	def _extract_string_pattern(self, pattern: Union[dict, list, str]):
		"""Extract a string pattern from a spaCy rule.

		Args:
			A pattern to search.

		Returns:
			A string pattern.
		"""
		if isinstance(pattern, list):
			key = list(pattern[0].keys())[0]
			pattern = pattern[0].get(key)
		elif isinstance(pattern, dict):
			key = list(pattern.keys())[0]
			pattern = pattern.get(key)
		return pattern


	def to_df(self, show_spacy_rules: bool = False) -> pd.DataFrame:
		"""Convert the data to a pandas dataframe.

		Args:
			show_spacy_rules: If True, use full spaCy rules for labels; otherwise use only the
			string pattern.
		"""
		if show_spacy_rules:
			patterns = self.patterns
		else:
			patterns = []
			# Extract strings from spaCy rules
			for pattern in self.patterns:
				if isinstance(pattern, list):
					patterns.append(self._extract_string_pattern(pattern))
				else:
					patterns.append(pattern)
		# Assign column labels
		cols = []
		for pattern in patterns:
			if not self.case_sensitive and isinstance(pattern, str):
				pattern = pattern.lower()
			elif not self.case_sensitive and isinstance(pattern, list):
				pattern = str(spacy_rule_to_lower(pattern))
			cols.append(str(pattern))
		# Generate dataframe
		return pd.DataFrame(self.data, columns=cols)


