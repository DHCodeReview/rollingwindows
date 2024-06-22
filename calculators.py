"""calculators.py.

Last update: June 11 2024
"""
import re
from typing import Any, Iterable, List, Union

import pandas as pd
import spacy
from spacy.matcher import Matcher

from rollingwindows.helpers import (Windows, flatten, regex_escape,
                                    spacy_rule_to_lower)


class BaseCalculator:
	"""The base calculator class."""

	@property
	def metadata(self) -> dict:
		"""Return metadata for the calculator."""
		exclude = ["data", "nlp", "windows"]
		metadata = {"id": self.id}
		return metadata | dict(
			(key, getattr(self, key))
			for key in dir(self)
			if key not in exclude and key not in dir(self.__class__)
		)


class RWCalculator(BaseCalculator):
	"""A calculator for rolling windows."""
	id: str = "rw_calculator"

	def __init__(self,
				 *,
				 patterns: Union[List, str] = None,
				 windows: Windows = None,
				 mode: bool = "exact", # regex, spacy_matcher, multi_token, multi_token_exact
				 case_sensitive: bool = False,
				 alignment_mode: str = "strict",
				 model: str = "xx_sent_ud_sm",
				 original_doc: spacy.tokens.doc.Doc = None,
				 query: str = "counts"):
		"""Instantiate calculator.

		Args:
			patterns (Union[list, str]): A pattern or list of patterns to search in windows.
			windows (Windows): A Windows object containing the windows to search.
			mode (str): The search method to use.
			case_sensitive (bool): Whether to make searches case-sensitive.
			alignment_mode (str): Whether to snap searches to token boundaries. Values are
				"strict", "contract", and "expand".
			model (str): The language model to be used for searching spaCy tokens/spans.
			original_doc (spacy.tokens.doc.Doc): A spaCy Doc. The "re_finditer" method returns character
				start and end indexes in the window. Access to the doc from which the windows was
				generated is necessary to map these to the token indexes in order to use `alignment_mode`.
			query (str): The type of data to return: "averages", "counts", or "ratios".
		"""
		self.patterns = patterns
		self.windows = windows
		self.mode = mode
		self.case_sensitive = case_sensitive
		self.alignment_mode = alignment_mode
		self.model = model
		self.original_doc = original_doc
		self.nlp = spacy.load(model)
		self.query = query
		self.data = []

	@property
	def regex_flags(self):
		"""Return regex flags based on case_sensitive setting."""
		if not self.case_sensitive:
			return re.IGNORECASE | re.UNICODE
		else:
			return re.UNICODE

	def _assign_variable(self, var_name: str, var: Any) -> Any:
		"""Try to use configured values if not passed by public functions.

		Args:
			var_name: The name of the variable.
			var: The variable to be evaluated.

		Returns:
			Either the original value or the instance value.
		"""
		if var is None:
			var = getattr(self, var_name)
		else:
			setattr(self, var_name, var)
		if var is None:
			raise Exception(f"You must supply a value for {var_name}.")
		return var



	def _count_character_patterns_in_character_windows(self, window: str, pattern: str) -> int:
		"""Use Python count() to count exact character matches in a character window.

		Args:
			window (str): A string window.
			pattern (str): A string pattern to search for.

		Returns:
			The number of occurrences of the pattern in the window.
		"""
		if self.mode == "regex":
			return len(re.findall(pattern, window, self.regex_flags))
		else:
			if not self.case_sensitive:
				window = window.lower()
				pattern = pattern.lower()
			return window.count(pattern)

	def _count_in_character_window(self, window: str, pattern: str) -> int:
		"""Choose function for counting matches in character windows.
		Args:
			window (str): A string window.
			pattern (str): A string pattern to search for.

		Returns:
			The number of occurrences of the pattern in the window.
		"""
		if self.mode in ["exact", "regex"]:
			return self._count_character_patterns_in_character_windows(window, pattern)
		else:
			raise Exception("Invalid mode for character windows.")

	def _count_token_patterns_in_token_lists(self, window: List[str], pattern: str) -> int:
		"""Count patterns in lists of token strings.

		Args:
			window (List[str]): A window consisting of a list of strings.
			pattern (str): A string pattern to search for.

		Returns:
			The number of occurrences of the pattern in the window.
		"""
		if self.mode == "regex":
			return sum([len(re.findall(pattern, token, self.regex_flags)) for token in window])
		else:
			if not self.case_sensitive:
				window = [token.lower() for token in window]
				pattern = pattern.lower()
			return window.count(pattern)

	def _count_token_patterns_in_span(self, window: spacy.tokens.span.Span, pattern: Union[list, str]) -> int:
		"""Count patterns in spans or docs.

		Args:
			window (spacy.tokens.span.Span): A window consisting of a list of spaCy spans or a spaCy doc.
			pattern (Union[list, str]): A string pattern or spaCy rule to search for.

		Returns:
			The number of occurrences of the pattern in the window.
		"""
		if self.mode == "exact":
			if not self.case_sensitive:
				window = [token.lower_ for token in window]
				pattern = pattern.lower()
			else:
				window = [token.text for token in window]
			return window.count(pattern)
		elif self.mode == "regex":
			return sum([len(re.findall(pattern, token.text, self.regex_flags)) for token in window])
		elif self.mode == "spacy_rule":
			if not self.case_sensitive:
				pattern = spacy_rule_to_lower(pattern)
			matcher = Matcher(self.nlp.vocab)
			matcher.add("Pattern", [pattern])
			return len(matcher(window))

	def _count_token_patterns_in_span_text(self, window: str, pattern: str) -> int:
		"""Count patterns in span or doc text with token alignment.
		Args:
			window (str): A string window.
			pattern (str): A string pattern to search for.

		Returns:
			The number of occurrences of the pattern in the window.
		"""
		if not self.original_doc:
			raise Exception("You must supply an `original_doc` to use `multi_token` mode.")
		count = 0
		if self.mode == "multi_token_exact":
			pattern = regex_escape(pattern)
		for match in re.finditer(pattern, window, self.regex_flags):
			start, end = match.span()
			span = self.original_doc.char_span(start, end, self.alignment_mode)
			if span is not None:
				count += 1
		return count

	def _count_in_token_window(self, window: Union[List[str], spacy.tokens.span.Span], pattern: Union[list, str]) -> int:
		"""Choose function for counting matches in token windows.

		Args:
			window (Union[List[str], spacy.tokens.span.Span]): A window consisting of a list of token strings, a list of spaCy spans, or a spaCy doc.
			pattern (Union[list, str]): A string pattern or spaCy rule to search for.

		Returns:
			The number of occurrences of the pattern in the window.
		"""
		if isinstance(window, (list, str)):
			if self.mode in ["multi_token", "spacy_rule"]:
				raise Exception("You cannot use spaCy rule or perform multi-token searches with a string or list of token strings.")
			return self._count_token_patterns_in_token_lists(window, pattern)
		elif isinstance(window, spacy.tokens.span.Span):
			# Iterate over the full text with token boundary alignment
			if self.mode.startswith("multi_token"):
				return self._count_token_patterns_in_span_text(window.text, pattern)
			# Match in single tokens
			else:
				return self._count_token_patterns_in_span(window, pattern)

	def _extract_string_pattern(self, pattern: Union[dict, list, str]) -> str:
		"""Extract a string pattern from a spaCy rule.

		Args:
			pattern (Union[dict, list, str]): A pattern to search.

		Returns:
			str: A string pattern.
		"""
		return "|".join(
			[
				item if isinstance(item, str)
				else list(item.values())[0]
				for item in list(flatten(pattern))
			]
		)

	def _get_ratio(self, counts: List[int]) -> float:
		"""Calculate the ratio between two counts.

		Args:
			counts (List[int]): A list of two counts.

		Returns:
			The calculated ratio.
		"""
		numerator = counts[0]
		denominator = counts[1]
		# Handle division by 0
		if denominator + numerator == 0:
			return 0
		else:
			return numerator / (denominator + numerator)

	def _get_window_count(self, window: Union[List[str], spacy.tokens.span.Span, str], pattern: Union[list, str]) -> int:
		"""Call character or token window methods, as appropriate.

		Args:
			window (Union[List[str], spacy.tokens.span.Span, str]): A window consisting of a list of token strings, a list of spaCy spans, a spaCy doc, or a string.
			pattern (Union[list, str]): A string pattern or spaCy rule to search for.

		Returns:
			The number of occurrences of the pattern in the window.
		"""
		if self.window_units == "characters":
			return self._count_in_character_window(window, pattern)
		else:
			return self._count_in_token_window(window, pattern)

	def get_averages(self,
			windows: Iterable = None,
			patterns: Union[List, str] = None,
		) -> None:
		"""Run the calculator and return averages.

		Args:
			windows (Iterable): A Windows object.
			patterns (Union[List, str]): A string pattern or spaCy rule, or a list of either.
		"""
		self.run(windows, patterns, "averages")

	def get_counts(self,
			windows: Iterable = None,
			patterns: Union[List, str] = None,
		) -> None:
		"""Run the calculator and return counts.

		Args:
			windows (Iterable): A Windows object.
			patterns (Union[List, str]): A string pattern or spaCy rule, or a list of either.
		"""
		self.run(windows, patterns, "counts")

	def get_ratios(self,
			windows: Iterable,
			patterns: list,
		) -> None:
		"""Run the calculator and return counts.

		Args:
			windows (Iterable): A Windows object.
			patterns (list): A string pattern or spaCy rule, or a list of either.
		"""
		self.run(windows, patterns, "ratios")

	def run(self,
			windows: Iterable = None,
			patterns: Union[List, str] = None,
			query: str = "counts" # averages | ratios
		):
		"""Run the calculator.

		Args:
			windows (Iterable): A Windows object.
			patterns (Union[List, str]): A string pattern or spaCy rule, or a list of either.
			query (str): String designating whether to return "counts", "averages", or "ratios".
		"""
		for var in [("patterns", patterns), ("windows", windows), ("query", query)]:
			self._assign_variable(var[0], var[1])
		self.window_units = self.windows.window_units
		self.n = self.windows.n
		# print(f"Calculating {self.query} of {self.mode} matches for {self.patterns} in windows of {self.n} {self.window_units}...")
		if self.query == "averages":
			self.data = [
				[self._get_window_count(window, pattern) / self.n for pattern in self.patterns]
				for window in self.windows
			]
		elif self.query == "counts":
			self.data = [
				[self._get_window_count(window, pattern) for pattern in self.patterns]
				for window in self.windows
			]
		elif self.query == "ratios":
			if not isinstance(patterns, list):
				raise Exception("You must supply a list of two patterns to calculate ratios.")
			if len(patterns) != 2:
				raise Exception("You can only calculate ratios for two patterns.")
			self.data = [
				self._get_ratio([self._get_window_count(window, pattern) for pattern in self.patterns])
				for window in self.windows
			]
		else:
			raise Exception("Invalid query type.")

	def to_df(self, show_spacy_rules: bool = False) -> pd.DataFrame:
		"""Convert the data to a pandas dataframe.

		Args:
			show_spacy_rules (bool): If True, use full spaCy rules for labels; otherwise use only the
			string pattern.

		Returns:
			pd.DataFrame: A pandas DataFrame.
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
		# Merge columns for ratios
		if self.query == "ratios":
			cols = [":".join(cols)]
		# Generate dataframe
		return pd.DataFrame(self.data, columns=cols)
		return pd.DataFrame(self.data, columns=cols)
