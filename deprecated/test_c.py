"""test_c.py.

Based on test_simple.py. As of 15 May 2024 this has 100% coverage for __init__.py. However, it could use some
cleaning up and some better mocking.
"""
import re
import tempfile
from enum import Enum
from pathlib import Path
from typing import Iterable, Protocol, runtime_checkable
from unittest.mock import MagicMock, PropertyMock, patch

import catalogue
import pandas as pd
import pytest
import spacy

from rollingwindows import (RollingWindows, calculators, filters,  # Windows,
                            get_rw_component, plotters, sliding_windows)
from rollingwindows.calculators import is_valid_spacy_rule, validate_config

# Fixtures


@pytest.fixture
def nlp():
	return spacy.load("en_core_web_sm")


@pytest.fixture
def text(nlp):
	return nlp("This is a test document. It has multiple sentences.")


# Fixtures


@pytest.fixture
def generator_type():
	return type(1 for i in "")


@pytest.fixture
def nlp():
	return spacy.load("en_core_web_sm")


@pytest.fixture
def doc(nlp):
	return nlp("This is a test document. It has multiple sentences.")


@pytest.fixture
def spans(doc):
	return list(doc.sents)


@pytest.fixture
def sliding_windows_generator():
	"""Create a generator to test the Windows class."""
	return (c for c in ["a", "b", "c", "d", "e"])


@pytest.fixture
def test_calculator():
	class TestCalculator(calculators.Calculator):
		id = "test_calculator"

	return TestCalculator()

@pytest.fixture
def mock_averages():
	class MockAverages:
		def __init__(self, patterns, windows):
			self.id = "mock_averages"
			self.patterns = patterns
			self.windows = windows
			self.search_method = "count",
			self.model = None,
			self.doc = None,
			self.alignment_mode = "strict",
			self.regex = False,
			self.case_sensitive = True,
			self.use_span_text = False,

		@property
		def metadata(self) -> dict:
			exclude = ["data", "windows"]
			metadata = {"id": self.id}
			return metadata | dict(
				(key, getattr(self, key))
				for key in dir(self)
				if key not in exclude and key not in dir(self.__class__)
			)

		@property
		def regex_flags(self) -> Enum:
			"""Return regex flags based on case_sensitive setting."""
			if not self.case_sensitive:
				return re.IGNORECASE | re.UNICODE
			else:
				return re.UNICODE

		def run(self):
			print("Running...")

		def to_df(self, **kwargs):
			return pd.DataFrame()

	return MockAverages("a", (1 for x in ""))


# Averages Properties

def test_mock_averages_metadata(mock_averages):
	assert isinstance(mock_averages.metadata, dict)
	assert mock_averages.metadata["id"] == "mock_averages"

def test_mock_averages_regex_flags_case_sensitive_is_true(mock_averages):
	assert str(mock_averages.regex_flags) == "re.UNICODE"

def test_mock_averages_regex_flags_case_sensitive_is_false(mock_averages):
	mock_averages.case_sensitive = False
	assert str(mock_averages.regex_flags) == "re.IGNORECASE|re.UNICODE"

# Averages Functions

@runtime_checkable
class WindowsProtocol(Protocol):
	window_units: str

def test_validate_config_with_valid_arguments(mock_averages):
	nlp = spacy.load("en_core_web_sm")
	validate_config("a", [], "count", "nlp")

# def test_validate_config_with_invalid_windows(mock_averages):
# 	windows = 3.27
# 	nlp = spacy.load("en_core_web_sm")
# 	with pytest.raises(Exception):
# 		validate_config(mock_averages.patterns, mock_averages.windows, "count", nlp)

# def test_validate_config_with_invalid_patterns_for_count_search_method():
#     windows = MagicMock(spec=WindowsProtocol)
#     windows.window_units = "tokens"
#     patterns = [123]
#     search_method = "count"
#     nlp = spacy.load("en_core_web_sm")
#     with pytest.raises(Exception, match="One or more patterns is not a valid string, which is required for the `count` search_method."):
#         validate_config(patterns, windows, search_method, nlp)

# def test_validate_config_with_spacy_matcher_search_method_and_character_windows():
#     windows = MagicMock(spec=WindowsProtocol)
#     windows.window_units = "characters"
#     patterns = ["test"]
#     search_method = "spacy_matcher"
#     nlp = spacy.load("en_core_web_sm")
#     with pytest.raises(Exception, match="You cannot use the `spacy_matcher` method to search character windows."):
#         validate_config(patterns, windows, search_method, nlp)

# def test_validate_config_with_spacy_matcher_search_method_and_invalid_patterns():
#     windows = MagicMock(spec=WindowsProtocol)
#     windows.window_units = "tokens"
#     patterns = ["invalid"]
#     search_method = "spacy_matcher"
#     nlp = spacy.load("en_core_web_sm")
#     with pytest.raises(Exception, match="invalid is not a valid spaCy rule."):
#         validate_config(patterns, windows, search_method, nlp)