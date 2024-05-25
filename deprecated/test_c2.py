"""test_c2.py.

Based on test_simple.py. As of 15 May 2024 this has 100% coverage for __init__.py. However, it could use some
cleaning up and some better mocking.
"""

# TODO: To test, I need to test each method as a function make sure that
# TODO: class attributes are provided as fixtures.

import re
from enum import Enum
from typing import Protocol, runtime_checkable
from unittest.mock import MagicMock, PropertyMock, patch

import pandas as pd
import pytest
import spacy

from rollingwindows.calculators import (Averages, Calculator,
                                        is_valid_spacy_rule,
                                        spacy_rule_to_lower)


class Windows(Protocol):
	...

@pytest.fixture
def nlp():
	return spacy.load("en_core_web_sm")


@pytest.fixture
def generator_type():
	return type(1 for i in "")


@pytest.fixture
def doc(nlp):
	return nlp("This is a test document. It has multiple sentences.")


@pytest.fixture
def sliding_windows_generator():
	"""Create a generator to test the Windows class."""
	return (c for c in ["a", "b", "c", "d", "e"])


@pytest.fixture
def mock_windows(sliding_windows_generator):
	class MockWindows:
		def __init__(self):
			self.windows = sliding_windows_generator
			self.window_units = "characters"
			self.n = 5
			self.alignment_mode = "strict"

	return MockWindows()

@pytest.fixture
def test_calculator():
	class TestCalculator(Calculator):
		id = "test_calculator"

	return TestCalculator()

@pytest.fixture
def mock_averages(sliding_windows_generator):
	class TestAverages(Calculator):
		id = "test_averages"

	return TestAverages()


# Functions

def test_is_valid_spacy_rule_with_valid_rule(nlp):
	pattern = [{"LOWER": "test"}]
	vocab = nlp.vocab
	assert is_valid_spacy_rule(pattern, vocab) == True

def test_is_valid_spacy_rule_with_invalid_rule(nlp):
	pattern = [{"INVALID": "rule"}]
	vocab = nlp.vocab
	assert is_valid_spacy_rule(pattern, vocab) == False

def test_spacy_rule_to_lower_with_dict_and_default_parameters():
	pattern = {"TEXT": "Hello", "ORTH": "World"}
	new_pattern = spacy_rule_to_lower(pattern)
	assert new_pattern == {"LOWER": "Hello", "LOWER": "World"}


def test_spacy_rule_to_lower_with_list_and_default_parameters():
	patterns = [{"TEXT": "Hello"}, {"ORTH": "World"}]
	new_patterns = spacy_rule_to_lower(patterns)
	assert new_patterns == [{"LOWER": "Hello"}, {"LOWER": "World"}]


def test_spacy_rule_to_lower_with_dict_and_custom_parameters():
	pattern = {"TEXT": "Hello", "ORTH": "World"}
	new_pattern = spacy_rule_to_lower(pattern, old_key="TEXT", new_key="LOWER")
	assert new_pattern == {"LOWER": "Hello", "ORTH": "World"}


def test_spacy_rule_to_lower_with_list_and_custom_parameters():
	patterns = [{"TEXT": "Hello"}, {"ORTH": "World"}]
	new_patterns = spacy_rule_to_lower(patterns, old_key="TEXT", new_key="LOWER")
	assert new_patterns == [{"LOWER": "Hello"}, {"LOWER": "World"}]

def test_calculator_protocol_metadata(mock_averages, mock_windows):
	calculator = mock_averages
	calculator.patterns = ["a", "b"]
	calculator.windows = mock_windows
	calculator.window_units = mock_windows.window_units
	calculator.n = mock_windows.n
	calculator.alignment_mode = mock_windows.alignment_mode
	assert isinstance(calculator.metadata, dict)

@pytest.mark.xfail(raises=NotImplementedError)
def test_calculator_protocol_run_method(test_calculator):
	test_calculator.run()

@pytest.mark.xfail(raises=NotImplementedError)
def test_calculator_protocol_to_df_method(test_calculator):
	test_calculator.to_df()

@runtime_checkable
class CaseSensitiveProtocol(Protocol):
	@property
	def case_sensitive(self) -> bool: ...

def test_regex_flags_when_case_sensitive_is_false():
	calculator = MagicMock(spec=CaseSensitiveProtocol)
	type(calculator).case_sensitive = PropertyMock(return_value=False)
	assert Averages.regex_flags.fget(calculator) == re.IGNORECASE | re.UNICODE

def test_regex_flags_when_case_sensitive_is_true():
	calculator = MagicMock(spec=CaseSensitiveProtocol)
	type(calculator).case_sensitive = PropertyMock(return_value=True)
	assert Averages.regex_flags.fget(calculator) == re.UNICODE

def test_averages_with_custom_parameters(mock_windows):
	averages = Averages(
		["test", "document"],
		mock_windows,
		search_method="re_search",
		alignment_mode="contract",
		regex=True,
		case_sensitive=False,
		use_span_text=True,
	)
	assert averages.patterns == ["test", "document"]
	assert averages.search_method == "find"
	assert averages.alignment_mode == "contract"
	assert averages.regex == True
	assert averages.case_sensitive == False
	assert averages.use_span_text == True
	assert isinstance(averages.windows, Windows)
