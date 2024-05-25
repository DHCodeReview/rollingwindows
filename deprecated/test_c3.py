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

# from rollingwindows import Windows
# from rollingwindows import (
# 	RollingWindows,
# 	Windows,
# 	calculators,
# 	filters,
# 	get_rw_component,
# 	plotters,
# 	sliding_windows,
# )
from rollingwindows.calculators import (  # , Windows, is_valid_spacy_rule, spacy_rule_to_lower
    Averages, Calculator)

# Fixtures

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
def test_averages():

	class TestAverages(Averages):
		id = "test_averages"

	return TestAverages("a", [])


# Functions

def test_calculator_protocol_metadata(test_calculator):
	assert test_calculator.metadata["id"] == "test_calculator"

@pytest.mark.xfail(raises=NotImplementedError)
def test_calculator_protocol_run_method(test_calculator):
	test_calculator.run()

@pytest.mark.xfail(raises=NotImplementedError)
def test_calculator_protocol_to_df_method(test_calculator):
	test_calculator.to_df()


# def test_calculate_with_callable_calculator(doc, mock_averages):
# 	rolling_windows = RollingWindows(doc, "en_core_web_sm")
# 	rolling_windows.set_windows(n=5, window_units="characters")
# 	rolling_windows.calculate(calculator=mock_averages)
# 	assert rolling_windows.metadata["calculator"] == mock_averages.metadata
# 	assert isinstance(rolling_windows.result, pd.DataFrame)


# def test_calculate_with_string_calculator(doc):
# 	rolling_windows = RollingWindows(doc, model="en_core_web_sm")
# 	rolling_windows.set_windows(n=5, window_units="characters")
# 	calculator = MagicMock(patterns=["a"], windows=[])
# 	# calculator.metadata = {"id": "averages"}
# 	with patch(
# 		"rollingwindows.calculators.Averages",
# 		return_value="Dunca",
# 	):
# 		# rolling_windows.calculate("averages")
# 		print(calculator.metadata)
# 		assert 1==2
# 		assert rolling_windows.metadata["calculator"]["id"] == "averages"

def test_averages_class():
	with patch('rollingwindows.calculators.Averages') as MockClass:
		instance = MockClass.return_value
		instance.id = "test_averages"
		instance.patterns = "a"
		instance.metadata = {"id": instance.id}
		print(instance.metadata)
		# instance.run()
		print(dir(instance))
		assert 1==2
		# instance.metadata.return_value = 'foo'
		# assert Averages() is instance
		# assert Averages().metadata == 'foo'


@runtime_checkable
class WindowsProtocol(Protocol):
	window_units: str
	n: int

@runtime_checkable
class CalculatorProtocol(Protocol):
	...

# def test_averages_initialization_with_minimal_arguments():
# 	windows = MagicMock(spec=WindowsProtocol)
# 	windows.window_units = "tokens"
# 	windows.n = 5
# 	averages = MagicMock(spec=Averages, patterns=["test"], windows=windows)
# 	assert averages.patterns == ["test"]
# 	assert averages.windows == windows
# 	assert averages.windows.window_units == "tokens"
# 	assert averages.windows.n == 5
# 	# assert averages.search_method == "count"
# 	# assert averages.alignment_mode == "strict"
# 	# assert averages.regex == False
# 	# assert averages.case_sensitive == True
# 	# assert averages.use_span_text == False
# 	# assert averages.doc == None
# 	# assert averages.data == []
# 	print(averages.metadata)
# 	assert 1 == 2

# def test_averages_initialization_with_all_arguments():
#     windows = MagicMock(spec=WindowsProtocol)
#     windows.window_units = "tokens"
#     windows.n = 5
#     averages = Averages("test", windows, search_method="find", model="en_core_web_sm", alignment_mode="contract", regex=True, case_sensitive=False, use_span_text=True)
#     assert averages.patterns == ["test"]
#     assert averages.windows == windows
#     assert averages.window_units == "tokens"
#     assert averages.n == 5
#     assert averages.search_method == "find"
#     assert isinstance(averages.nlp, spacy.language.Language)
#     assert averages.alignment_mode == "contract"
#     assert averages.regex == True
#     assert averages.case_sensitive == False
#     assert averages.use_span_text == True
#     assert averages.doc == None
#     assert averages.data == []

# def test_averages_initialization_with_model_argument():
#     windows = MagicMock(spec=WindowsProtocol)
#     windows.window_units = "tokens"
#     windows.n = 5
#     averages = Averages("test", windows, model="en_core_web_sm")
#     assert isinstance(averages.nlp, spacy.language.Language)
