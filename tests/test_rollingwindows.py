"""test_rollingwindows.py.

As of 12 June 2024 this has 98% coverage for __init__.py. However, it could use some
cleaning up and some better mocking.
"""

import re
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import catalogue
import pandas as pd
import pytest
import spacy

from rollingwindows import (RollingWindows, Windows, calculators, filters,
                            get_rw_component, plotters, sliding_str_windows,
                            sliding_windows)

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
def test_filter():
	class TestFilter(filters.Filter):
		id = "test_filter"
		name = "TestFilter"
		doc = "Test document"

	return TestFilter()

@pytest.fixture
def mock_counts():
	calculator = calculators.RWCalculator(
		patterns=["love"],
		mode="exact",
		query="counts"
	)
	return calculator

@pytest.fixture
def plotter():
	return plotters.RWSimplePlotter()

@pytest.fixture
def doc_sliding_str(nlp):
	return nlp("This is a test document for sliding window functionality.")

@pytest.fixture
def spacy_sents(nlp):
	text = "When shall we three meet again? In thunder, lightning, or in rain? When the hurly-burly’s done, when the battle’s lost and won. That will be ere the set of sun. Where the place? Upon the heath. There to meet with Macbeth."
	return nlp(text).sents

# Test RW Components
def test_get_rw_component_with_valid_id():
	component_id = "rw_calculator"
	component = get_rw_component(component_id)
	assert component is not None


def test_get_rw_component_with_invalid_id():
	with pytest.raises(catalogue.RegistryError):
		get_rw_component("invalid_component_id")


# Filter Protocol


def test_filter_protocol(test_filter):
	metadata = test_filter.metadata
	assert metadata["id"] == "test_filter"
	assert "doc" not in metadata


@pytest.mark.xfail(raises=NotImplementedError)
def test_filter_protocol_apply_method(test_filter):
	test_filter.apply()

# Sliding Windows Function


def windows_list_has_strings_or_spans(windows_list):
	span = spacy.tokens.span.Span
	for window in windows_list:
		if (
			not isinstance(window, str)
			and not isinstance(window, span)
			and not isinstance(window, list)
		):
			return False
		else:
			if isinstance(window, list) and not all(
				isinstance(x, (str, spacy.tokens.span.Span)) for x in window
			):
				return False
	return True


def windows_list_has_strings(windows_list):
	for window in windows_list:
		if not isinstance(window, str):
			return False
		else:
				return True

def test_sliding_windows_with_sentences(spans, generator_type, n=2):
    spans = list(spans)
    windows = sliding_windows(spans, n=n, window_units="sentences")
    assert type(windows) == generator_type
    windows_list = list(windows)
    assert len(windows_list) == len(spans)
    assert len(windows_list[0]) == n
    assert windows_list_has_strings_or_spans(windows_list) == True


def test_sliding_windows_with_spacy_doc(doc, generator_type, n=10):
    windows = sliding_windows(doc, n=n, window_units="characters")
    assert type(windows) == generator_type
    windows_list = list(windows)
    assert len(windows_list) == len(doc.text)
    assert len(windows_list[0]) == n
    assert windows_list_has_strings_or_spans(windows_list) == True


def test_sliding_windows_with_different_window_sizes(doc, generator_type, n=5):
    windows = sliding_windows(doc, n=n, window_units="characters")
    assert type(windows) == generator_type
    windows_list = list(windows)
    assert len(windows_list) == len(doc.text)
    assert len(windows_list[0]) == n
    assert windows_list_has_strings_or_spans(windows_list) == True


def test_sliding_windows_with_tokens(doc, generator_type, n=5):
    windows = sliding_windows(doc, n=n, window_units="tokens")
    assert type(windows) == generator_type
    windows_list = list(windows)
    assert len(windows_list) == len(doc)
    assert len(windows_list[0]) == n
    assert windows_list_has_strings_or_spans(windows_list) == True


def test_sliding_windows_with_invalid_window_units(doc):
    with pytest.raises(Exception):
        list(sliding_windows(doc, n=10, window_units="invalid"))


def test_sliding_windows_with_different_alignment_modes(doc, generator_type, n=5):
    windows = sliding_str_windows(doc, n=n, alignment_mode="expand")
    assert type(windows) == generator_type
    windows_list = list(windows)
    # Note: There is no way to predict the number or length of the windows
    assert windows_list_has_strings(windows_list) == True
    windows = sliding_str_windows(doc, n=n, alignment_mode="contract")
    windows_list = list(windows)
    assert windows_list_has_strings(windows_list) == True

def test_sliding_str_windows_with_str(nlp):
    input_str = "This is a test string for sliding windows."
    doc = nlp(input_str)
    windows_size = 10
    expected_windows = ['This is a test string for sliding windows.', 'is a test string for sliding windows.', 'a test string for sliding windows.', 'test string for sliding windows.', 'string for sliding windows.', 'for sliding windows.', 'sliding windows.', 'windows.', '.', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '']
    windows = list(sliding_str_windows(doc, n=windows_size, alignment_mode="strict"))
    assert windows == expected_windows, "The windows generated from a string do not match the expected output."

def test_sliding_str_windows_with_spacy_doc(doc_sliding_str):
    windows_size = 50
    # Expected windows might need adjustment based on the actual spaCy tokenization and the chosen window size
    expected_windows = ['This is a test document for sliding window functionality', 'This is a test document for sliding window functionality', 'This is a test document for sliding window functionality', 'This is a test document for sliding window functionality', 'is a test document for sliding window functionality', 'is a test document for sliding window functionality', 'is a test document for sliding window functionality', 'a test document for sliding window functionality.']
    windows = list(sliding_str_windows(doc_sliding_str, n=windows_size, alignment_mode="expand"))
    assert windows == expected_windows

def test_sliding_str_windows_with_spacy_spans(spacy_sents):
	windows_size = 3
	expected_windows = ["When shall we three meet again? In thunder, lightning, or in rain? When the hurly-burly’s done, when the battle’s lost and won.", "In thunder, lightning, or in rain? When the hurly-burly’s done, when the battle’s lost and won. That will be ere the set of sun.", "When the hurly-burly’s done, when the battle’s lost and won. That will be ere the set of sun. Where the place?", "That will be ere the set of sun. Where the place? Upon the heath.", "Where the place? Upon the heath. There to meet with Macbeth.", "Upon the heath. There to meet with Macbeth.", "There to meet with Macbeth."]
	windows = sliding_str_windows(list(spacy_sents), n=windows_size, alignment_mode="expand")
	print(list(windows))
	for i, window in list(windows):
		assert window == expected_windows[i]

# Windows Class


def test_windows_class():
    windows = Windows(sliding_windows_generator, "characters", 3)
    assert windows.window_units == "characters"
    assert windows.n == 3
    assert windows.alignment_mode == "strict"


def test_windows_class_with_different_alignment_mode():
    windows = Windows(sliding_windows_generator, "characters", 3, "expand")
    assert windows.alignment_mode == "expand"


def test_windows_class_iter(sliding_windows_generator, generator_type):
    windows = Windows(sliding_windows_generator, "characters", 3)
    assert type(windows.windows) == generator_type
    for window in windows:
        assert window is not None


# Rolling Windows Class


def test_rolling_windows_class_initialization(doc):
    rolling_windows = RollingWindows(doc, "en_core_web_sm")
    assert rolling_windows.doc == doc
    assert rolling_windows.metadata["model"] == "en_core_web_sm"
    assert rolling_windows.patterns == []


def test_rolling_windows_class_with_patterns(doc):
    patterns = ["pattern1", "pattern2"]
    rolling_windows = RollingWindows(doc, "en_core_web_sm", patterns=patterns)
    assert rolling_windows.patterns == patterns


def test_rolling_windows_class_without_patterns(doc):
    rolling_windows = RollingWindows(doc, "en_core_web_sm")
    assert rolling_windows.patterns == []


def test_get_search_method_with_valid_inputs():
    rolling_windows = RollingWindows(doc, "en_core_web_sm")
    assert rolling_windows._get_search_method("characters") == "count"
    assert rolling_windows._get_search_method("tokens") == "spacy_matcher"
    assert rolling_windows._get_search_method("lines") == "spacy_matcher"
    assert rolling_windows._get_search_method("sentences") == "spacy_matcher"


def test_get_search_method_with_invalid__or_no_input():
    rolling_windows = RollingWindows(doc, "en_core_web_sm")
    assert rolling_windows._get_search_method() == "re_finditer"
    assert rolling_windows._get_search_method("invalid") == "re_finditer"


def test_get_units_with_sentences(doc):
    rolling_windows = RollingWindows(doc, "en_core_web_sm")
    units = rolling_windows._get_units(doc, "sentences")
    assert len(units) == 2
    assert all(isinstance(unit, spacy.tokens.span.Span) for unit in units)


def test_get_units_with_lines(nlp):
    doc = nlp("This is a test document.\nIt has multiple lines.")
    rolling_windows = RollingWindows(doc, "en_core_web_sm")
    units = rolling_windows._get_units(doc, "lines")
    assert len(units) == 2
    assert all(isinstance(unit, spacy.tokens.span.Span) for unit in units)


def test_get_units_with_other(doc):
    rolling_windows = RollingWindows(doc, "en_core_web_sm")
    units = rolling_windows._get_units(doc, "tokens")
    assert isinstance(units, spacy.tokens.doc.Doc)

def test_get_units_with_no_windows(doc):
    rolling_windows = RollingWindows(doc, "en_core_web_sm")
    units = rolling_windows._get_units(doc, "tokens")
    assert isinstance(units, spacy.tokens.doc.Doc)

# Set Windows


def test_set_windows_with_filter_id(doc):
    rolling_windows = RollingWindows(doc, "en_core_web_sm")
    rolling_windows.set_windows(filter="word_filter")
    assert rolling_windows.metadata["filter"]["id"] == "word_filter"


def test_set_windows_with_filter_object(doc):
    rolling_windows = RollingWindows(doc, "en_core_web_sm")
    word_filter = filters.WordFilter(rolling_windows.doc, exclude_digits=True)
    rolling_windows.set_windows(filter=word_filter)
    assert rolling_windows.metadata["filter"]["id"] == "word_filter"
    assert rolling_windows.metadata["filter"]["exclude_digits"] == True


def test_set_windows_without_filter(doc):
    rolling_windows = RollingWindows(doc, "en_core_web_sm")
    rolling_windows.set_windows()
    assert "filter" not in rolling_windows.metadata


def test_set_windows_with_different_parameters(doc):
    rolling_windows = RollingWindows(doc, "en_core_web_sm")
    rolling_windows.set_windows(n=5, window_units="lines", alignment_mode="expand")
    assert rolling_windows.metadata["n"] == 5
    assert rolling_windows.metadata["window_units"] == "lines"
    assert rolling_windows.metadata["alignment_mode"] == "expand"
    assert rolling_windows.metadata["search_method"] in [
        "count",
        "spacy_matcher",
        "re_finditer",
    ]


def test_calculate_with_callable_calculator(doc, mock_counts):
    rolling_windows = RollingWindows(doc, "en_core_web_sm")
    rolling_windows.set_windows(n=5, window_units="characters")
    mock_counts.windows = rolling_windows.windows
    rolling_windows.calculate(calculator=mock_counts)
    assert rolling_windows.metadata["calculator"] == mock_counts.metadata
    assert isinstance(rolling_windows.result, pd.DataFrame)


def test_calculate_with_string_calculator(doc):
    rolling_windows = RollingWindows(doc, model="en_core_web_sm")
    rolling_windows.set_windows(n=5, window_units="characters")
    calculator = MagicMock()
    calculator.metadata = {"id": "rw_calculator"}
    with patch(
        "rollingwindows.get_rw_component",
        return_value=lambda *args, **kwargs: calculator,
    ):
        rolling_windows.calculate("rw_calculator")
    assert rolling_windows.metadata["calculator"]["id"] == "rw_calculator"
    assert rolling_windows.result is not None


def test_calculate_with_no_windows_set():
    rolling_windows = RollingWindows(doc, model="en_core_web_sm")
    calculator = MagicMock()
    calculator.metadata = {"id": "rw_calculator"}
    with patch(
        "rollingwindows.get_rw_component",
        return_value=lambda *args, **kwargs: calculator,
    ):
        with pytest.raises(
            Exception,
            match=re.escape("You must call set_windows() before running calculations."),
        ):
            rolling_windows.calculate("rw_calculator")


def test_calculate_with_invalid_calculator():
    rolling_windows = RollingWindows(doc, model="en_core_web_sm")
    rolling_windows.set_windows(n=5, window_units="characters")
    with patch("rollingwindows.get_rw_component", side_effect=catalogue.RegistryError):
        with pytest.raises(catalogue.RegistryError):
            rolling_windows.calculate("invalid_calculator")


def test_plot_with_callable_plotter(doc, plotter):
    rolling_windows = RollingWindows(doc, "en_core_web_sm")
    rolling_windows.result = pd.DataFrame([1], columns=["test"])
    rolling_windows.plot(plotter=plotter)
    assert rolling_windows.metadata["plotter"] == plotter.metadata


def test_calculate_with_string_plotter(doc):
    rolling_windows = RollingWindows(doc, model="en_core_web_sm")
    rolling_windows.result = pd.DataFrame([1], columns=["test"])
    plotter = MagicMock()
    plotter.metadata = {"id": "rw_simple_plotter"}
    with patch(
        "rollingwindows.get_rw_component", return_value=lambda *args, **kwargs: plotter
    ):
        rolling_windows.plot()
    assert rolling_windows.metadata["plotter"]["id"] == "rw_simple_plotter"
    assert rolling_windows.plot is not None


def test_plot_with_invalid_plotter(doc):
    rolling_windows = RollingWindows(doc, "en_core_web_sm")
    rolling_windows.result = pd.DataFrame([1], columns=["test"])
    with pytest.raises(Exception):
        rolling_windows.plot(plotter="invalid_plotter")


def test_plot_with_file(doc, plotter):
    rolling_windows = RollingWindows(doc, "en_core_web_sm")
    rolling_windows.result = pd.DataFrame([1], columns=["test"])
    with tempfile.TemporaryDirectory() as tmp_path:
        file = f"{tmp_path}/plot.png"
        rolling_windows.plot(plotter=plotter, file=file)
        assert Path(file).is_file()


def test_plot_with_valid_plotter_and_show_and_file_false(doc):
    rolling_windows = RollingWindows(doc, "en_core_web_sm")
    rolling_windows.result = pd.DataFrame([1], columns=["test"])
    plotter = MagicMock()
    plotter.metadata = {"id": "rw_simple_plotter"}
    with patch(
        "rollingwindows.get_rw_component", return_value=lambda *args, **kwargs: plotter
    ):
        rolling_windows.plot("rw_simple_plotter")
    assert rolling_windows.metadata["plotter"]["id"] == "rw_simple_plotter"
    assert rolling_windows.fig is not None


def test_plot_with_valid_plotter_and_show_true_and_file_false(doc):
    rolling_windows = RollingWindows(doc, "en_core_web_sm")
    rolling_windows.result = pd.DataFrame([1], columns=["test"])
    plotter = MagicMock()
    plotter.metadata = {"id": "rw_simple_plotter"}
    with patch(
        "rollingwindows.get_rw_component", return_value=lambda *args, **kwargs: plotter
    ) as mock_show:
        rolling_windows.plot("rw_simple_plotter", show=True)
    assert rolling_windows.metadata["plotter"]["id"] == "rw_simple_plotter"
    assert rolling_windows.fig is not None
    mock_show.assert_called_once()


def test_plot_with_valid_plotter_and_show_false_and_file_valid(doc):
    rolling_windows = RollingWindows(doc, "en_core_web_sm")
    rolling_windows.result = pd.DataFrame([1], columns=["test"])
    plotter = MagicMock()
    plotter.metadata = {"id": "rw_simple_plotter"}
    with patch(
        "rollingwindows.get_rw_component", return_value=lambda *args, **kwargs: plotter
    ):
        rolling_windows.plot("rw_simple_plotter", file="test.png")
    assert rolling_windows.metadata["plotter"]["id"] == "rw_simple_plotter"
    assert rolling_windows.fig is not None


def test_plot_with_no_result_set(doc):
    rolling_windows = RollingWindows(doc, "en_core_web_sm")
    plotter = MagicMock()
    plotter.metadata = {"id": "rw_simple_plotter"}
    with pytest.raises(
        Exception,
        match="You must run a calculator on your data before generating a plot.",
    ):
        rolling_windows.plot("rw_simple_plotter")
