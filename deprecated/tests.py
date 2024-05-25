"""
Here is a plan for the unit tests:
"""
# TODO: Just test with the Protocol classes here.
# Create separate tests for individual filters, calculators, plotters.
# This will require instances as fixtures to use as the result of these classes.


import pandas as pd
import pytest
import spacy

from rollingwindows import (RollingWindows, Windows, calculators, filters,
                            plotters, sliding_windows)

# Fixtures

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
    """Create a generator to test the Windows clas."""
    return (c for c in ["a", "b", "c", "d", "e"])

@pytest.fixture
def test_filter():
    class TestFilter(filters.Filter):
        id = 1
        name = "TestFilter"
        doc = "Test document"

    return TestFilter()

@pytest.fixture
def test_calculator():
    class TestCalculator(calculators.TestCalculator):
        id = "test_calculator"

    return TestCalculator()

# Filter Protocol

def test_filter_protocol(test_filter):
    metadata = test_filter.metadata
    assert metadata["id"] == "test_filter"
    assert metadata["name"] == "TestFilter"
    assert "doc" not in metadata
    with pytest.raises(NotImplementedError):
        test_filter.apply()

# Calculator Protocol

def test_calcutor_protocol(test_calculator):
    metadata = test_calculator.metadata
    assert metadata["id"] == 1
    assert metadata["name"] == "TestCalculator"
    assert "doc" not in metadata
    with pytest.raises(NotImplementedError):
        test_calculator.apply()

# Sliding Windows Function

def test_sliding_windows_with_spacy_spans(spans):
    """Test `sliding_windows` function with spans as input."""
    windows = list(sliding_windows(spans, n=2, window_units="spans"))
    assert len(windows) == len(spans) - 1

def test_sliding_windows_with_spacy_doc(doc):
    """Test `sliding_windows` function with spaCy doc as input."""
    windows = list(sliding_windows(doc, n=10, window_units="characters"))
    assert len(windows) == len(doc.text) - 9

def test_sliding_windows_with_different_window_sizes(doc):
    """Test `sliding_windows` function with different window sizes."""
    windows = list(sliding_windows(doc, n=5, window_units="characters"))
    assert len(windows) == len(doc.text) - 4

def test_sliding_windows_with_different_window_units(doc):
    """Test `sliding_windows` function with sentence window_units."""
    windows = list(sliding_windows(doc, n=1, window_units="sentences"))
    assert len(windows) == len(list(doc.sents)) - 0

def test_sliding_windows_with_different_alignment_modes(doc):
    """Test `sliding_windows` function with different alignment modes."""
    windows = list(sliding_windows(doc, n=10, window_units="characters", alignment_mode="expand"))
    assert len(windows) == len(doc.text) - 9

def test_sliding_windows_with_invalid_window_units(doc):
    """Test `sliding_windows` function with invalid window_units."""
    with pytest.raises(Exception):
        list(sliding_windows(doc, n=10, window_units="invalid"))

# Windows Class

def test_windows_class():
    windows = Windows(sliding_windows_generator, "characters", 3)
    assert windows.window_units == "characters"
    assert windows.n == 3
    assert windows.alignment_mode == "strict"

def test_windows_class_with_different_alignment_mode():
    windows = Windows(sliding_windows_generator, "characters", 3, "expand")
    assert windows.alignment_mode == "expand"

def test_windows_class_iter():
    """Test whether a `Windows` instance contains a generator."""
    windows = Windows(sliding_windows_generator, "characters", 3)
    generator_type = type(1 for i in "")
    assert type(windows) == generator_type

# Rolling Windows Class

def test_rolling_windows_class_initialization(doc):
    """Test the `RollingWindows` class initialisation."""
    rolling_windows = RollingWindows(doc, "en_core_web_sm")
    assert rolling_windows.doc == doc
    assert rolling_windows.metadata["model"] == "en_core_web_sm"
    assert rolling_windows.patterns == []

def test_rolling_windows_class_with_patterns(doc):
    """Test the `RollingWindows` class initialisation with patterns."""
    patterns = ["pattern1", "pattern2"]
    rolling_windows = RollingWindows(doc, "en_core_web_sm", patterns=patterns)
    assert rolling_windows.patterns == patterns

def test_rolling_windows_class_without_patterns(doc):
    """Test the `RollingWindows` class initialisation without patterns."""
    rolling_windows = RollingWindows(doc, "en_core_web_sm")
    assert rolling_windows.patterns == []

def test_get_search_method_with_valid_inputs():
    """Test _get_search_method() with valid inputs()."""
    rolling_windows = RollingWindows(doc, "en_core_web_sm")
    assert rolling_windows._get_search_method("characters") == "count"
    assert rolling_windows._get_search_method("tokens") == "spacy_matcher"
    assert rolling_windows._get_search_method("lines") == "spacy_matcher"
    assert rolling_windows._get_search_method("sentences") == "spacy_matcher"

def test_get_search_method_with_invalid__or_no_input():
    """Test _get_search_method() with valid inputs()."""
    rolling_windows = RollingWindows(doc, "en_core_web_sm")
    assert rolling_windows._get_search_method() == "re_finditer"
    assert rolling_windows._get_search_method("invalid") == "re_finditer"

def test_get_units_with_sentences(doc):
    rolling_windows = RollingWindows(doc, "en_core_web_sm")
    units = rolling_windows._get_units(doc, "sentences")
    assert len(units) == 2
    assert all(isinstance(unit, spacy.tokens.span.Span) for unit in units)

def test_get_units_with_lines(doc):
    rolling_windows = RollingWindows(doc, "en_core_web_sm")
    units = rolling_windows._get_units(doc, "lines")
    assert len(units) == 2
    assert all(isinstance(unit, spacy.tokens.span.Span) for unit in units)

def test_get_units_with_other(doc):
    rolling_windows = RollingWindows(doc, "en_core_web_sm")
    units = rolling_windows._get_units(doc, "tokens")
    assert isinstance(units, spacy.tokens.doc.Doc)

# Set Windows

def test_set_windows_with_filter_name(doc, test_filter):
    rolling_windows = RollingWindows(doc)
    rolling_windows.set_windows(filter=test_filter.id)
    assert rolling_windows.metadata["filter"] == test_filter.metadata

def test_set_windows_with_filter_object(doc, test_filter):
    rolling_windows = RollingWindows(doc)
    rolling_windows.set_windows(filter=test_filter)
    assert rolling_windows.metadata["filter"] == test_filter.metadata

def test_set_windows_without_filter(doc):
    rolling_windows = RollingWindows(doc)
    rolling_windows.set_windows()
    assert "filter" not in rolling_windows.metadata

def test_set_windows_with_different_parameters(doc):
    rolling_windows = RollingWindows(doc)
    rolling_windows.set_windows(n=500, window_units="lines", alignment_mode="expand")
    assert rolling_windows.metadata["n"] == 500
    assert rolling_windows.metadata["window_units"] == "lines"
    assert rolling_windows.metadata["alignment_mode"] == "expand"
    assert rolling_windows.metadata["search_method"] in ["count", "spacy_matcher", "re_finditer"]

@pytest.fixture
def averages():
    return calculators.Averages()

def test_calculate_with_callable_calculator(doc, calculator):
    rolling_windows = RollingWindows(doc)
    rolling_windows.set_windows()
    rolling_windows.calculate(calculator=averages)
    assert rolling_windows.metadata["calculator"] == calculator.metadata
    assert isinstance(rolling_windows.result, pd.DataFrame)

def test_calculate_with_string_calculator(doc):
    rolling_windows = RollingWindows(doc)
    rolling_windows.set_windows()
    rolling_windows.calculate(calculator="averages")
    assert rolling_windows.metadata["calculator"]["name"] == "averages"
    assert isinstance(rolling_windows.result, pd.DataFrame)

def test_calculate_without_calculator(doc):
    rolling_windows = RollingWindows(doc)
    rolling_windows.set_windows()
    rolling_windows.calculate()
    assert rolling_windows.metadata["calculator"]["name"] == "averages"
    assert isinstance(rolling_windows.result, pd.DataFrame)

def test_calculate_without_set_windows(doc):
    rolling_windows = RollingWindows(doc)
    with pytest.raises(Exception):
        rolling_windows.calculate()

@pytest.fixture
def plotter():
    return plotters.RWSimplePlotter()

def test_plot_with_callable_plotter(doc, plotter):
    rolling_windows = RollingWindows(doc)
    rolling_windows.set_windows()
    rolling_windows.calculate()
    rolling_windows.plot(plotter=plotter)
    assert rolling_windows.metadata["plotter"] == plotter.metadata

def test_plot_with_string_plotter(doc):
    rolling_windows = RollingWindows(doc)
    rolling_windows.set_windows()
    rolling_windows.calculate()
    rolling_windows.plot(plotter="rw_simple_plotter")
    assert rolling_windows.metadata["plotter"]["name"] == "rw_simple_plotter"

def test_plot_without_plotter(doc):
    rolling_windows = RollingWindows(doc)
    rolling_windows.set_windows()
    rolling_windows.calculate()
    with pytest.raises(Exception):
        rolling_windows.plot()

def test_plot_without_calculate(doc, plotter):
    rolling_windows = RollingWindows(doc)
    with pytest.raises(Exception):
        rolling_windows.plot(plotter=plotter)

def test_plot_with_show_true(doc, plotter):
    rolling_windows = RollingWindows(doc)
    rolling_windows.set_windows()
    rolling_windows.calculate()
    rolling_windows.plot(plotter=plotter, show=True)
    assert rolling_windows.fig is not None

def test_plot_with_show_false(doc, plotter):
    rolling_windows = RollingWindows(doc)
    rolling_windows.set_windows()
    rolling_windows.calculate()
    rolling_windows.plot(plotter=plotter, show=False)
    assert rolling_windows.fig is not None

def test_plot_with_file(doc, plotter, tmp_path):
    rolling_windows = RollingWindows(doc)
    rolling_windows.set_windows()
    rolling_windows.calculate()
    file = tmp_path / "plot.png"
    rolling_windows.plot(plotter=plotter, file=str(file))
    assert file.exists()

