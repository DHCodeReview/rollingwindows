"""plotter_tests.py."""

from unittest.mock import patch

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest
import spacy
from spacy.tokens import Token

from rollingwindows import Windows, sliding_windows
from rollingwindows.plotters import BasePlotter, RWSimplePlotter, interpolate

# Fixtures


@pytest.fixture
def nlp():
    return spacy.load("en_core_web_sm")


@pytest.fixture
def doc(nlp):
    return nlp("This is a test document. It has multiple sentences.")


@pytest.fixture
def test_plotter():
    class TestPlotter(BasePlotter):
        id = "test_plotter"

    return TestPlotter()


@pytest.fixture
def windows():
    return Windows(sliding_windows(doc), "characters", 10)


@pytest.fixture
def token_windows():
    return Windows(sliding_windows(doc), "tokens", 10)


@pytest.fixture
def sentence_windows():
    return Windows(sliding_windows(doc.sents), "sentences", 2)


# Plotter Protocol

def test_base_plotter_protocol(test_plotter):
    metadata = test_plotter.metadata
    assert metadata["id"] == "test_plotter"
    with pytest.raises(NotImplementedError):
        test_plotter.run()
    with pytest.raises(NotImplementedError):
        test_plotter.file()
    with pytest.raises(NotImplementedError):
        test_plotter.show()

def test_metadata_with_additional_attributes(test_plotter):
    test_plotter.extra_attribute = "extra"
    assert test_plotter.metadata == {"id": "test_plotter", "extra_attribute": "extra"}

# Functions

def test_interpolate_with_pchip():
    x = np.array([1, 2, 3])
    y = np.array([1, 2, 3])
    xx = np.array([1.5, 2.5])
    result = interpolate(x, y, xx, "pchip")
    # TODO: I am not sure np.allclose is the proper evaluation
    assert np.allclose(result, [1.5, 2.5])

def test_interpolate_with_linear():
    x = np.array([1, 2, 3])
    y = np.array([1, 2, 3])
    xx = np.array([1.5, 2.5])
    result = interpolate(x, y, xx, "linear")
    assert np.allclose(result, [1.5, 2.5])

def test_interpolate_with_not_pchip_or_interp1d():
    x = np.array([1, 2, 3])
    y = np.array([1, 2, 3])
    xx = np.array([1.5, 2.5])
    result = interpolate(x, y, xx, "not_pchip_or_interp1d")
    assert np.allclose(result, [1.5, 2.5])

def test_interpolate_with_none():
    x = np.array([1, 2, 3])
    y = np.array([1, 2, 3])
    xx = np.array([1.5, 2.5])
    result = interpolate(x, y, xx, None)
    assert np.allclose(result, [1.5, 2.5])

# RWSimplePlotter Class

def test_rw_simple_plotter_init_custom():
    plotter = RWSimplePlotter(width=7.0, height=5.0, title="Custom Title")
    assert plotter.width == 7.0
    assert plotter.height == 5.0
    assert plotter.title == "Custom Title"

def test_rw_simple_plotter_init_kwargs():
    plotter = RWSimplePlotter(extra_param="extra")
    assert plotter.extra_param == "extra"

def test_check_duplicate_labels_with_unique_keys():
    plotter = RWSimplePlotter()
    locations = [{"label1": 1}, {"label2": 2}, {"label3": 3}]
    result = plotter._check_duplicate_labels(locations)
    assert result == locations

def test_check_duplicate_labels_with_duplicate_keys():
    plotter = RWSimplePlotter()
    locations = [{"label": 1}, {"label": 2}, {"label": 3}]
    result = plotter._check_duplicate_labels(locations)
    assert result == [{"label_1": 1}, {"label_2": 2}, {"label_3": 3}]

def test_get_label_height_with_different_lengths_and_no_rotation():
    plotter = RWSimplePlotter()
    labels = [{"label1": 1}, {"longer_label2": 2}, {"longest_label3": 3}]
    shortest = plotter._get_label_height([labels[0]], 0)
    longest = plotter._get_label_height(labels, 0)
    assert longest == shortest

def test_get_label_height_with_different_lengths_and_rotation():
    plotter = RWSimplePlotter()
    labels = [{"label1": 1}, {"longer_label2": 2}, {"longest_label3": 3}]
    shortest = plotter._get_label_height([labels[0]], 0)
    longest = plotter._get_label_height(labels, 45)
    assert longest > shortest

def test_get_label_height_with_same_lengths_and_no_rotation():
    plotter = RWSimplePlotter()
    labels = [{"label": 1}, {"label": 2}, {"label": 3}]
    shortest = plotter._get_label_height([labels[0]], 0)
    longest = plotter._get_label_height(labels, 0)
    assert longest == shortest

def test_get_label_height_with_same_lengths_and_rotation():
    plotter = RWSimplePlotter()
    labels = [{"label1": 1}, {"label2": 2}, {"label3": 3}]
    shortest = plotter._get_label_height([labels[0]], 0)
    longest = plotter._get_label_height(labels, 0)
    assert longest == shortest

def test_get_label_height_with_empty_labels():
    plotter = RWSimplePlotter()
    labels = []
    height = plotter._get_label_height(labels, 0)
    assert height == 0

def test_run_with_single_column():
    plotter = RWSimplePlotter()
    df = pd.DataFrame({"column1": [1, 2, 3]})
    plotter.run(df)
    assert isinstance(plotter.plot, plt.figure)

def test_run_with_multiple_columns():
    plotter = RWSimplePlotter()
    df = pd.DataFrame({"column1": [1, 2, 3], "column2": [4, 5, 6]})
    plotter.run(df)
    assert isinstance(plotter.plot, plt.figure)

def test_run_with_no_columns():
    plotter = RWSimplePlotter()
    df = pd.DataFrame()
    plotter.run(df)
    assert isinstance(plotter.plot, plt.figure)

def test_run_with_show_milestones():
    plotter = RWSimplePlotter(show_milestones=True)
    df = pd.DataFrame({"column1": [1, 2, 3]})
    plotter.run(df)
    assert isinstance(plotter.plot, plt.figure)

def test_run_with_show_milestone_labels():
    plotter = RWSimplePlotter(show_milestone_labels=True)
    df = pd.DataFrame({"column1": [1, 2, 3]})
    plotter.run(df)
    assert isinstance(plotter.plot, plt.figure)

def test_run_with_interpolation():
    plotter = RWSimplePlotter(use_interpolation=True)
    df = pd.DataFrame({"column1": [1, 2, 3]})
    plotter.run(df)
    assert isinstance(plotter.plot, plt.figure)

def test_save_with_valid_path_and_plot_created(tmp_path):
    plotter = RWSimplePlotter()
    df = pd.DataFrame({"column1": [1, 2, 3]})
    plotter.run(df)
    path = tmp_path / "plot.png"
    plotter.save(str(path))
    assert path.exists()

def test_save_with_valid_path_and_no_plot_created(tmp_path):
    plotter = RWSimplePlotter()
    path = tmp_path / "plot.png"
    with pytest.raises(Exception, match="There is no plot to save, try calling `plotter.run()`."):
        plotter.save(str(path))

def test_save_with_invalid_path_and_plot_created():
    plotter = RWSimplePlotter()
    df = pd.DataFrame({"column1": [1, 2, 3]})
    plotter.run(df)
    path = "/invalid/path/plot.png"
    with pytest.raises(FileNotFoundError):
        plotter.save(path)

def test_show_with_plot_created():
    plotter = RWSimplePlotter()
    df = pd.DataFrame({"column1": [1, 2, 3]})
    plotter.run(df)
    with patch.object(plotter.plot, "show", return_value=None) as mock_show:
        plotter.show()
    mock_show.assert_called_once()

def test_show_with_no_plot_created():
    plotter = RWSimplePlotter()
    with patch.object(plotter, "run", return_value=None) as mock_run:
        plotter.show()
    mock_run.assert_called_once()

def test_show_with_plot_created_and_user_warning_raised():
    plotter = RWSimplePlotter()
    df = pd.DataFrame({"column1": [1, 2, 3]})
    plotter.run(df)
    with patch.object(plotter.plot, "show", side_effect=UserWarning) as mock_show:
        plotter.show()
    mock_show.assert_called_once()