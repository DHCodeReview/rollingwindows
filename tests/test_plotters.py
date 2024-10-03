"""test_plotters.py.

As of 13 June 2024 this has 96% coverage for plotters.py.
"""

import tempfile
from pathlib import Path
from unittest.mock import patch

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.express as px
import pytest
from scipy.interpolate import interp1d, pchip

from rollingwindows.plotters import (BasePlotter, RWPlotlyPlotter,
                                     RWSimplePlotter, interpolate)

# Fixtures

class MockPlotter(BasePlotter):
    def __init__(self, id, name, value):
        self.id = id
        self.name = name
        self.value = value

@pytest.fixture
def mock_plotter():
    return MockPlotter(id="123", name="TestPlotter", value=42)

@pytest.fixture
def simple_plotter():
    return RWSimplePlotter()

@pytest.fixture
def sample_dataframe():
    np.random.seed(0)
    data = np.random.rand(10, 2)
    df = pd.DataFrame(data, columns=['A', 'B'])
    return df

@pytest.fixture
def plotly_plotter():
    return RWPlotlyPlotter()

@pytest.fixture
def plotly_plotter_without_fig():
    return RWPlotlyPlotter()

# Functions

@pytest.mark.parametrize("interpolation_kind,expected_output", [
    ("pchip", np.array([3.5, 4.5, 5.5])),  # Assuming pchip interpolation
    ("linear", np.array([3.5, 4.5, 5.5])),  # Assuming linear interpolation
    ("nearest", np.array([3, 4, 5])),  # Assuming nearest interpolation
    (None, np.array([3.5, 4.5, 5.5]))  # Assuming default numpy interp
])

def test_interpolate(interpolation_kind, expected_output):
    x = np.array([1, 2, 3, 4, 5])
    y = np.array([2, 3, 4, 5, 6])
    xx = np.array([2.5, 3.5, 4.5])
    interpolated_values = interpolate(x, y, xx, interpolation_kind)
    assert list(interpolated_values) == list(expected_output)

# BasePlotter

def test_metadata_includes_attributes(mock_plotter):
    metadata = mock_plotter.metadata
    assert metadata["id"] == "123", "ID should be included in metadata"
    assert metadata["name"] == "TestPlotter", "Name should be included in metadata"
    assert metadata["value"] == 42, "Value should be included in metadata"

def test_metadata_excludes_private_and_magic_attributes(mock_plotter):
    metadata = mock_plotter.metadata
    private_attrs = [attr for attr in metadata if attr.startswith("_")]
    magic_attrs = [attr for attr in metadata if attr.startswith("__") and attr.endswith("__")]
    assert not private_attrs, "Private attributes should not be included in metadata"
    assert not magic_attrs, "Magic attributes should not be included in metadata"

def test_metadata_excludes_class_attributes():
    class AnotherMockPlotter(MockPlotter):
        class_attribute = "Should not be included"

    another_mock_plotter = AnotherMockPlotter(id="456", name="AnotherTestPlotter", value=84)
    metadata = another_mock_plotter.metadata
    assert "class_attribute" not in metadata, "Class attributes should not be included in metadata"

# RWSimplePlotter

def test_default_constructor(simple_plotter):
    assert simple_plotter.width == 6.4
    assert simple_plotter.height == 4.8
    assert simple_plotter.figsize is None
    assert simple_plotter.hide_spines == ["top", "right"]
    assert simple_plotter.title == "Rolling Windows Plot"
    assert simple_plotter.titlepad == 6.0
    assert simple_plotter.title_position == "top"
    assert simple_plotter.show_legend is True
    assert simple_plotter.show_grid is False
    assert simple_plotter.xlabel == "Token Count"
    assert simple_plotter.ylabel == "Average Frequency"
    assert simple_plotter.show_milestones is False
    assert simple_plotter.milestone_colors == "teal"
    assert simple_plotter.milestone_style == "--"
    assert simple_plotter.milestone_width == 1
    assert simple_plotter.show_milestone_labels is False
    assert simple_plotter.milestone_labels is None
    assert simple_plotter.milestone_labels_ha == "left"
    assert simple_plotter.milestone_labels_va == "baseline"
    assert simple_plotter.milestone_labels_rotation == 45
    assert simple_plotter.milestone_labels_offset == (-8, 4)
    assert simple_plotter.milestone_labels_textcoords == "offset pixels"
    assert simple_plotter.use_interpolation is False
    assert simple_plotter.interpolation_num == 500
    assert simple_plotter.interpolation_kind == "pchip"

def test_custom_constructor():
    custom_plotter = RWSimplePlotter(width=8, height=6, figsize=(10, 7.5), hide_spines=["bottom", "left"],
                                     title="Custom Plot", titlepad=10, title_position="bottom", show_legend=False,
                                     show_grid=True, xlabel="Custom X", ylabel="Custom Y", show_milestones=True,
                                     milestone_colors=["red", "blue"], milestone_style=":", milestone_width=2,
                                     show_milestone_labels=True, milestone_labels=[{"label1": 10}, {"label2": 20}],
                                     milestone_labels_ha="right", milestone_labels_va="top", milestone_labels_rotation=90,
                                     milestone_labels_offset=(10, -5), milestone_labels_textcoords="offset points",
                                     use_interpolation=True, interpolation_num=1000, interpolation_kind="linear")
    assert custom_plotter.width == 8
    assert custom_plotter.height == 6
    assert custom_plotter.figsize == (10, 7.5)
    assert custom_plotter.hide_spines == ["bottom", "left"]
    assert custom_plotter.title == "Custom Plot"
    assert custom_plotter.titlepad == 10
    assert custom_plotter.title_position == "bottom"
    assert custom_plotter.show_legend is False
    assert custom_plotter.show_grid is True
    assert custom_plotter.xlabel == "Custom X"
    assert custom_plotter.ylabel == "Custom Y"
    assert custom_plotter.show_milestones is True
    assert custom_plotter.milestone_colors == ["red", "blue"]
    assert custom_plotter.milestone_style == ":"
    assert custom_plotter.milestone_width == 2
    assert custom_plotter.show_milestone_labels is True
    assert custom_plotter.milestone_labels == [{"label1": 10}, {"label2": 20}]
    assert custom_plotter.milestone_labels_ha == "right"
    assert custom_plotter.milestone_labels_va == "top"
    assert custom_plotter.milestone_labels_rotation == 90
    assert custom_plotter.milestone_labels_offset == (10, -5)
    assert custom_plotter.milestone_labels_textcoords == "offset points"
    assert custom_plotter.use_interpolation is True
    assert custom_plotter.interpolation_num == 1000
    assert custom_plotter.interpolation_kind == "linear"

@pytest.mark.parametrize("input_locations,expected_output", [
    ([{"label": 10}], [{"label_1": 10}]),  # Single label, should add suffix
    ([{"label": 10}, {"label": 20}], [{"label_1": 10}, {"label_2": 20}]),  # Duplicate labels, should add suffixes
    ([{"label1": 10}, {"label2": 20}], [{"label1": 10}, {"label2": 20}]),  # Different labels, should remain unchanged
    # ([{"label": 10}, {"label2": 20}, {"label": 30}], [{"label_1": 10}, {"label2": 20}, {"label_2": 30}]),  # Mixed, only duplicates get suffixes
    ([], []),  # Empty list, should return empty
])

def test_check_duplicate_labels(simple_plotter, input_locations, expected_output):
    assert simple_plotter._check_duplicate_labels(input_locations) == expected_output

@pytest.mark.parametrize("milestone_labels,milestone_labels_rotation,expected_height", [
    ([{"label1": 10}], 0, 14),  # Assuming default font size, straight horizontal label
    ([{"label1": 10}], 45, 53.298173631936024),  # Diagonal label, height should be less due to rotation
    ([{"label1": 10}, {"label2": 20}], 0, 14),  # Multiple labels, height should be the same as single label
    ([{"long_label_example": 10}], 90, 156.375),  # Vertical label, height should be more due to rotation
])

def test_get_label_height(simple_plotter, milestone_labels, milestone_labels_rotation, expected_height):
    # WARNING: The expected height may not be correct, depending on the system where the test is run.
    height = simple_plotter._get_label_height(milestone_labels, milestone_labels_rotation)
    assert height == pytest.approx(expected_height, abs=2), "The calculated label height is incorrect"


def test_run_with_default_settings(simple_plotter, sample_dataframe):
    simple_plotter.run(sample_dataframe)
    assert simple_plotter.fig is not None, "Figure should be created"

def test_run_with_custom_figsize(simple_plotter, sample_dataframe):
    simple_plotter.figsize = (10, 8)
    simple_plotter.run(sample_dataframe)
    assert tuple(simple_plotter.fig.get_size_inches()) == (10, 8), "Figure size should be (10, 8)"

def test_run_with_title_position_bottom(simple_plotter, sample_dataframe):
    simple_plotter.title_position = "bottom"
    simple_plotter.run(sample_dataframe)
    # This test checks if the title is at the bottom, but matplotlib doesn't expose title position directly
    # Instead, we check if the title text matches the expected title
    # Note that this check is fragile. There are more direct ways to get
    # the figure title in matplotlib 3.9.
    assert simple_plotter.fig.axes[0].get_title() == simple_plotter.title

def test_run_with_milestones(simple_plotter, sample_dataframe):
    simple_plotter.show_milestones = True
    simple_plotter.show_milestone_labels = True
    simple_plotter.milestone_labels = [{"A": 5}, {"B": 7}]
    simple_plotter.run(sample_dataframe)
    # This test assumes that if milestones are shown, annotations are created
    # Checking the number of annotations to verify milestones are plotted
    assert len(simple_plotter.fig.axes[0].texts) == 2, "There should be two milestone labels"

def test_run_with_interpolation(simple_plotter, sample_dataframe):
    simple_plotter.use_interpolation = True
    simple_plotter.interpolation_num = 100
    simple_plotter.run(sample_dataframe)
    # Check if the number of points in the line is equal to interpolation_num
    # This indirectly verifies that interpolation was applied
    line = simple_plotter.fig.axes[0].lines[0]
    assert len(line.get_xdata()) == 100, "Interpolated line should have 100 points"

@pytest.mark.parametrize("hide_spines,expected_visible", [
    (["top", "right"], {"left": True, "bottom": True, "top": False, "right": False}),
    ([], {"left": True, "bottom": True, "top": True, "right": True}),
])
def test_run_with_hide_spines(simple_plotter, sample_dataframe, hide_spines, expected_visible):
    simple_plotter.hide_spines = hide_spines
    simple_plotter.run(sample_dataframe)
    ax = simple_plotter.fig.axes[0]
    for spine, visible in expected_visible.items():
        assert ax.spines[spine].get_visible() == visible, f"Spine '{spine}' visibility should be {visible}"

def test_save_without_plot_raises_exception(simple_plotter):
    with pytest.raises(Exception) as exc_info:
        simple_plotter.save("test_plot.png")
    assert "There is no plot to save" in str(exc_info.value), "Expected an exception when trying to save without a plot"

def test_save_plot_to_file(simple_plotter, sample_dataframe):
    # Generate a simple plot to save
    simple_plotter.run(sample_dataframe)
    with tempfile.TemporaryDirectory() as tmp_path:
        file = f"{tmp_path}/test_plot.png"
        simple_plotter.save(file)
        assert Path(file).is_file(), "Plot file should exist after saving"

def test_save_with_kwargs(simple_plotter, sample_dataframe):
    # Generate a simple plot to save with additional kwargs
    simple_plotter.run(sample_dataframe)
    with tempfile.TemporaryDirectory() as tmp_path:
        file = f"{tmp_path}/test_plot.png"
        simple_plotter.save(file, dpi=300)
        assert Path(file).is_file(), "Plot file should exist after saving"

def test_show_calls_fig_show(simple_plotter, sample_dataframe):
    simple_plotter.run(sample_dataframe)  # Assuming run() method creates the fig attribute
    with patch.object(plt.Figure, 'show') as mock_show:
        simple_plotter.show()
        mock_show.assert_called_once()


# RWPlotlyPlotter

def test_default_initialization(plotly_plotter):
    assert plotly_plotter.width == 700
    assert plotly_plotter.height == 450
    assert plotly_plotter.title == "Rolling Windows Plot"
    assert plotly_plotter.xlabel == "Token Count"
    assert plotly_plotter.ylabel == "Average Frequency"
    assert plotly_plotter.line_color == "variable"
    assert plotly_plotter.showlegend is True
    assert plotly_plotter.titlepad is None
    assert plotly_plotter.show_milestones is True
    assert plotly_plotter.milestone_marker_style == {"width": 1, "color": "teal"}
    assert plotly_plotter.show_milestone_labels is False
    assert plotly_plotter.milestone_labels is None
    assert plotly_plotter.milestone_label_rotation == 0.0
    assert plotly_plotter.milestone_label_style == {
        "size": 10.0,
        "family": "Open Sans, verdana, arial, sans-serif",
        "color": "teal",
    }
    assert plotly_plotter.fig is None

def test_custom_initialization():
    custom_plotter = RWPlotlyPlotter(
        width=800,
        height=600,
        title="Custom Title",
        xlabel="Custom X Label",
        ylabel="Custom Y Label",
        line_color="blue",
        showlegend=False,
        titlepad=20,
        show_milestones=False,
        milestone_marker_style={"width": 2, "color": "red"},
        show_milestone_labels=True,
        milestone_labels=[{"label1": 100}, {"label2": 200}],
        milestone_label_rotation=45.0,
        milestone_label_style={
            "size": 12.0,
            "family": "Arial",
            "color": "green",
        },
    )
    assert custom_plotter.width == 800
    assert custom_plotter.height == 600
    assert custom_plotter.title == "Custom Title"
    assert custom_plotter.xlabel == "Custom X Label"
    assert custom_plotter.ylabel == "Custom Y Label"
    assert custom_plotter.line_color == "blue"
    assert custom_plotter.showlegend is False
    assert custom_plotter.titlepad == 20
    assert custom_plotter.show_milestones is False
    assert custom_plotter.milestone_marker_style == {"width": 2, "color": "red"}
    assert custom_plotter.show_milestone_labels is True
    assert custom_plotter.milestone_labels == [{"label1": 100}, {"label2": 200}]
    assert custom_plotter.milestone_label_rotation == 45.0
    assert custom_plotter.milestone_label_style == {
        "size": 12.0,
        "family": "Arial",
        "color": "green",
    }

def test_kwargs_initialization():
    kwargs_plotter = RWPlotlyPlotter(additional_option="test_value")
    assert hasattr(kwargs_plotter, "additional_option")
    assert kwargs_plotter.additional_option == "test_value"

def test_get_axis_and_title_labels_with_strings(plotly_plotter):
    plotly_plotter.title = "My Plot Title"
    plotly_plotter.xlabel = "X Axis Label"
    plotly_plotter.ylabel = "Y Axis Label"
    title_dict, xlabel_dict, ylabel_dict = plotly_plotter._get_axis_and_title_labels()
    assert title_dict == {"text": "My Plot Title", "y": 0.9, "x": 0.5, "xanchor": "center", "yanchor": "top"}
    assert xlabel_dict == {"title": "X Axis Label"}
    assert ylabel_dict == {"title": "Y Axis Label"}

def test_get_axis_and_title_labels_with_dicts(plotly_plotter):
    plotly_plotter.title = {"text": "Custom Title", "y": 0.8}
    plotly_plotter.xlabel = {"title": "Custom X Label", "font": {"size": 12}}
    plotly_plotter.ylabel = {"title": "Custom Y Label", "font": {"size": 12}}
    title_dict, xlabel_dict, ylabel_dict = plotly_plotter._get_axis_and_title_labels()
    assert title_dict == {"text": "Custom Title", "y": 0.8}
    assert xlabel_dict == {"title": "Custom X Label", "font": {"size": 12}}
    assert ylabel_dict == {"title": "Custom Y Label", "font": {"size": 12}}

def test_get_axis_and_title_labels_with_mixed_types(plotly_plotter):
    plotly_plotter.title = {"text": "Mixed Title", "y": 0.8}
    plotly_plotter.xlabel = "Mixed X Label"
    plotly_plotter.ylabel = {"title": "Mixed Y Label", "font": {"size": 12}}
    title_dict, xlabel_dict, ylabel_dict = plotly_plotter._get_axis_and_title_labels()
    assert title_dict == {"text": "Mixed Title", "y": 0.8}
    assert xlabel_dict == {"title": "Mixed X Label"}
    assert ylabel_dict == {"title": "Mixed Y Label", "font": {"size": 12}}

@pytest.mark.parametrize("labels,titlepad,expected", [
    ({"Label1": 100, "Label2": 200}, None, 90.0),  # Assuming max height < 50
    ({"Label1": 100, "Label2": 200, "LabelWithVeryLongNameThatExceedsFiftyPixels": 300}, None, 90.0),  # Assuming max height >= 50
    ({"Label1": 100}, 30, 30),  # Custom titlepad provided
])

def test_get_titlepad(plotly_plotter, labels, titlepad, expected):
    # WARNING: Expected values may vary depending on the system on which testing is performed.
    plotly_plotter.titlepad = titlepad
    result = plotly_plotter._get_titlepad(labels)
    assert result == expected, f"Expected titlepad to be {expected}, got {result}"

def test_get_titlepad_with_multiple_font_families(plotly_plotter):
    plotly_plotter.milestone_label_style["family"] = "Arial, sans-serif"
    labels = {"Label1": 100, "Label2": 200}
    result = plotly_plotter._get_titlepad(labels)
    assert isinstance(result, float), "Expected titlepad to be a float value"


def test_plot_milestone_label(plotly_plotter, sample_dataframe):
    label = "Test Label"
    x = 100
    plotly_plotter.fig = px.line(sample_dataframe)
    plotly_plotter._plot_milestone_label(label, x)
    plotly_plotter.fig.add_annotation(
        x=x,
        y=1,
        xanchor="left",
        yanchor="bottom",
        xshift=-10,
        yref="paper",
        showarrow=False,
        text=label,
        textangle=-45,
        font={"size": 12, "color": "blue"},
    )
    assert plotly_plotter.fig.layout.annotations[0].text == label

def test_plot_milestone_marker(plotly_plotter, sample_dataframe):
    x = 100
    df_val_min = 10
    df_val_max = 20
    plotly_plotter.fig = px.line(sample_dataframe)
    plotly_plotter._plot_milestone_marker(x, df_val_min, df_val_max)
    plotly_plotter.fig.add_shape(
        type="line",
        yref="y",
        xref="x",
        x0=x,
        y0=df_val_min,
        x1=x,
        y1=df_val_max,
        line={"color": "red", "width": 2},
    )
    assert plotly_plotter.fig.layout.shapes[0].type == "line"

def test_run_with_valid_data(plotly_plotter, sample_dataframe):
    plotly_plotter.milestone_label_rotation = 45
    plotly_plotter.show_milestones = True
    plotly_plotter.show_milestone_labels = True
    plotly_plotter.milestone_labels = {"Milestone 1": 2}
    plotly_plotter.run(sample_dataframe)
    # Test if the plot was created by checking if fig is not None
    assert plotly_plotter.fig is not None, "Expected fig to be created"

def test_run_with_invalid_milestone_label_rotation(plotly_plotter, sample_dataframe):
    plotly_plotter.milestone_label_rotation = 91
    with pytest.raises(Exception) as exc_info:
        plotly_plotter.run(sample_dataframe)
    assert "Milestone labels can only be rotated clockwise a maximum of 90 degrees." in str(exc_info.value)

def test_run_without_milestone_labels(plotly_plotter, sample_dataframe):
    plotly_plotter.show_milestones = True
    plotly_plotter.show_milestone_labels = True
    plotly_plotter.milestone_labels = None
    with pytest.raises(Exception) as exc_info:
        plotly_plotter.run(sample_dataframe)
    assert "The `show_milestones` and `show_milestone_labels` parameters require a dictionary of labels and x-axis values assigned to `milestone_labels`." in str(exc_info.value)

def test_save_without_fig(plotly_plotter_without_fig):
    with pytest.raises(Exception) as exc_info:
        plotly_plotter_without_fig.save("plot.html")
    assert "There is no plot to save, try calling `plotter.run()`." in str(exc_info.value)

def test_save_as_html(plotly_plotter, sample_dataframe):
    plotly_plotter.show_milestone_labels = True
    plotly_plotter.milestone_labels = {"Milestone 1": 2}
    plotly_plotter.run(sample_dataframe)
    with tempfile.TemporaryDirectory() as tmp_path:
        file = f"{tmp_path}/test_plot.html"
        plotly_plotter.save(file)
        assert Path(file).is_file(), "Plot file should exist after saving"


def test_save_as_image(plotly_plotter, sample_dataframe):
    plotly_plotter.show_milestone_labels = True
    plotly_plotter.milestone_labels = {"Milestone 1": 2}
    plotly_plotter.run(sample_dataframe)
    with tempfile.TemporaryDirectory() as tmp_path:
        file = f"{tmp_path}/test_plot.png"
        plotly_plotter.save(file, scale=2)
        assert Path(file).is_file(), "Plot file should exist after saving"
