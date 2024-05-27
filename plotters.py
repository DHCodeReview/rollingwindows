"""plotters.py.

Last Update: May 25 2024
"""

from typing import Dict, List, Protocol, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.express as px
from scipy.interpolate import interp1d, pchip


def interpolate(
    x: np.ndarray, y: np.ndarray, xx: np.ndarray, interpolation_kind: str = None
) -> np.ndarray:
    """Get interpolated points for plotting.

    Args:
        x: The x values
        y: The y values
        xx: The projected interpolation range
        interpolation_kind: The interpolation function to use.

    Returns:
        The interpolated points.

    Note:
        The interpolation function may be either
        [scipy.interpolate.pchip_interpolate](https://docs.scipy.org/doc/scipy/reference/generated/scipy.interpolate.pchip_interpolate.html#scipy.interpolate.pchip_interpolate),
        [numpy.interp](https://numpy.org/devdocs/reference/generated/numpy.interp.html#numpy.interp),
        or one of the options for [scipy.interpolate.interp1d](https://docs.scipy.org/doc/scipy/reference/generated/scipy.interpolate.interp1d.html).
        Note however, that `scipy.interpolate.interp1d` is [deprecated](https://docs.scipy.org/doc/scipy/tutorial/interpolate/1D.html#piecewise-linear-interpolation).
    """
    legacy_interp1d = [
        "linear",
        "nearest",
        "nearest-up",
        "zero",
        "slinear",
        "quadratic",
        "cubic",
        "previous",
        "next",
    ]
    # Return the values interpolated with the specified function
    if interpolation_kind == "pchip":
        interpolator = pchip(x, y)
        return interpolator(xx)
    elif interpolation_kind in legacy_interp1d:
        interpolator = interp1d(x, y, kind=interpolation_kind)
        return interpolator(xx)
    else:
        return np.interp(xx, x, y)


class BasePlotter(Protocol):
    """BasePlotter class to enable type hinting."""

    @property
    def metadata(self) -> dict:
        exclude = []
        metadata = {"id": self.id}
        return metadata | dict(
            (key, getattr(self, key))
            for key in dir(self)
            if key not in exclude and key not in dir(self.__class__)
        )

    def run(self):
        """Use a dataframe to plot the rolling means with pyplot."""
        ...

    def file(self, path):
        """Save the plot to a file."""
        ...

    def show(self):
        """Display a plot."""
        ...


class RWSimplePlotter(BasePlotter):
    """Simple plotter using pyplot."""

    id: str = "rw_simple_plotter"

    def __init__(
        self,
        width: Union[float, int] = 6.4,
        height: Union[float, int] = 4.8,
        figsize: tuple = None,
        hide_spines: List[str] = ["top", "right"],
        title: str = "Rolling Windows Plot",
        titlepad: Union[float, int] = 6.0,
        title_position: str = "top",
        show_legend: bool = True,
        show_grid: bool = False,
        xlabel: str = "Token Count",
        ylabel: str = "Average Frequency",
        show_milestones: bool = False,
        milestone_colors: Union[List[str], str] = "teal",
        milestone_style: str = "--",
        milestone_width: int = 1,
        show_milestone_labels: bool = False,
        milestone_labels: List[dict] = None,
        milestone_labels_ha: str = "left",
        milestone_labels_va: str = "baseline",
        milestone_labels_rotation: int = 45,
        milestone_labels_offset: tuple = (-8, 4),
        milestone_labels_textcoords: str = "offset pixels",
        use_interpolation: bool = False,
        interpolation_num: int = 500,
        interpolation_kind: str = "pchip",
        **kwargs,
    ) -> None:
        """Initialise object.

        Args:
        width: The width in inches.
        height: The height in inches.
        figsize: A tuple containing the width and height in inches (overrides the previous keywords).
        hide_spines: A list of ["top", "right", "bottom", "left"] indicating which spines to hide
        title: The title to use for the plot.
        titlepad: The padding in points to place between the title and the plot. May need to be increased
                        if you are showing milestone labels. Default is 6.0 points.
        title_position: Show the title on the "bottom" or the "top" of the figure.
        show_legend: Whether to show the legend.
        show_grid: Whether to show the grid.
        xlabel: The text to display along the x axis.
        ylabel: The text to display along the y axis.
        show_milestones: Whether to show the milestone markers.
        milestone_colors: The colour or colours to use for milestone markers. See pyplot.vlines().
        milestone_style: The style of the milestone markers. See pyplot.vlines().
        milestone_width: The width of the milestone markers. See pyplot.vlines().
        show_milestone_labels: Whether to show the milestone labels.
        milestone_labels: A list of dicts with keys as milestone labels and values as token indexes.
        milestone_labels_ha: The horizontal alignment of the milestone labels. See pyplot.annotate().
        milestone_labels_va: The vertical alignment of the milestone labels. See pyplot.annotate().
        milestone_labels_rotation: The rotation of the milestone labels. See pyplot.annotate().
        milestone_labels_offset: A tuple containing the number of pixels along the x and y axes
            to offset the milestone labels. See pyplot.annotate().
        milestone_labels_textcoords: Whether to offset milestone labels by pixels or points.
            See pyplot.annotate().
        use_interpolation: Whether to use interpolation on values.
        interpolation_num: Number of values to add between points.
        interpolation_kind: Algorithm to use for interpolation.
        """
        self.width = width
        self.height = height
        self.figsize = figsize
        self.hide_spines = hide_spines
        self.title = title
        self.titlepad = titlepad
        self.title_position = title_position
        self.show_legend = show_legend
        self.show_grid = show_grid
        self.xlabel = xlabel
        self.ylabel = ylabel
        self.show_milestones = show_milestones
        self.milestone_colors = milestone_colors
        self.milestone_style = milestone_style
        self.milestone_width = milestone_width
        self.show_milestone_labels = show_milestone_labels
        self.milestone_labels = milestone_labels
        self.milestone_labels_ha = milestone_labels_ha
        self.milestone_labels_va = milestone_labels_va
        self.milestone_labels_rotation = milestone_labels_rotation
        self.milestone_labels_offset = milestone_labels_offset
        self.milestone_labels_textcoords = milestone_labels_textcoords
        self.use_interpolation = use_interpolation
        self.interpolation_num = interpolation_num
        self.interpolation_kind = interpolation_kind
        self.plot = None
        for k, v in kwargs.items():
            setattr(self, k, v)

    def _check_duplicate_labels(
        self, locations: List[Dict[str, int]]
    ) -> List[Dict[str, int]]:
        """Add numeric suffixes for duplicate milestone labels.

        Args:
            locations: A list of location dicts.

        Returns:
            A list of location dicts.
        """
        keys = set().union(*(d.keys() for d in locations))
        if len(keys) == 1:
            locations = [
                {f"{k}_{i}": v}
                for i, location in enumerate(locations, start=1)
                for k, v in location.items()
            ]
        return locations

    def _get_label_height(
        self, milestone_labels: List[dict], milestone_labels_rotation: int
    ) -> float:
        """Calculate the height of the longest milestone label.

        Args:
            milestone_labels: A list of milestone_label dicts
            milestone_labels_rotation: The rotation in degrees of the labels

        Returns:
            The height of the longest label

        Note:
            This method is a hack to calculate the label height using a separate plot.
        """
        labels = [list(item.keys()) for item in milestone_labels]
        tmp_fig, tmp_ax = plt.subplots()
        r = tmp_fig.canvas.get_renderer()
        heights = set()
        for x in labels:
            t = tmp_ax.annotate(
                x,
                xy=(0, 0),
                xytext=(0, 0),
                textcoords="offset points",
                rotation=milestone_labels_rotation,
            )
            bb = t.get_window_extent(renderer=r)
            heights.add(bb.height)
        plt.close()
        return max(list(heights))

    def run(self, df: pd.DataFrame) -> None:
        """Use a dataframe to plot the rolling means with pyplot.

        Args:
            df: A dataframe containing the data to plot.

        Note:
            The dataframe is normally generated by a RollingWindows
            calculator and stored in `RollingWindows.result`.
        """
        if self.figsize:
            width = self.figsize[0]
            height = self.figsize[1]
        else:
            width = self.width
            height = self.height
        titlepad = self.titlepad

        # Hack to move the title above the labels
        fig, ax = plt.subplots(figsize=(width, height))
        plt.close()
        if self.show_milestone_labels and self.title_position == "top":
            # Only override self.titlepad if it is the default value
            if self.titlepad == 6.0:
                titlepad = self._get_label_height(
                    self.milestone_labels, self.milestone_labels_rotation
                )

        # Now generate the plot
        fig, ax = plt.subplots(figsize=(width, height))
        ax.spines[self.hide_spines].set_visible(False)
        plt.margins(x=0, y=0)
        plt.ticklabel_format(axis="both", style="plain")
        if self.title_position == "bottom":
            plt.title(self.title, y=-0.25)
        else:
            plt.title(self.title, pad=titlepad)
        # TODO: plt.xlabel(self.xlabel, fontsize=10)
        plt.xlabel(self.xlabel)
        plt.ylabel(self.ylabel)
        if self.show_grid:
            plt.grid(visible=True)
        if self.use_interpolation:
            x = np.arange(df.shape[0])
            xx = np.linspace(x[0], x[-1], self.interpolation_num)
            for term in df.columns:
                y = np.array(df[term].values.tolist())
                interpolated = interpolate(x, y, xx, self.interpolation_kind)
                plt.plot(xx, interpolated, "-", label=term)
        else:
            for term in df.columns:
                plt.plot(df[term].values.tolist(), label=term)
        if self.show_legend:
            plt.legend()
        # If milestones have been set, plot them
        if self.show_milestones or self.show_milestone_labels:
            # Get milestone locations
            if self.milestone_labels:
                locations = [
                    {label: index[0] for label, index in item.items()}
                    for item in self.milestone_labels
                ]
            else:
                # WARNING! The plotter object has no doc attribute. For now, raise an exception.
                raise Exception(
                    "The `show_milestones` and `show_milestone_labels` parameters require a list of dicts assigned to `milestone_labels`."
                )
                # locations = self._check_duplicate_labels(
                #     [{t.text: t.i} for t in self.doc if t._.milestone_iob == "B"]
                # )
            # Plot the milestones with adjustments to the margin and spines
            # This looks like it is the highest value
            ymax = df.to_numpy().max()
            for milestone in locations:
                for k, v in milestone.items():
                    if self.show_milestones:
                        plt.vlines(
                            x=v,
                            ymin=0,
                            ymax=ymax,
                            colors=self.milestone_colors,
                            ls=self.milestone_style,
                            lw=self.milestone_width,
                        )
                    if self.show_milestone_labels:
                        ax.annotate(
                            k,
                            xy=(v, ymax),
                            ha=self.milestone_labels_ha,
                            va=self.milestone_labels_va,
                            rotation=self.milestone_labels_rotation,
                            xytext=self.milestone_labels_offset,
                            textcoords=self.milestone_labels_textcoords,
                        )
        plt.close()
        # Assign the plot
        self.plot = fig

    def save(self, path: str, **kwargs) -> None:
        """Save the plot to a file (wrapper for `pyplot.savefig()`).

        Args:
            path: The path to the file to save.

        Returns:
            None
        """
        if not self.plot:
            raise Exception("There is no plot to save, try calling `plotter.run()`.")
        self.plot.savefig(path, **kwargs)

    def show(self, **kwargs) -> None:
        """Display a plot.

        Note:
            This method calls pyplot.show(), but it won't work with
            an inline backend like Jupyter notebooks. It tries to
            detect this via a UserWarning and then just calls the
            `plot` attribute.
        """
        if not self.plot:
            self.run(kwargs)
        try:
            self.plot.show()
        except UserWarning:
            return self.plot


class RWPlotlyPlotter(BasePlotter):
    """Plotter using Plotly."""

    id: str = "rw_plotly_plotter"

    def __init__(
        self,
        width: int = 700,
        height: int = 450,
        title: Union[dict, str] = "Rolling Windows Plot",
        xlabel: str = "Token Count",
        ylabel: str = "Average Frequency",
        line_color: str = "variable",
        showlegend: bool = True,
        titlepad: float = None,
        show_milestones: bool = True,
        milestone_marker_style: dict = {"width": 1, "color": "teal"},
        show_milestone_labels: bool = False,
        milestone_labels: List[dict] = None,
        milestone_label_rotation: float = 0.0,
        milestone_label_style: dict = {
            "size": 10.0,
            "family": "Open Sans, verdana, arial, sans-serif",
            "color": "teal",
        },
        **kwargs,
    ) -> None:
        """Initialise object.

        Args:
            width: The width of the graph.
            height: The height of the graph. Note that if you change the height, you will need
                to adjust the titelpad manually to show the title above milestone labels.
            title: The title to use for the plot. If can be styled with a dict containing any of the
                keywords listed at https://plotly.com/python/reference/layout/#layout-title.
            xlabel: The text to display along the x axis.
            ylabel: The text to display along the y axis.
            line_color: The colour pattern to use for lines on the graph.
            showlegend: Whether or not to show the legend.
            titlepad: The margin in pixels between the title and the top of the graph.
            show_milestones: Whether to show the milestone markers.
            milestone_marker_style: A dict containing the styles to apply to the milestone marker. For
                valid properties, see https://plotly.com/python-api-reference/generated/plotly.graph_objects.layout.shape.html#plotly.graph_objects.layout.shape.Line.
            show_milestone_labels: Whether to show the milestone labels.
            milestone_label_rotation: The number of degrees clockwise to rotate the milestone labels (maximum 90).
            milestone_labels: A dicts containing the milestone labels and their values on the x-axis.
        """
        self.width = width
        self.height = height
        self.title = title
        self.xlabel = xlabel
        self.ylabel = ylabel
        self.line_color = line_color
        self.showlegend = showlegend
        self.titlepad = titlepad
        self.show_milestones = show_milestones
        self.milestone_marker_style = milestone_marker_style
        self.show_milestone_labels = show_milestone_labels
        self.milestone_labels = milestone_labels
        self.milestone_label_rotation = milestone_label_rotation
        self.milestone_label_style = milestone_label_style
        self.plot = None
        for k, v in kwargs.items():
            setattr(self, k, v)

    def _check_duplicate_labels(
        self, locations: List[Dict[str, int]]
    ) -> List[Dict[str, int]]:
        """Add numeric suffixes for duplicate milestone labels."""
        # TODO: Not yet implemented. It should be substantially the same as in RWSimplePlotter.
        ...

    def _get_axis_and_title_labels(self) -> Tuple[bool, str]:
        """Ensure that the title, xlabel, and ylabel values are dicts."""
        if isinstance(self.title, str):
            title_dict = dict(
                text=self.title, y=0.9, x=0.5, xanchor="center", yanchor="top"
            )
        else:
            title_dict = self.title
        if isinstance(self.xlabel, str):
            xlabel_dict = dict(title=self.xlabel)
        else:
            xlabel_dict = self.xlabel
        if isinstance(self.ylabel, str):
            ylabel_dict = dict(title=self.ylabel)
        else:
            ylabel_dict = self.ylabel
        return title_dict, xlabel_dict, ylabel_dict

    def _get_titlepad(self, labels: Dict[str, int]) -> float:
        """Get a titlepad value based on the height of the longest milestone label.

        Args:
            labels: A dict with the labels as keys.

        Returns:
            A float.
        """
        if self.titlepad:
            return self.titlepad
        fontfamily = self.milestone_label_style["family"]
        if "," in self.milestone_label_style["family"]:
            fontfamily = self.milestone_label_style["family"].split(",")
            fontfamily = [x.strip() for x in fontfamily]
        tmp_fig, tmp_ax = plt.subplots()
        r = tmp_fig.canvas.get_renderer()
        heights = []
        for x in list(labels.keys()):
            t = tmp_ax.annotate(
                x,
                xy=(0, 0),
                xytext=(0, 0),
                textcoords="offset points",
                rotation=self.milestone_label_rotation,
                fontfamily=fontfamily,
                fontsize=self.milestone_label_style["size"],
            )
            bb = t.get_window_extent(renderer=r)
            heights.append(bb.height)
        plt.close()
        if max(heights) < 50:
            return max(heights) + 75
        else:
            return max(heights) + 50

    def _plot_milestone_label(self, label: str, x: int) -> None:
        """Add a milestone label to the Plotly figure.

        Args:
            label: The label text.
            x: The location on the x axis.
        """
        self.fig.add_annotation(
            x=x,
            y=1,
            xanchor="left",
            yanchor="bottom",
            xshift=-10,
            yref="paper",
            showarrow=False,
            text=label,
            textangle=-self.milestone_label_rotation,
            font=self.milestone_label_style,
        )

    def _plot_milestone_marker(
        self, x: int, df_val_min: Union[float, int], df_val_max: Union[float, int]
    ) -> None:
        """Add a milestone marker to the Plotly figure.

        Args:
            x: The location on the x axis.
            df_val_min: The minimum value in the pandas DataFrame.
            df_val_max: The maximum value in the pandas DataFrame.
        """
        self.fig.add_shape(
            type="line",
            yref="y",
            xref="x",
            x0=x,
            y0=df_val_min,
            x1=x,
            y1=df_val_max,
            line=self.milestone_marker_style,
        )

    def run(
        self,
        df: pd.DataFrame,
    ) -> None:
        """Use a dataframe to plot the rolling means with pyplot.

        Args:
            df: A dataframe containing the data to plot.

        Note:
            The dataframe is normally generated by a RollingWindows
            calculator and stored in `RollingWindows.result`.
        """
        # Validation
        if self.milestone_label_rotation > 90:
            raise Exception(
                "Milestone labels can only be rotated clockwise a maximum of 90 degrees."
            )
        if self.show_milestones or self.show_milestone_labels:
            if not self.milestone_labels:
                raise Exception(
                    "The `show_milestones` and `show_milestone_labels` parameters require a dictionary of labels and x-axis values assigned to `milestone_labels`."
                )

        # Massage the DataFrame for Plotly Express
        df["id"] = df.index
        df = pd.melt(df, id_vars="id", value_vars=df.columns[:-1])

        # Create plotly line figure
        self.fig = px.line(
            df,
            x="id",
            y="value",
            color=self.line_color,
            width=self.width,
            height=self.height,
        )
        title_dict, xlabel_dict, ylabel_dict = self._get_axis_and_title_labels()
        self.fig.update_layout(
            title=title_dict,
            xaxis=xlabel_dict,
            yaxis=ylabel_dict,
            showlegend=self.showlegend,
        )

        if self.show_milestone_labels and not self.milestone_labels:
            # Generate milestone labels for use below
            pass

        if self.show_milestones:
            # Add milestones using absolute references
            for label, x in self.milestone_labels.items():
                df_val_min = df["value"].min() * 1.2
                df_val_max = df["value"].max() * 1.2
                self._plot_milestone_marker(x, df_val_min, df_val_max)
                if self.show_milestone_labels:
                    self._plot_milestone_label(label, x)

        # Increase the margin from the top to accommodate the milestone labels
        if self.show_milestone_labels:
            titlepad = self._get_titlepad(self.milestone_labels)
            self.fig.update_layout(margin=dict(t=titlepad))

    def save(self, path: str, **kwargs) -> None:
        """Save the plot to a file.

        Args:
            path: The path to the file to save.

        Returns:
            None
        """
        if not self.plot:
            raise Exception("There is no plot to save, try calling `plotter.run()`.")
        # Try first to save as HTML; if that doesn't work, try to save as a static image
        if not path.endswith(".html"):
            self.fig.write_html(path, **kwargs)
        else:
            import plotly.io as pio

            pio.write_image(self.fig, path)

    def show(self, **kwargs) -> None:
        """Display a plot."""
        self.fig.show(config={"displaylogo": False})
