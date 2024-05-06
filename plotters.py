"""plotters.py.

Working draft: April 20 2024

Sample usage:

rw = rollingwindows.RollingWindows(doc, patterns="the")
rw.set_windows(1000, "characters")
averages = calculators2.Averages(
	patterns=rw.patterns,
	windows=rw.windows,
	search_method=rw.metadata["search_method"],
)
rw.calculate(calculator=averages)
rw.result # A dataframe

plotter = plotters.RWSimplePlotter()
rw.plot(plotter=plotter)d
rw.fig


interpolation: https://stackoverflow.com/questions/40085300/how-to-smooth-lines-in-a-figure-in-python
interpolation with pandas: https://stackoverflow.com/questions/61338944/how-to-smooth-line-from-a-pandas-dataframe
"""

from typing import List, Protocol, Union
import pandas as pd
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d, pchip
import numpy as np


def interpolate(df, key: str, num: int = 50, kind: List[str] = ["linear"]):
	"""Plot a dataframe with interpolations.

	Args:
		df: A dataframe.
		num: Number of samples to generate. Default is 50.
		kind: The kind of interpolation specifying the interpolator to use.

		This is my current best attempt at an interpolation function. The function
		is not currently implemented because it is easily integrated into the
		SimplePlotter class with a few lines of code. However, it is useful for
		testing.
	"""
	# Get number of x ticks
	x = np.linspace(1, df.shape[0], num=df.shape[0], endpoint=True)
	x_smoothed = np.linspace(1, df.shape[0], num=num, endpoint=True)
	# Needs to be converted to use all columns
	y = np.array(df[key].values.tolist())

	# Define interpolators - only cubic and quadratic seem to create smooth curves
	interpolators = {
		"cubic": interp1d(x, y, kind="cubic"),
		"linear": interp1d(x, y),
		"nearest": interp1d(x, y, kind="nearest"),
		"nearest-up": interp1d(x, y, kind="nearest-up"),
		"next": interp1d(x, y, kind="next"),
		"quadratic": interp1d(x, y, kind="quadratic"),
		"previous": interp1d(x, y, kind="previous"),
		"slinear": interp1d(x, y, kind="slinear"),
		"zero": interp1d(x, y, kind="zero"),
	}
	# Plot
	plt.plot(x, y, "o", label="data")
	plt.figure(figsize=(15, 6))
	for func in kind:
		plt.plot(x_smoothed, interpolators[func](x_smoothed), "-", label=func)
	plt.legend(loc="best")
	plt.show()


class BasePlotter(Protocol):
	"""RWCalculator class to enable type hinting."""
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

	def __init__(self,
			width: int = 6.4,
			height: int = 4.8,
			figsize: tuple = None,
			hide_spines: List[str] = ["top", "right"],
			title: str = "Rolling Windows Plot",
			titlepad: float = 6.0,
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
			**kwargs
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
		df = None
		self.plot = None
		for k, v in kwargs.items():
			setattr(self, k, v)

	def _check_duplicate_labels(self, locations: list) -> list:
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

	def _get_label_height(self, milestone_labels: List[dict], milestone_labels_rotation: int) -> float:
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
			t = tmp_ax.annotate(x, xy=(0, 0), xytext=(0, 0), textcoords="offset points", rotation=milestone_labels_rotation)
			bb = t.get_window_extent(renderer=r)
			heights.add(bb.height)
		plt.close()
		return max(list(heights))

	def run(self, df: pd.DataFrame) -> None:
		"""Use a dataframe to plot the rolling means with pyplot.

		Args:
			df: A dataframe containing the data to plot. This is normally generated
				by a RollingWindows calculator and stored in the RollingWindows object's
				`result` attribute.
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
				titlepad = self._get_label_height(self.milestone_labels, self.milestone_labels_rotation)

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
			# Original x range
			x = np.arange(df.shape[0])

			# Projected interpolations
			xx = np.linspace(x[0], x[-1], self.interpolation_num)
			for term in df.columns:
				# Get y values
				y = np.array(df[term].values.tolist())
				# Define interpolation function
				if self.interpolation_kind == "pchip":
					interpolator = pchip(x, y)
				else:
					interpolator = interp1d(x, y, kind=self.interpolation_kind)
				plt.plot(xx, interpolator(xx), "-", label=term)
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
				locations = self._check_duplicate_labels(
					[{t.text: t.i} for t in self.doc if t._.milestone_iob == "B"]
				)
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
			raise Exception(
				"There is no plot to save, try calling `plotter.run()`."
			)
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
