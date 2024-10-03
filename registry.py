"""registry.py.

Last Update: May 25 2024
"""
import catalogue

# from . import calculators, filters, plotters
from rollingwindows import calculators, filters, plotters

# Create the registry
rw_components = catalogue.create("rolling_windows", "rw_components")

# Register default calculators
rw_components.register("rw_calculator", func=calculators.RWCalculator)

# Register default filters
rw_components.register("word_filter", func=filters.WordFilter)
rw_components.register("non_stopword_filter", func=filters.NonStopwordFilter)

# Register default plotters
rw_components.register("rw_simple_plotter", func=plotters.RWSimplePlotter)
rw_components.register("rw_plotly_plotter", func=plotters.RWPlotlyPlotter)
