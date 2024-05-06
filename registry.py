"""registry.py."""
import catalogue

from . import calculators, filters, plotters

# Create the registry
rw_components = catalogue.create("rolling_windows2", "rw_components")

# Register default calculators
rw_components.register("averages", func=calculators.Averages)

# Register default filters
rw_components.register("word_filter", func=filters.WordFilter)
rw_components.register("non_stopword_filter", func=filters.NonStopwordFilter)

# Register default plotters
rw_components.register("rw_simple_plotter", func=plotters.RWSimplePlotter)
