"""__init__.py.

Last Update: May 29 2024
"""

import re
from collections.abc import Iterator
from dataclasses import dataclass
from typing import Callable, Iterable, List, Union

import spacy
from spacy.tokens.doc import Doc
from timer import timer

from rollingwindows import helpers
from rollingwindows.registry import rw_components


def get_rw_component(id: str):
    """Get a component from the registry by id.

    Args:
        id (str): The registry id of the component

    Returns:
        The component class.
    """
    return rw_components.get(id)


def sliding_str_windows(
    input: Union[List[spacy.tokens.span.Span], spacy.tokens.doc.Doc, str],
    n: int = 1000,
    alignment_mode: str = "contract",
) -> Iterator:
    """Return a generator of string windows.

    Args:
        input (Union[List[spacy.tokens.span.Span], spacy.tokens.doc.Doc, str]): A spaCy doc or a list of spaCy spans.
        n (int): The size of the window.
        window_units (str): The type of units to use ("characters", "tokens", "lines", "sentences", "spans").
        alignment_mode (str): How character indices snap to token boundaries.
        - "strict" (no snapping)
        - "contract" (span of all tokens completely within the character span)
        - "expand" (span of all tokens at least partially covered by the character span)

    Returns:
        A generator of window strings.

    Note:
        Window boundaries are snapped to token boundaries in the original doc.
        "Contract" means that the window will contain all tokens completely
        within the boundaries of `i:i + n`. "Expand" means that window will
        contain all tokens partially withn those boundaries. Setting
        `alignment_mode="strict"` in `doc.char_span()` is not advised
        because it returns `None` for any string that cannot be aligned to
        token boundaries. As a result, a slice method is used if you want
        to simply cut the text strictly on `n` characters.
    """
    # TODO: We have to iterate through the input twice to get the boundaries.
    if isinstance(input, list):
        input_spans = [span.as_doc() for span in input]
        boundaries = [(i, i + n) for i in range(len(input_spans))]
        for start, end in boundaries:
            yield Doc.from_docs(input_spans[start:end]).text.strip()
    else:
        if isinstance(input, str):
            alignment_mode = "strict"
            boundaries = [(i, i + n) for i in range(len(input))]
        else:
            boundaries = [(i, i + n) for i in range(len(input.text))]
        if alignment_mode == "strict":
            for start_char, end_char in boundaries:
                span = input[start_char:end_char]
                if span is not None:
                    yield span.text
        else:
            for start_char, end_char in boundaries:
                span = input.char_span(
                    start_char, end_char, alignment_mode=alignment_mode
                )
                if span is not None:
                    yield span.text


def sliding_windows(
    input: Union[List[spacy.tokens.span.Span], spacy.tokens.doc.Doc],
    n: int = 1000,
    window_units: str = "characters",
    alignment_mode: str = "strict",
) -> Iterator:
    """Create the windows generator.

    Args:
        input (Union[List[spacy.tokens.span.Span], spacy.tokens.doc.Doc]): A spaCy doc or a list of spaCy spans.
        n (int): The size of the window.
        window_units (str): The type of units to use ("characters", "tokens", "lines", "sentences", "spans").
        alignment_mode (str): How character indices snap to token boundaries.
        - "strict" (no snapping)
        - "contract" (span of all tokens completely within the character span)
        - "expand" (span of all tokens at least partially covered by the character span)

    Yields:
        Iterator: A generator of sliding windows.
    """
    # Process character windows
    if window_units == "characters":
        boundaries = [(i, i + n) for i in range(len(input.text))]
        if alignment_mode == "strict":
            for start_char, end_char in boundaries:
                yield input.text[start_char:end_char]
        else:
            for start_char, end_char in boundaries:
                window = input.char_span(
                    start_char, end_char, alignment_mode=alignment_mode
                )
                if window is not None:
                    yield window.text

    # Process span and token windows
    elif window_units in ["lines", "sentences", "spans", "tokens"]:
        boundaries = [(i, i + n) for i in range(len(input))]
        for start, end in boundaries:
            yield input[start:end]
    else:
        raise Exception("Invalid window unit.")


# Windows class
@dataclass
class Windows:
    """A dataclass for storing rolling windows."""
    windows: Iterable
    window_units: str
    n: int
    alignment_mode: str = "strict"

    def __iter__(self):
        """Iterate over the windows."""
        return iter(self.windows)


# RollingWindows class
class RollingWindows:
    """A class for managing a rolling windows workflow."""

    def __init__(
        self,
        doc: spacy.tokens.doc.Doc,
        model: str,
        *,
        patterns: Union[list, str] = None,
    ):
        """Initialise a RollingWindows object.

        Args:
            doc (spacy.tokens.doc.Doc): A spaCy Doc object.
            model (str): The name of a spaCy model.
            patterns (Union[list, str]): The patterns to match.
        """
        self.doc = doc
        self.nlp = spacy.load(model)
        if patterns:
            self.patterns = helpers.ensure_list(patterns)
        else:
            self.patterns = []
        self.metadata = {"model": model}

    def _get_search_method(self, window_units: str = None) -> str:
        """Get the search method based on the window type.

        Args:
            window_units (str): The type of window unit.

        Returns:
            str: The preliminary search method
        """
        methods = {
            "characters": "count",
            "tokens": "spacy_matcher",
            "lines": "spacy_matcher",
            "sentences": "spacy_matcher",
        }
        return methods.get(window_units, "re_finditer")

    def _get_units(
        self, doc: spacy.tokens.doc.Doc, window_units: str = "characters"
    ) -> Union[List[spacy.tokens.span.Span], spacy.tokens.doc.Doc]:
        """Get a list of characters, sentences, lines, or tokens.

        Args:
            doc (spacy.tokens.doc.Doc): A list of spaCy spans or docs.
            window_units (str): "characters", "lines", "sentences", or "tokens".

        Returns:
            Union[List[spacy.tokens.span.Span], spacy.tokens.doc.Doc]: A list of spaCy spans or the original doc
        """
        if window_units == "sentences":
            if doc.has_annotation("SENT_START"):
                return list(doc.sents)
        elif window_units == "lines":
            regex = r"^(.+)\n+|(.+)\n+|(.+)$"
            lines = []
            for match in re.finditer(regex, doc.text):
                start, end = match.span()
                span = doc.char_span(start, end)
                if span is not None:
                    lines.append(span)
            return lines
        else:
            return doc

    @timer
    def calculate(
        self,
        patterns: Union[list, str] = None,
        calculator: Union[Callable, str] = "rw_calculator",
        query: str = "counts",
        show_spacy_rules: bool = False,
    ) -> None:
        """Set up a calculator.

        Args:
            patterns: (Union[list, str]): The patterns to search for.
                        calculator (Union[Callable, str]): The calculator to use.
            query (str): String designating whether to return "counts", "averages", or "ratios".
                        show_spacy_rules (bool): Whether to use spaCy rules or strings in column labels
        """
        if not hasattr(self, "windows"):
            raise Exception("You must call set_windows() before running calculations.")
        else:
            if calculator:
                # Use the "averages" calculator with the default config
                if isinstance(calculator, str):
                    if patterns is not None:
                        self.patterns = patterns
                    calculator = get_rw_component(calculator)
                    calculator = calculator(
                        patterns=self.patterns, windows=self.windows, query=query
                    )
            calculator.run(query=calculator.query)
            self.metadata["calculator"] = calculator.metadata
            self.result = calculator.to_df(show_spacy_rules=show_spacy_rules)

    def plot(
        self,
        plotter: Union[Callable, str] = "rw_simple_plotter",
        show: bool = False,
        file: str = None,
        **kwargs,
    ) -> None:
        """Set up the plotter.

        Args:
            plotter (Union[Callable, str]): The plotter to use.
            show (bool): Whether to show the generated figure.
            file (str): The filepath to save the file, if desired.
        """
        if not hasattr(self, "result") or self.result is None:
            raise Exception(
                "You must run a calculator on your data before generating a plot."
            )
        # Use the "rw_simple_plotter" plotter with the default config
        if isinstance(plotter, str):
            plotter = get_rw_component(plotter)
            plotter = plotter()
        plotter.run(self.result)
        self.metadata["plotter"] = plotter.metadata
        self.fig = plotter.fig
        if show:
            plotter.show(**kwargs)
        if file:
            plotter.save(file, **kwargs)

    # @timer
    def set_windows(
        self,
        n: int = 1000,
        window_units: str = "characters",
        *,
        alignment_mode: str = "strict",
        filter: Union[Callable, str] = None,
    ) -> None:
        """Set the object's windows.

        Args:
            n (int): The number of windows to calculate
            window_units (str): "characters", "lines", "sentences", or "tokens".
            alignment_mode (str): How character indices snap to token boundaries.
            - "strict" (no snapping)
            - "contract" (span of all tokens completely within the character span)
            - "expand" (span of all tokens at least partially covered by the character span)
            filter (Union[Callable, str]): The name of a filter or a filter object to apply to the document.
        """
        if filter:
            # Use the filter with the default config
            if isinstance(filter, str):
                filter = get_rw_component(filter)
                filter = filter(self.doc)
            doc = filter.apply()
        else:
            doc = self.doc
        # _get_units() returns either a doc or a list of spans. The doc is used to slide over
        # characters or tokens, and the list is used to slide over sentences or lines.
        input = self._get_units(doc, window_units)
        # sliding_windows() returns a generator containing with string or span windows.
        if window_units == "characters":
            if isinstance(input, list):
                input = " ".join([span.text for span in input])
            windows = sliding_str_windows(input, n, alignment_mode)
        else:
            windows = sliding_windows(input, n, window_units, alignment_mode)
        # Since spans windows are lists of multiple spans, we need to get the first and last
        # token from the original doc to get a window that combines them into a single span.
        if window_units in ["lines", "sentences", "spans"]:
            span_windows = (doc[window[0].start : window[-1].end] for window in windows)
            self.windows = Windows(span_windows, window_units, n, alignment_mode)
        else:
            self.windows = Windows(windows, window_units, n, alignment_mode)
        # For convenience's sake, we detect the search method here, but the calculator
        # will override it based on the pattern.
        search_method = self._get_search_method(window_units)
        metadata = {
            "n": n,
            "window_units": window_units,
            "alignment_mode": alignment_mode,
            "search_method": search_method,
        }
        if filter:
            metadata["filter"] = filter.metadata
        else:
            self.metadata.pop("filter", None)
        self.metadata = self.metadata | metadata
