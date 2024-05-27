# Rolling Windows

This is a development repository for a new `RollingWindows` module for the [Lexos API](https://github.com/scottkleinman/lexos). This code is not production ready. The primary purpose of this repo is to make the code available to reviewers.

## Goals of the Project

The Lexos API is a library of methods for programmatically implementing and extending the functionality in the [Lexos](http://lexos.wheatoncollege.edu/) text analysis tool. Eventually, the web app will be rewritten to use the API directly. The goal of this alpha stage of development is to reproduce (and in some cases extend) the functionality of the current web app.

The project is motivated by the general difficulty of maintaining the web app without the sustained involvement of a team of developers familiar with the code base. Working with a well-developed API makes it possible for individuals to develop new features which can more easily be plugged into the UI at future date or serve as the basis for "mini-apps" that combine established and new features such as NLP functions. It will also enable Lexos functions to be used in other environments such as notebooks.

This hybrid goal requires a balancing act between consistency and flexibility. The code should be able to both replicate and extend the functionality of the web app. Where possible, it should function as a thin wrapper over other Python libraries (e.g. spaCy, scipy, matplotlib), providing functions that aid the management of these libraries within a Lexos workflow.

The [Lexos Rolling Windows tool](http://lexos.wheatoncollege.edu/rolling-window) was not included in the initial development of the Lexos API because of its complex code. Rolling window analysis (also called sliding-window analysis) is a method for tracing the frequency of terms within a designated window of tokens over the course of a document. It can be used to identify small- and large-scale patterns of individual features or to compare these patterns for multiple features. Rolling window analysis tabulates term frequency as part of a continuously moving metric, rather than in discrete segments. Beginning with the selection of a window, say 100 tokens, rolling window analysis traces the frequency of a term's occurrence first within tokens 1-100, then 2 to 101, then 3, 102, and so on until the end of the document is reached. The result can be plotted as a line graph so that it is possible to observe gradual changes in a token’s frequency as the text progresses.

Work was first undertaken on a `rollingwindows` in 2023 after a request from a Lexos user to find a way to label structural divisions (milestones) on the line graph. The opportunity was taken to reconceive of the tool to allow it to count features only available from NLP processing such as lemmas or parts of speech. This meant that the fundamental unit of input was changed from raw text to a spaCy `Doc` object. The output was also changed from a dynamic line graph generated with `Plotly` to a static graph generated with `matplotlib`. A rough equivalent using Plotly has been developed but has not yet been documented.

The `rollingwindows` module is meant to be extensible, with more features added over time in the form of new "calculators" (responsible for computing the rolling windows analysis) and "plotters" (responsible for visualising the results). The Lexos web app calculates either the average term frequency in each window or the ratio of term frequencies between multiple terms. At present, the module only has a class for calculating the average term frequency.

## Goals for Code Review

Feedback would be useful in the following areas:

- Overall usability of the code.
- How the code might be streamlined.
- Whether there are any broken functions or if the code fails any stress tests.
- How the processing speed of the `Averages` calculator could be improved (whilst keeping a fairly consistent API). This is a particular concern. The change from raw text to spaCy docs leads to some overhead, so that processing a text like a novel takes much longer with the API than it does in the web app.
- Whether there are additional features that users of the module might find useful.

## Installation

1. Create a Python environment (v3.9+) and activate it.
2. Install the dependencies:

```python
pip install timer
pip install lexos
python -m spacy download en_core_web_sm
```

You can learn the basic workflow by launching `tutorial_notebook.ipynb`. Full documentation of the API can be referenced in [rolling_windows_api_docs.md](https://github.com/scottkleinman/rollingwindows/blob/main/rolling_windows_api_docs.md) and [milestones_api_docs.md](https://github.com/scottkleinman/rollingwindows/blob/main/milestones_api_docs.md).

A sample document is provided in `sample_docs`, and its use is explained in the tutorial notebook.
