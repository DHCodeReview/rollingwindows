# Rolling Windows

This is a development repository for a new `RollingWindows` module for the [Lexos API](https://github.com/scottkleinman/lexos). This code is not production ready. The primary purpose of this repo is to make the code available to reviewers.

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
