# Notes for DHTech Code Review

See [https://dhcodereview.github.io/](https://dhcodereview.github.io/).

## Requirements for Code Review

When getting ready to submit code for a review, please make sure to have the following information available (you will need them for the submission form):

### Link to the GitHub repository that contains the code to be reviewed

Ideally the repository would only contain the code that should be reviewed. If, however, there is more code that shouldn't be reviewed please note so in the form.

[https://github.com/scottkleinman/rollingwindows](https://github.com/scottkleinman/rollingwindows)

### What is the utilized programming language?

Python

### Who are the authors and contributors and what did they contribute?

The main developer for both the `Lexos API` and the `rollingwindows` module is Scott Kleinman. The work is based on the Lexos app, developed by a large team of developers over a number of years (with Scott Kleinman as one of the project leaders).

### Are there supporting papers (that, for example, describe employed algorithms)?

The use of rolling windows is described in [Michael D.C. Drout and Elie Chauvet (2015). “Tracking the Moving Ratio of _þ_ to _ð_ in Anglo-Saxon Texts: A New Method, and Evidence for a lost Old English version of the ‘Song of the Three Youths.’” _Anglia_ 133.2: 278-319](https://www.degruyter.com/document/doi/10.1515/ang-2015-0024/html).

The reviewer is encouraged to try out the Rolling Windows tool (under `Visualize > Rolling Window`) in the [Lexos web app](https://lexos.wheatoncollege.edu/). Instructions can be found by clicking "Help" at the top right of the user interface. Note that Lexos has been experiencing some server outages of late, so please contact me if it is not available.

### For which field was the code developed?

The code was developed for use in literary or similar text analysis with small to medium-sized corpora. The Lexos API is an attempt to create a flexible programming interface that can be used for scripting, in notebooks, or for further development of the Lexos web app. Part the intent is also to go beyond raw text analysis by enabling the use of language models and NLP tools.

### Are there any standards the reviewer should know?

No domain knowledge is required to experiment with the code, but some familiarity with [spaCy](https://spacy.io/) is helpful. The [Lexos API documentation](https://scottkleinman.github.io/lexos/) also provides extensive background about the underlying philosophy of the API and the integration of spaCy.

The project follows the [Black](https://black.readthedocs.io/en/stable/) and [isort](https://pycqa.github.io/isort/) formatting conventions. The project API documentation is automatically generated from the code docstrings using [mkdocs](https://www.mkdocs.org/). As the `rollingwindows` module is still under development, two separate API documentation files, `rolling_windows_api_docs.md` and `milestones_api_docs.md` for reference.

### Are there any requirements that need to be fulfilled for the code to run?

The code requires Python 3.9 or greater as well as the following dependencies in the Python environment.

- The Lexos API installed with `pip install lexos`. This will install most of the dependencies needed by `rollingwindows`.
- The Python `timer` module installed with `pip install timer`. There may be one or two other Python packages that you will need to install if you encounter errors, but this is the only one currently known.
- You must have downloaded spaCy's small English language model: `python -m spacy download en_core_web_sm`. The tutorial notebook assumes for demonstration purposes that you will use an English-language text with the sample data file.

### Anything else we need to know?

The code for `rollingwindows` was developed over the course of 2023, and then much of it was lost due to an unexplained corruption of my Python environment before I could push the code to GitHub. As a result, much of the code had to be reconstructed. As a result, there is still a certain amount of legacy code in the module, and function and variable names may not be as clean or informative as desired. Unit tests are also a bit messy, although the coverage is pretty good.

At present, I need some outside perspective on the following topics:

- Overall usability of the code
- How it might be streamlined
- Whether there are any broken functions or if the code fails any stress tests
- How the processing speed of the `Averages` calculator could be improved (whilst keeping a fairly consistent API).
- Whether there are additional features that users of the module might find useful.
