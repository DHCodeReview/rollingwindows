# Rolling Windows API Documentation

The Lexos Rolling Windows module is used to analyse the frequency of patterns across rolling windows (also known as sliding windows) of units in a text. The Rolling Windows module under development for the next release of the [Lexos API](https://scottkleinman.github.io/lexos/). It implements a new programming interface for the Rolling Windows tool currently available in the [Lexos web app](http://lexos.wheatoncollege.edu/rolling-window), but with added functionality. For further information on the use if Rolling Windows, users are encouraged to lick the "Help" button at the top right of the Lexos user interface and try out the web app.

## Architecture of the Module

The `__init__.py` file contains the main logic for the file, including the main `RollingWindows` class used to manage the workflow. There are three main submodules, the functions of which are listed below:

- `filters`: Classes to manage the filtering of documents prior to analysis.
- `calculators`: Classes to manage statistical calculations.
- `plotters`: Classes to manage the plotting of the results of Rolling Windows analyses.

Each submodule contains at least one class, which is registered in `registry.py`, allowing it to be treated as "built-in". Other built-in classes can be added in future releases, but users can also integrate their own custom classes to manage project-specific tasks not handled by built-in classes.

A fourth submodule, `milestones`, manages the labelling of structural divisions within documents. Since its use is not limited to Rolling Windows, it will become a component of the main Lexos library in the next release.

The file `helpers.py` contains functions used by more than one file in the module.

> [!NOTE]
> Development of the `rollingwindows` module suffered from a (still unexplained) malfunction of the development environment, which caused a catastrophic loss of much of the code before it could be pushed to GitHub. The version here is a reconstruction which works but may not be as elegant or efficient as the original code. There may also be legacy code blocks with no function in the current code that have not yet been identified.

Each component of the module is documented separately below.

## `rollingwindows.__init__`

This is the main component of the `rollingwindows` module, containing the `RollingWindows` class and associated functions.

### `rollingwindows.__init__.get_rw_component`

Gets a component from the registry using a string id. Note that this is a near duplicate of `rollingwindows.scrubber.registry.load_component`.

```python
def get_rw_component(id: str)
```

| Parameter   | Description                     | Required |
|-------------|---------------------------------|----------|
| `id`: _str_ | The string id of the component. | Yes      |

### `rollingwindows.RollingWindows`

The main class for managing the workflow and state of a Rolling Windows analysis.

```python
class RollingWindows(doc: spacy.tokens.doc.Doc, model: str, *, patterns: Union[list, str] = None)
```

#### Attributes

| Attribute                      | Description                                                                                                                                 | Required |
|--------------------------------|---------------------------------------------------------------------------------------------------------------------------------------------|----------|
| `doc`: _spacy.tokens.doc.Doc_  | A spaCy doc.                                                                                                                                | Yes      |
| `model`: _str_                 | The name of the language model. Default: `xx_sent_ud_sm`.                                                                                   | Yes      |
| `patterns`: _Union[list, str]_ | A pattern or list of patterns to search in each window. Patterns can be strings, regex patterns, or spaCy Matcher rules. Default is `None`. | No       |

The `RollingWindows.metadata` property returns dictionary recording the current configuration state.

#### Private Methods

##### `rollingwindows.RollingWindows.\_get_search_method`

Gets a preliminary search method based on the type of window unit.

```python
def _get_search_method(self, window_units: str = None) -> str
```

| Parameter             | Description                                                                                              | Required |
|-----------------------|----------------------------------------------------------------------------------------------------------|----------|
| `window_units`: _str_ | The units counted to construct windows: `characters`, `lines`, `sentences`, `tokens`. Default is `None`. | Yes      |

##### `rollingwindows.RollingWindows._get_units`

Gets a list of characters, sentences, lines, or tokens from the doc.

```python
def _get_units(self, doc: spacy.tokens.doc.Doc, window_units: str = "characters") -> Union[List[spacy.tokens.span.Span], spacy.tokens.doc.Doc]
```

| Parameter                     | Description                                                                                                    | Required |
|-------------------------------|----------------------------------------------------------------------------------------------------------------|----------|
| `doc`: _spacy.tokens.doc.Doc_ | A spaCy `Doc` object.                                                                                          | Yes      |
| `window_units`: _str_         | The units counted to construct windows: `characters`, `lines`, `sentences`, `tokens`. Default is `characters`. | Yes      |

#### Public Methods

##### `rollingwindows.RollingWindows.calculate`

Uses a calculator to generates a rolling windows analysis and assigns the result to `RollingWindows.result`.

```python
RollingWindows.calculate(calculator: Union[Callable, str] = "averages", show_spacy_rules: bool = False) -> None
```

| Parameter                            | Description                                                                                                                                                                                                            | Required |
|--------------------------------------|------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|----------|
| `calculator`: _Union[Callable, str]_ | The calculator to use. Default is the built-in "averages" calculator.                                                                                                                                                  | Yes      |
| `show_spacy_rules`: _bool_           | If the calculator uses a spaCy `Matcher` rule, tell the calculator's `to_df` method to display the rule as a column header; otherwise, only the value matched by the calculator will be displayed. Default is `False`. | Yes      |

> [!NOTE]
> For development purposes, `RollingWindows.calculate()` has a timer decorator, which will display the time elapsed when the windows are generated.

##### `rollingwindows.RollingWindows.plot`

Uses a plotter to generates a plot of rolling windows analysis and assigns the result to `RollingWindows.fig`.

```python
RollingWindows.plot(calculator: Union[Callable, str] = "rw_simple_plotter", file: str = None) -> None
```

| Parameter                         | Description                                                      | Required |
|-----------------------------------|------------------------------------------------------------------|----------|
| `plotter`: _Union[Callable, str]_ | The plotter to use. Default is the built-in "rw_simple_plotter". | Yes      |
| `file`: _str_                     | The path to a file to save the plot. Default is `None`.          | No       |

##### `rollingwindows.RollingWindows.set_windows`

Generates rolling windows, creates a `rollingwindows.Windows` object, and assigns it to `RollingWindows.windows`.

```python
RollingWindows.set_windows(n: int = 100, window_units: str = "characters", alignment_mode: str = "strict", filter:  Union[Callable, str] = None)
```

| Parameter               | Description                                                                                                                                                                                                                                                                                                                             | Required |
|-------------------------|-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|----------|
| `n`: _int_              | The number of windows to generate. Default is `1000`.                                                                                                                                                                                                                                                                                   | Yes      |
| `window_units`: _str_   | The units counted to construct windows: `characters`, `lines`, `sentences`, `tokens`. Default is `characters`.                                                                                                                                                                                                                          | Yes      |
| `alignment_mode`: _str_ | How character-based windows snap to token boundaries.<br><br>- `strict`: No snapping<br>- `contract`: Window contains all tokens _completely_ within the window's assigned start and end indices.<br>- `expand`: Window contains all tokens _partially_ within the window's assigned start and end indices.<br><br>Default is `strict`. | No       |
| `filter`: _str_         | The name of a filter or an instance of a filter to be applied to the doc before windows are generated. Default is `None`.                                                                                                                                                                                                               | No       |

!!! Note
For development purposes, `RollingWindows.set_windows()` has a timer decorator, which will display the time elapsed when the windows are generated.

### `rollingwindows.sliding_windows`

Function to create a generator of sliding windows.

```
def sliding_windows(input: Union[List[spacy.tokens.span.Span], spacy.tokens.doc.Doc], n: int = 1000, window_units: str = "characters", alignment_mode: str = "strict") -> Iterator:
```

| Parameter                                                          | Description                                                                                                                                                                                                                                                                                                                             | Required |
|--------------------------------------------------------------------|-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|----------|
| `input`: Union[List[spacy.tokens.span.Span], spacy.tokens.doc.Doc] | Either a list of spaCy `Span` objects or a spaCy `Doc` object..                                                                                                                                                                                                                                                                         | Yes      |
| `n`: _int_                                                         | The number of units per window.                                                                                                                                                                                                                                                                                                         | Yes      |
| `window_units`: _str_                                              | The units counted to construct windows: `characters`, `lines`, `sentences`, `tokens`. Default is `characters`.                                                                                                                                                                                                                          | Yes      |
| `alignment_mode`: _str_                                            | How character-based windows snap to token boundaries.<br><br>- `strict`: No snapping<br>- `contract`: Window contains all tokens _completely_ within the window's assigned start and end indices.<br>- `expand`: Window contains all tokens _partially_ within the window's assigned start and end indices.<br><br>Default is `strict`. | Yes      |

### `rollingwindows.RollingWindows.Windows`

A dataclass for storing a generator of rolling windows and associated metadata.

```python
class Windows(windows: Iterable, window_units: str, n: int, alignment_mode: str = "strict")



    def __iter__(self):

        return self.windows
```

| Parameter               | Description                                                                                                                                                                                                                                                                                                                             | Required |
|-------------------------|-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|----------|
| `windows`: _Iterable_   | The number of windows to generate. Default is `1000`.                                                                                                                                                                                                                                                                                   | Yes      |
| `window_units`: _str_   | The units counted to construct windows: `characters`, `lines`, `sentences`, `tokens`. Default is `characters`.                                                                                                                                                                                                                          | Yes      |
| `n`: _int_              | The number of units per window.                                                                                                                                                                                                                                                                                                         | Yes      |
| `alignment_mode`: _str_ | How character-based windows snap to token boundaries.<br><br>- `strict`: No snapping<br>- `contract`: Window contains all tokens _completely_ within the window's assigned start and end indices.<br>- `expand`: Window contains all tokens _partially_ within the window's assigned start and end indices.<br><br>Default is `strict`. | Yes      |

#### Private Methods

##### `rollingwindows.Windows.__iter__`

Returns the value of `self.windows`.

> [!NOTE]
> This value is a generator, so iterating through the `Windows` object will empty the values

## `rollingwindows.calculators`

Contains registered calculators. There is currently one registered calculators: `Averages`. Each calculators is an implementation of the `Calculator` protocol, which has a `metadata` property and two methods: `run` and `to_df`.

### `rollingwindows.calculators.is_valid_spacy_rule`

Applies a calculator to a document and returns a new document.

```python
def is_valid_spacy_rule(pattern: list, vocab: spacy.vocab.Vocab) -> bool
```

| Parameter                    | Description                            | Required |
|------------------------------|----------------------------------------|----------|
| `pattern`: _list_            | A pattern to test.                     | Yes      |
| `vocab`: _spacy.vocab.Vocab_ | The language model to use for testing. | Yes      |

### `rollingwindows.calculators.spacy_rule_to_lower`

Converts a spaCy `Matcher` rule to lower case.

```python
def spacy_rule_to_lower(patterns: Union[Dict, List[Dict]], old_key: Union[List[str], str] = ["TEXT", "ORTH"], new_key: str = "LOWER") -> list
```

| Parameter                             | Description                                                                | Required |
|---------------------------------------|----------------------------------------------------------------------------|----------|
| `patterns`: _Union[Dict, List[Dict]]_ | A string to match against the Roman numerals pattern.                      | Yes      |
| `old_key`: _Union[List[str], str]_    | A dictionary key or list of keys to rename. Default is `["TEXT", "ORTH"]`. | No       |
| `new_key`: _str_                      | The new key name. Default is `LOWER`.                                      | No       |

### `rollingwindows.calculators.Averages`

A calculator class to calculate rolling averages of a matched pattern in a document. The property `Averages.regex_flags` returns the flags used with methods that call the Python `re` module..

`rollingwindows.calculators.Averages` has a class attribute `id`, the value of which is "averages". This the `id` registered in the registry.

```python
class Averages(patterns: Union[list, str], windows: Windows, *, search_method: str = "count", model: str = None, doc: spacy.tokens.doc.Doc = None, alignment_mode: str = "strict", regex: bool = False, case_sensitive: bool = True, use_span_text: bool = False)
```

| Parameter                      | Description                                                                                                                                                                                                                                                                  | Required                                              |
|--------------------------------|------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|-------------------------------------------------------|
| `patterns`: _Union[list, str]_ | A pattern or list of patterns to search in windows.                                                                                                                                                                                                                          | Yes                                                   |
| `windows`: _Windows_           | A `rollingwindows.Windows` object containing the windows to search.                                                                                                                                                                                                          | Yes                                                   |
| `search_method`: _str_         | The preliminary search method to use. Default is `count`. See the explanation below.                                                                                                                                                                                         | No                                                    |
| `model`: _bool_                | The name of language model to be used with spaCy's [Matcher](https://spacy.io/usage/rule-based-matching#matcher) class. Default is `None`.                                                                                                                                   | No                                                    |
| `doc`: _spacy.tokens.doc.Doc_  | A copy of the original doc. Default is `None`.                                                                                                                                                                                                                               | No, except if `search_method` is set to `re_finditer` |
| `alignment_mode`: _str_        | Whether to snap searches to token boundaries:<br><br>- `strict`: No snapping.<br>- `contract`: Count all matches that fall _completely_ within token boundaries.<br>- `expand`: Count all matches that fall _partially_ within token boundaries.<br><br>Default is `strict`. | No                                                    |
| `regex`: _bool_                | Whether the search pattern is a to be processed as a regex string. Default is `False`.                                                                                                                                                                                       | No                                                    |
| `case_sensitive`: _bool_       | Whether to make searches case-sensitive. Default is `True`.                                                                                                                                                                                                                  | No                                                    |
| `use_span_text`: _bool_        | Use the `re_finditer` search method on `line` or `sentence` windows to search for patterns across tokens. Default is `false`.                                                                                                                                                | No                                                    |

The default `search_method` (`count`) finds exact matches within window text strings using Python's List `count()` function. The `regex` method does the same thing but allows for matching by regex pattern using Python's `re.findall()` function. If the window consists of tokens, these options will only find matches _within_ token boundaries. The `spacy_matcher` option allows for complex searches of tokens or token combinations using spaCy's [Matcher](https://spacy.io/usage/rule-based-matching#matcher) class. The `re_finditer` method performs a regex search of the full text of the window using Python's `re.finditer()` function. It can be configured with `alignment_mode` to make matches sensitive to token boundaries. Some of these options require additional configuration as specified in the table above. Note that since the `re_finditer` search method returns character start and end indices, the original doc is needed to map these onto token indices in order to use `alignment_mode`.

> [!NOTE]
> If a search methods cannot be used with the specified configuration, the calculator will try to override the value of `Averages.search_method` with a compatible search method or it will raise an error.

#### Private Methods

##### `rollingwindows.calculators.Averages._configure_search_method`

Override the initial search method based on the class configuration.

```python
def _configure_search_method(self)
```

##### `rollingwindows.calculators.Averages._count_pattern_matches`

Counts the matches for a single pattern in a single window using the configured search method.

```python
def _count_pattern_matches(self, pattern: Union[list, str], window: Union[list, spacy.tokens.span.Span, str]) -> int
```

| Parameter                                            | Description           | Required |
|------------------------------------------------------|-----------------------|----------|
| `pattern`: _Union[list, str]_                        | The pattern to match. | Yes      |
| `window`: _Union[list, spacy.tokens.span.Span, str]_ | The window to search. | Yes      |

##### `rollingwindows.calculators.Averages._extract_string_pattern`

Get the string pattern to match from a spaCy rule. For instance, if the rule is `[{"TEXT": "hello"}]`, this method will return `"hello"`.

```python
def _count_pattern_matches(self, pattern: Union[dict, list, str]) -> str
```

| Parameter                           | Description            | Required |
|-------------------------------------|------------------------|----------|
| `pattern`: _Union[dict, list, str]_ | A spaCy rule to parse. | Yes      |

##### `rollingwindows.calculators.Averages._validate_config`

Ensures that the `Averages` object is instantiated with a valid configuration.

```python
def validate_config(self, patterns: Union[list, str], windows: Windows, search_method: str) -> None
```

| Parameter                      | Description                                         | Required |
|--------------------------------|-----------------------------------------------------|----------|
| `patterns`: _Union[list, str]_ | A pattern or list of patterns to search in windows. | Yes      |
| `windows`: _Windows_           | A Windows object containing the windows to search.  | Yes      |
| `search_method`: _str_         | The name of the `search_method` to use.             | Yes      |

#### Public Methods

##### `rollingwindows.calculators.Averages.run`

Runs the calculator, which performs calculations and saves the result to `Averages.data`.

```python
def runs(self) -> None
```

##### `rollingwindows.calculators.Averages.to_df`

Converts the data in `Averages.data` to a pandas DataFrame.

```python
def to_df(self, show_spacy_rules: bool = False) -> pd.DataFrame
```

| Parameter                  | Description                                                                                                                                                                                                            | Required |
|----------------------------|------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|----------|
| `show_spacy_rules`: _bool_ | If the calculator uses a spaCy `Matcher` rule, tell the calculator's `to_df` method to display the rule as a column header; otherwise, only the value matched by the calculator will be displayed. Default is `False`. | No       |

## `rollingwindows.filters`

Contains registered filters. There are currently two registered filters: `WordFilter` and `NonStopwordFilter`. Each filter is an implementation of the `Filter` protocol, which has a `metadata` property and an `apply` method.

### `rollingwindows.filters.filter_doc`

Applies a filter to a document and returns a new document.

```python
def filter_doc(input: Union[List[spacy.tokens.span.Span], spacy.tokens.doc.Doc], n: int = 1000, window_units: str = "characters", alignment_mode: str = "strict") -> Iterator
```

| Parameter                     | Description                                                                                                                    | Required |
|-------------------------------|--------------------------------------------------------------------------------------------------------------------------------|----------|
| `doc`: _spacy.tokens.doc.Doc_ | A spaCy `Doc` object.                                                                                                          | Yes      |
| `keep_ids`: _int_             | A list of spaCy `Token` ids to keep in the filtered `Doc`.                                                                     | Yes      |
| `spacy_attrs`: _List[str]_    | A list of spaCy `Token` attributes to keep in the filtered `Doc`. Default is the `SPACY_ATTRS` list imported with `filters`.\* | No       |
| `force_ws`: _bool_            | Force a whitespace at the end of every token except the last. Default is `True`.                                               | No       |

\* The default list of spaCy token attributes can be inspected by calling `filters.SPACY_ATTRS`.

### `rollingwindows.filters.get_doc_array`

Converts a spaCy `Doc` object into a `numpy` array.

```python
def get_doc_array(doc: spacy.tokens.doc.Doc, spacy_attrs: List[str] = SPACY_ATTRS, force_ws: bool = True) -> np.ndarray
```

| Parameter                     | Description                                                                                                                    | Required |
|-------------------------------|--------------------------------------------------------------------------------------------------------------------------------|----------|
| `doc`: _spacy.tokens.doc.Doc_ | A spaCy `Doc` object.                                                                                                          | Yes      |
| `keep_ids`: _int_             | A list of spaCy `Token` ids to keep in the filtered `Doc`.                                                                     | Yes      |
| `spacy_attrs`: _List[str]_    | A list of spaCy `Token` attributes to keep in the filtered `Doc`. Default is the `SPACY_ATTRS` list imported with `filters`.\* | No       |
| `force_ws`: _bool_            | Force a whitespace at the end of every token except the last. Default is `True`.                                               | No       |

\* The default list of spaCy token attributes can be inspected by calling `filters.SPACY_ATTRS`.

The following options are available for handling whitespace:

1. `force_ws=True` ensures that `token_with_ws` and `whitespace_` attributes are preserved, but all tokens will be separated by whitespaces in the text of a doc created from the array.
2. `force_ws=False` with `SPACY` in `spacy_attrs` preserves the `token_with_ws` and `whitespace_` attributes and their original values. This may cause tokens to be merged if subsequent processing operates on the `doc.text`.
3. `force_ws=False` without `SPACY` in `spacy_attrs` does not preserve the `token_with_ws` and `whitespace_` attributes or their values. By default, `doc.text` displays a single space between each token.

### `rollingwindows.filters.is_not_roman_numeral`

Returns `True` if a token is not a Roman numeral. Works only on upper-case Roman numerals.

```python
def is_not_roman_numeral(s: str) -> bool
```

| Parameter  | Description                                           | Required |
|------------|-------------------------------------------------------|----------|
| `s`: _str_ | A string to match against the Roman numerals pattern. | Yes      |

### `rollingwindows.filters.NonStopwordFilter`

A filter class to remove stop words from a document. This is a minimal function that strips punctuation and returns the ids of words not flagged as stop words by the language model or in a list of `additional_stopwords`. The property `NonStopwordFilter.word_ids` returns the token ids for all tokens in the document that are not stop words according to these criteria.

`rollingwindows.filters.NonStopwordFilter` has a class attribute `id`, the value of which is "non_stopword_filter". This the `id` registered in the registry.

```python
class NonStopwordFilter(doc: spacy.tokens.doc.Doc, *, spacy_attrs: List[str]: SPACY_ATTRS, additional_stopwords: List[str] = None, case_sensitive: bool = False)
```

| Parameter                           | Description                                                                                                                  | Required |
|-------------------------------------|------------------------------------------------------------------------------------------------------------------------------|----------|
| `doc`: _spacy.tokens.doc.Doc_       | A spaCy `Doc` object.                                                                                                        | Yes      |
| `spacy_attrs`: _List[str]_          | A list of spaCy `Token` attributes to keep in the filtered `Doc`. Default is the `SPACY_ATTRS` list imported with `filters`. | No       |
| `additional_stopwords`: _List[str]_ | A list of stop words to add to those labelled as stop words by the model. Default is `None`.                                 | No       |
| `case_sensitive`: _bool_            | Use only lower case forms if `False`. Default is `True`.                                                                     | No       |

#### Private Methods

##### `rollingwindows.filters.NonStopwordFilter._is_non_stopword`

Returns `True` if a token is not a stop word.

```python
def _is_non_stopword(self, token: spacy.tokens.Token) -> bool
```

| Parameter                     | Description             | Required |
|-------------------------------|-------------------------|----------|
| `token`: _spacy.tokens.Token_ | A spaCy `Token` object. | Yes      |

#### Public Methods

##### `rollingwindows.filters.NonStopwordFilter.apply`

Applies the filter and returns a new, filtered doc.

```python
def apply(self) -> spacy.tokens.doc.Doc
```

### `rollingwindows.filters.WordFilter`

A filter class to remove non-words from a document. The property `WordFilter.word_ids` returns the token ids for all tokens in the document that are identified as words according to supplied criteria.

`rollingwindows.filters.WordFilter` has a class attribute `id`, the value of which is "word_filter". This the `id` registered in the registry.

```python
class WordFilter(doc: spacy.tokens.doc.Doc, *, spacy_attrs: List[str]: SPACY_ATTRS, exclude: Union[List[str], str] = [" ", "\n"], exclude_digits: bool = False, exclude_roman_numerals: bool = False, exclude_pattern: Union[List[str], str] = None)
```

| Parameter                        | Description                                                                                                                         | Required |
|----------------------------------|-------------------------------------------------------------------------------------------------------------------------------------|----------|
| `doc`: _spacy.tokens.doc.Doc_    | A spaCy `Doc` object.                                                                                                               | Yes      |
| `spacy_attrs`: _List[str]_       | A list of spaCy `Token` attributes to keep in the filtered `Doc`. Default is the `SPACY_ATTRS` list imported with `filters`.        | No       |
| `exclude`: _List[str]_           | A string/regex or list of strings/regex patterns to exclude. Default is `[" ", "\n"]`.                                              | No       |
| `exclude_digits`: _bool_         | If True, digits will not be treated as words. Default is `False`.                                                                   | No       |
| `exclude_roman_numerals`: _bool_ | If True, Roman numerals will not be treated as words. However, this only works with capitalised Roman numerals. Default is `False`. | No       |
| `exclude_pattern`: _bool_        | Additional regex patterns to add to the default `exclude` list. Default is `None`                                                   | No       |

#### Public Methods

##### `rollingwindows.filters.WordFilter.apply`

Applies the filter and returns a new, filtered doc.

```python
def apply(self) -> spacy.tokens.doc.Doc
```

## `rollingwindows.helpers`

Contains helper functions used by multiple files in the module. `rollingwindows.helpers.ensure_doc` may be legacy code that is not used in the current version.

### `rollingwindows.helpers.ensure_doc`

Converts input into a spaCy `Doc` object. The returned `Doc` is unannotated if it is derived from a string or a list of tokens.

```python
def ensure_doc(input: Union[str, List[str], spacy.tokens.doc.Doc], nlp: Union[Language, str], batch_size: int = 1000) -> spacy.tokens.doc.Doc
```

| Parameter                                              | Description                                                                 | Required |
|--------------------------------------------------------|-----------------------------------------------------------------------------|----------|
| `input`: _Union[str, List[str], spacy.tokens.doc.Doc]_ | string, list of tokens, or a spaCy doc.                                     | Yes      |
| `nlp`: \_Union[Language, str]                          | The language model to use.                                                  | Yes      |
| `batch_size`: _int_                                    | The number of texts to accumulate in an internal buffer. Default is `1000`. | No       |

### `rollingwindows.helpers.ensure_list`

Wraps any input in a list if it is not already a list.

```python
def ensure_list(input: Any) -> list
```

| Parameter      | Description        | Required |
|----------------|--------------------|----------|
| `input`: _Any_ | An input variable. | Yes      |

## `rollingwindows.plotters`

Contains registered plotters. There are currently two registered plotters: `RWSimplePlotter` and `RWPlotlyPlotter`. Each plotter is an implementation of the `BasePlotter` protocol, which has a `metadata` property and three methods: `run` , `file`, and `show`.

### `rollingwindows.plotters.interpolate`

Returns interpolated points for plots that use interpolation. The interpolation function may be either [scipy.interpolate.pchip_interpolate](https://docs.scipy.org/doc/scipy/reference/generated/scipy.interpolate.pchip_interpolate.html#scipy.interpolate.pchip_interpolate), [numpy.interp](https://numpy.org/devdocs/reference/generated/numpy.interp.html#numpy.interp), or one of the options for [scipy.interpolate.interp1d](https://docs.scipy.org/doc/scipy/reference/generated/scipy.interpolate.interp1d.html). Note however, that `scipy.interpolate.interp1d` is [deprecated](https://docs.scipy.org/doc/scipy/tutorial/interpolate/1D.html#piecewise-linear-interpolation).

```python
def interpolate(x: np.ndarray, y: np.ndarray, xx: np.ndarray, interpolation_kind: str = None) -> np.ndarray
```

| Parameter                   | Description                                           | Required |
|-----------------------------|-------------------------------------------------------|----------|
| `x`: _np.ndarray_           | The x values.                                         | Yes      |
| `y`: _np.ndarray_           | The x values.                                         | Yes      |
| `xx`: _np.ndarray_          | The projected interpolation range.                    | Yes      |
| `interpolation_kind`: _str_ | The interpolation function to use. Default is `None`. | No       |

### `rollingwindows.plotters.RWPlotlyPlotter`

Generates a plot using [Plotly](https://plotly.com/).

`rollingwindows.plotters.RWPlotlyPlotter` has a class attribute `id`, the value of which is "rw_plotly_plotter". This the `id` registered in the registry.

```python
class RWPlotlyPlotter(width: int = 700, height: int = 450, title: Union[dict, str] = "Rolling Windows Plot", xlabel: str = "Token Count", ylabel: str = "Average Frequency", line_color: str = "variable", showlegend: bool = True, titlepad: float = None, show_milestones: bool = True, milestone_marker_style: dict = {"width": 1, "color": "teal"}, show_milestone_labels: bool = False, milestone_labels: List[dict] = None, milestone_label_rotation: float = 0.0, milestone_label_style: dict = {"size": 10.0, "family": "Open Sans, verdana, arial, sans-serif", "color": "teal"}, **kwargs)
```

| Attribute                                       | Description                                                                                                                                                                                                                                                                                                                  | Required |
| ----------------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | -------- |
| `width`: _int_                                  | The figure width in pixels. Default is `700`.                                                                                                                                                                                                                                                                                | No       |
| `height`: _int_                                 | The figure height in pixels. Default is `450`.                                                                                                                                                                                                                                                                               | No       |
| `title`: _Union[dict, str]_                     | The title of the figure. Styling can be added by passing a dict with the keywords described in Plotly's [documentation](https://plotly.com/python/reference/layout/#layout-title). Default is `Rolling Windows Plot`.                                                                                                        | No       |
| `xlabel`: _str_                                 | The text to display along the x axis. Default is `Token Count`.                                                                                                                                                                                                                                                              | No       |
| `ylabel`: _str_                                 | The text to display along the y axis. Default is `Average Frequency`.                                                                                                                                                                                                                                                        | No       |
| `line_color`: _float_                           | The colour to be used for the lines on the line graph. Default is `variable`.                                                                                                                                                                                                                                                | No       |
| `showlegend`: _bool_                            | Whether to show the legend. Default is `True`.                                                                                                                                                                                                                                                                               | No       |
| `titlepad`: _float_                             | The margin in pixels between the title and the top of the graph. If not set, the margin will be calculated automatically from milestone label heights if the are shown. Default is `None`.                                                                                                                                   | No       |
| `xlabel`: _str_                                 | The text to display along the x axis. Default is `Token Count`.                                                                                                                                                                                                                                                              | No       |
| `ylabel`: _str_                                 | The text to display along the y axis. Default is `Average Frequency`.                                                                                                                                                                                                                                                        | No       |
| `show_milestones`: _bool_                       | Whether to show the milestone markers. Default is `False`.                                                                                                                                                                                                                                                                   | No       |
| `milestone_marker_style`: _dict_                | A dict containing the styles to apply to the milestone marker. For valid properties, see the Plotly [documentation](https://plotly.com/python-api-reference/generated/plotly.graph_objects.layout.shape.html#plotly.graph_objects.layout.shape.Line). Default is `{"width": 1, "color": "teal"}`.                            | No       |
| `show_milestone_labels`: _bool_                 | Whether to show the milestone labels. Default is `False`.                                                                                                                                                                                                                                                                    | No       |
| `milestone_labels`: _Dict[str, int]_            | A dict with keys as milestone labels and values as points on the x-axis. Default is `None`.                                                                                                                                                                                                                                  | No       |
| `milestone_label_rotation`: _Union[float, int]_ | The clockwise rotation of the milestone labels up to 90 degrees. Default is `0.0`.                                                                                                                                                                                                                                           | No       |
| `milestone_label_style`: _dict_                 | A dict containing the styling information for the milestone labels. For valid properties, see the Plotly [documentation](https://plotly.com/python/reference/layout/annotations/#layout-annotations-items-annotation-font). Default is `{"size": 10.0, "family": "Open Sans, verdana, arial, sans-serif", "color": "teal"}`. | No       |

> [!NOTE]
> The class seems to require a `self.plot` attribute set to `None`, although it is not used. This is something to be debugged.
>
> [!TIP]
> When milestone labels are shown and `titlepad` is not set manually, the class attempts to detect a suitable margin by using the same trick as `RWSimplePlotter`: it constructs a plot in `matplotlib` and measures the longest label to use as a guide. This produces reasonable results unless you change the figure height. In that case, it is advisable to set `titlepad` manually.
>
> Once the figure is generated, it can be accessed with `self.fig`. You can then call `self.fig.update_layout()` and modify the figure using any of the parameters available in the Plotly [documentation](https://plotly.com/python/reference/layout/). This is useful to make changes not enabled by the Lexos API.

#### Private Methods

##### `rollingwindows.plotters.RWPlotlyPlotter._check_duplicate_labels`

Adds numeric suffixes for duplicate milestone labels. Returns a dictionary containing unique keys.

```python
def _check_duplicate_labels(self, locations: List[Dict[str, int]]) -> List[Dict[str, int]]
```

| Parameter                                            | Description               | Required |
| ---------------------------------------------------- | ------------------------- | -------- |
| `locations`: _List[Dict[str, int]]_                  | A list of location dicts. | Yes      |

> [!NOTE]
> The method is not yet implemented. The documentation here is copied from `RWSimplePlotter` since it should be substantially the same. That said, the class currently requires milestones to be submitted as a dictionary, which requires unique keys. So this needs some further thought.

##### `rollingwindows.plotters.RWPlotlyPlotter._get_axis_and_title_labels`

Ensures that the `title`, `xlabel`, and `ylabel` values are dicts.

```python
def _get_axis_and_title_labels(self) -> Tuple[bool, str]
```

##### `rollingwindows.plotters.RWPlotlyPlotter._get_titlepad`

Get a titlepad value based on the height of the longest milestone label if the `titlepad` class attribute is not set.

```python
def _get_titlepad(self, labels: Dict[str, int]) -> float
```

| Parameter                                            | Description               | Required |
| ---------------------------------------------------- | ------------------------- | -------- |
| `labels`: _Dict[str, int]_                  | A dict with the labels as keys. | Yes      |

##### `rollingwindows.plotters.RWPlotlyPlotter._plot_milestone_label`

Adds a milestone label to the Plotly figure.

```python
def _plot_milestone_label(self, label: str, x: int) -> None
```

| Parameter                                            | Description               | Required |
| ---------------------------------------------------- | ------------------------- | -------- |
| `label`: _str_                  | The text of a milestone label. | Yes      |
| `x`: _int_                  | The location of the milestone label on the x axis. | Yes      |

##### `rollingwindows.plotters.RWPlotlyPlotter._plot_milestone_marker`

Adds a milestone marker (vertical line) to the Plotly figure.

```python
def _plot_milestone_marker(self, x: int, df_val_min: Union[float, int], df_val_max: Union[float, int]) -> None
```

| Parameter                                            | Description               | Required |
| ---------------------------------------------------- | ------------------------- | -------- |
| `x`: _int_                  | The location of the milestone label on the x axis. | Yes      |
| `df_val_min`: _Union[float, int]_                  | The minimum value in the pandas DataFrame. | Yes      |
| `df_val_max`: _Union[float, int]_                  | The maximum value in the pandas DataFrame. | Yes      |

#### Public Methods

##### `rollingwindows.plotters.RWPlotlyPlotter.run`

Runs the plotter saves the figure to `RWPlotlyPlotter.fig`.

```python
def runs(self, df: pd.DataFrame) -> None
```

| Parameter                | Description                                                     | Required |
|--------------------------|-----------------------------------------------------------------|----------|
| `df`: _pandas.DataFrame_ | A pandas DataFrame, normally stored in `RollingWindows.result`. | Yes      |

##### `rollingwindows.plotters.RWPlotlyPlotter.save`

Saves the plot to a file.

```python
def save(self, path: str, **kwargs) -> None
```

| Parameter     | Description                                                                                                                                        | Required |
|---------------|----------------------------------------------------------------------------------------------------------------------------------------------------|----------|
| `path`: _str_ | The path to the file where the figure is to be saved. | Yes      |

> [NOTE]
> If the path ends in `.html`, this method will attempt to save the figure as a dynamic HTML file. The method accepts any keyword available for Plotly's [`Figure.write_html`](https://plotly.github.io/plotly.py-docs/generated/plotly.io.write_html.html) method.
>
> Otherwise, it will attempt to save the figure as a static file in the format suggested the the extension in the filename  (e.g. `.png`, `.jpg`, `.pdf`). The method accepts any keyword available for Plotly's [`Figure.write_image`](https://plotly.github.io/plotly.py-docs/generated/plotly.io.write_image.html) method.

##### `rollingwindows.plotters.RWPlotlyPlotter.show`

Displays a generated figure. This method calls `matplotlib.pyplot.show`. However, since this does not work with an inline backend like Jupyter notebooks, the method tried to detect this environment via a UserWarning and then just calls the `plot` attribute.

```python
def show(self, config={"displaylogo": False}, **kwargs) -> None
```

| Parameter     | Description                                                                                                                                        | Required |
|---------------|----------------------------------------------------------------------------------------------------------------------------------------------------|----------|
| `config`: _dict_ | A dictionary supply Plotly configuration values. | No      |
```

### `rollingwindows.plotters.RWSimplePlotter`

Generates a plot using [matplotlib.pyplot](https://matplotlib.org/stable/api/pyplot_summary.html#module-matplotlib.pyplot).

`rollingwindows.plotters.RWSimplePlotter` has a class attribute `id`, the value of which is "rw_simple_plotter". This the `id` registered in the registry.

```python
class RWSimplePlotter(width: Union[float, int] = 6.4, height: Union[float, int] = 4.8, figsize: tuple = None, hide_spines: List[str] = ["top", "right"], title: str = "Rolling Windows Plot", titlepad: float = 6.0, title_position: str = "top", show_legend: bool = True, show_grid: bool = False, xlabel: str = "Token Count", ylabel: str = "Average Frequency", show_milestones: bool = False, milestone_colors: Union[List[str], str] = "teal", milestone_style: str = "--", milestone_width: int = 1, show_milestone_labels: bool = False, milestone_labels: List[dict] = None, milestone_labels_ha: str = "left", milestone_labels_va: str = "baseline", milestone_labels_rotation: int = 45, milestone_labels_offset: tuple = (-8, 4), milestone_labels_textcoords: str = "offset pixels", use_interpolation: bool = False, interpolation_num: int = 500, interpolation_kind: str = "pchip", **kwargs)
```

| Attribute                                   | Description                                                                                                                                                                                                              | Default                |
|---------------------------------------------|--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|------------------------|
| `width`: _Union[float, int]_                | The figure width in inches. Default is `6.4`.                                                                                                                                                                            | `6.4`                  |
| `height`: _Union[float, int]_               | The figure height in inches. Default is `6.4`.                                                                                                                                                                           | `6.4`                  |
| `fig_size`: _tuple_                         | A tuple containing the figure width and height in inches (overrides the `width` and `height` settings). Default is `None`.                                                                                               | `None`                 |
| `hide_spines`: _List[str]_                  | A list of ["top", "right", "bottom", "left"] indicating which spines to hide. Default is `["top", "right"]`.                                                                                                             | `["top", "right"]`     |
| `title`: _str_                              | The title to use for the plot. Default is `Rolling Windows Plot`.                                                                                                                                                        | `Rolling Windows Plot` |
| `titlepad`: _float_                         | The padding in points to place between the title and the plot, which may need to be increased if you are showing milestone labels. Default is `6.0`.                                                                     | `6.0`                  |
| `title_position`: _str_                     | Show the title on the "bottom" or the "top" of the figure. Default is `top`.                                                                                                                                             | `top`                  |
| `show_legend`: _bool_                       | Whether to show the legend. Default is `True`.                                                                                                                                                                           | `True`                 |
| `show_grid`: _bool_                         | Whether to show the grid. Default is `False`.                                                                                                                                                                            | `False`                |
| `xlabel`: _str_                             | The text to display along the x axis. Default is `Token Count`.                                                                                                                                                          | `Token Count`          |
| `ylabel`: _str_                             | The text to display along the y axis. Default is `Average Frequency`.                                                                                                                                                    | `Average Frequency`    |
| `show_milestones`: _bool_                   | Whether to show the milestone markers. Default is `False`.                                                                                                                                                               | `False`                |
| `milestone_colors`: _Union[List[str], str]_ | The colour or colours to use for milestone markers. See [pyplot.vlines()](https://matplotlib.org/stable/api/_as_gen/matplotlib.pyplot.vlines.html). Default is `teal`.                                                   | `teal`                 |
| `milestone_style`: _str_                    | The style of the milestone markers. See [pyplot.vlines()](https://matplotlib.org/stable/api/_as_gen/matplotlib.pyplot.vlines.html). Default is `--`.                                                                     | `--`                   |
| `milestone_width`: _int_                    | The width of the milestone markers. See [pyplot.vlines()](https://matplotlib.org/stable/api/_as_gen/matplotlib.pyplot.vlines.html). Default is `1`.                                                                      | `1`                    |
| `show_milestone_labels`: _bool_             | Whether to show the milestone labels. Default is `False`.                                                                                                                                                                | `False`                |
| `milestone_labels`: _List[dict]_            | A list of dicts with keys as milestone labels and values as token indexes. Default is `None`.                                                                                                                            | `None`                 |
| `milestone_labels_ha`: _str_                | The horizontal alignment of the milestone labels. See [pyplot.annotate()](https://matplotlib.org/stable/api/_as_gen/matplotlib.pyplot.annotate.html). Default is `left`.                                                 | `left`                 |
| `milestone_labels_va`: _str_                | The vertical alignment of the milestone labels. See [pyplot.annotate()](https://matplotlib.org/stable/api/_as_gen/matplotlib.pyplot.annotate.html). Default is `baseline`.                                               | `baseline`             |
| `milestone_labels_rotation`: _int_          | The rotation of the milestone labels in degrees. See [pyplot.annotate()](https://matplotlib.org/stable/api/_as_gen/matplotlib.pyplot.annotate.html). Default is `45`.                                                    | `45`                   |
| `milestone_labels_offset`: _tuple_          | A tuple containing the number of pixels along the x and y axes to offset the milestone labels. See [pyplot.annotate()](https://matplotlib.org/stable/api/_as_gen/matplotlib.pyplot.annotate.html). Default is `(-8, 4)`. | `(-8, 4)`              |
| `milestone_labels_textcoords`: _str_        | Whether to offset milestone labels by pixels or points. See [pyplot.annotate()](https://matplotlib.org/stable/api/_as_gen/matplotlib.pyplot.annotate.html). Default is `offset pixels`.                                  | `offset pixels`        |
| `use_interpolation`: _bool_                 | Whether to use interpolation on values. Default is `False`.                                                                                                                                                              | `False`                |
| `interpolation_num`: _int_                  | Number of values to add between points. Default is `500`.                                                                                                                                                                | `500`                  |
| `interpolation_kind`: _str_                 | Algorithm to use for interpolation. Default is `pchip`.                                                                                                                                                                  | `pchip`                |

If your `RollingWindows.doc` has milestones, you can display them as vertical lines on the graph. If `show_milestone_labels` is set to `True`, the first token in each milestone will be displayed as a label above the vertical line. If the labels are the same, they will be numbered consecutively ("milestone1", "milestone2", etc.). You can also submit custom labels and locations using the `milestone_labels` keyword. The other `milesone_labels_` parameters control the rotation and location of the labels.

Rolling Windows plots can often produce unattractive, squarish lines, rather than the smooth curves you often see in line graphs for some types of data. This is because there tend to be very abrupt shifts in the frequencies of patterns, rather than gradual changes. With `use_interpolation`, you can attempt to introduce smoothing by interpolating points between the values calculated by the calculator to produce a more aesthetically pleasing graph. However, the resulting plots should only be used for presentation purposes where the interpretive value is established in a non-interpolated plot. This is because interpolations can introduce distortions which may be deceptive. The user is encouraged to compare interpolated and non-interpolation plots of their analysis. The value of `interpolation_num` is the number of points to interpolate between points in your data. The `interpolation_kind` refers to the function used to interpolate the points. The default is scipy's [`interpolate.pchip_interpolate` function](https://docs.scipy.org/doc/scipy/reference/generated/scipy.interpolate.pchip_interpolate.html#scipy.interpolate.pchip_interpolate). You can also supply any of the kinds allowed by the scipy's [`interpolate.interp1d`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.interpolate.interp1d.html#scipy.interpolate.interp1d) method, although in practice, only "cubic" and "quadratic" are likely to make a difference.

#### Private Methods

##### `rollingwindows.plotters.RWSimplePlotter._check_duplicate_labels`

Adds numeric suffixes for duplicate milestone labels. Returns a list of unique location dictionaries.

```python
def _check_duplicate_labels(self, locations: List[Dict[str, int]]) -> List[Dict[str, int]]
```

| Parameter                           | Description               | Required |
|-------------------------------------|---------------------------|----------|
| `locations`: _List[Dict[str, int]]_ | A list of location dicts. | Yes      |

##### `rollingwindows.plotters.RWSimplePlotter._get_label_height`

Returns the height of the longest milestone label by using a separate plot to calculate the label height. The method is used to estimate how high to place the title above the plot.

```python
def _get_label_height(self, milestone_labels: List[dict], milestone_labels_rotation: int) -> float
```

| Parameter                          | Description                            | Required |
|------------------------------------|----------------------------------------|----------|
| `milestone_labels`: _List[dict]_   | A list of milestone_label dicts.       | Yes      |
| `milestone_labels_rotation`: _int_ | The rotation of the labels in degrees. | Yes      |

#### Public Methods

##### `rollingwindows.plotters.RWSimplePlotter.run`

Runs the plotter saves the figure to `RWSimplePlotter.plot`.

```python
def runs(self, df: pd.DataFrame) -> None
```

| Parameter                | Description                                                     | Required |
|--------------------------|-----------------------------------------------------------------|----------|
| `df`: _pandas.DataFrame_ | A pandas DataFrame, normally stored in `RollingWindows.result`. | Yes      |

##### `rollingwindows.plotters.RWSimplePlotter.save`

Saves the plot to a file. This method is a wrapper for `matplotlib.pyplot.savefig()`.

```python
def save(self, path: str) -> None
```

| Parameter     | Description                                                                                                                                        | Required |
|---------------|----------------------------------------------------------------------------------------------------------------------------------------------------|----------|
| `path`: _str_ | The path to the file where the figure is to be saved. The image type (e.g. `.png`, `.jpg`, `.pdf`) is determined by the extension on the filename. | Yes      |

##### `rollingwindows.plotters.RWSimplePlotter.show`

Displays a generated figure. This method calls `matplotlib.pyplot.show`. However, since this does not work with an inline backend like Jupyter notebooks, the method tried to detect this environment via a UserWarning and then just calls the `plot` attribute.

```python
def show(self, **kwargs) -> None
```

## `rollingwindows.registry`

A registry of "built-in" rolling windows calculators, filters, and plotters. These can be loaded using their string `id` attributes with the Python `catalogue` module.

## Custom Components

### Custom Calculators

Calculators are implemented with the `Calculator` protocol, which allows you to produce custom calculator classes. A skeleton calculator is given below.

```python
class MyCustomCalculator(Calculator):
   id: str = "my_custom_calculator"

   def __init__(
      self,
      patterns: Union[list, str],
      windows: Iterable
   ):
   """Create an instance of the calculator."""
   self.patterns = patterns
   self.windows = windows
self.data = None

   def run(self) -> spacy.tokens.doc.Doc:
   """Run the calculator."""
   ...

   def to_df(self) -> pd.DataFrame:
   """Convert the data to a pandas DataFrame."""
   ...
```

The `Calculator` protocol automatically builds a `metadata` dictionary when the class is instantiated. It requires a `run()` method to perform calculations and save the data to the object's `data` attribute. It also requires a `to_df()` method to convert the data to a pandas DataFrame. The data and DataFrame can take any format, as required for your purpose. However, if the data must be compatible with the chosen plotter. For instance, if using `rollingwindows.plotters.RWSimplePlotter`, the DataFrame must be organised with each pattern in a separate column and each window in a separate row.

### Custom Filters

Calculators are implemented with the `Filter` protocol, which allows you to produce custom filter classes. A skeleton filter is given below.

```python
class MyCustomFilter(Filter):
    id: str = "my_custom_filter"

    def __init__(
      self,
      doc: spacy.tokens.doc.Doc,
      *,
      spacy_attrs: List[str] = SPACY_ATTRS
    ):
      self.doc = doc
      self.spacy_attrs = spacy_attrs

    @property
    def filtered_token_ids(self) -> set:
        """Get a set of token_ids to keep after filtering."""
        return {
            token.i for token in self.doc
            if token.text.startswith("a")
        }

    def apply(self) -> spacy.tokens.doc.Doc:
        """Apply the filter."""
        return filter_doc(
            self.doc,
            self.filtered_token_ids,
            self.spacy_attrs
        )
```

The name of the filter is stored in the class attribute `id`. The `filtered_token_ids` property retrieves a list of token ids to keep. The `apply()` method returns a new document with all tokens not in the `filtered_token_ids` list removed. Notice that it calls the `filter_doc()` function, which is imported with `filters`. This function returns a new document in which the attribute labels have been copied from the old one. However, you may call your own function if you wish to adopt different procedure. Once you have a filtered document, you can use it to create a new `RollingWindows` instance.

!!! Note
If you wish to pass an arbitrary list of token indexes to `filter_doc()`, it is wise to pass these indexes as a set. Although, `filter_doc()` will accept a Python list, this can increase processing times from less than a second to several minutes, depending on the length of the document.

### Custom Plotters

Plotters are implemented with the `BasePlotter` protocol, which allows you to produce custom plotter classes. A skeleton plotter is given below.

```python
class MyCustomPlotter(BasePlotter):
   id: str = "my_custom_plotter"

   def __init__(self, **kwargs):
   """Create an instance of the plotter."""
   # Define any attributes here

   def file(self) -> None:
   """Save the figure to a file."""
   ...

   def run(self, data: Any) -> None:
   """Run the plotter on a set of input data."""
   ...

   def show(self) -> None:
   """Display the plot."""
   ...
```

The `Plotter` protocol automatically builds a `metadata` dictionary when the class is instantiated. The data can be passed to the `run()` method in any format as long as the `run()` method handles the logic of generating a plot from it. However, if the data is to be compatible with a built-in calculator, it must take the form of a pandas DataFrame organised with each pattern in a separate column and each window in a separate row.
