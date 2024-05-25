# `Milestones` API Documentation

Milestones are typically location designators (e.g. chapter numbers, lines numbers, or groups of sentences) that help you identify structural divisions within documents.

> [!NOTE]
> The `milestones` submodule provides useful functionality for other features of Lexos and will eventually be made into a separate module. For now, it can be accessed as a submodule of Rolling Windows.

Since milestones can span multiple tokens, each token in the document is classified using the **IOB** method (also used by spaCy's named entity recognition component). The value `I` assigned to `milestone_iob` indicates that a token is "inside" (part of) a milestone. The value `O` indicates that the token is "outside" of (not part of) a milestone. The value `B` indicates that the token is the "beginning" of (the first token in) a milestone. The `milestone_label` attribute provides a text representation of the combined tokens. Note, however, that by default it is truncated after twenty characters. Its main function is thus as a point of reference for the user.

> [!NOTE]
> Custom attributes in spaCy are accessed with the `._.` prefix, so the `milestone_iob` and `milestone_iob` values for the first token in the document would be accessed with `ms.doc[0]._.milestone_iob` and `ms.doc[0]._.milestone_label`.

If you already have a doc with milestone attributes, you can simple initialise a `Milestones` object using that doc, and the `milestone_iob` and `milestone_label` attributes will be available. If you have used the `Milestones` class to create these attributes, in most cases you will want to replace your original doc with the one in your `Milestones` object with `doc = ms.doc`.

## `lexos.milestones.helpers`

`lexos.milestones.helpers` mostly consists of deprecated functions. The only one currently used is `lexos.milestones.helpers.ensure_list`. The deprecated functions are not documented below.

### `lexos.milestones.helpers.ensure_list`

Wraps any input in a list if it is not already a list.

```python
def ensure_list(input: Any) -> list
```

| Parameter      | Description        | Required |
| -------------- | ------------------ | -------- |
| `input`: _Any_ | An input variable. | Yes      |

### `lexos.milestones.get_multiple_milestones`

Get a list of Milestone objects from a list of docs. **This function may be deprecated.**

```python
def get_multiple_milestones(docs: List[spacy.tokens.doc.Doc], nlp: str = "xx_sent_ud_sm", patterns: Any = None, case_sensitive: bool = True, mode: str = None, skip_token: bool = False, remove_token: bool = False, split_lines: bool = False, split_sentences: bool = False, step: int = None, remove_milestone: bool = True) -> List[Milestones]
```

| Parameter                            | Description                                                                                                                                                                                 | Required |
| ------------------------------------ | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | -------- |
| `docs`: _List[spacy.tokens.doc.Doc]_ | A list of spaCy `Doc` objects.                                                                                                                                                              | Yes      |
| `nlp`: _str_                         | The name of a spaCy language model. Default is `xx_sent_ud_sm`.                                                                                                                             | No       |
| `patterns`: _Any_                    | The list of patterns to match milestone spans or line breaks. If nothing is supplied, `get_line_spans()` will use the default pattern for line breaks. Default is `None`.                   | No       |
| `case_sensitive`: _bool_             | Whether to use case sensitive matching. Default is `True`.                                                                                                                                  | No       |
| `mode`: _str_                        | The mode to use for token matching. Default is `None`.                                                                                                                                      | No       |
| `skip_token`: _bool_                 | Set milestone start to the token following the milestone span. Default is `False`.                                                                                                          | No       |
| `remove_token`: _bool_               | Set milestone start to the token following the milestone span and remove the milestone span. Default is `False`.                                                                            | No       |
| `split_lines`: _bool_                | Use `set_line_spans()` instead of `set_milestones()`. Default is `False`.                                                                                                                   | No       |
| `split_sentences`: _bool_            | Use `set_sentence_spans()` instead of `set_milestones()`. Default is `False`.                                                                                                               | No       |
| `step`: _int_                        | The number of lines or sentences to include in the spans. By default, all are included.<br>remove_milestone: Whether or not to remove the linebreak using `split_lines`. Default is `None`. | No       |
| `remove_milestone`: _bool_           | Whether or not to remove the linebreak using `split_lines`. Default is `True`.                                                                                                              | No       |

### `lexos.milestones.Milestones`

Creates a `Milestones` object. The object has the property `spans`, which returns the value of Returns `Milestones.doc.spans["milestones"]`.

```python
class Milestones(doc: spacy.tokens.doc.Doc, *, nlp: str = "xx_sent_ud_sm", patterns: Any = None, case_sensitive: bool = True)
```

| Attribute                     | Description                                                              | Required |
| ----------------------------- | ------------------------------------------------------------------------ | -------- |
| `doc`: _spacy.tokens.doc.Doc_ | A spaCy `Doc` object.                                                    | Yes      |
| `nlp`: _str_                  | The name of a spaCy language model. Default is `xx_sent_ud_sm`.          | No       |
| `patterns`: _Any_             | A pattern or list of patterns to match to milestones. Default is `None`. | No       |
| `case_sensitive`: _bool_      | Whether to use case sensitive matching. Default is `True`.               | No       |

#### Private Methods

##### `lexos.milestones.Milestones.__iter__`

Returns a `Milestones` generator of `Milestones.spans`.

```python
def __iter__(self)
```

##### `lexos.milestones.Milestones._assign_token_attributes`

Assign token attributes in the doc based on spans.

```python
def _assign_token_attributes(self, spans: List[spacy.tokens.span.Span])
```

| Parameter                               | Description                     | Required |
| --------------------------------------- | ------------------------------- | -------- |
| `spans`: _List[spacy.tokens.span.Span]_ | A list of spaCy `Span` objects. | Yes      |

##### `lexos.milestones.Milestones._autodetect_mode`

Autodetect mode for matching milestones if not supplied (experimental). Returns a string to supply to the mode parameter of `lexos.milestones.Milestones.get_matches`.

```python
def _autodetect_mode(self, patterns: Any) -> str
```

| Parameter         | Description              | Required |
| ----------------- | ------------------------ | -------- |
| `patterns`: _Any_ | The pattern(s) to match. | Yes      |

##### `lexos.milestones.Milestones._get_string_matches`

Get matches to milestone patterns in strings. Returns a list of spaCy spans matching the pattern.

```python
def _get_string_matches(self, patterns: Any, flags: Enum) -> List[spacy.tokens.Span]
```

| Parameter         | Description                           | Required |
| ----------------- | ------------------------------------- | -------- |
| `patterns`: _Any_ | The pattern(s) to match.              | Yes      |
| `flags`: _Enum_   | An enum containing Python `re` flags. | Yes      |

##### `lexos.milestones.Milestones._get_phrase_matches`

Get matches to milestone patterns in phrases. Returns a list of spaCy spans matching the pattern.

```python
def _get_phrase_matches(self, patterns: Any, attr: str = "ORTH") -> List[spacy.tokens.Span]
```

| Parameter         | Description                                                                | Required |
| ----------------- | -------------------------------------------------------------------------- | -------- |
| `patterns`: _Any_ | The pattern(s) to match.                                                   | Yes      |
| `attr`: _str_     | A string indicating the spaCy token attribute to match. Default is `ORTH`. | No       |

##### `lexos.milestones.Milestones._get_rule_matches`

Get matches to milestone patterns in phrases. Returns a list of spaCy spans matching the pattern.

```python
def _get_rule_matches(self, patterns: Any) -> List[spacy.tokens.Span]
```

| Parameter         | Description                                                                | Required |
| ----------------- | -------------------------------------------------------------------------- | -------- |
| `patterns`: _Any_ | The pattern(s) to match.                                                   | Yes      |

##### `lexos.milestones.Milestones._remove_duplicate_spans`

Remove duplicate spans, generally created when a pattern is added.

```python
def _remove_duplicate_spans(self, spans: List[spacy.tokens.Span]) -> List[spacy.tokens.Span]
```

| Parameter                          | Description                     | Required |
| ---------------------------------- | ------------------------------- | -------- |
| `spans`: _List[spacy.tokens.Span]_ | A list of spaCy `Span` objects. | Yes      |

##### `lexos.milestones.Milestones._set_case_sensitivity`

Set the object's case sensitivity.

```python
def _set_case_sensitivity(self, case_sensitive: bool = True)
```

| Parameter                | Description                                                            | Required |
| ------------------------ | ---------------------------------------------------------------------- | -------- |
| `case_sensitive`: _bool_ | Whether or not to perform case-sensitive searching. Default is `True`. | Yes      |

##### `lexos.milestones.Milestones._to_spacy_span`

Convert a `re.match` object to a spaCy `Span` object.

```python
def _to_spacy_span(self, match: Match) -> spacy.tokens.Span
```

| Parameter           | Description          | Required |
| ------------------- | -------------------- | -------- |
| `match`: _re.match_ | A `re.match` object. | Yes      |

#### Public Methods

##### `lexos.milestones.Milestones.add`

Add patterns. Note that the resulting patterns are unsorted. Depending on what you are doing, you may need to call `ms.patterns = sorted(ms.patterns)`.

```python
def add(self, patterns: Any, mode: str = "string") -> None
```

| Parameter         | Description                                        | Required |
| ----------------- | -------------------------------------------------- | -------- |
| `patterns`: _Any_ | The pattern(s) to match.                           | Yes      |
| `mode`: _str_     | The mode to use for matching. Default is `string`. | No       |

##### `lexos.milestones.Milestones.get_matches`

Get matches to milestone patterns. Returns a list of spaCy spans matching the pattern.

```python
def get_matches(self, patterns: Any = None, mode: str = None, case_sensitive: bool = True)
```

| Parameter                | Description                                                                                                                                                                                                                                                                                         | Required |
| ------------------------ | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | -------- |
| `patterns`: _Any_        | The pattern(s) to match.                                                                                                                                                                                                                                                                            | Yes      |
| `mode`: _str_            | The mode to use for matching:<br><br>- `string`: Match milestone patterns in the document text.<br>- `phrase`: Match to milestone patterns in phrases.<br>- `rule`: Match to milestone patterns with spaCy rules.<br>- `sentence`: Match milestone patterns in sentences.<br><br>Default is `None`. | No       |
| `case_sensitive`: _bool_ | Whether to use case sensitive matching. Default is `True`.                                                                                                                                                                                                                                          | No       |

The `mode` parameter identifies the function to use for matching patterns. The `string` mode matches character sequences in the document's text. The `phrase` mode matches token sequences in the document using [spaCy's Phrase Matcher](https://spacy.io/usage/rule-based-matching#phrasematcher). The `rule` mode matches a spaCy Rule Matcher pattern. The `sentence` mode works somewhat differently, it uses returns a list of sentences in the document. Since it uses spaCy's sentence detection component, it will only work if that component is available in the selected language model. If no `mode` is provided, Lexos will attempt to auto-detect the most appropriate mode based on the pattern.

Pattern matching may not work as desired in RTL languages like Arabic and Hebrew. Some functions to handle RTL languages have been prototyped but are not part of this version of `Milestones`.

> [!TIP]
> The `string` mode matches patterns using regular expressions, which may occasionally cause mismatches. For instance, matching "Mr. Darcy" will return matches to "Mrs Darcy" since "." indicates any single character in regular expressions. Typically, this problem can be avoided by selecting the `phrase` mode.

> [!CAUTION]
> Calling `Milestones.get_matches()` will overwrite any pre-existing patterns. If you wish to add patterns to existing ones, use the `Milestones.add()` method, which updates the list of patterns and sets the milestones matching both the previous and the new milestones. You can also remove patterns with the `Milestones.remove()` method. Both methods accept the `mode` parameter. Finally, you can clear the pattern list by calling the `Milestones.reset()` method. This will also reset all `milestone_iob` values to "O" and all `milestone_label` values to empty strings.

##### `lexos.milestones.Milestones.remove`

Remove patterns.

```python
def remove(self, patterns: Any, mode: str = "string") -> None
```

| Parameter         | Description                                        | Required |
| ----------------- | -------------------------------------------------- | -------- |
| `patterns`: _Any_ | The pattern(s) to match.                           | Yes      |
| `mode`: _str_     | The mode to use for matching. Default is `string`. | No       |

##### `lexos.milestones.Milestones.reset`

Reset all `milestone` values to defaults. Does not modify patterns or any other settings.

```python
def reset(self)
```

##### `lexos.milestones.Milestones.set_custom_spans`

Generate spans based on a custom list. Returns a list of spaCy spans.

```python
def set_custom_spans(self, spans: List[spacy.tokens.Span], step: int = None, type: str = "custom") -> List[spacy.tokens.Span])
```

| Parameter                            | Description                                                                                                   | Required |
| ------------------------------------ | ------------------------------------------------------------------------------------------------------------- | -------- |
| `pattern`: _List[spacy.tokens.Span]_ | The string or regex pattern to use to identify the milestone.                                                 | Yes      |
| `step`: _str_                        | The number of spans to group into each milestone span. By default, all spans are included. Default is `None`. | No       |
| `step`: _str_                        | The type of span used. Default is `custom`.                                                                   | No       |

##### `lexos.milestones.Milestones.set_line_spans`

Generate spans based on line breaks. Returns a list of spaCy spans.

```python
def set_line_spans(self, pattern: str = r".+?\n", step: int = None, remove_milestone: bool = True) -> List[spacy.tokens.Span])
```

| Parameter                  | Description                                                                                                   | Required |
| -------------------------- | ------------------------------------------------------------------------------------------------------------- | -------- |
| `pattern`: _str_           | The string or regex pattern to use to identify the milestone. Default is `r".+?\n"`.                          | No       |
| `step`: _str_              | The number of spans to group into each milestone span. By default, all lines are included. Default is `None`. | No       |
| `remove_milestone`: _bool_ | Whether or not to remove the line break character. Default is `True`.                                         | No       |

##### `lexos.milestones.Milestones.set_milestones`

Commit milestones to the object instance.

```python
def set_milestones(self, spans: List[spacy.tokens.span.Span], skip_token: bool = False, remove_token: bool = False) -> None
```

| Parameter                               | Description                                                                                                      | Required |
| --------------------------------------- | ---------------------------------------------------------------------------------------------------------------- | -------- |
| `spans`: _List[spacy.tokens.span.Span]_ | The span(s) to use for identifying token attributes.                                                             | Yes      |
| `skip_token`: _bool_                    | Set milestone start to the token following the milestone span. Default is `False`.                               | No       |
| `remove_token`: _bool_                  | Set milestone start to the token following the milestone span and remove the milestone span. Default is `False`. | No       |

##### `lexos.milestones.Milestones.set_sentence_spans`

Generate spans with n sentences per span. Returns a list of spaCy spans.

```python
def set_sentence_spans(self, step: int = None) -> List[spacy.tokens.Span])
```

| Parameter                  | Description                                                                                                   | Required |
| -------------------------- | ------------------------------------------------------------------------------------------------------------- | -------- |
| `step`: _str_              | The number of spans to group into each milestone span. By default, all lines are included. Default is `None`. | No       |

##### `lexos.milestones.Milestones.to_list`

Get a list of milestone dictionaries. Some language models include a final punctuation mark in the token string, particularly at the end of a sentence. The `strip_punct` argument is a somewhat hacky convenience method to remove it. However, the user may wish instead to do some post-processing in order to use the output for their own purposes.

```python
def to_list(self, strip_punct: bool = True) -> List[dict]
```

| Parameter             | Description                                                                          | Required |
| --------------------- | ------------------------------------------------------------------------------------ | -------- |
| `strip_punct`: _bool_ | Strip single punctuation mark at the end of the character string. Default is `True`. | No       |

### `lexos.milestones.util`

`lexos.milestones.helpers` mostly consists of deprecated functions. The only one currently used is `lexos.milestones.helpers.ensure_list`. The deprecated functions are not documented below.

#### `lexos.milestones.util.chars_to_tokens`

Generate a characters to tokens mapping. Returns a dictionary mapping character indexes to token indexes.

```python
def chars_to_tokens(doc: spacy.tokens.doc.Doc) -> Dict[int, int]
```

| Parameter                     | Description           | Required |
| ----------------------------- | --------------------- | -------- |
| `doc`: _spacy.tokens.doc.Doc_ | A spaCy `Doc` object. | Yes      |

#### `lexos.milestones.util.lowercase_spacy_rules`

Converts a spaCy `Matcher` rule to lower case. Performs the same function as `rollingwindows.calculators.spacy_rule_to_lower`.

```python
def spacy_rule_to_lower(patterns: Union[Dict, List[Dict]], old_key: Union[List[str], str] = ["TEXT", "ORTH"], new_key: str = "LOWER") -> list
```

| Parameter                             | Description                                                                | Required |
| ------------------------------------- | -------------------------------------------------------------------------- | -------- |
| `patterns`: _Union[Dict, List[Dict]]_ | A string to match against the Roman numerals pattern.                      | Yes      |
| `old_key`: _Union[List[str], str]_    | A dictionary key or list of keys to rename. Default is `["TEXT", "ORTH"]`. | No       |
| `new_key`: _str_                      | The new key name. Default is `LOWER`.                                      | No       |

### `lexos.milestones.util.filter_doc`

Applies a filter to a document and returns a new document. This function is a duplicate of `rollingwindows.filters.filter_doc`.

```python
def filter_doc(input: Union[List[spacy.tokens.span.Span], spacy.tokens.doc.Doc], n: int = 1000, window_units: str = "characters", alignment_mode: str = "strict") -> Iterator
```

| Parameter                     | Description                                                                                                                | Required |
| ----------------------------- | -------------------------------------------------------------------------------------------------------------------------- | -------- |
| `doc`: _spacy.tokens.doc.Doc_ | A spaCy `Doc` object.                                                                                                      | Yes      |
| `keep_ids`: _int_             | A list of spaCy `Token` ids to keep in the filtered `Doc`.                                                                 | Yes      |
| `spacy_attrs`: _List[str]_    | A list of spaCy `Token` attributes to keep in the filtered `Doc`. Default is the `SPACY_ATTRS` list imported with `util`.\* | No       |
| `force_ws`: _bool_            | Force a whitespace at the end of every token except the last. Default is `True`.                                           | No       |

\* The default list of spaCy token attributes can be inspected by calling `util.SPACY_ATTRS`.

### `rollingwindows.filters.get_doc_array`

Converts a spaCy `Doc` object into a `numpy` array. This function is a duplicate of `rollingwindows.filters.get_doc_array`.

```python
def get_doc_array(doc: spacy.tokens.doc.Doc, spacy_attrs: List[str] = SPACY_ATTRS, force_ws: bool = True) -> np.ndarray
```

| Parameter                     | Description                                                                                                                | Required |
| ----------------------------- | -------------------------------------------------------------------------------------------------------------------------- | -------- |
| `doc`: _spacy.tokens.doc.Doc_ | A spaCy `Doc` object.                                                                                                      | Yes      |
| `keep_ids`: _int_             | A list of spaCy `Token` ids to keep in the filtered `Doc`.                                                                 | Yes      |
| `spacy_attrs`: _List[str]_    | A list of spaCy `Token` attributes to keep in the filtered `Doc`. Default is the `SPACY_ATTRS` list imported with `util`.\* | No       |
| `force_ws`: _bool_            | Force a whitespace at the end of every token except the last. Default is `True`.                                           | No       |

\* The default list of spaCy token attributes can be inspected by calling `util.SPACY_ATTRS`.

The following options are available for handling whitespace:

1. `force_ws=True` ensures that `token_with_ws` and `whitespace_` attributes are preserved, but all tokens will be separated by whitespaces in the text of a doc created from the array.
2. `force_ws=False` with `SPACY` in `spacy_attrs` preserves the `token_with_ws` and `whitespace_` attributes and their original values. This may cause tokens to be merged if subsequent processing operates on the `doc.text`.
3. `force_ws=False` without `SPACY` in `spacy_attrs` does not preserve the `token_with_ws` and `whitespace_` attributes or their values. By default, `doc.text` displays a single space between each token.
