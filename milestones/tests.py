"""tests.py.

Some tests may not work in the version included in `rollingwindows`.

Last Update: May 25 2024
"""

import en_core_web_sm
import pytest
import spacy
import xx_sent_ud_sm
from smart_open import open
from spacy.matcher import Matcher, PhraseMatcher

nlp = xx_sent_ud_sm.load()

from lexos.milestones import Milestones, get_multiple_milestones


@pytest.fixture
def doc():
    filename = (
        "C:/Users/scott/Documents/apis/lexos/tests/test_data/txt/Austen_Pride_sm.txt"
    )
    with open(filename, "r") as f:
        doc = nlp(f.read().strip())
    return doc


def ttest_mode_detection(doc):
    """Check that a mode can be detected."""
    patterns = ["Chapter"]
    ms = Milestones(doc)
    spans = ms.get_matches(patterns)
    ms.set_milestones(spans)
    assert ms.mode == "string"
    assert ms.patterns == patterns
    assert ms.type == "tokens"
    patterns = ["Chapter 1"]
    ms = Milestones(doc)
    spans = ms.get_matches(patterns)
    ms.set_milestones(spans)
    assert ms.mode == "phrase"
    assert ms.patterns == patterns
    assert ms.type == "tokens"
    patterns = [[{"TEXT": "Chapter"}]]
    ms = Milestones(doc)
    spans = ms.get_matches(patterns)
    ms.set_milestones(spans)
    assert ms.mode == "rule"
    assert ms.patterns == patterns
    assert ms.type == "tokens"


def ttest_string_mode(doc, patterns=["Chapter"]):
    """Check that a string pattern can be set."""
    ms = Milestones(doc)
    spans = ms.get_matches(patterns, mode="string")
    ms.set_milestones(spans)
    assert ms.mode == "string"
    assert ms.patterns == patterns
    assert ms.type == "tokens"
    string_matches = [token for token in ms.doc if token.text == patterns[0]]
    for token in string_matches:
        assert token._.milestone_iob == "B"
        assert token._.milestone_label == patterns[0]


def ttest_string_mode_case_insensitive(doc, patterns=["Chapter"]):
    """Check that a string pattern can be set with `case_insensitive=False`."""
    ms = Milestones(doc, case_sensitive=False)
    assert ms.case_sensitive == False
    spans = ms.get_matches(patterns, mode="string")
    ms.set_milestones(spans)
    if ms.case_sensitive:
        ms_patterns = ms.patterns
        input_patterns = patterns
    else:
        ms_patterns = [pattern.lower() for pattern in ms.patterns]
        input_patterns = [pattern.lower() for pattern in patterns]
    assert ms_patterns == input_patterns


def ttest_skip_token(doc, patterns=["Chapter"]):
    """Check that IOB labels are correct with `skip_token`."""
    ms = Milestones(doc)
    spans = ms.get_matches(patterns, mode="string")
    ms.set_milestones(spans, skip_token=True)
    string_matches = [token for token in ms.doc if token.text == patterns[0]]
    for token in string_matches:
        next_token = ms.doc[token.i + 1]
        assert token._.milestone_iob == "O"
        assert next_token._.milestone_iob == "B"


def test_remove_token(doc, patterns=[[{"TEXT": "Chapter"}, {"IS_DIGIT": True}]]):
    """Check that IOB labels are correct with `remove_token`."""
    ms = Milestones(doc)
    spans = ms.get_matches(patterns, mode="rule")
    before = [doc[span.start + len(span) + 1].text for span in spans]
    # for span in spans:
    #     # print(span.start, span.end + len(span))
    #     print(doc[span.start + len(span)].text)
    ms.set_milestones(spans, remove_token=True)
    print("=====")
    after = [token for token in ms.doc if token._.milestone_iob == "B"]
    print(after)
    # for i, text in enumerate(before):
    #     print((text, f"-{after[i + 2]}-"))
    # assert text == after[i]
    # print(span.start, span.end)
    # print(ms.doc[span.start + 1].text)
    assert 1 == 2


def ttest_phrase_mode(doc, patterns=["Chapter I"]):
    """Check that a phrase pattern can be set."""
    ms = Milestones(doc)
    spans = ms.get_matches(patterns, mode="phrase")
    ms.set_milestones(spans)
    assert ms.mode == "phrase"
    assert ms.patterns == patterns
    assert ms.type == "tokens"
    matcher = PhraseMatcher(nlp.vocab)
    matcher.add("Test", [nlp.make_doc(text) for text in patterns])
    matches = matcher(ms.doc)
    phrase_matches = [list(range(start, end)) for _, start, end in matches]
    label_matches = [ms.doc[start:end].text for _, start, end in matches]
    for match in phrase_matches:
        assert ms.doc[match[0]]._.milestone_iob == "B"
        for i in match[1:]:
            assert ms.doc[match[i]]._.milestone_iob == "I"
    assert spans == label_matches


def ttest_phrase_mode_case_insensitive(doc, patterns=["Chapter"]):
    """Check that a phrase pattern can be set with `case_insensitive=False`."""
    ms = Milestones(doc, case_sensitive=False)
    assert ms.case_sensitive == False
    spans = ms.get_matches(patterns, mode="phrase")
    span_matches = [span.text.lower() for span in spans]
    matcher = PhraseMatcher(nlp.vocab)
    matcher.add("Test", [nlp.make_doc(text) for text in patterns])
    matches = matcher(ms.doc)
    label_matches = [ms.doc[start:end].text.lower() for _, start, end in matches]
    assert span_matches == label_matches


def ttest_rule_mode(doc, patterns=[[{"TEXT": "Chapter"}, {"TEXT": "I"}]]):
    """Check that a rule pattern can be set."""
    ms = Milestones(doc)
    spans = ms.get_matches(patterns, mode="rule")
    ms.set_milestones(spans)
    assert ms.mode == "rule"
    assert ms.patterns == patterns
    assert ms.type == "tokens"
    matcher = Matcher(nlp.vocab)
    matcher.add("Test", patterns)
    matches = matcher(ms.doc)
    rule_matches = [list(range(start, end)) for _, start, end in matches]
    label_matches = [ms.doc[start:end].text for _, start, end in matches]
    for match in rule_matches:
        assert ms.doc[match[0]]._.milestone_iob == "B"
        for i in match[1:]:
            assert ms.doc[match[i]]._.milestone_iob == "I"
    assert spans == label_matches


def ttest_rule_mode_case_insensitive(
    doc, patterns=[[{"TEXT": "Chapter"}, {"TEXT": "I"}]]
):
    """Check that a phrase pattern can be set with `case_insensitive=False`."""
    ms = Milestones(doc, case_sensitive=False)
    assert ms.case_sensitive == False
    spans = ms.get_matches(patterns, mode="rule")
    span_matches = [span.text.lower() for span in spans]
    matcher = Matcher(nlp.vocab)
    matcher.add("Test", patterns)
    matches = matcher(ms.doc)
    label_matches = [ms.doc[start:end].text.lower() for _, start, end in matches]
    assert span_matches == label_matches


def ttest_add(doc, patterns=["Chapter I"]):
    """Check the add method."""
    ms = Milestones(doc, patterns=patterns)
    assert ms.patterns == patterns
    add_pattern = ["Chapter II"]
    ms.add(add_pattern)
    if isinstance(add_pattern, list):
        patterns = patterns + add_pattern
    else:
        patterns = patterns + [add_pattern]
    patterns = sorted(list(set(patterns)))
    assert sorted(ms.patterns) == patterns


def ttest_remove(doc, patterns=["Chapter I", "Chapter II"]):
    """Check the remove method."""
    ms = Milestones(doc, patterns=patterns)
    matcher = PhraseMatcher(nlp.vocab)
    matcher.add("Test", [nlp.make_doc(text) for text in patterns])
    matches = matcher(ms.doc)
    assert ms.patterns == patterns
    patterns = ["Chapter II"]
    for pattern in patterns:
        ms.remove(pattern)
    for match in matches:
        assert match.text not in ms.patterns


def ttest_reset(doc, patterns=["Chapter I"]):
    """Check the reset method."""
    ms = Milestones(doc)
    spans = ms.get_matches(patterns, mode="string")
    ms.set_milestones(spans)
    assert ms.mode == "string"
    assert ms.patterns == patterns
    assert ms.type == "tokens"
    string_matches = [token for token in ms.doc if token.text == patterns[0]]
    for token in string_matches:
        assert token._.milestone_iob == "B"
        assert token._.milestone_label == patterns[0]
    ms.reset()
    for token in string_matches:
        assert token._.milestone_iob == "O"
        assert token._.milestone_label == ""


def ttest_to_list(doc, patterns=["Chapter"]):
    """Check that a string pattern can be set."""
    ms = Milestones(doc)
    spans = ms.get_matches(patterns, mode="string")
    ms.set_milestones(spans)
    matcher = PhraseMatcher(nlp.vocab)
    matcher.add("Test", [nlp.make_doc(text) for text in patterns])
    matches = matcher(ms.doc)
    output = ms.to_list()
    assert isinstance(output, list)
    assert output[0]["text"] == patterns[0]
    for i, match in enumerate(matches):
        assert "text" in output[i]
        assert "characters" in output[i]
        assert "start_char" in output[i]
        assert "end_char" in output[i]
        assert output[i]["start_token"] == match[1]
        assert output[i]["end_token"] == match[2]


def ttest_sentence_spans(doc, step=50, nlp="en_core_web_sm"):
    """Check that sentence spans matches sentences."""
    ms = Milestones(doc, nlp=nlp)
    ms.set_sentence_spans(step=step)
    # Ensure that the last token index for the span step is the same as
    # expected for the doc's sents.
    last_span_token_index = ms.spans[0][-1].i
    last_sent_token_index = list(ms.doc.sents)[step - 1][-1].i
    assert last_span_token_index == last_sent_token_index


def ttest_line_spans(doc, step=50):
    """Check that line spans."""
    ms = Milestones(doc)
    ms.set_line_spans(step=step)
    # Ensure that the last token index for the span step is the same as
    # expected for the doc's sents.
    last_span_token = ms.spans[0][-1].text
    assert last_span_token == "\n"


def ttest_custom_spans(doc, step=2, type="custom"):
    """Check custom spans."""
    # Extract a list of custom spans (every 10 tokens) from doc
    spans = [doc[0:10], doc[10:20], doc[20:30], doc[30:40]]
    step_starts = [span.start for span in spans][::step]
    expected_length = len(step_starts)
    ms = Milestones(doc)
    ms.set_custom_spans(spans, step=step, type=type)
    assert ms.type == type
    spans_length = len(ms.spans)
    assert spans_length == expected_length
    for i, span in enumerate(ms.spans):
        assert span.start == step_starts[i]


def ttest_get_multiple_milestones(docs=[doc, doc], nlp=nlp):
    """Check get_multiple_milestones.

    Args:
        patterns: The list of patterns to match milestone spans or linebreaks. If nothing is supplied,
            `get_line_spans()` will use the default pattern for linebreaks.
        case_sensitive: Whether to perform case-sensitive pattern matching.
        mode: The mode to use for token matching.
        # For set_milestones
        skip_token: Set milestone start to the token following the milestone span
        remove_token: Set milestone start to the token following the milestone span and
            remove the milestone span
        # If these exist, call the appropriate method
        split_lines: Use `set_line_spans()` instead of `set_milestones()`.
        split_sentences: bool = Use `set_sentence_spans()` instead of `set_milestones()`.
        step: The number of lines or sentences to include in the spans. By default, all are included.
        remove_milestone: Whether or not to remove the linebreak using `split_lines`.
    """
    patterns = ["Chapter"]
    case_sensitive = True
    mode = "string"
    skip_token = True
    remove_token = True
    split_lines = True
    split_sentences = True
    step = 50
    remove_milestone = True
    if split_lines:
        milestones = get_multiple_milestones(
            docs,
            nlp,
            patterns,
            case_sensitive,
            mode,
            split_lines,
            step,
            remove_milestone,
        )
    elif split_sentences:
        milestones = get_multiple_milestones(
            docs,
            nlp,
            patterns,
            case_sensitive,
            mode,
            split_sentences,
            step,
            remove_milestone,
        )
    else:
        milestones = get_multiple_milestones(
            docs,
            nlp,
            patterns,
            case_sensitive,
            mode,
            skip_token,
            remove_token,
            step,
            remove_milestone,
        )
