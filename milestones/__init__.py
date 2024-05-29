"""milestones.py.

Last Update: May 25 2024
"""

import re
from enum import Enum
from string import punctuation

# TODO: Clean up method and variable names
from typing import Any, Iterator, List, Match, Protocol

import spacy
from spacy.matcher import Matcher, PhraseMatcher
from spacy.tokens import Token

from . import helpers, util


class Milestones(Protocol): ...


class Milestones:
    """Milestones class."""

    def __init__(
        self,
        doc: spacy.tokens.doc.Doc,
        *,
        nlp: str = "xx_sent_ud_sm",
        patterns: Any = None,
        case_sensitive: bool = True,
    ):
        """Instantiate the object."""
        self.doc = doc
        self.doc.spans["milestones"] = []
        self.nlp = spacy.load(nlp)
        self.patterns = patterns
        self.character_map = None
        self.mode = None
        self.type = None
        if not Token.has_extension("milestone_iob"):
            Token.set_extension("milestone_iob", default="O", force=True)
        if not Token.has_extension("milestone_label"):
            Token.set_extension("milestone_label", default="", force=True)
        self._set_case_sensitivity(case_sensitive)

    @property
    def spans(self) -> List[spacy.tokens.span.Span]:
        """Return the doc.spans.

        Returns:
            List[spacy.tokens.span.Span]: A list of spaCy spans.
        """
        return self.doc.spans["milestones"]

    def __iter__(self) -> Iterator:
        """Make the class iterable.

        Returns:
            Iterator: A generator containing the object's spans.
        """
        return (span for span in self.spans)

    def _autodetect_mode(self, patterns: Any) -> str:
        """Autodetect mode for matching milestones if not supplied (experimental).

        Args:
            patterns (Any): A pattern to match.

        Returns:
            str: A string to supply to the get_matches() mode argument.
        """
        for pattern in patterns:
            if isinstance(pattern, str):
                if re.search(r"\s", pattern):
                    self.mode = "phrase"
                else:
                    self.mode = "string"
            else:
                try:
                    matcher = Matcher(self.nlp.vocab, validate=True)
                    matcher.add("Pattern", [pattern])
                    self.mode = "rule"
                except:
                    raise Exception(
                        f"The pattern `{pattern}` could not be matched automatically. Check that the pattern is correct and try setting the `mode` argument in `get_matches()`."
                    )
        return self.mode

    def _get_string_matches(
        self, patterns: Any, flags: Enum
    ) -> List[spacy.tokens.span.Span]:
        """Get matches to milestone patterns.

        Args:
            patterns (Any): A pattern to match.
            flags (Enum): An enum of regex flags.

        Returns:
            List[spacy.tokens.span.Span]: A list of spaCy spans matching the pattern.
        """
        if self.character_map is None:
            self.character_map = util.chars_to_tokens(self.doc)
        pattern_matches = []
        for pattern in patterns:
            matches = re.finditer(pattern, self.doc.text, flags=flags)
            for match in matches:
                pattern_matches.append(match)
        return [self._to_spacy_span(match) for match in pattern_matches]

    def _get_phrase_matches(
        self, patterns: Any, attr: str = "ORTH"
    ) -> List[spacy.tokens.span.Span]:
        """Get matches to milestone patterns in phrases.

        Args:
            patterns (Any): A pattern to match.
            attr (str): A spaCy Token attribute to search.

        Returns:
            List[spacy.tokens.span.Span]: A list of spaCy spans matching the pattern.
        """
        matcher = PhraseMatcher(self.nlp.vocab, attr=attr)
        patterns = [self.nlp.make_doc(text) for text in patterns]
        matcher.add("PatternList", patterns)
        matches = matcher(self.doc)
        return [self.doc[start:end] for _, start, end in matches]

    def _get_rule_matches(self, patterns: Any) -> List[spacy.tokens.span.Span]:
        """Get matches to milestone patterns with spaCy rules.

        Args:
            patterns (Any): A pattern to match.

        Returns:
            List[spacy.tokens.span.Span]: A list of spaCy spans matching the pattern.
        """
        spans = []
        if not self.case_sensitive:
            patterns = util.lowercase_spacy_rules(patterns)
        for pattern in patterns:
            matcher = Matcher(self.nlp.vocab, validate=True)
            matcher.add("Pattern", [pattern])
            matches = matcher(self.doc)
            spans.extend([self.doc[start:end] for _, start, end in matches])
        return spans

    def _remove_duplicate_spans(
        self, spans: List[spacy.tokens.span.Span]
    ) -> List[spacy.tokens.span.Span]:
        """Remove duplicate spans, generally created when a pattern is added.

        Args:
            spans (List[spacy.tokens.span.Span]): A list of spans

        Returns:
            List[spacy.tokens.span.Span]: A list of de-duplicated spans.
        """
        result = []
        seen = []
        for span in spans:
            if not f"{span.start},{span.end}" in seen:
                result.append(span)
                seen.append(f"{span.start},{span.end}")
        return result

    def _set_case_sensitivity(self, case_sensitive: bool = True) -> None:
        """Set the object's case sensitivity.

        Args:
            case_sensitive (bool): Whether or not to use case-sensitive searching.
        """
        if not case_sensitive:
            self.case_sensitive = False
            self.flags: Enum = re.DOTALL | re.IGNORECASE | re.MULTILINE | re.UNICODE
            self.attr = "LOWER"
        else:
            self.case_sensitive = True
            self.flags: Enum = re.DOTALL | re.MULTILINE | re.UNICODE
            self.attr = "ORTH"

    def _to_spacy_span(self, match: Match) -> spacy.tokens.span.Span:
        """Convert a re.match object to a spaCy span.

        Args:
            match (Match): A re.match object.

        Returns:
            spacy.tokens.span.Span: A spaCy span.
        """
        if self.character_map is None:
            self.character_map = util.chars_to_tokens(self.doc)
        start_char, end_char = match.span()
        span = self.doc.char_span(start_char, end_char)
        if span is None:
            start_token = self.character_map.get(start_char)
            end_token = self.character_map.get(end_char)
            if start_token is not None and end_token is not None:
                span = self.doc[start_token : end_token + 1]
        return span

    def _assign_token_attributes(self, spans: List[spacy.tokens.span.Span]):
        """Assign token attributes in the doc based on spans.

        Args:
            spans (List[spacy.tokens.span.Span]): A list of spaCy spans.
        """
        span_iter = 0
        milestone_token_starts = set()
        milestone_token_ranges = set()
        for span in spans:
            milestone_token_starts.add(span.start)
            for token in span[1:]:
                milestone_token_ranges.add(token.i)
        for token in self.doc:
            if token.i in milestone_token_starts:
                self.doc[token.i]._.milestone_iob = "B"
                span = spans[span_iter]
                text = f"{span.text:.20}{'...' if len(span.text) > 20 else ''}"
                self.doc[token.i]._.milestone_label = text
                span_iter += 1
            elif token.i in milestone_token_ranges:
                self.doc[token.i]._.milestone_iob = "I"
                self.doc[token.i]._.milestone_label = ""
            else:
                self.doc[token.i]._.milestone_iob = "O"
                self.doc[token.i]._.milestone_label = ""

    def add(self, patterns: Any, mode: str = "string") -> None:
        """Add patterns.

        Args:
            patterns (Any): The pattern(s) to match
            mode (str): The mode to use for matching

        Note:
            Resulting patterns are unsorted. Depending on what you are doing,
            you may need to call `ms.patterns = sorted(ms.patterns)`.
        """
        spans = self.get_matches(helpers.ensure_list(patterns), mode=mode)
        spans = self._remove_duplicate_spans(spans)
        self.set_milestones(spans)
        if self.patterns is None:
            self.patterns = [patterns]
        elif isinstance(patterns, str) and patterns not in self.patterns:
            self.patterns.append(patterns)
        elif isinstance(patterns, list):
            if mode == "rule":
                self.patterns.append(patterns)
            else:
                for pattern in patterns:
                    if pattern not in self.patterns:
                        self.patterns.append(pattern)

    def get_matches(
        self, patterns: Any = None, mode: str = None, case_sensitive: bool = True
    ) -> List[spacy.tokens.span.Span]:
        """Get matches to milestone patterns.

        Args:
            patterns (Any): The pattern(s) to match
            mode (str): The mode to use for matching
            case_sensitive (bool): Whether to use case sensitive matching

        Returns:
            List[spacy.tokens.span.Span]: A list of spaCy spans matching the pattern.
        """
        if case_sensitive:
            self._set_case_sensitivity(case_sensitive)
        patterns = helpers.ensure_list(patterns)
        if self.patterns:
            self.patterns += patterns
        else:
            self.patterns = patterns
        if mode == "string":
            self.mode = "string"
            spans = self._get_string_matches(patterns, self.flags)
        elif mode == "phrase":
            self.mode = "phrase"
            spans = self._get_phrase_matches(patterns, self.attr)
        elif mode == "rule":
            self.mode = "rule"
            spans = self._get_rule_matches(patterns)
        elif mode == "sentence":
            spans = self.doc.sents
        else:
            mode = self._autodetect_mode(patterns)
            return self.get_matches(patterns, mode=mode)
        return self._remove_duplicate_spans(spans)

    def remove(self, patterns: Any, mode: str = "string") -> None:
        """Remove patterns.

        Args:
            patterns (Any): The pattern(s) to match
            mode (str): The mode to use for matching
        """
        spans = self.get_matches(helpers.ensure_list(patterns), mode=mode)
        # Get a list spans to remove
        remove_spans = [f"{span.start},{span.end}" for span in spans]
        # Get a list of doc spans with the above spans removed
        new_spans = [
            span
            for span in self.doc.spans["milestones"]
            if f"{span.start},{span.end}" not in remove_spans
        ]
        # Reset the token attributes for the spans to be removed
        for span in spans:
            for i in range(span.start, span.end):
                self.doc[i]._.milestone_iob = "O"
                self.doc[i]._.milestone_label = ""
        # Re-set the milestones with the remaining spans
        self.set_milestones(new_spans)
        # Remove the patterns from the object patterns list
        if patterns in self.patterns:
            self.patterns.remove(patterns)

    def reset(self):
        """Reset all `milestone` values to defaults.

        Note: Does not modify patterns or any other settings.
        """
        self.doc.spans["milestones"] = []
        for i, _ in enumerate(self.doc):
            self.doc[i]._.milestone_iob = "O"
            self.doc[i]._.milestone_label = ""

    def set_custom_spans(
        self,
        spans: List[spacy.tokens.span.Span],
        step: int = None,
        type: str = "custom",
    ) -> List[spacy.tokens.span.Span]:
        """Generate spans based on a custom list.

        Args:
            spans (List[spacy.tokens.span.Span]): A list of spaCy spans.
            step (int): The number of spans to group into each milestone span. By default, all spans are included.
            type (str): The type of span used.

        Returns:
            List[spacy.tokens.span.Span]: A list of spaCy spans.
        """
        self.reset()
        if step:
            segments = [
                [spans[i].start, spans[i].end] for i in range(0, len(spans), step)
            ]
            # Change the end indexes to fill in gaps
            for i, segment in enumerate(segments):
                if i > 0 and segment[0] > segments[i - 1][1]:
                    segments[i - 1][1] = segment[0]
            # Use the segment start and end indexes to generate new spans
            self.doc.spans["milestones"] = [
                self.doc[segment[0] : segment[1]] for segment in segments
            ]
        else:
            self.doc.spans["milestones"] = spans
        # Set the token attributes
        for span in self.doc.spans["milestones"]:
            self.doc[span.start]._.milestone_iob = "B"
            # Truncate labels larger than 20 characters
            self.doc[span.start]._.milestone_label = (
                f"{span.text:.20}{'...' if len(span.text) > 20 else ''}"
            )
        self.type = type

    def set_line_spans(
        self, pattern: str = r".+?\n", step: int = None, remove_milestone: bool = True
    ) -> List[spacy.tokens.span.Span]:
        """Generate spans based on line breaks.

        Args:
            pattern (str): The string or regex pattern to use to identify the milestone
            step (int): The number of lines to include in the spans. By default, all lines are included.
            remove_milestone (bool): Whether or not to remove the linebreak character.

        Returns:
            List[spacy.tokens.span.Span]: A list of spaCy spans.
        """
        self.reset()
        spans = []
        for match in re.finditer(pattern, self.doc.text):
            start, end = match.span()
            if remove_milestone:
                end -= 1
            span = self.doc.char_span(start, end)
            if span is not None:
                spans.append(span)
        if step:
            segments = [
                [spans[i].start, spans[i].end] for i in range(0, len(spans), step)
            ]
            # Change the end indexes to fill in gaps
            for i, segment in enumerate(segments):
                if i > 0 and segment[0] > segments[i - 1][1]:
                    segments[i - 1][1] = segment[0]
            # Use the segment start and end indexes to generate new spans
            self.doc.spans["milestones"] = [
                self.doc[segment[0] : segment[1]] for segment in segments
            ]
        else:
            self.doc.spans["milestones"] = spans
        # Set the token attributes
        for span in self.doc.spans["milestones"]:
            self.doc[span.start]._.milestone_iob = "B"
            # Truncate labels larger than 20 characters
            self.doc[span.start]._.milestone_label = (
                f"{span.text:.20}{'...' if len(span.text) > 20 else ''}"
            )
        self.type = "lines"

    def set_milestones(
        self,
        spans: List[spacy.tokens.span.Span],
        skip_token: bool = False,
        remove_token: bool = False,
    ) -> None:
        """Commit milestones to the object instance.

        Args:
            spans (List[spacy.tokens.span.Span]): The span(s) to use for identifying token attributes.
            skip_token (bool): Set milestone start to the token following the milestone span.
            remove_token (bool): Set milestone start to the token following the milestone span and
                remove the milestone span.
        """
        if skip_token or remove_token:
            milestone_length = len(spans[0])
            # Unset all tokens
            for i, token in enumerate(self.doc):
                self.doc[i]._.milestone_iob = "O"
                self.doc[i]._.milestone_label = ""
            # Remove the milestone token
            if remove_token:
                remove_ids = set()
                for span in spans:
                    # Set the new milestone start token
                    self.doc[span.end]._.milestone_iob = "B"
                    self.doc[span.end]._.milestone_label = ""
                    # Get token ids to remove
                    for token in span:
                        remove_ids.add(token.i)
                # Filter the doc
                keep_ids = [token.i for token in self.doc if token.i not in remove_ids]
                self.doc = util.filter_doc(self.doc, keep_ids)
            else:
                # Set the span end token to B
                for span in spans:
                    self.doc[span.end]._.milestone_iob = "B"
                    self.doc[span.end]._.milestone_label = ""
        if skip_token or remove_token:
            # Unset all tokens
            for i, token in enumerate(self.doc):
                self.doc[i]._.milestone_iob = "O"
            # Set first token after each span to "B" and list ids to remove
            remove_ids = []
            milestone_length = len(spans[0])
            for span in spans:
                for token in span:
                    remove_ids.append(token.i)
                self.doc[span.end]._.milestone_iob = "B"
            # Create new spans from the tokens with "B" attributes
            new_milestones = [
                token.i for token in self.doc if token._.milestone_iob == "B"
            ]
            new_spans = [self.doc[i : i + milestone_length] for i in new_milestones]
        else:
            new_spans = spans
        self.doc.spans["milestones"] = new_spans
        self._assign_token_attributes(new_spans)
        self.type = "tokens"

    def set_sentence_spans(self, step: int = 10) -> None:
        """Generate spans with n sentences per span.

        Args:
            step (int): The number of sentences to group under a single milestone
        """
        self.reset()
        # Get a list of segments with start and end indexes
        sents = list(self.doc.sents)
        segments = [[sents[i].start, sents[i].end] for i in range(0, len(sents), step)]
        # Change the end indexes to fill in gaps
        for i, segment in enumerate(segments):
            if i > 0 and segment[0] > segments[i - 1][1]:
                segments[i - 1][1] = segment[0]
        # Use the segment start and end indexes to generate spans
        self.doc.spans["milestones"] = [
            self.doc[segment[0] : segment[1]] for segment in segments
        ]
        # Set the token attributes
        for span in self.doc.spans["milestones"]:
            self.doc[span.start]._.milestone_iob = "B"
            self.doc[span.start]._.milestone_label = (
                f"{span.text:.20}{'...' if len(span.text) > 20 else ''}"
            )
        self.type = "sentences"

    def to_list(self, strip_punct: bool = True) -> List[dict]:
        """Get a list of milestone dicts.

        Args:
            strip_punct (bool): Strip single punctation mark at the end of the character string.

        Returns:
            List[dict]: A list of milestone dicts.

        Note:
            Some language models include a final punctuation mark in the token string,
            particularly at the end of a sentence. The strip_punct argument is a
            somewhat hacky convenience method to remove it. However, the user may wish
            instead to do some post-processing in order to use the output for their
            own purposes.
        """
        milestone_dicts = []
        for span in self.doc.spans["milestones"]:
            start_char = self.doc[span.start].idx
            end_char = start_char + len(span.text)
            chars = self.doc.text[start_char:end_char]
            if strip_punct:
                chars = chars.rstrip(punctuation)
                end_char -= 1
            milestone_dicts.append(
                {
                    "text": span.text,
                    "characters": chars,
                    "start_token": span.start,
                    "end_token": span.end,
                    "start_char": start_char,
                    "end_char": end_char,
                }
            )

        return milestone_dicts


def get_multiple_milestones(
    docs: List[spacy.tokens.doc.Doc],
    nlp: str = "xx_sent_ud_sm",
    patterns: Any = None,
    case_sensitive: bool = True,
    mode: str = None,
    skip_token: bool = False,
    remove_token: bool = False,
    split_lines: bool = False,
    split_sentences: bool = False,
    step: int = None,
    remove_milestone: bool = True,
) -> List[Milestones]:
    """Get a list of Milestone objects from a list of docs.

    Args:
        docs (List[spacy.tokens.doc.Doc]): A list of docs.
        nlp (str): The language model used.
        patterns (Any): The list of patterns to match milestone spans or linebreaks. If nothing is supplied,
            `get_line_spans()` will use the default pattern for linebreaks.
        case_sensitive (bool): Whether to perform case-sensitive pattern matching.
        mode (str): The mode to use for token matching.
        skip_token (bool): Set milestone start to the token following the milestone span
        remove_token (remove_token): Set milestone start to the token following the milestone span and
            remove the milestone span
        split_lines (bool): Use `set_line_spans()` instead of `set_milestones()`.
        split_sentences (bool): Use `set_sentence_spans()` instead of `set_milestones()`.
        step (int): The number of lines or sentences to include in the spans. By default, all are included.
        remove_milestone (bool): Whether or not to remove the linebreak using `split_lines`.

    Returns:
        List[Milestones]: A list of Milestones objects.
    """
    milestone_objects = []
    for doc in docs:
        ms = Milestones(doc=doc, nlp=nlp, case_sensitive=case_sensitive)
        if split_lines:
            ms.set_line_spans(
                pattern=patterns, step=step, remove_milestone=remove_milestone
            )
        elif split_sentences:
            ms.set_sentence_spans(step=step)
        else:
            spans = ms.get_matches(patterns=patterns, mode=mode)
            ms.set_milestones(spans, skip_token=skip_token, remove_token=remove_token)
        milestone_objects.append(ms)
    return milestone_objects
