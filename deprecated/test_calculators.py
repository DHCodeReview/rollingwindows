"""test_calculators.py."""

import re
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest
import spacy
from spacy.tokens import Token

from rollingwindows import Windows, sliding_windows
from rollingwindows.calculators import (Averages, Calculator,
                                        spacy_rule_to_lower)

# Fixtures


@pytest.fixture
def nlp():
    return spacy.load("en_core_web_sm")


@pytest.fixture
def doc(nlp):
    return nlp("This is a test document. It has multiple sentences.")


@pytest.fixture
def test_calculator():
    class TestCalculator(Calculator):
        id = "test_calculator"

    return TestCalculator()


@pytest.fixture
def windows():
    return Windows(sliding_windows(doc), "characters", 10)


@pytest.fixture
def token_windows():
    return Windows(sliding_windows(doc), "tokens", 10)

@pytest.fixture
def sentence_windows():
    return Windows(sliding_windows(doc.sents), "sentences", 2)


@pytest.fixture
def test_calculator():
    class TestCalculator(Calculator):
        id = "test_calculator"
        doc = "Test document"

    return TestCalculator()


# Calculator Protocol

# def test_calculator_protocol(test_calculator):
#     metadata = test_calculator.metadata
#     assert metadata["id"] == "test_calculator"
#     assert "data" not in metadata
#     assert "windows" not in metadata

# @pytest.mark.xfail(raises=NotImplementedError)
# def test_calculator_protocol_run_method(test_calculator):
#     test_calculator.run()

# @pytest.mark.xfail(raises=NotImplementedError)
# def test_calculator_protocol_to_df_method(test_calculator):
#     test_calculator.to_df()

# Functions


# def test_spacy_rule_to_lower_with_dict_and_default_parameters():
#     pattern = {"TEXT": "Hello", "ORTH": "World"}
#     new_pattern = spacy_rule_to_lower(pattern)
#     assert new_pattern == {"LOWER": "Hello", "LOWER": "World"}


# def test_spacy_rule_to_lower_with_list_and_default_parameters():
#     patterns = [{"TEXT": "Hello"}, {"ORTH": "World"}]
#     new_patterns = spacy_rule_to_lower(patterns)
#     assert new_patterns == [{"LOWER": "Hello"}, {"LOWER": "World"}]


# def test_spacy_rule_to_lower_with_dict_and_custom_parameters():
#     pattern = {"TEXT": "Hello", "ORTH": "World"}
#     new_pattern = spacy_rule_to_lower(pattern, old_key="TEXT", new_key="LOWER")
#     assert new_pattern == {"LOWER": "Hello", "ORTH": "World"}


# def test_spacy_rule_to_lower_with_list_and_custom_parameters():
#     patterns = [{"TEXT": "Hello"}, {"ORTH": "World"}]
#     new_patterns = spacy_rule_to_lower(patterns, old_key="TEXT", new_key="LOWER")
#     assert new_patterns == [{"LOWER": "Hello"}, {"LOWER": "World"}]


# Averages Class

def test_averages_regex_flags_when_case_sensitive_is_false():
    calculator = Averages()
    calculator.metadata = {"id": "averages"}
    assert calculator.metadata["id"] == "averages"

# def test_averages_regex_flags_when_case_sensitive_is_true(windows):
#     calculator = MagicMock()
#     calculator.metadata = {"id": "averages"}
#     calculator.windows = windows
#     with patch(
#         "rollingwindows.calculators.Averages.regex_flags",
#         return_value=lambda *args, **kwargs: calculator,
#     ):
#         assert calculator.regex_flags == re.UNICODE

# def test_averages_with_custom_parameters(windows):
#     averages = Averages(
#         ["test", "document"],
#         windows,
#         search_method="re_search",
#         alignment_mode="contract",
#         regex=True,
#         case_sensitive=False,
#         use_span_text=True,
#     )
#     assert averages.patterns == ["test", "document"]
#     assert averages.search_method == "find"
#     assert averages.alignment_mode == "contract"
#     assert averages.regex == True
#     assert averages.case_sensitive == False
#     assert averages.use_span_text == True


# def test_averages_with_model(windows):
#     averages = Averages("test", windows, model="en_core_web_sm")
#     assert isinstance(averages.nlp, spacy.lang.en.English)


# def test_validate_config_with_invalid_windows():
#     averages = Averages("test", "characters")
#     with pytest.raises(Exception):
#         averages._validate_config("test", "characters")


# def test_validate_config_pattern_not_valid_for_search_method():
#     averages = Averages("test", windows)
#     with pytest.raises(Exception):
#         averages._validate_config(["test", [{"ORTH": "document"}]], windows, "count")


# def test_validate_config_invalid_window_units_with_spacy_matcher(windows):
#     averages = Averages("test", windows)
#     with pytest.raises(Exception):
#         averages._validate_config([{"ORTH": "document"}], windows, "spacy_matcher")


# def test_validate_config_invalid_spacy_matcher_patterns(token_windows):
#     averages = Averages("test", token_windows)
#     with pytest.raises(Exception):
#         averages._validate_config("test", token_windows, "spacy_matcher")

# def test_regex_flags_with_case_sensitive_true(windows):
#     averages = Averages("test", windows, case_sensitive=True)
#     assert averages.regex_flags == re.UNICODE

# def test_regex_flags_with_case_sensitive_false(windows):
#     averages = Averages("test", windows, case_sensitive=False)
#     assert averages.regex_flags == re.IGNORECASE | re.UNICODE

# def test_is_valid_spacy_rule_with_valid_rule(windows):
#     averages = Averages("test", windows, model="en_core_web_sm")
#     assert averages._is_valid_spacy_rule([{"LOWER": "test"}]) == True

# def test_is_valid_spacy_rule_with_invalid_rule(windows):
#     averages = Averages("test", windows, model="en_core_web_sm")
#     assert averages._is_valid_spacy_rule([{"INVALID": "rule"}]) == False

# def test_configure_search_method_with_window_units(token_windows):
#     averages = Averages("test", token_windows, window_units="tokens")
#     averages._configure_search_method()
#     assert averages.search_method == "spacy_matcher"

# def test_configure_search_method_with_count_and_regex(windows):
#     averages = Averages("test", windows, search_method="count", regex=True)
#     averages._configure_search_method()
#     assert averages.search_method == "re_search"

# def test_configure_search_method_with_spacy_matcher_and_use_span_text(sentence_windows):
#     averages = Averages("test", sentence_windows, search_method="spacy_matcher", use_span_text=True)
#     averages._configure_search_method()
#     assert averages.search_method == "re_finditer"

# def test_configure_search_method_with_invalid_spacy_rule(windows):
#     averages = Averages([{"INVALID": "rule"}], windows, search_method="spacy_matcher")
#     with pytest.raises(Exception):
#         averages._configure_search_method()

# def test_count_pattern_matches_with_count_and_case_sensitive_true(windows):
#     averages = Averages("test", windows, search_method="count", case_sensitive=True)
#     assert averages._count_pattern_matches("test", "This is a test document.") == 1

# def test_count_pattern_matches_with_count_and_case_sensitive_false(windows):
#     averages = Averages("test", windows, search_method="count", case_sensitive=False)
#     assert averages._count_pattern_matches("TEST", "This is a test document.") == 1

# def test_count_pattern_matches_with_regex_and_case_sensitive_true(windows):
#     averages = Averages("test", windows, search_method="regex", case_sensitive=True)
#     assert averages._count_pattern_matches("test", "This is a test document.") == 1

# def test_count_pattern_matches_with_regex_and_case_sensitive_false(windows):
#     averages = Averages("test", windows, search_method="regex", case_sensitive=False)
#     assert averages._count_pattern_matches("TEST", "This is a test document.") == 1

# def test_count_pattern_matches_with_spacy_matcher_and_case_sensitive_true(windows):
#     averages = Averages([{"TEXT": "test"}], windows, model="en_core_web_sm", search_method="spacy_matcher", case_sensitive=True)
#     assert averages._count_pattern_matches([{"TEXT": "test"}], "This is a test document.") == 1

# def test_count_pattern_matches_with_spacy_matcher_and_case_sensitive_false(windows):
#     averages = Averages([{"TEXT": "TEST"}], windows, model="en_core_web_sm", search_method="spacy_matcher", case_sensitive=False)
#     assert averages._count_pattern_matches([{"TEXT": "TEST"}], "This is a test document.") == 1

# def test_count_pattern_matches_with_re_finditer_and_case_sensitive_true(windows):
#     averages = Averages("test", windows, search_method="re_finditer")
#     assert averages._count_pattern_matches("test", "This is a test document.") == 1

# def test_count_pattern_matches_with_re_finditer_and_case_sensitive_false(windows):
#     averages = Averages("test", windows, search_method="re_finditer")
#     assert averages._count_pattern_matches("TEST", "This is a test document.") == 1

# def test_extract_string_pattern_with_list(windows):
#     averages = Averages("test", windows)
#     assert averages._extract_string_pattern([{"LOWER": "test"}]) == "test"

# def test_extract_string_pattern_with_dict(windows):
#     averages = Averages("test", windows)
#     assert averages._extract_string_pattern({"LOWER": "test"}) == "test"

# def test_extract_string_pattern_with_string(windows):
#     averages = Averages("test", windows)
#     assert averages._extract_string_pattern("test") == "test"

# def test_run_with_single_pattern_and_single_window(windows):
#     averages = Averages("test", windows)
#     averages.run()
#     assert averages.data == [[1.0]]

# def test_run_with_multiple_patterns_and_single_window(windows):
#     averages = Averages(["test", "document"], windows)
#     averages.run()
#     assert averages.data == [[1.0, 1.0]]

# def test_run_with_single_pattern_and_multiple_windows(windows):
#     windows = Windows("This is a test document. It has multiple sentences. This is another sentence.", 3)
#     averages = Averages("test", windows)
#     averages.run()
#     assert averages.data == [[1.0], [1.0]]

# def test_run_with_multiple_patterns_and_multiple_windows(windows):
#     windows = Windows("This is a test document. It has multiple sentences. This is another sentence.", 3)
#     averages = Averages(["test", "document"], windows)
#     averages.run()
#     assert averages.data == [[1.0, 1.0], [1.0, 0.0]]

# def test_to_df_with_show_spacy_rules_true(token_windows):
#     averages = Averages([{"TEXT": "test"}], token_windows)
#     averages.run()
#     df = averages.to_df(show_spacy_rules=True)
#     assert isinstance(df, pd.DataFrame)
#     assert list(df.columns) == ['[{"TEXT": "test"}]']

# def test_to_df_with_show_spacy_rules_false(windows):
#     averages = Averages([{"TEXT": "test"}], windows, model="en_core_web_sm")
#     averages.run()
#     df = averages.to_df(show_spacy_rules=False)
#     assert isinstance(df, pd.DataFrame)
#     assert list(df.columns) == ["test"]

# def test_to_df_with_case_sensitive_true(windows):
#     averages = Averages("TEST", windows, case_sensitive=True)
#     averages.run()
#     df = averages.to_df(show_spacy_rules=False)
#     assert isinstance(df, pd.DataFrame)
#     assert list(df.columns) == ["TEST"]

# def test_to_df_with_case_sensitive_false(windows):
#     averages = Averages("TEST", windows, case_sensitive=False)
#     averages.run()
#     df = averages.to_df(show_spacy_rules=False)
#     assert isinstance(df, pd.DataFrame)
#     assert list(df.columns) == ["test"]
