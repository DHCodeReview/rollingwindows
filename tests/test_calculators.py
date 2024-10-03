"""test_calculators.py.

As of 12 June 2024 this has 99% coverage for calcuators.py.
"""
import re
from enum import Enum

import pandas as pd
import pytest
import spacy
from spacy.tokens import Doc

from rollingwindows import Windows, sliding_str_windows, sliding_windows
from rollingwindows.calculators import BaseCalculator, RWCalculator


@pytest.fixture
def doc(model="en_core_web_sm"):
    path = "../sample_docs/Austen_Pride"
    model = "en_core_web_sm"
    nlp = spacy.load(model)
    with open(path, "rb") as f:
        doc_bytes = f.read()
    doc = Doc(nlp.vocab).from_bytes(doc_bytes)
    return doc[0:500].as_doc()

@pytest.fixture
def char_win_100_contract(doc):
    sliding_windows = sliding_str_windows(doc)
    return Windows(sliding_windows, "characters", 100, "contract")

@pytest.fixture
def token_win_100(doc):
    sliding_win = sliding_windows(doc)
    return Windows(sliding_win, "tokens", 100)

@pytest.fixture
def sent_win_100(doc):
    sliding_win = sliding_windows(doc)
    return Windows(sliding_win, "sentences", 100)

@pytest.fixture
def token_list_win_100(doc):
    sliding_win = sliding_windows(doc, 100, "tokens")
    sliding_token_lists = ([str(token) for token in window] for window in sliding_win)
    return Windows(sliding_token_lists, "tokens", 100)

@pytest.fixture
def token_win_100(doc):
    sliding_win = sliding_windows(doc, 100, "tokens")
    return Windows(sliding_win, "tokens", 100)

@pytest.fixture
def love_marriage():
    return ["love", "marriage"]

# Class Instantiation

def test_instantiate_base_calculator():
    calculator = BaseCalculator()
    assert isinstance(calculator, BaseCalculator)

def test_instantiate_with_no_arguments():
    calculator = RWCalculator()
    assert(calculator.id == "rw_calculator")
    assert isinstance(calculator.metadata, dict)

def test_instantiate_with_arguments(char_win_100_contract, love_marriage):
    calculator = RWCalculator(windows=char_win_100_contract, patterns=love_marriage)
    assert isinstance(calculator, RWCalculator)
    assert(calculator.id == "rw_calculator")
    assert isinstance(calculator.metadata, dict)
    assert isinstance(calculator.regex_flags, Enum)

def test_regex_flags(char_win_100_contract, love_marriage):
    calculator = RWCalculator(windows=char_win_100_contract, patterns=love_marriage, case_sensitive=False)
    assert calculator.regex_flags == re.IGNORECASE | re.UNICODE
    calculator = RWCalculator(windows=char_win_100_contract, patterns=love_marriage, case_sensitive=True)
    assert calculator.regex_flags == re.UNICODE

# Public Methods

def test_run_invalid_query(char_win_100_contract, love_marriage):
    calculator = RWCalculator()
    with pytest.raises(Exception, match=r"Invalid query type."):
        calculator.run(char_win_100_contract, love_marriage, "invalid")
        raise Exception("Invalid query type.")

def test_run_averages(char_win_100_contract, love_marriage):
    calculator = RWCalculator()
    calculator.run(char_win_100_contract, love_marriage, "averages")
    assert calculator.data is not None
    assert isinstance(calculator.to_df(), pd.DataFrame)

def test_run_counts(char_win_100_contract, love_marriage):
    calculator = RWCalculator()
    calculator.run(char_win_100_contract, love_marriage, "counts")
    assert calculator.data is not None
    assert isinstance(calculator.to_df(), pd.DataFrame)

def test_run_ratios(char_win_100_contract, love_marriage):
    calculator = RWCalculator()
    calculator.run(char_win_100_contract, love_marriage, "ratios")
    assert calculator.data is not None
    assert isinstance(calculator.to_df(), pd.DataFrame)

def test_get_averages(char_win_100_contract, love_marriage):
    calculator = RWCalculator()
    calculator.get_averages(char_win_100_contract, love_marriage)
    assert calculator.data is not None
    assert isinstance(calculator.to_df(), pd.DataFrame)

def test_get_counts(char_win_100_contract, love_marriage):
    calculator = RWCalculator()
    calculator.get_counts(char_win_100_contract, love_marriage)
    assert calculator.data is not None
    assert isinstance(calculator.to_df(), pd.DataFrame)

def test_get_ratios(char_win_100_contract, love_marriage):
    calculator = RWCalculator()
    calculator.get_ratios(char_win_100_contract, love_marriage)
    assert calculator.data is not None
    assert isinstance(calculator.to_df(), pd.DataFrame)

def test_show_spacy_rules(token_win_100):
    calculator = RWCalculator(mode="spacy_rule")
    calculator.get_counts(token_win_100, [[{"LOWER": "love"}]])
    df = calculator.to_df()
    assert df.columns[0] == "love"
    df = calculator.to_df(show_spacy_rules=True)
    assert df.columns[0].startswith("[")

def test_mix_class_attributes_and_keyword_parameters(char_win_100_contract, love_marriage):
    calculator = RWCalculator(windows=char_win_100_contract)
    calculator.get_counts(patterns=love_marriage)

def test_missing_windows_and_patterns():
    calculator = RWCalculator()
    with pytest.raises(Exception, match=r"You must supply a value.+$"):
        calculator.get_counts()
        raise Exception("You must supply a value")

# Private Methods

def test_exact_counts_character_windows(char_win_100_contract, love_marriage):
    calculator = RWCalculator()
    calculator.get_counts(char_win_100_contract, love_marriage)
    assert calculator.data is not None

def test_regex_counts_character_windows(char_win_100_contract):
    calculator = RWCalculator(mode="regex")
    calculator.get_counts(char_win_100_contract, ["lov.{1,3}", "marriage"])
    assert calculator.data is not None

def test_invalid_mode_in_character_windows(char_win_100_contract, love_marriage):
    calculator = RWCalculator(mode="spacy_rule")
    with pytest.raises(Exception, match=r"Invalid mode for character windows."):
        calculator.get_counts(char_win_100_contract, love_marriage)
        raise Exception("Invalid mode for character windows.")

def test_exact_counts_token_lists_case_sensitive_true(token_list_win_100):
    calculator = RWCalculator(case_sensitive=True)
    calculator.get_counts(token_list_win_100, ["love"])
    assert calculator.data is not None

def test_exact_counts_token_lists_case_sensitive_false(token_list_win_100):
    calculator = RWCalculator(case_sensitive=False)
    calculator.get_counts(token_list_win_100, ["love"])
    assert calculator.data is not None

def test_regex_counts_token_lists_case_sensitive_true(token_list_win_100):
    calculator = RWCalculator(mode="regex", case_sensitive=True)
    calculator.get_counts(token_list_win_100, ["lov.{1,3}"])
    assert calculator.data is not None

def test_regex_counts_token_lists_case_sensitive_false(token_list_win_100):
    calculator = RWCalculator(mode="regex", case_sensitive=False)
    calculator.get_counts(token_list_win_100, ["lov.{1,3}"])
    assert calculator.data is not None

def test_calculate_exact_search_in_doc_tokens_string_pattern(token_win_100):
    calculator = RWCalculator()
    calculator.get_counts(token_win_100, "love")
    assert calculator.data is not None

def test_calculate_exact_search_in_doc_tokens_case_sensitive_true(token_win_100):
    calculator = RWCalculator(case_sensitive=True)
    calculator.get_counts(token_win_100, ["this"])
    assert calculator.data is not None

def test_calculate_exact_search_in_doc_tokens_case_sensitive_false(token_win_100):
    calculator = RWCalculator(case_sensitive=False)
    calculator.get_counts(token_win_100, ["love"])
    assert calculator.data is not None

def test_regex_counts_token_windows_case_sensitive_true(token_win_100):
    calculator = RWCalculator(mode="regex")
    calculator.get_counts(token_win_100, ["lov.{1,3}"])
    assert calculator.data is not None

def test_regex_counts_token_windows_case_sensitive_false(token_win_100):
    calculator = RWCalculator(mode="regex", case_sensitive=False)
    calculator.get_counts(token_win_100, ["lov.{1,3}"])
    assert calculator.data is not None

def test_spacy_rule_counts_token_windows_case_sensitive_true(token_win_100):
    calculator = RWCalculator(mode="spacy_rule")
    calculator.get_counts(token_win_100, [[{'ORTH': 'love'}]])
    assert calculator.data is not None

def test_spacy_rule_counts_token_windows_case_sensitive_false(token_win_100):
    calculator = RWCalculator(mode="spacy_rule", case_sensitive=False)
    calculator.get_counts(token_win_100, [[{'ORTH': 'love'}]])
    assert calculator.data is not None

def test_multi_token_without_original_doc(token_win_100):
    calculator = RWCalculator(mode="multi_token")
    with pytest.raises(Exception, match=r"You must supply an `original_doc` to use `multi_token` mode."):
        calculator.get_counts(token_win_100, "love")
        raise Exception("You must supply an `original_doc` to use `multi_token` mode.")

def test_multi_token_with_original_doc_case_sensitive_true(token_win_100, doc):
    calculator = RWCalculator(mode="multi_token", case_sensitive=True, original_doc=doc)
    calculator.get_counts(token_win_100, "love")
    assert calculator.data is not None

def test_multi_token_with_original_doc_case_sensitive_false(token_win_100, doc):
    calculator = RWCalculator(mode="multi_token", case_sensitive=False, original_doc=doc)
    calculator.get_counts(token_win_100, "love")
    assert calculator.data is not None

def test_multi_token_with_token_list(token_list_win_100, doc):
    calculator = RWCalculator(mode="multi_token", original_doc=doc)
    with pytest.raises(Exception, match=r"You cannot use spaCy rule or perform multi-token searches with a string or list of token strings."):
        calculator.get_counts(token_list_win_100, ["love"])
        raise Exception("You cannot use spaCy rule or perform multi-token searches with a string or list of token strings.")

def test_multi_token_exact_with_original_doc_case_sensitive_true(token_win_100, doc):
    calculator = RWCalculator(mode="multi_token_exact", case_sensitive=True, original_doc=doc)
    calculator.get_counts(token_win_100, "love")
    assert calculator.data is not None

def test_multi_token_exact_with_original_doc_case_sensitive_false(token_win_100, doc):
    calculator = RWCalculator(mode="multi_token_exact", case_sensitive=False, original_doc=doc)
    calculator.get_counts(token_win_100, "love")
    assert calculator.data is not None

def test_ratios_with_invalid_patterns(char_win_100_contract):
    calculator = RWCalculator()
    # with pytest.raises(Exception, match=r"You must supply a list of two patterns to calculate ratios."):
    with pytest.raises(Exception):
        calculator.get_ratios(char_win_100_contract, "love")
        raise Exception("You must supply a list of two patterns to calculate ratios.")
    with pytest.raises(Exception, match=r"You can only calculate ratios for two patterns."):
        calculator.get_ratios(char_win_100_contract, ["love"])
        raise Exception("You must supply a list of two patterns to calculate ratios.")
    with pytest.raises(Exception, match=r"You can only calculate ratios for two patterns."):
        calculator.get_ratios(char_win_100_contract, ["love", "marriage", "riches"])
        raise Exception("You can only calculate ratios for two patterns.")

def test_calculate_exact_search_in_sent_windows_string_pattern(sent_win_100):
    calculator = RWCalculator()
    calculator.get_counts(sent_win_100, "love")
    assert calculator.data is not None
