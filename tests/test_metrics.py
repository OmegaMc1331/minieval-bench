from __future__ import annotations

import pytest

from minieval_bench.metrics import contains, evaluate, exact_match, normalize


def test_normalize_strips_lowercases_and_removes_punctuation():
    assert normalize("  Hello,   World!! ") == "hello world"


def test_exact_match_ignores_case_and_punctuation():
    assert exact_match("The Answer is 42.", "the answer is 42")


def test_exact_match_rejects_different_content():
    assert not exact_match("the answer is 42", "the answer is 43")


def test_exact_match_collapses_whitespace():
    assert exact_match("a\tb\nc", "a b c")


def test_contains_matches_after_normalization():
    assert contains("The capital of France is Paris.", "paris")


def test_contains_rejects_absent_reference():
    assert not contains("the capital is berlin", "paris")


def test_regex_matches_raw_response():
    assert evaluate("regex", "order-12345 shipped", r"order-\d+")


def test_regex_is_case_sensitive_on_raw_text():
    assert not evaluate("regex", "ABC", "abc")


def test_regex_supports_anchors():
    assert evaluate("regex", "mock-1a2b3c4d", "^mock-[0-9a-f]{8}$")
    assert not evaluate("regex", "x mock-1a2b3c4d", "^mock-[0-9a-f]{8}$")


def test_evaluate_dispatches_all_metrics():
    assert evaluate("exact_match", "yes", "Yes")
    assert evaluate("contains", "he said yes", "said")
    assert evaluate("regex", "12345", r"\d+")


def test_evaluate_rejects_unknown_metric():
    with pytest.raises(ValueError, match="unknown metric"):
        evaluate("bleu", "response", "reference")
