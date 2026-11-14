from __future__ import annotations

import pytest

from minieval_bench.evaluation.structured import (
    check_schema,
    evaluate_json_schema,
    extract_json,
)

USER_SCHEMA = {
    "type": "object",
    "properties": {
        "name": {"type": "string"},
        "age": {"type": "integer", "minimum": 0},
        "email": {"type": "string"},
    },
    "required": ["name", "age"],
    "additionalProperties": False,
}


def test_extract_json_parses_direct_and_fenced_content():
    assert extract_json('{"a": 1}')[0] == {"a": 1}
    assert extract_json('```json\n{"a": 1}\n```')[0] == {"a": 1}


def test_extract_json_finds_first_balanced_block():
    value, error = extract_json('Sure, here it is:\n{"a": {"b": 2}} hope this helps')

    assert value == {"a": {"b": 2}}
    assert error is None


def test_extract_json_rejects_non_json():
    value, error = extract_json("no json here at all")

    assert value is None
    assert error is not None


def test_extract_json_handles_braces_inside_strings():
    value, _ = extract_json('{"text": "a } b { c", "n": 1}')

    assert value == {"text": "a } b { c", "n": 1}


def test_check_schema_accepts_valid_schemas():
    check_schema(USER_SCHEMA)
    check_schema({"type": "array", "items": {"type": "integer"}})


def test_check_schema_rejects_remote_references():
    with pytest.raises(ValueError, match="remote"):
        check_schema({"$ref": "https://evil.example.com/schema.json"})
    with pytest.raises(ValueError, match="remote"):
        check_schema({"properties": {"a": {"$ref": "http://schemas.example.com/a.json"}}})


def test_check_schema_rejects_invalid_schemas():
    with pytest.raises(ValueError, match="invalid json schema"):
        check_schema({"type": "not-a-type"})


def test_valid_response_conforms():
    outcome = evaluate_json_schema('{"name": "Ada", "age": 36, "email": "a@b.c"}', USER_SCHEMA)

    assert outcome.status == "completed"
    assert outcome.passed is True
    assert outcome.score == 1.0
    assert outcome.details["json_valid"] is True
    assert outcome.details["errors"] == []


def test_missing_required_property_is_reported():
    outcome = evaluate_json_schema('{"age": 36}', USER_SCHEMA)

    assert outcome.passed is False
    paths = [error["path"] for error in outcome.details["errors"]]
    assert "name" in paths


def test_wrong_type_and_constraint_violations_are_reported():
    outcome = evaluate_json_schema('{"name": "Ada", "age": -5}', USER_SCHEMA)

    assert outcome.passed is False
    messages = " ".join(error["message"] for error in outcome.details["errors"])
    assert "is not of type" in messages or "-5" in messages


def test_additional_properties_are_rejected():
    outcome = evaluate_json_schema('{"name": "Ada", "age": 1, "extra": true}', USER_SCHEMA)

    assert outcome.passed is False
    assert any("extra" in error["message"] for error in outcome.details["errors"])


def test_nested_structures_are_validated():
    schema = {
        "type": "object",
        "properties": {
            "user": {
                "type": "object",
                "properties": {"tags": {"type": "array", "items": {"type": "string"}}},
                "required": ["tags"],
            }
        },
        "required": ["user"],
    }
    outcome = evaluate_json_schema('{"user": {"tags": ["a", 3]}}', schema)

    assert outcome.passed is False
    assert any(
        "tags" in error["path"] or "3" in error["message"]
        for error in outcome.details["errors"]
    )


def test_invalid_json_syntax_is_a_completed_failure():
    outcome = evaluate_json_schema('{"name": ', USER_SCHEMA)

    assert outcome.status == "completed"
    assert outcome.passed is False
    assert outcome.details["json_valid"] is False
    assert outcome.details["errors"][0]["path"] == "$"


def test_json_wrapped_in_prose_is_extracted():
    outcome = evaluate_json_schema('Here is the user: {"name": "Ada", "age": 36}', USER_SCHEMA)

    assert outcome.details["json_valid"] is True
    assert outcome.passed is True
