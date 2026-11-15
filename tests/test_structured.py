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


def test_json_null_response_validates_against_null_schema():
    outcome = evaluate_json_schema("null", {"type": "null"})

    assert outcome.passed is True
    assert outcome.details["json_valid"] is True
    assert outcome.details["errors"] == []


def test_json_null_response_fails_object_schema_with_path():
    outcome = evaluate_json_schema("null", {"type": "object"})

    assert outcome.passed is False
    assert outcome.details["json_valid"] is True
    assert outcome.details["errors"][0]["path"] == "$"


def test_nested_objects_arrays_and_special_characters():
    schema = {
        "type": "object",
        "properties": {
            "items": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {"name": {"type": "string"}},
                    "required": ["name"],
                },
            },
            "note": {"type": "string"},
        },
        "required": ["items"],
    }
    response = '{"items": [{"name": "a\\"b"}, {"name": "é中🙂"}], "note": "ok"}'

    outcome = evaluate_json_schema(response, schema)

    assert outcome.passed is True


def test_numeric_types_and_missing_required_property():
    schema = {
        "type": "object",
        "properties": {"count": {"type": "integer"}, "price": {"type": "number"}},
        "required": ["count", "price"],
    }

    missing = evaluate_json_schema('{"count": 3}', schema)
    wrong_type = evaluate_json_schema('{"count": "three", "price": 1.5}', schema)

    assert missing.passed is False
    assert missing.details["errors"][0]["path"] == "price"
    assert wrong_type.passed is False


def test_local_refs_are_resolved():
    schema = {
        "type": "object",
        "properties": {"child": {"$ref": "#/$defs/node"}},
        "$defs": {"node": {"type": "object", "properties": {"value": {"type": "integer"}}}},
    }

    valid = evaluate_json_schema('{"child": {"value": 1}}', schema)
    invalid = evaluate_json_schema('{"child": {"value": "x"}}', schema)

    assert valid.passed is True
    assert invalid.passed is False


def test_recursive_schema_validates():
    schema = {
        "type": "object",
        "properties": {"next": {"$ref": "#"}},
        "additionalProperties": False,
    }

    outcome = evaluate_json_schema('{"next": {"next": {}}}', schema)

    assert outcome.passed is True


def test_malformed_schema_is_rejected_at_validation_time():
    import pytest

    with pytest.raises(ValueError, match="invalid json schema"):
        check_schema({"type": "not-a-real-type"})


def test_fenced_null_json_is_extracted():
    parsed, error = extract_json("```json\nnull\n```")

    assert parsed is None
    assert error is None
