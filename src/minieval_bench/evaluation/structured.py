from __future__ import annotations

import json
import re
from typing import Any

import jsonschema
from jsonschema import Draft202012Validator

from minieval_bench.models import EvaluationOutcome

MAX_VALIDATION_ERRORS = 20


def check_schema(schema: dict) -> None:
    if not isinstance(schema, dict):
        raise ValueError("json schema must be an object")
    for ref in _collect_refs(schema):
        if ref.startswith(("http://", "https://", "//")):
            raise ValueError(f"remote $ref is not allowed: {ref}")
    try:
        Draft202012Validator.check_schema(schema)
    except jsonschema.SchemaError as exc:
        raise ValueError(f"invalid json schema: {exc.message}") from exc


def _collect_refs(schema: Any) -> list[str]:
    refs: list[str] = []
    if isinstance(schema, dict):
        ref = schema.get("$ref")
        if isinstance(ref, str):
            refs.append(ref)
        for value in schema.values():
            refs.extend(_collect_refs(value))
    elif isinstance(schema, list):
        for item in schema:
            refs.extend(_collect_refs(item))
    return refs


def extract_json(text: str) -> tuple[Any | None, str | None]:
    stripped = text.strip()
    if stripped.startswith("```"):
        lines = stripped.splitlines()
        lines = [line for line in lines if not line.strip().startswith("```")]
        stripped = "\n".join(lines).strip()
    try:
        return json.loads(stripped), None
    except json.JSONDecodeError:
        pass
    block = _first_json_block(stripped)
    if block is None:
        return None, "response is not valid JSON"
    try:
        return json.loads(block), None
    except json.JSONDecodeError as exc:
        return None, f"invalid JSON: {exc.msg}"


def _first_json_block(text: str) -> str | None:
    start = None
    for index, char in enumerate(text):
        if char in "{[":
            start = index
            break
    if start is None:
        return None
    opener = text[start]
    closer = "}" if opener == "{" else "]"
    depth = 0
    in_string = False
    escaped = False
    for index in range(start, len(text)):
        char = text[index]
        if in_string:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == '"':
                in_string = False
            continue
        if char == '"':
            in_string = True
        elif char == opener:
            depth += 1
        elif char == closer:
            depth -= 1
            if depth == 0:
                return text[start : index + 1]
    return None


def evaluate_json_schema(response: str, schema: dict) -> EvaluationOutcome:
    parsed, json_error = extract_json(response)
    if parsed is None:
        return EvaluationOutcome(
            evaluator="json_schema",
            status="completed",
            passed=False,
            score=0.0,
            details={"json_valid": False, "errors": [{"path": "$", "message": json_error}]},
        )

    validator = Draft202012Validator(schema)
    errors = sorted(validator.iter_errors(parsed), key=lambda error: list(error.absolute_path))
    validation_errors = [
        {"path": _error_path(error), "message": error.message[:300]}
        for error in errors[:MAX_VALIDATION_ERRORS]
    ]
    schema_ok = not errors
    return EvaluationOutcome(
        evaluator="json_schema",
        status="completed",
        passed=schema_ok,
        score=1.0 if schema_ok else 0.0,
        details={"json_valid": True, "errors": validation_errors},
    )


def _error_path(error) -> str:
    base = ".".join(str(part) for part in error.absolute_path)
    if error.validator == "required":
        match = re.search(r"'([^']+)' is a required property", error.message)
        if match:
            return f"{base}.{match.group(1)}" if base else match.group(1)
    return base or "$"
