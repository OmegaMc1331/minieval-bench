from __future__ import annotations

import io
from contextlib import redirect_stdout

import pytest

from minieval_bench.evaluation.code import (
    DockerCodeRunner,
    build_harness_script,
    outcome_from_harness,
    parse_harness_output,
)
from minieval_bench.models import CodeEvalConfig

ADD_TESTS = (
    "def test_add():\n    assert add(2, 3) == 5\n\n"
    "def test_add_zero():\n    assert add(0, 0) == 0\n"
)


def run_harness_on_host(submission: str, tests: str) -> dict:
    script = build_harness_script(submission, tests, 5)
    buffer = io.StringIO()
    with redirect_stdout(buffer):
        try:
            exec(compile(script, "harness", "exec"), {"__name__": "harness"})
        except SystemExit:
            pass
    return parse_harness_output(buffer.getvalue())


def test_harness_reports_passed_tests():
    submission = "def add(a, b):\n    return a + b\n"

    result = run_harness_on_host(submission, ADD_TESTS)

    assert result["status"] == "completed"
    assert [entry["status"] for entry in result["tests"]] == ["passed", "passed"]
    outcome = outcome_from_harness(result)
    assert outcome.passed is True
    assert outcome.score == 1.0


def test_harness_reports_failed_assertions():
    submission = "def add(a, b):\n    return a + b + 1\n"

    result = run_harness_on_host(submission, ADD_TESTS)

    assert [entry["status"] for entry in result["tests"]] == ["failed", "failed"]
    outcome = outcome_from_harness(result)
    assert outcome.passed is False
    assert outcome.score == 0.0


def test_harness_reports_runtime_exceptions():
    submission = "def add(a, b):\n    raise ValueError('boom')\n"

    result = run_harness_on_host(submission, ADD_TESTS)

    assert all(entry["status"] == "error" for entry in result["tests"])
    assert "ValueError" in result["tests"][0]["error"]


def test_harness_reports_submission_syntax_errors():
    submission = "def add(a, b:\n    return a + b\n"

    result = run_harness_on_host(submission, ADD_TESTS)

    assert result["status"] == "submission_error"
    outcome = outcome_from_harness(result)
    assert outcome.status == "completed"
    assert outcome.passed is False
    assert outcome.details["overall"] == "submission_error"


def test_parse_harness_output_finds_json_line():
    result = parse_harness_output('noise\n{"status": "completed", "tests": []}\nmore noise\n')

    assert result["status"] == "completed"


def test_parse_harness_output_rejects_empty_output():
    with pytest.raises(ValueError):
        parse_harness_output("")


def test_outcome_from_harness_marks_tests_module_errors_as_evaluator_failures():
    outcome = outcome_from_harness({"status": "tests_error", "submission_error": "boom"})

    assert outcome.status == "failed"
    assert outcome.passed is None


def test_partial_pass_score_is_a_ratio():
    result = {
        "status": "completed",
        "tests": [
            {"name": "a", "status": "passed", "error": None},
            {"name": "b", "status": "failed", "error": None},
        ],
    }

    outcome = outcome_from_harness(result)

    assert outcome.score == 0.5
    assert outcome.passed is False


def test_runner_reports_unavailable_when_docker_is_down(monkeypatch):
    import minieval_bench.evaluation.code as code_module

    monkeypatch.setattr(
        code_module, "docker_available", lambda: (False, "docker daemon is not reachable")
    )
    runner = DockerCodeRunner(CodeEvalConfig())

    outcome = runner.run("def add(a, b):\n    return a + b\n", ADD_TESTS)

    assert outcome.status == "unavailable"
    assert outcome.passed is None
    assert "unavailable" in outcome.error_message


def test_runner_reports_container_timeout(monkeypatch):
    import minieval_bench.evaluation.code as code_module

    monkeypatch.setattr(code_module, "docker_available", lambda: (True, "29.6.1"))
    calls = []

    def fake_run(*args, **kwargs):
        calls.append(args)
        if len(calls) == 1:
            raise code_module.subprocess.TimeoutExpired(["docker"], 5)
        return type("Completed", (), {"returncode": 0, "stdout": b"", "stderr": b""})()

    monkeypatch.setattr(code_module.subprocess, "run", fake_run)

    runner = DockerCodeRunner(CodeEvalConfig(timeout_seconds=5))
    outcome = runner.run("while True:\n    pass\n", ADD_TESTS)

    assert outcome.status == "completed"
    assert outcome.passed is False
    assert outcome.details["overall"] == "timeout"
    assert len(calls) == 2


def test_runner_reports_memory_limit_kill(monkeypatch):
    import minieval_bench.evaluation.code as code_module

    monkeypatch.setattr(code_module, "docker_available", lambda: (True, "29.6.1"))
    completed = type("Completed", (), {"returncode": 137, "stdout": b"", "stderr": b""})()
    monkeypatch.setattr(code_module.subprocess, "run", lambda *a, **k: completed)

    runner = DockerCodeRunner(CodeEvalConfig())
    outcome = runner.run("x = [0] * 10**10\n", ADD_TESTS)

    assert outcome.status == "completed"
    assert outcome.details["overall"] == "memory_limit"


def test_runner_reports_docker_start_failure(monkeypatch):
    import minieval_bench.evaluation.code as code_module

    monkeypatch.setattr(code_module, "docker_available", lambda: (True, "29.6.1"))

    def failing_run(*args, **kwargs):
        raise OSError("docker binary vanished")

    monkeypatch.setattr(code_module.subprocess, "run", failing_run)

    runner = DockerCodeRunner(CodeEvalConfig())
    outcome = runner.run("def add(a, b):\n    return a + b\n", ADD_TESTS)

    assert outcome.status == "unavailable"
    assert "failed to start docker" in outcome.error_message


def test_harness_script_is_valid_python():
    script = build_harness_script("x = 1\n", "def test_x():\n    assert x == 1\n", 5)

    compile(script, "harness", "exec")
