from __future__ import annotations

import io

from minieval_bench.evaluation.code import (
    DockerCodeRunner,
    build_controller_script,
    expected_test_names,
    outcome_from_verdict,
    parse_controller_output,
    read_bounded,
    validate_worker_report,
)
from minieval_bench.models import CodeEvalConfig

ADD_TESTS = (
    "def test_add():\n    assert add(2, 3) == 5\n\n"
    "def test_add_zero():\n    assert add(0, 0) == 0\n"
)
EXPECTED = ["test_add", "test_add_zero"]


def completed_verdict(tests=None, **overrides):
    verdict = {
        "status": "completed",
        "submission_error": None,
        "tests": tests if tests is not None else [],
        "exit_code": 0,
        "stdout_bytes": 0,
        "stderr_bytes": 0,
        "stdout_truncated": False,
        "stderr_truncated": False,
        "timed_out": False,
    }
    return {**verdict, **overrides}


def passed_tests():
    return [
        {"name": "test_add", "status": "passed", "error": None},
        {"name": "test_add_zero", "status": "passed", "error": None},
    ]


def test_expected_test_names_collects_module_level_functions():
    source = (
        "import math\n\n"
        "def helper():\n    pass\n\n"
        "def test_b():\n    assert True\n\n"
        "def test_a():\n    assert True\n\n"
        "def wrapper():\n    def test_nested():\n        pass\n"
    )

    assert expected_test_names(source) == ["test_a", "test_b"]


def test_expected_test_names_rejects_async_tests():
    import pytest

    with pytest.raises(ValueError, match="async test function"):
        expected_test_names("async def test_async():\n    assert True\n")


def test_expected_test_names_empty_when_no_tests():
    assert expected_test_names("def helper():\n    pass\n") == []


def test_validate_worker_report_accepts_matching_tests():
    assert validate_worker_report(completed_verdict(passed_tests()), EXPECTED) is None


def test_validate_worker_report_rejects_unknown_status():
    assert validate_worker_report(completed_verdict(status="hacked"), EXPECTED) is not None


def test_validate_worker_report_rejects_bad_test_status():
    tests = [{"name": "test_add", "status": "flawless", "error": None}]
    error = validate_worker_report(completed_verdict(tests), EXPECTED)
    assert error is not None and "invalid status" in error


def test_validate_worker_report_rejects_duplicates():
    tests = passed_tests() + [{"name": "test_add", "status": "passed", "error": None}]
    assert "duplicate" in validate_worker_report(completed_verdict(tests), EXPECTED)


def test_validate_worker_report_rejects_name_mismatch():
    tests = [{"name": "test_fake", "status": "passed", "error": None}]
    error = validate_worker_report(completed_verdict(tests), EXPECTED)
    assert error is not None and "do not match" in error


def test_validate_worker_report_skips_tests_check_for_error_statuses():
    assert validate_worker_report(completed_verdict(status="submission_error"), EXPECTED) is None


def test_outcome_completed_full_pass():
    outcome = outcome_from_verdict(completed_verdict(passed_tests()), EXPECTED)

    assert outcome.status == "completed"
    assert outcome.passed is True
    assert outcome.score == 1.0
    assert outcome.details["passed"] == 2


def test_outcome_completed_partial_pass():
    tests = [
        {"name": "test_add", "status": "passed", "error": None},
        {"name": "test_add_zero", "status": "failed", "error": "assertion failed"},
    ]

    outcome = outcome_from_verdict(completed_verdict(tests), EXPECTED)

    assert outcome.passed is False
    assert outcome.score == 0.5


def test_outcome_zero_tests_is_not_a_perfect_score():
    outcome = outcome_from_verdict(completed_verdict([]), [])

    assert outcome.status == "completed"
    assert outcome.passed is False
    assert outcome.score is None
    assert outcome.details["overall"] == "no_tests"


def test_outcome_zero_reported_tests_against_expected_is_integrity_failure():
    outcome = outcome_from_verdict(completed_verdict([]), EXPECTED)

    assert outcome.status == "failed"
    assert "harness integrity" in outcome.error_message


def test_outcome_submission_error():
    outcome = outcome_from_verdict(
        completed_verdict(status="submission_error", submission_error="SyntaxError"), EXPECTED
    )

    assert outcome.status == "completed"
    assert outcome.passed is False
    assert outcome.score == 0.0
    assert outcome.details["overall"] == "submission_error"


def test_outcome_tests_error_is_evaluator_failure():
    outcome = outcome_from_verdict(
        completed_verdict(status="tests_error", submission_error="boom"), EXPECTED
    )

    assert outcome.status == "failed"
    assert outcome.passed is None


def test_outcome_worker_timeout():
    outcome = outcome_from_verdict(completed_verdict(status="worker_timeout"), EXPECTED)

    assert outcome.status == "completed"
    assert outcome.passed is False
    assert outcome.details["overall"] == "timeout"


def test_outcome_output_overflow():
    outcome = outcome_from_verdict(
        completed_verdict(status="output_overflow", stdout_bytes=999999), EXPECTED
    )

    assert outcome.status == "completed"
    assert outcome.passed is False
    assert outcome.details["overall"] == "output_overflow"


def test_outcome_worker_crash():
    outcome = outcome_from_verdict(
        completed_verdict(status="worker_crashed", exit_code=-9), EXPECTED
    )

    assert outcome.status == "completed"
    assert outcome.passed is False
    assert outcome.details["overall"] == "worker_crashed"


def test_outcome_report_invalid_and_controller_error_are_failures():
    invalid = outcome_from_verdict(completed_verdict(status="report_invalid"), EXPECTED)
    controller = outcome_from_verdict(
        completed_verdict(status="controller_error", error="boom"), EXPECTED
    )

    assert invalid.status == "failed"
    assert controller.status == "failed"


def test_outcome_integrity_mismatch_is_evaluator_failure():
    forged = completed_verdict([{"name": "test_fake", "status": "passed", "error": None}])

    outcome = outcome_from_verdict(forged, EXPECTED)

    assert outcome.status == "failed"
    assert "harness integrity" in outcome.error_message
    assert outcome.passed is None


def test_read_bounded_under_limit():
    stream = io.BytesIO(b"abc")

    data, overflow, total = read_bounded(stream, 10)

    assert data == b"abc"
    assert overflow is False
    assert total == 3


def test_read_bounded_at_exact_limit():
    stream = io.BytesIO(b"abcdef")

    data, overflow, total = read_bounded(stream, 6)

    assert data == b"abcdef"
    assert overflow is False
    assert total == 6


def test_read_bounded_detects_overflow_without_accumulating():
    stream = io.BytesIO(b"x" * 300)

    data, overflow, total = read_bounded(stream, 100)

    assert len(data) <= 100
    assert overflow is True
    assert total == 300


def test_parse_controller_output_takes_last_json_line():
    verdict = parse_controller_output(b'noise\n{"status": "completed"}\nnot json\n')

    assert verdict == {"status": "completed"}


def test_parse_controller_output_rejects_noise_only():
    assert parse_controller_output(b"no json here") is None
    assert parse_controller_output(b"") is None


def test_templates_compile_as_python():
    from minieval_bench.evaluation.code import CONTROLLER_TEMPLATE, WORKER_TEMPLATE

    controller = build_controller_script(
        "x = 1\n", "def test_x():\n    assert x == 1\n", 5, 30, 65536
    )

    compile(controller, "controller", "exec")
    compile(WORKER_TEMPLATE.replace("__REPORT_FD__", "9"), "worker", "exec")
    compile(
        CONTROLLER_TEMPLATE.replace("__PAYLOAD__", "'{}'")
        .replace("__WORKER__", repr(WORKER_TEMPLATE))
        .replace("__REPORT_LIMIT__", "1000000"),
        "controller-template",
        "exec",
    )


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


class FakePopen:
    def __init__(self, returncode=0, stdout=b"", stderr=b"", timeout=None):
        self.returncode = None
        self.pid = 1234
        self._final_returncode = returncode
        self.stdin = io.BytesIO()
        self.stdout = io.BytesIO(stdout)
        self.stderr = io.BytesIO(stderr)
        self._timeout = timeout

    def wait(self, timeout=None):
        if self._timeout is not None and timeout is not None:
            import minieval_bench.evaluation.code as code_module

            self.returncode = 137
            raise code_module.subprocess.TimeoutExpired(["docker"], timeout)
        self.returncode = self._final_returncode
        return self.returncode


VERDICT_LINE = (
    b'{"status": "completed", "submission_error": null, "tests": '
    b'[{"name": "test_add", "status": "passed", "error": null}, '
    b'{"name": "test_add_zero", "status": "passed", "error": null}], '
    b'"exit_code": 0, "stdout_bytes": 0, "stderr_bytes": 0, '
    b'"stdout_truncated": false, "stderr_truncated": false, "timed_out": false}'
)


def install_fake_popen(monkeypatch, fake):
    import minieval_bench.evaluation.code as code_module

    executed = []

    def fake_run(command, **kwargs):
        executed.append(command)
        return type("Completed", (), {"returncode": 0, "stdout": b"", "stderr": b""})()

    monkeypatch.setattr(code_module, "docker_available", lambda: (True, "29.6.1"))
    monkeypatch.setattr(code_module.subprocess, "Popen", lambda *a, **k: fake)
    monkeypatch.setattr(code_module.subprocess, "run", fake_run)
    return executed


def test_runner_maps_controller_verdict_to_outcome(monkeypatch):
    fake = FakePopen(returncode=0, stdout=VERDICT_LINE)
    install_fake_popen(monkeypatch, fake)

    runner = DockerCodeRunner(CodeEvalConfig())
    outcome = runner.run("def add(a, b):\n    return a + b\n", ADD_TESTS)

    assert outcome.status == "completed"
    assert outcome.passed is True
    assert outcome.score == 1.0


def test_runner_reports_container_timeout_and_cleans_up(monkeypatch):
    fake = FakePopen(timeout=True)
    executed = install_fake_popen(monkeypatch, fake)

    runner = DockerCodeRunner(CodeEvalConfig(timeout_seconds=1))
    outcome = runner.run("while True:\n    pass\n", ADD_TESTS)

    assert outcome.status == "completed"
    assert outcome.passed is False
    assert outcome.details["overall"] == "timeout"
    kill_commands = [c for c in executed if c[:2] == ["docker", "kill"]]
    rm_commands = [c for c in executed if c[:2] == ["docker", "rm"]]
    assert kill_commands and rm_commands


def test_runner_reports_memory_limit_kill(monkeypatch):
    fake = FakePopen(returncode=137)
    install_fake_popen(monkeypatch, fake)

    runner = DockerCodeRunner(CodeEvalConfig())
    outcome = runner.run("x = [0] * 10**10\n", ADD_TESTS)

    assert outcome.status == "completed"
    assert outcome.details["overall"] == "memory_limit"


def test_runner_reports_docker_start_failure(monkeypatch):
    import minieval_bench.evaluation.code as code_module

    monkeypatch.setattr(code_module, "docker_available", lambda: (True, "29.6.1"))

    def failing_popen(*args, **kwargs):
        raise OSError("docker binary vanished")

    monkeypatch.setattr(code_module.subprocess, "Popen", failing_popen)

    runner = DockerCodeRunner(CodeEvalConfig())
    outcome = runner.run("def add(a, b):\n    return a + b\n", ADD_TESTS)

    assert outcome.status == "unavailable"
    assert "failed to start docker" in outcome.error_message


def test_runner_reports_docker_error_without_output(monkeypatch):
    fake = FakePopen(returncode=125, stderr=b"docker: invalid option")
    install_fake_popen(monkeypatch, fake)

    runner = DockerCodeRunner(CodeEvalConfig())
    outcome = runner.run("def add(a, b):\n    return a + b\n", ADD_TESTS)

    assert outcome.status == "failed"
    assert "code 125" in outcome.error_message


def test_runner_rejects_unparseable_controller_output(monkeypatch):
    fake = FakePopen(returncode=0, stdout=b"total garbage")
    install_fake_popen(monkeypatch, fake)

    runner = DockerCodeRunner(CodeEvalConfig())
    outcome = runner.run("def add(a, b):\n    return a + b\n", ADD_TESTS)

    assert outcome.status == "failed"
    assert "cannot parse" in outcome.error_message
