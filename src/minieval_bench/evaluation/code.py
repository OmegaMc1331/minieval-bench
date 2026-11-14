from __future__ import annotations

import json
import subprocess
import uuid

from minieval_bench.models import CodeEvalConfig, EvaluationOutcome

HARNESS_TEMPLATE = '''
import json as _json
import signal as _signal
import sys as _sys
import traceback as _traceback

SUBMISSION_CODE = {submission!r}
TESTS_CODE = {tests!r}
PER_TEST_TIMEOUT = {per_test_timeout!r}

_result = {{"status": "completed", "submission_error": None, "tests": []}}
_namespace = {{"__name__": "submission"}}
try:
    exec(compile(SUBMISSION_CODE, "submission.py", "exec"), _namespace)
except BaseException:
    _result["status"] = "submission_error"
    _result["submission_error"] = _traceback.format_exc(limit=2)[-1500:]
    print(_json.dumps(_result))
    _sys.exit(0)

try:
    exec(compile(TESTS_CODE, "tests.py", "exec"), _namespace)
except BaseException:
    _result["status"] = "tests_error"
    _result["submission_error"] = _traceback.format_exc(limit=2)[-1500:]
    print(_json.dumps(_result))
    _sys.exit(0)

_tests = [
    (name, obj)
    for name, obj in sorted(_namespace.items())
    if name.startswith("test_") and callable(obj)
]
_has_alarm = hasattr(_signal, "SIGALRM")


def _alarm_handler(signum, frame):
    raise TimeoutError("per-test timeout")


if _has_alarm:
    _signal.signal(_signal.SIGALRM, _alarm_handler)

for _name, _func in _tests:
    _entry = {{"name": _name, "status": "passed", "error": None}}
    try:
        if _has_alarm:
            _signal.alarm(max(1, int(PER_TEST_TIMEOUT)))
        _func()
    except AssertionError:
        _entry["status"] = "failed"
        _entry["error"] = "assertion failed"
    except TimeoutError:
        _entry["status"] = "timeout"
        _entry["error"] = "exceeded the per-test timeout"
    except BaseException as _exc:
        _entry["status"] = "error"
        _entry["error"] = f"{{type(_exc).__name__}}: {{_exc}}"[:300]
    finally:
        if _has_alarm:
            _signal.alarm(0)
    _result["tests"].append(_entry)

print(_json.dumps(_result))
'''


def build_harness_script(submission: str, tests: str, per_test_timeout: float) -> str:
    return HARNESS_TEMPLATE.format(
        submission=submission, tests=tests, per_test_timeout=per_test_timeout
    )


def parse_harness_output(stdout: str) -> dict:
    for line in stdout.splitlines():
        line = line.strip()
        if line.startswith("{"):
            return json.loads(line)
    raise ValueError("no harness result found in container output")


def docker_available() -> tuple[bool, str]:
    try:
        completed = subprocess.run(
            ["docker", "version", "--format", "{{.Server.Version}}"],
            capture_output=True,
            text=True,
            timeout=20,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        return False, f"docker is not usable: {exc}"
    if completed.returncode != 0 or not completed.stdout.strip():
        return False, "docker daemon is not reachable"
    return True, completed.stdout.strip()


class DockerCodeRunner:
    def __init__(self, config: CodeEvalConfig) -> None:
        self.config = config

    def run(self, code: str, tests: str) -> EvaluationOutcome:
        available, reason = docker_available()
        if not available:
            return EvaluationOutcome(
                evaluator="code_tests",
                status="unavailable",
                error_message=f"code evaluation unavailable: {reason}",
            )

        container = f"minieval-eval-{uuid.uuid4().hex[:12]}"
        command = [
            "docker", "run", "--rm", "--name", container,
            "--network", "none",
            "--read-only",
            "--user", "65534:65534",
            "--cap-drop", "ALL",
            "--security-opt", "no-new-privileges",
            "--memory", f"{self.config.memory_mb}m",
            "--cpus", str(self.config.cpus),
            "--pids-limit", str(self.config.pids_limit),
            "--tmpfs", "/tmp:rw,noexec,nosuid,size=16m",
            "-i",
            self.config.image,
            "python", "-I", "-",
        ]
        harness = build_harness_script(
            code, tests, self.config.per_test_timeout_seconds
        ).encode("utf-8")

        try:
            completed = subprocess.run(
                command,
                input=harness,
                capture_output=True,
                timeout=self.config.timeout_seconds,
            )
        except subprocess.TimeoutExpired:
            subprocess.run(
                ["docker", "kill", container], capture_output=True, timeout=20
            )
            return EvaluationOutcome(
                evaluator="code_tests",
                status="completed",
                passed=False,
                score=None,
                details={
                    "overall": "timeout",
                    "error": (
                        f"container exceeded the {self.config.timeout_seconds:g}s wall-clock "
                        "timeout and was killed"
                    ),
                },
            )
        except OSError as exc:
            return EvaluationOutcome(
                evaluator="code_tests",
                status="unavailable",
                error_message=f"failed to start docker: {exc}",
            )

        stdout = completed.stdout[: self.config.max_output_bytes].decode("utf-8", "replace")
        if completed.returncode == 137:
            return EvaluationOutcome(
                evaluator="code_tests",
                status="completed",
                passed=False,
                score=None,
                details={
                    "overall": "memory_limit",
                    "error": f"container was killed by the {self.config.memory_mb} MB memory limit",
                },
            )
        if completed.returncode != 0 and not stdout.strip():
            return EvaluationOutcome(
                evaluator="code_tests",
                status="failed",
                error_message=(
                    f"docker exited with code {completed.returncode}: "
                    + completed.stderr[:300].decode("utf-8", "replace")
                ),
            )

        try:
            result = parse_harness_output(stdout)
        except (ValueError, json.JSONDecodeError) as exc:
            return EvaluationOutcome(
                evaluator="code_tests",
                status="failed",
                error_message=f"cannot parse harness output: {exc}",
            )
        return outcome_from_harness(result)


def outcome_from_harness(result: dict) -> EvaluationOutcome:
    if result.get("status") == "submission_error":
        return EvaluationOutcome(
            evaluator="code_tests",
            status="completed",
            passed=False,
            score=0.0,
            details={
                "overall": "submission_error",
                "submission_error": result.get("submission_error"),
                "tests": [],
                "passed": 0,
                "total": 0,
            },
        )
    if result.get("status") == "tests_error":
        return EvaluationOutcome(
            evaluator="code_tests",
            status="failed",
            error_message=f"test module raised: {result.get('submission_error')}",
        )

    tests = result.get("tests", [])
    passed_count = sum(1 for entry in tests if entry.get("status") == "passed")
    total = len(tests)
    return EvaluationOutcome(
        evaluator="code_tests",
        status="completed",
        passed=total > 0 and passed_count == total,
        score=passed_count / total if total else None,
        details={
            "overall": "completed",
            "tests": tests,
            "passed": passed_count,
            "total": total,
        },
    )
