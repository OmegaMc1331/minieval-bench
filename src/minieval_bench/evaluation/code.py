from __future__ import annotations

import ast
import json
import subprocess
import threading
import uuid

from minieval_bench.models import CodeEvalConfig, EvaluationOutcome

VERDICT_STATUSES = frozenset(
    {
        "completed",
        "submission_error",
        "tests_error",
        "worker_timeout",
        "output_overflow",
        "worker_crashed",
        "worker_killed",
        "report_invalid",
        "controller_error",
    }
)
TEST_STATUSES = frozenset({"passed", "failed", "timeout", "error"})
REPORT_LIMIT_BYTES = 1_000_000
READ_CHUNK_BYTES = 65536
HOST_TIMEOUT_GRACE_SECONDS = 15.0

WORKER_TEMPLATE = r'''
import json
import os
import signal
import sys
import traceback

REPORT_FD = __REPORT_FD__
PAYLOAD = json.loads(sys.stdin.read())

report = {"status": "completed", "submission_error": None, "tests": []}


def emit():
    os.write(REPORT_FD, json.dumps(report).encode())
    os.close(REPORT_FD)


namespace = {"__name__": "submission"}
try:
    exec(compile(PAYLOAD["submission"], "submission.py", "exec"), namespace)
except BaseException:
    report["status"] = "submission_error"
    report["submission_error"] = traceback.format_exc(limit=2)[-1500:]
    emit()
    sys.exit(0)

try:
    exec(compile(PAYLOAD["tests"], "tests.py", "exec"), namespace)
except BaseException:
    report["status"] = "tests_error"
    report["submission_error"] = traceback.format_exc(limit=2)[-1500:]
    emit()
    sys.exit(0)

tests = [
    (name, namespace[name])
    for name in sorted(namespace)
    if name.startswith("test_") and callable(namespace[name])
]
has_alarm = hasattr(signal, "SIGALRM")


def alarm_handler(signum, frame):
    raise TimeoutError("per-test timeout")


if has_alarm:
    signal.signal(signal.SIGALRM, alarm_handler)

sys.settrace(None)
sys.setprofile(None)

for name, func in tests:
    entry = {"name": name, "status": "passed", "error": None}
    try:
        if has_alarm:
            signal.alarm(max(1, int(PAYLOAD["per_test_timeout"])))
        func()
    except AssertionError:
        entry["status"] = "failed"
        entry["error"] = "assertion failed"
    except TimeoutError:
        entry["status"] = "timeout"
        entry["error"] = "exceeded the per-test timeout"
    except BaseException as exc:
        entry["status"] = "error"
        entry["error"] = f"{type(exc).__name__}: {exc}"[:300]
    finally:
        if has_alarm:
            signal.alarm(0)
    report["tests"].append(entry)

emit()
'''

CONTROLLER_TEMPLATE = r'''
import json
import os
import signal
import subprocess
import sys
import threading
import time

PAYLOAD = __PAYLOAD__
WORKER_SOURCE = __WORKER__
OUTPUT_CAP = PAYLOAD["output_cap_bytes"]
REPORT_LIMIT = __REPORT_LIMIT__
READ_CHUNK = 65536

verdict = {
    "status": "worker_crashed",
    "submission_error": None,
    "tests": [],
    "exit_code": None,
    "stdout_bytes": 0,
    "stderr_bytes": 0,
    "stdout_truncated": False,
    "stderr_truncated": False,
    "timed_out": False,
}


def read_bounded(stream, cap):
    chunks = []
    total = 0
    overflow = False
    while True:
        chunk = stream.read(READ_CHUNK)
        if not chunk:
            break
        total += len(chunk)
        if total > cap:
            overflow = True
            break
        chunks.append(chunk)
    return b"".join(chunks), overflow, total


def finish():
    sys.stdout.write(json.dumps(verdict) + "\n")
    sys.stdout.flush()
    sys.exit(0)


try:
    report_read, report_write = os.pipe()
    worker_source = WORKER_SOURCE.replace("__REPORT_FD__", str(report_write))
    worker = subprocess.Popen(
        [sys.executable, "-I", "-c", worker_source],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        pass_fds=(report_write,),
        start_new_session=True,
    )
    os.close(report_write)
    payload_bytes = json.dumps(PAYLOAD).encode()
    holders = {"report": {}, "stdout": {}, "stderr": {}}

    def feed_stdin():
        try:
            worker.stdin.write(payload_bytes)
            worker.stdin.close()
        except Exception:
            pass

    def read_into(holder, stream, cap):
        data, overflow, total = read_bounded(stream, cap)
        holder["data"] = data
        holder["overflow"] = overflow
        holder["total"] = total

    threads = [
        threading.Thread(target=feed_stdin, daemon=True),
        threading.Thread(
            target=read_into, args=(holders["report"], os.fdopen(report_read, "rb"), REPORT_LIMIT),
            daemon=True,
        ),
        threading.Thread(
            target=read_into, args=(holders["stdout"], worker.stdout, OUTPUT_CAP), daemon=True
        ),
        threading.Thread(
            target=read_into, args=(holders["stderr"], worker.stderr, OUTPUT_CAP), daemon=True
        ),
    ]
    for thread in threads:
        thread.start()

    timed_out = False
    try:
        worker.wait(timeout=PAYLOAD["timeout_seconds"])
    except subprocess.TimeoutExpired:
        timed_out = True
        try:
            os.killpg(os.getpgid(worker.pid), signal.SIGKILL)
        except Exception:
            worker.kill()
        worker.wait(timeout=10)
    for thread in threads:
        thread.join(timeout=10)

    verdict["timed_out"] = timed_out
    verdict["exit_code"] = worker.returncode
    for channel in ("stdout", "stderr"):
        verdict[channel + "_bytes"] = holders[channel].get("total", 0)
        verdict[channel + "_truncated"] = bool(holders[channel].get("overflow"))

    report = None
    report_data = holders["report"].get("data")
    if report_data is not None and not holders["report"].get("overflow"):
        try:
            report = json.loads(report_data)
        except ValueError:
            report = None

    valid_statuses = ("completed", "submission_error", "tests_error")
    if isinstance(report, dict) and report.get("status") in valid_statuses:
        verdict["status"] = report["status"]
        verdict["submission_error"] = report.get("submission_error")
        verdict["tests"] = report.get("tests") or []
    elif report_data is not None and holders["report"].get("overflow"):
        verdict["status"] = "report_invalid"
    elif verdict["stdout_truncated"] or verdict["stderr_truncated"]:
        verdict["status"] = "output_overflow"
    elif timed_out:
        verdict["status"] = "worker_timeout"
    elif worker.returncode == -9:
        verdict["status"] = "worker_killed"
    finish()
except SystemExit:
    raise
except BaseException as exc:
    sys.stdout.write(
        json.dumps({"status": "controller_error", "error": str(exc)[:500]}) + "\n"
    )
    sys.exit(0)
'''


def expected_test_names(tests_source: str) -> list[str]:
    try:
        tree = ast.parse(tests_source)
    except SyntaxError as exc:
        raise ValueError(f"invalid tests module: {exc}") from exc
    names: set[str] = set()
    for node in tree.body:
        if not isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef):
            continue
        if not node.name.startswith("test_"):
            continue
        if isinstance(node, ast.AsyncFunctionDef):
            raise ValueError(f"async test function '{node.name}' is not supported")
        names.add(node.name)
    return sorted(names)


def build_controller_script(
    submission: str,
    tests: str,
    per_test_timeout: float,
    timeout_seconds: float,
    output_cap_bytes: int,
) -> str:
    payload = {
        "submission": submission,
        "tests": tests,
        "per_test_timeout": per_test_timeout,
        "timeout_seconds": timeout_seconds,
        "output_cap_bytes": output_cap_bytes,
    }
    return (
        CONTROLLER_TEMPLATE.replace("__PAYLOAD__", repr(payload))
        .replace("__WORKER__", repr(WORKER_TEMPLATE))
        .replace("__REPORT_LIMIT__", str(REPORT_LIMIT_BYTES))
    )


def read_bounded(stream, cap: int) -> tuple[bytes, bool, int]:
    chunks = []
    total = 0
    overflow = False
    while True:
        chunk = stream.read(READ_CHUNK_BYTES)
        if not chunk:
            break
        total += len(chunk)
        if total > cap:
            overflow = True
            break
        chunks.append(chunk)
    return b"".join(chunks), overflow, total


def parse_controller_output(stdout: bytes) -> dict | None:
    verdict = None
    for line in stdout.decode("utf-8", "replace").splitlines():
        line = line.strip()
        if not line.startswith("{"):
            continue
        try:
            verdict = json.loads(line)
        except json.JSONDecodeError:
            continue
    return verdict if isinstance(verdict, dict) else None


def validate_worker_report(verdict: dict, expected_names: list[str]) -> str | None:
    status = verdict.get("status")
    if status not in VERDICT_STATUSES:
        return f"unknown verdict status: {status!r}"
    if status != "completed":
        return None
    tests = verdict.get("tests")
    if not isinstance(tests, list):
        return "completed verdict must contain a 'tests' list"
    names = []
    for entry in tests:
        if not isinstance(entry, dict):
            return "each reported test must be an object"
        name = entry.get("name")
        if not isinstance(name, str) or not name:
            return "each reported test must have a name"
        if entry.get("status") not in TEST_STATUSES:
            return f"test '{name}' has an invalid status: {entry.get('status')!r}"
        names.append(name)
    if len(names) != len(set(names)):
        return "reported tests contain duplicate names"
    if sorted(names) != expected_names:
        return (
            f"reported tests {sorted(names)} do not match the dataset tests {expected_names}"
        )
    return None


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

        expected_names = expected_test_names(tests)
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
        controller = build_controller_script(
            code,
            tests,
            self.config.per_test_timeout_seconds,
            self.config.timeout_seconds,
            self.config.max_output_bytes,
        ).encode("utf-8")

        try:
            process = subprocess.Popen(
                command, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE
            )
        except OSError as exc:
            return EvaluationOutcome(
                evaluator="code_tests",
                status="unavailable",
                error_message=f"failed to start docker: {exc}",
            )

        holders = {"stdout": {}, "stderr": {}}
        threads = [
            threading.Thread(target=self._feed_stdin, args=(process, controller), daemon=True),
            threading.Thread(
                target=self._read_into,
                args=(holders["stdout"], process.stdout, self.config.max_output_bytes),
                daemon=True,
            ),
            threading.Thread(
                target=self._read_into,
                args=(holders["stderr"], process.stderr, self.config.max_output_bytes),
                daemon=True,
            ),
        ]
        try:
            for thread in threads:
                thread.start()
            timed_out = False
            try:
                process.wait(
                    timeout=self.config.timeout_seconds + HOST_TIMEOUT_GRACE_SECONDS
                )
            except subprocess.TimeoutExpired:
                timed_out = True
                subprocess.run(
                    ["docker", "kill", container], capture_output=True, timeout=20
                )
                try:
                    process.wait(timeout=20)
                except subprocess.TimeoutExpired:
                    pass
            for thread in threads:
                thread.join(timeout=10)
        finally:
            subprocess.run(
                ["docker", "rm", "-f", container], capture_output=True, timeout=20
            )

        if timed_out:
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

        stdout = holders["stdout"].get("data", b"")
        stderr = holders["stderr"].get("data", b"")
        if process.returncode == 137:
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
        if process.returncode != 0 and not stdout.strip():
            return EvaluationOutcome(
                evaluator="code_tests",
                status="failed",
                error_message=(
                    f"docker exited with code {process.returncode}: "
                    + stderr[:300].decode("utf-8", "replace")
                ),
            )

        verdict = parse_controller_output(stdout)
        if verdict is None:
            return EvaluationOutcome(
                evaluator="code_tests",
                status="failed",
                error_message="cannot parse the evaluation controller output",
            )
        return outcome_from_verdict(verdict, expected_names)

    @staticmethod
    def _feed_stdin(process: subprocess.Popen, payload: bytes) -> None:
        try:
            process.stdin.write(payload)
            process.stdin.close()
        except (OSError, ValueError):
            pass

    @staticmethod
    def _read_into(holder: dict, stream, cap: int) -> None:
        data, overflow, total = read_bounded(stream, cap)
        holder["data"] = data
        holder["overflow"] = overflow
        holder["total"] = total


def outcome_from_verdict(verdict: dict, expected_names: list[str]) -> EvaluationOutcome:
    invalid = validate_worker_report(verdict, expected_names)
    if invalid:
        return EvaluationOutcome(
            evaluator="code_tests",
            status="failed",
            error_message=f"harness integrity failure: {invalid}",
            details={"overall": "invalid_verdict", "reported": verdict.get("status")},
        )
    status = verdict["status"]

    if status == "controller_error":
        return EvaluationOutcome(
            evaluator="code_tests",
            status="failed",
            error_message=f"controller failed: {verdict.get('error', 'unknown error')}",
        )
    if status == "submission_error":
        return EvaluationOutcome(
            evaluator="code_tests",
            status="completed",
            passed=False,
            score=0.0,
            details={
                "overall": "submission_error",
                "submission_error": verdict.get("submission_error"),
                "tests": [],
                "passed": 0,
                "total": 0,
                "exit_code": verdict.get("exit_code"),
            },
        )
    if status == "tests_error":
        return EvaluationOutcome(
            evaluator="code_tests",
            status="failed",
            error_message=f"test module raised: {verdict.get('submission_error')}",
        )
    if status == "worker_timeout":
        return EvaluationOutcome(
            evaluator="code_tests",
            status="completed",
            passed=False,
            score=None,
            details={
                "overall": "timeout",
                "error": "the evaluated program exceeded the wall-clock timeout",
            },
        )
    if status == "output_overflow":
        return EvaluationOutcome(
            evaluator="code_tests",
            status="completed",
            passed=False,
            score=None,
            details={
                "overall": "output_overflow",
                "error": (
                    "the evaluated program produced more output than the capture limit; "
                    "it was killed and cannot be scored"
                ),
                "stdout_bytes": verdict.get("stdout_bytes"),
                "stderr_bytes": verdict.get("stderr_bytes"),
            },
        )
    if status == "worker_crashed":
        return EvaluationOutcome(
            evaluator="code_tests",
            status="completed",
            passed=False,
            score=None,
            details={
                "overall": "worker_crashed",
                "error": "the evaluated program terminated the test worker before any "
                "test result was produced",
                "exit_code": verdict.get("exit_code"),
            },
        )
    if status == "worker_killed":
        return EvaluationOutcome(
            evaluator="code_tests",
            status="completed",
            passed=False,
            score=None,
            details={
                "overall": "worker_killed",
                "error": "the test worker was killed by SIGKILL, most likely by the "
                "container memory limit",
                "exit_code": verdict.get("exit_code"),
            },
        )
    if status == "report_invalid":
        return EvaluationOutcome(
            evaluator="code_tests",
            status="failed",
            error_message="the test worker produced an unreadable result report",
        )

    tests = verdict.get("tests", [])
    passed_count = sum(1 for entry in tests if entry.get("status") == "passed")
    total = len(tests)
    if total == 0:
        return EvaluationOutcome(
            evaluator="code_tests",
            status="completed",
            passed=False,
            score=None,
            details={
                "overall": "no_tests",
                "error": "no test function was executed for this submission",
                "tests": [],
                "passed": 0,
                "total": 0,
            },
        )
    return EvaluationOutcome(
        evaluator="code_tests",
        status="completed",
        passed=passed_count == total,
        score=passed_count / total,
        details={
            "overall": "completed",
            "tests": tests,
            "passed": passed_count,
            "total": total,
            "stdout_bytes": verdict.get("stdout_bytes"),
            "stderr_bytes": verdict.get("stderr_bytes"),
        },
    )
