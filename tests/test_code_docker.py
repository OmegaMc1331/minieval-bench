from __future__ import annotations

import os

import pytest

from minieval_bench.evaluation.code import DockerCodeRunner, docker_available
from minieval_bench.models import CodeEvalConfig

pytestmark = pytest.mark.docker

ADD_TESTS = "def test_add():\n    assert add(2, 3) == 5\n"
DOCKER_ENABLED = os.environ.get("MINIEVAL_DOCKER_TESTS") == "1"

if not DOCKER_ENABLED:
    pytest.skip(
        "docker integration tests require MINIEVAL_DOCKER_TESTS=1 and a reachable daemon",
        allow_module_level=True,
    )


@pytest.fixture(scope="module")
def runner() -> DockerCodeRunner:
    available, reason = docker_available()
    if not available:
        pytest.skip(f"docker daemon not reachable: {reason}")
    return DockerCodeRunner(CodeEvalConfig(timeout_seconds=30, per_test_timeout_seconds=5))


def test_correct_function_passes_in_container(runner):
    outcome = runner.run("def add(a, b):\n    return a + b\n", ADD_TESTS)

    assert outcome.status == "completed"
    assert outcome.passed is True
    assert outcome.score == 1.0


def test_incorrect_function_fails_in_container(runner):
    outcome = runner.run("def add(a, b):\n    return a * b\n", ADD_TESTS)

    assert outcome.passed is False
    assert outcome.score == 0.0


def test_syntax_error_is_reported(runner):
    outcome = runner.run("def add(a, b:\n    pass\n", ADD_TESTS)

    assert outcome.status == "completed"
    assert outcome.details["overall"] == "submission_error"


def test_runtime_exception_is_reported(runner):
    outcome = runner.run("def add(a, b):\n    raise RuntimeError('nope')\n", ADD_TESTS)

    assert outcome.passed is False
    assert any(entry["status"] == "error" for entry in outcome.details["tests"])


def test_infinite_loop_hits_per_test_timeout(runner):
    outcome = runner.run("def add(a, b):\n    while True:\n        pass\n", ADD_TESTS)

    assert outcome.passed is False
    assert any(entry["status"] == "timeout" for entry in outcome.details["tests"])


def test_memory_bomb_is_killed_by_limit(runner):
    outcome = runner.run(
        "def add(a, b):\n    blob = bytearray(512 * 1024 * 1024)\n    return len(blob)\n",
        ADD_TESTS,
    )

    assert outcome.passed is False
    assert outcome.details["overall"] in ("memory_limit", "timeout")


def test_network_access_fails_inside_container(runner):
    outcome = runner.run(
        "def add(a, b):\n"
        "    import urllib.request\n"
        "    urllib.request.urlopen('http://example.com', timeout=5)\n"
        "    return a + b\n",
        ADD_TESTS,
    )

    assert outcome.passed is False
    statuses = [entry["status"] for entry in outcome.details["tests"]]
    assert "error" in statuses


def test_filesystem_write_is_blocked(runner):
    outcome = runner.run(
        "def add(a, b):\n"
        "    with open('/etc/pwned', 'w') as handle:\n"
        "        handle.write('x')\n"
        "    return a + b\n",
        ADD_TESTS,
    )

    assert outcome.passed is False
    statuses = [entry["status"] for entry in outcome.details["tests"]]
    assert "error" in statuses
