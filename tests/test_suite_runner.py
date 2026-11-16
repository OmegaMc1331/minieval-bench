from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml

from minieval_bench.suites.lockfile import LOCKFILE_NAME, lockfile_bytes
from minieval_bench.suites.registry import load_suite_dir
from minieval_bench.suites.runner import (
    SUITE_MANIFEST_NAME,
    SUITE_RESULTS_NAME,
    SuiteRunError,
    SuiteRunOptions,
    UserOverrides,
    dry_run_details,
    preflight_problems,
    prepare_run,
    run_suite,
)
from minieval_bench.suites.sampling import SamplingPlan

MODELS_YAML = """
models:
  - name: mock-a
    provider: mock
  - name: mock-b
    provider: mock
"""


def write_jsonl(path: Path, entries: list[dict]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "\n".join(json.dumps(entry) for entry in entries) + "\n", encoding="utf-8"
    )
    return path


@pytest.fixture
def suite_dir(tmp_path: Path) -> Path:
    root = tmp_path / "suites" / "demo-suite"
    definition = {
        "id": "demo-suite",
        "version": "1.0.0",
        "name": "Demo Suite",
        "description": "Two deterministic tasks.",
        "license": "MIT",
        "defaults": {"repeats": 2, "seed": 5, "concurrency": 2},
        "tasks": [
            {"id": "alpha", "dataset": "datasets/alpha.jsonl", "weight": 2.0},
            {"id": "beta", "dataset": "datasets/beta.jsonl", "weight": 1.0},
        ],
    }
    root.mkdir(parents=True)
    (root / "suite.yaml").write_text(yaml.safe_dump(definition), encoding="utf-8")
    write_jsonl(
        root / "datasets" / "alpha.jsonl",
        [
            {"id": "a1", "prompt": "one", "metric": "exact_match", "reference": "1"},
            {"id": "a2", "prompt": "two", "metric": "exact_match", "reference": "2"},
            {"id": "a3", "prompt": "three", "metric": "exact_match", "reference": "3"},
        ],
    )
    write_jsonl(
        root / "datasets" / "beta.jsonl",
        [
            {"id": "b1", "prompt": "x", "metric": "contains", "reference": "mock"},
        ],
    )
    return root


@pytest.fixture
def models_file(tmp_path: Path) -> Path:
    path = tmp_path / "models.yaml"
    path.write_text(MODELS_YAML, encoding="utf-8")
    return path


@pytest.fixture
def options(suite_dir: Path, models_file: Path, tmp_path: Path) -> SuiteRunOptions:
    return SuiteRunOptions(
        models_path=models_file,
        output_dir=tmp_path / "results" / "demo-suite",
    )


def test_run_suite_executes_every_task(suite_dir: Path, options: SuiteRunOptions) -> None:
    outcome = run_suite(load_suite_dir(suite_dir), options)

    assert [result.status for result in outcome.tasks] == ["completed", "completed"]
    assert [result.reused for result in outcome.tasks] == [False, False]
    alpha, beta = outcome.tasks
    assert alpha.planned_calls == 3 * 2 * 2
    assert beta.planned_calls == 1 * 2 * 2
    assert len(alpha.trials) == 12
    assert {trial.model for trial in alpha.trials} == {"mock-a", "mock-b"}

    output = options.output_dir
    assert (output / SUITE_MANIFEST_NAME).is_file()
    assert (output / SUITE_RESULTS_NAME).is_file()
    assert (output / LOCKFILE_NAME).is_file()
    for task_id in ("alpha", "beta"):
        task_dir = output / "tasks" / task_id
        for name in ("results.json", "manifest.json", "report.md", "report.html"):
            assert (task_dir / name).is_file(), name
        manifest = json.loads((task_dir / "manifest.json").read_text(encoding="utf-8"))
        assert manifest["benchmark"] == f"demo-suite/{task_id}"


def test_suite_manifest_records_tasks_and_protocol(
    suite_dir: Path, options: SuiteRunOptions
) -> None:
    outcome = run_suite(load_suite_dir(suite_dir), options)
    payload = json.loads(
        (options.output_dir / SUITE_MANIFEST_NAME).read_text(encoding="utf-8")
    )
    assert payload["suite"]["id"] == "demo-suite"
    assert payload["suite"]["version"] == "1.0.0"
    assert payload["protocol_modified"] is False
    assert payload["sampled"] is False
    assert [task["status"] for task in payload["tasks"]] == ["completed", "completed"]
    assert payload["generation"]["defaults"]["repeats"] == 2
    assert [model["name"] for model in payload["models"]] == ["mock-a", "mock-b"]
    assert outcome.interrupted is False


def test_lockfile_is_identical_for_identical_protocols(
    suite_dir: Path, options: SuiteRunOptions, tmp_path: Path
) -> None:
    first = run_suite(load_suite_dir(suite_dir), options)
    second_options = SuiteRunOptions(
        models_path=options.models_path,
        output_dir=tmp_path / "results" / "second",
    )
    run_suite(load_suite_dir(suite_dir), second_options)
    first_bytes = (options.output_dir / LOCKFILE_NAME).read_bytes()
    second_bytes = (second_options.output_dir / LOCKFILE_NAME).read_bytes()
    assert first_bytes == second_bytes
    assert first.lockfile["generation"]["defaults"]["seed"] == 5


def test_lockfile_changes_with_model_parameters(
    suite_dir: Path, options: SuiteRunOptions, tmp_path: Path
) -> None:
    run_suite(load_suite_dir(suite_dir), options)
    other_models = tmp_path / "models-hot.yaml"
    other_models.write_text(
        "models:\n  - name: mock-a\n    provider: mock\n    temperature: 0.4\n"
        "  - name: mock-b\n    provider: mock\n",
        encoding="utf-8",
    )
    other_options = SuiteRunOptions(
        models_path=other_models,
        output_dir=tmp_path / "results" / "other",
    )
    run_suite(load_suite_dir(suite_dir), other_options)
    first = json.loads((options.output_dir / LOCKFILE_NAME).read_text(encoding="utf-8"))
    second = json.loads((other_options.output_dir / LOCKFILE_NAME).read_text(encoding="utf-8"))
    assert first["generation"]["models"] != second["generation"]["models"]


def test_resume_reuses_matching_tasks(suite_dir: Path, options: SuiteRunOptions) -> None:
    loaded = load_suite_dir(suite_dir)
    run_suite(loaded, options)
    marker = options.output_dir / "tasks" / "alpha" / "results.json"
    original = marker.read_text(encoding="utf-8")

    resume_options = SuiteRunOptions(
        models_path=options.models_path,
        output_dir=options.output_dir,
        resume=True,
    )
    outcome = run_suite(loaded, resume_options)
    assert [result.reused for result in outcome.tasks] == [True, True]
    assert marker.read_text(encoding="utf-8") == original

    payload = json.loads(
        (options.output_dir / SUITE_MANIFEST_NAME).read_text(encoding="utf-8")
    )
    assert payload["resume_used"] is True


def test_resume_reruns_changed_datasets(suite_dir: Path, options: SuiteRunOptions) -> None:
    loaded = load_suite_dir(suite_dir)
    run_suite(loaded, options)

    write_jsonl(
        suite_dir / "datasets" / "beta.jsonl",
        [
            {"id": "b1", "prompt": "x changed", "metric": "contains", "reference": "mock"},
            {"id": "b2", "prompt": "y", "metric": "contains", "reference": "mock"},
        ],
    )
    reloaded = load_suite_dir(suite_dir)
    resume_options = SuiteRunOptions(
        models_path=options.models_path,
        output_dir=options.output_dir,
        resume=True,
    )
    outcome = run_suite(reloaded, resume_options)
    assert [result.reused for result in outcome.tasks] == [True, False]
    reused, rerun = outcome.tasks
    assert reused.task_id == "alpha"
    assert rerun.planned_calls == 2 * 2 * 2


def test_resume_ignores_missing_task_results(suite_dir: Path, options: SuiteRunOptions) -> None:
    loaded = load_suite_dir(suite_dir)
    run_suite(loaded, options)
    (options.output_dir / "tasks" / "alpha" / "results.json").unlink()

    resume_options = SuiteRunOptions(
        models_path=options.models_path,
        output_dir=options.output_dir,
        resume=True,
    )
    outcome = run_suite(loaded, resume_options)
    assert [result.reused for result in outcome.tasks] == [False, True]


def test_interruption_keeps_completed_tasks(
    suite_dir: Path, options: SuiteRunOptions, monkeypatch: pytest.MonkeyPatch
) -> None:
    from minieval_bench.suites import runner as runner_module

    original_execute = runner_module.execute_task

    def interrupting_execute(plan, run_options):
        if plan.task.id == "beta":
            raise KeyboardInterrupt
        return original_execute(plan, run_options)

    monkeypatch.setattr(runner_module, "execute_task", interrupting_execute)
    outcome = run_suite(load_suite_dir(suite_dir), options)
    monkeypatch.setattr(runner_module, "execute_task", original_execute)

    assert outcome.interrupted is True
    assert [result.status for result in outcome.tasks] == ["completed", "not_executed"]
    assert (options.output_dir / "tasks" / "alpha" / "results.json").is_file()
    assert not (options.output_dir / "tasks" / "beta").exists()


def test_provider_errors_make_task_partially_completed(
    suite_dir: Path, options: SuiteRunOptions, monkeypatch: pytest.MonkeyPatch
) -> None:
    from minieval_bench import runner as engine_runner

    class FailingProvider:
        def __init__(self, model):
            self.model = model

        def complete(self, prompt, system, seed):
            raise ConnectionError("simulated network failure")

    monkeypatch.setattr(engine_runner, "build_provider", lambda model: FailingProvider(model))
    outcome = run_suite(load_suite_dir(suite_dir), options)

    assert [result.status for result in outcome.tasks] == [
        "partially_completed",
        "partially_completed",
    ]
    alpha = outcome.tasks[0]
    assert all(not trial.ok for trial in alpha.trials)
    assert all(trial.error_type == "ConnectionError" for trial in alpha.trials)


def test_infrastructure_failure_is_isolated(
    suite_dir: Path, options: SuiteRunOptions, monkeypatch: pytest.MonkeyPatch
) -> None:
    from minieval_bench.suites import runner as runner_module

    original_run_benchmark = runner_module.run_benchmark

    def failing_run_benchmark(config, entries, show_progress=True):
        if config.benchmark.endswith("/alpha"):
            raise RuntimeError("docker daemon exploded")
        return original_run_benchmark(config, entries, show_progress=False)

    monkeypatch.setattr(runner_module, "run_benchmark", failing_run_benchmark)
    outcome = run_suite(load_suite_dir(suite_dir), options)

    statuses = [result.status for result in outcome.tasks]
    assert statuses == ["infrastructure_failed", "completed"]
    assert "RuntimeError" in outcome.tasks[0].error


def test_code_task_without_flag_is_unavailable(tmp_path: Path, models_file: Path) -> None:
    root = tmp_path / "suites" / "code-suite"
    definition = {
        "id": "code-suite",
        "version": "0.1.0",
        "name": "Code",
        "description": "Experimental code task.",
        "license": "MIT",
        "tasks": [{"id": "code", "dataset": "datasets/code.jsonl"}],
        "evaluators": {"code": {"runtime": "docker"}},
    }
    root.mkdir(parents=True)
    (root / "suite.yaml").write_text(yaml.safe_dump(definition), encoding="utf-8")
    write_jsonl(
        root / "datasets" / "code.jsonl",
        [
            {
                "id": "c1",
                "prompt": "Write add.",
                "metric": "code_tests",
                "tests": "def test_add():\n    assert add(1, 1) == 2\n",
            }
        ],
    )
    options = SuiteRunOptions(
        models_path=models_file,
        output_dir=tmp_path / "results" / "code-suite",
    )
    outcome = run_suite(load_suite_dir(root), options)

    assert outcome.tasks[0].status == "unavailable"
    assert "--allow-code-execution" in outcome.tasks[0].error
    assert outcome.tasks[0].stats == []


def test_preflight_refuses_over_max_calls(suite_dir: Path, options: SuiteRunOptions) -> None:
    loaded = load_suite_dir(suite_dir)
    plans, _, _, _ = prepare_run(loaded, options)
    limited = SuiteRunOptions(
        models_path=options.models_path,
        output_dir=options.output_dir,
        max_calls=4,
    )
    limited_plans, _, _, _ = prepare_run(loaded, limited)
    problems = preflight_problems(loaded, limited_plans, limited)
    assert any("exceed --max-calls" in problem for problem in problems)
    assert not preflight_problems(loaded, plans, options)


def test_preflight_refuses_code_without_flag(tmp_path: Path, models_file: Path) -> None:
    root = tmp_path / "suites" / "code-suite"
    definition = {
        "id": "code-suite",
        "version": "0.1.0",
        "name": "Code",
        "description": "d",
        "license": "MIT",
        "tasks": [{"id": "code", "dataset": "datasets/code.jsonl"}],
        "evaluators": {"code": {"runtime": "docker"}},
    }
    root.mkdir(parents=True)
    (root / "suite.yaml").write_text(yaml.safe_dump(definition), encoding="utf-8")
    write_jsonl(
        root / "datasets" / "code.jsonl",
        [
            {
                "id": "c1",
                "prompt": "Write add.",
                "metric": "code_tests",
                "tests": "def test_add():\n    assert add(1, 1) == 2\n",
            }
        ],
    )
    loaded = load_suite_dir(root)
    options = SuiteRunOptions(
        models_path=models_file, output_dir=tmp_path / "r", allow_code_execution=False
    )
    plans, _, _, _ = prepare_run(loaded, options)
    problems = preflight_problems(loaded, plans, options)
    assert any("--allow-code-execution" in problem for problem in problems)
    options.allow_code_execution = True
    assert not preflight_problems(loaded, plans, options)


def test_models_file_validation(tmp_path: Path) -> None:
    from minieval_bench.suites.runner import load_models_file

    missing = tmp_path / "absent.yaml"
    with pytest.raises(SuiteRunError, match="not found"):
        load_models_file(missing)

    empty = tmp_path / "empty.yaml"
    empty.write_text("models: []\n", encoding="utf-8")
    with pytest.raises(SuiteRunError, match="defines no model"):
        load_models_file(empty)

    duplicate = tmp_path / "dup.yaml"
    duplicate.write_text(
        "models:\n  - name: a\n    provider: mock\n  - name: a\n    provider: mock\n",
        encoding="utf-8",
    )
    with pytest.raises(SuiteRunError, match="duplicate model names"):
        load_models_file(duplicate)

    keyless = tmp_path / "keyless.yaml"
    keyless.write_text("models:\n  - name: real\n    provider: litellm\n", encoding="utf-8")
    with pytest.raises(SuiteRunError, match="requires a model string"):
        load_models_file(keyless)


def test_dry_run_details_report_costs_and_docker(
    suite_dir: Path, models_file: Path, tmp_path: Path
) -> None:
    priced = tmp_path / "models-priced.yaml"
    priced.write_text(
        "models:\n  - name: mock-priced\n    provider: mock\n"
        "    input_cost_per_1m: 1.0\n    output_cost_per_1m: 2.0\n"
        "    max_tokens: 64\n",
        encoding="utf-8",
    )
    loaded = load_suite_dir(suite_dir)
    options = SuiteRunOptions(
        models_path=priced, output_dir=tmp_path / "r", allow_code_execution=False
    )
    plans, _, _, _ = prepare_run(loaded, options)
    details = dry_run_details(plans)
    assert details["docker_required"] is False
    assert details["docker_available"] is None
    alpha_bounds = details["max_cost_upper_bounds"]["alpha"]["mock-priced"]
    assert alpha_bounds is not None and alpha_bounds > 0
    beta_bounds = details["max_cost_upper_bounds"]["beta"]["mock-priced"]
    assert beta_bounds is not None and beta_bounds < alpha_bounds

    unpriced_options = SuiteRunOptions(
        models_path=models_file, output_dir=tmp_path / "r2"
    )
    unpriced_plans, _, _, _ = prepare_run(loaded, unpriced_options)
    unpriced = dry_run_details(unpriced_plans)
    assert unpriced["max_cost_upper_bounds"]["alpha"]["mock-a"] is None


def test_sampling_subset_is_deterministic_and_recorded(
    suite_dir: Path, options: SuiteRunOptions, tmp_path: Path
) -> None:
    loaded = load_suite_dir(suite_dir)
    sampled = SuiteRunOptions(
        models_path=options.models_path,
        output_dir=tmp_path / "results" / "sampled",
        sampling=SamplingPlan(method="random", count=2, seed=11),
    )
    outcome = run_suite(loaded, sampled)
    lock = json.loads((sampled.output_dir / LOCKFILE_NAME).read_text(encoding="utf-8"))
    alpha_section = next(t for t in lock["tasks"] if t["id"] == "alpha")
    assert alpha_section["sampling"]["method"] == "random"
    assert alpha_section["sampling"]["selected_count"] == 2
    assert len(alpha_section["sampling"]["selected_ids"]) == 2
    assert alpha_section["dataset"]["entry_count"] == 3
    assert outcome.tasks[0].planned_calls == 2 * 2 * 2

    payload = json.loads(
        (sampled.output_dir / SUITE_MANIFEST_NAME).read_text(encoding="utf-8")
    )
    assert payload["sampled"] is True

    rerun = SuiteRunOptions(
        models_path=options.models_path,
        output_dir=tmp_path / "results" / "sampled-again",
        sampling=SamplingPlan(method="random", count=2, seed=11),
    )
    run_suite(loaded, rerun)
    other_lock = json.loads((rerun.output_dir / LOCKFILE_NAME).read_text(encoding="utf-8"))
    assert other_lock["tasks"] == lock["tasks"]

    other_seed = SuiteRunOptions(
        models_path=options.models_path,
        output_dir=tmp_path / "results" / "sampled-seed2",
        sampling=SamplingPlan(method="random", count=2, seed=12),
    )
    run_suite(loaded, other_seed)
    seed_lock = json.loads((other_seed.output_dir / LOCKFILE_NAME).read_text(encoding="utf-8"))
    seed_section = next(t for t in seed_lock["tasks"] if t["id"] == "alpha")
    assert seed_section["sampling"]["selected_ids"] != alpha_section["sampling"]["selected_ids"]


def test_user_overrides_are_recorded(suite_dir: Path, options: SuiteRunOptions) -> None:
    overridden = SuiteRunOptions(
        models_path=options.models_path,
        output_dir=options.output_dir,
        overrides=UserOverrides(repeats=1, seed=99),
    )
    outcome = run_suite(load_suite_dir(suite_dir), overridden)
    assert outcome.user_overrides == {"repeats": 1, "seed": 99}
    payload = json.loads(
        (options.output_dir / SUITE_MANIFEST_NAME).read_text(encoding="utf-8")
    )
    assert payload["protocol_modified"] is True
    assert payload["generation"]["user_overrides"] == {"repeats": 1, "seed": 99}
    assert payload["generation"]["defaults"]["repeats"] == 1


def test_lockfile_bytes_are_canonical(suite_dir: Path, options: SuiteRunOptions) -> None:
    loaded = load_suite_dir(suite_dir)
    plans, models, effective, overrides = prepare_run(loaded, options)
    from minieval_bench.suites.lockfile import build_lockfile

    lock = build_lockfile(
        loaded, models, effective, overrides, [plan.lock_section for plan in plans]
    )
    text = lockfile_bytes(lock).decode()
    assert text == lockfile_bytes(json.loads(text)).decode()
