from __future__ import annotations

import hashlib
import json
from pathlib import Path

from minieval_bench import __version__
from minieval_bench.models import (
    BenchmarkConfig,
    DatasetEntry,
    Defaults,
    ModelConfig,
    StatisticsConfig,
)
from minieval_bench.reports import judge_manifest_block
from minieval_bench.suites.models import SuiteTask
from minieval_bench.suites.registry import LoadedSuite

LOCKFILE_NAME = "benchmark.lock.json"


def build_lockfile(
    loaded: LoadedSuite,
    models: list[ModelConfig],
    effective_defaults: Defaults,
    user_overrides: dict,
    task_sections: list[dict],
) -> dict:
    definition = loaded.definition
    return {
        "minieval_version": __version__,
        "suite": {
            "id": definition.id,
            "version": definition.version,
            "definition_fingerprint": definition.fingerprint(),
            "name": definition.name,
            "license": definition.license,
        },
        "generation": {
            "defaults": effective_defaults.model_dump(mode="json"),
            "user_overrides": user_overrides,
            "models": [model.model_dump(mode="json", exclude_none=True) for model in models],
        },
        "evaluators": _evaluator_block(loaded),
        "statistics": definition.statistics.model_dump(mode="json"),
        "tasks": task_sections,
    }


def _evaluator_block(loaded: LoadedSuite) -> dict | None:
    evaluators = loaded.definition.evaluators
    if evaluators is None:
        return None
    block = {}
    if evaluators.judge is not None:
        block["judge"] = judge_manifest_block(_stub_config(loaded))
    if evaluators.code is not None:
        block["code"] = evaluators.code.model_dump(mode="json")
    return block


def _stub_config(loaded: LoadedSuite) -> BenchmarkConfig:
    return BenchmarkConfig(
        benchmark=loaded.definition.id,
        dataset="n/a",
        models=[ModelConfig(name="n/a")],
        evaluators=loaded.definition.evaluators,
    )


def task_lock_section(
    task: SuiteTask,
    dataset_sha256: str,
    entries: list[DatasetEntry],
    sampling_record: dict,
) -> dict:
    return {
        "id": task.id,
        "weight": task.weight,
        "dataset": {
            "reference": task.dataset,
            "sha256": dataset_sha256,
            "entry_count": len(entries),
        },
        "sampling": sampling_record,
        "metrics": _metric_counts(entries),
        "entry_fingerprints": _entry_fingerprints(entries),
    }


def _metric_counts(entries: list[DatasetEntry]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for entry in entries:
        counts[entry.metric] = counts.get(entry.metric, 0) + 1
    return counts


def _entry_fingerprints(entries: list[DatasetEntry]) -> dict:
    schemas = {}
    tests = {}
    for entry in entries:
        if entry.json_schema is not None:
            schemas[entry.id] = canonical_sha(entry.json_schema)
        if entry.tests is not None:
            tests[entry.id] = hashlib.sha256(entry.tests.encode()).hexdigest()
    return {"json_schemas": schemas, "code_tests": tests}


def canonical_sha(payload: dict) -> str:
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode()).hexdigest()


def lockfile_bytes(lockfile: dict) -> bytes:
    return (json.dumps(lockfile, sort_keys=True, indent=2) + "\n").encode()


def write_lockfile(path: Path, lockfile: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_bytes(lockfile_bytes(lockfile))
    temporary.replace(path)


def read_lockfile(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def lockfile_differences(stored: dict, current: dict) -> list[str]:
    differences: list[str] = []

    stored_suite = stored.get("suite", {})
    current_suite = current.get("suite", {})
    if stored_suite.get("id") != current_suite.get("id"):
        differences.append(
            f"suite id changed: {stored_suite.get('id')} -> {current_suite.get('id')}"
        )
    if stored_suite.get("version") != current_suite.get("version"):
        differences.append(
            f"suite version changed: {stored_suite.get('version')} ->"
            f" {current_suite.get('version')}"
        )
    if stored_suite.get("definition_fingerprint") != current_suite.get("definition_fingerprint"):
        differences.append(
            "suite definition changed "
            f"({stored_suite.get('definition_fingerprint')} -> "
            f"{current_suite.get('definition_fingerprint')})"
        )

    stored_tasks = {task.get("id"): task for task in stored.get("tasks", [])}
    current_tasks = {task.get("id"): task for task in current.get("tasks", [])}
    for task_id in sorted(set(stored_tasks) - set(current_tasks)):
        differences.append(f"task '{task_id}' exists in the stored lockfile but not anymore")
    for task_id in sorted(set(current_tasks) - set(stored_tasks)):
        differences.append(f"task '{task_id}' was added to the suite")
    for task_id in sorted(set(stored_tasks) & set(current_tasks)):
        differences.extend(
            _task_differences(task_id, stored_tasks[task_id], current_tasks[task_id])
        )

    if stored.get("evaluators") != current.get("evaluators"):
        differences.append(
            "evaluator configuration changed "
            "(judge rubric or model, code runtime settings)"
        )
    if stored.get("statistics") != current.get("statistics"):
        differences.append("statistics configuration changed")

    stored_generation = stored.get("generation", {})
    current_generation = current.get("generation", {})
    if stored_generation.get("models") != current_generation.get("models"):
        differences.append("model configuration changed")
    if stored_generation.get("defaults") != current_generation.get("defaults"):
        differences.append("run parameters changed (repeats, seed, concurrency, ...)")
    return differences


def _task_differences(task_id: str, stored: dict, current: dict) -> list[str]:
    differences: list[str] = []
    stored_dataset = stored.get("dataset", {})
    current_dataset = current.get("dataset", {})
    if stored_dataset.get("sha256") != current_dataset.get("sha256"):
        differences.append(
            f"task '{task_id}': dataset content changed "
            f"({stored_dataset.get('sha256')} -> {current_dataset.get('sha256')})"
        )
    if stored.get("sampling") != current.get("sampling"):
        differences.append(f"task '{task_id}': sampling selection changed")
    stored_prints = stored.get("entry_fingerprints", {})
    current_prints = current.get("entry_fingerprints", {})
    for kind in ("json_schemas", "code_tests"):
        shared = sorted(set(stored_prints.get(kind, {})) & set(current_prints.get(kind, {})))
        for entry_id in shared:
            if stored_prints[kind][entry_id] != current_prints[kind][entry_id]:
                differences.append(
                    f"task '{task_id}': {kind.removesuffix('s')} of entry '{entry_id}' changed"
                )
    if stored.get("weight") != current.get("weight"):
        differences.append(
            f"task '{task_id}': weight changed "
            f"({stored.get('weight')} -> {current.get('weight')})"
        )
    return differences


def statistics_from_lockfile(lockfile: dict) -> StatisticsConfig:
    return StatisticsConfig.model_validate(lockfile.get("statistics") or {})
