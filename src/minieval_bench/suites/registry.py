from __future__ import annotations

import hashlib
import os
from dataclasses import dataclass
from importlib.resources import files
from importlib.resources.abc import Traversable
from pathlib import Path

import yaml
from pydantic import ValidationError

from minieval_bench.config import format_validation_error, parse_dataset_content
from minieval_bench.evaluation.judge import load_rubric
from minieval_bench.models import DatasetEntry
from minieval_bench.suites.models import SuiteDefinition, SuiteError

BUILTIN_ORIGIN = "builtin"
SUITE_FILE_NAME = "suite.yaml"
MAX_DATASET_BYTES = 64 * 1024 * 1024

SuiteDir = Path | Traversable


@dataclass
class SuiteSummary:
    suite_id: str
    version: str | None
    name: str | None
    origin: str
    location: str
    task_count: int | None
    error: str | None


@dataclass
class TaskDataset:
    reference: str
    sha256: str
    entries: list[DatasetEntry]

    def metric_counts(self) -> dict[str, int]:
        counts: dict[str, int] = {}
        for entry in self.entries:
            counts[entry.metric] = counts.get(entry.metric, 0) + 1
        return counts

    def category_counts(self) -> dict[str, int] | None:
        if any(entry.category is None for entry in self.entries):
            return None
        counts: dict[str, int] = {}
        for entry in self.entries:
            counts[entry.category or ""] = counts.get(entry.category or "", 0) + 1
        return counts


@dataclass
class LoadedSuite:
    definition: SuiteDefinition
    origin: str
    location: str
    datasets: dict[str, TaskDataset]

    def task_ids(self) -> list[str]:
        return [task.id for task in self.definition.tasks]


def registry_roots(extra_paths: list[Path] | None = None) -> list[tuple[str, SuiteDir]]:
    roots: list[tuple[str, SuiteDir]] = [(BUILTIN_ORIGIN, builtin_suites_root())]
    search_paths: list[Path] = []
    env_paths = os.environ.get("MINIEVAL_SUITE_PATH", "")
    search_paths.extend(Path(part) for part in env_paths.split(os.pathsep) if part)
    search_paths.extend(extra_paths or [])
    for path in search_paths:
        roots.append((str(path), path))
    return roots


def builtin_suites_root():
    return files("minieval_bench").joinpath("builtin_suites")


def discover_suites(extra_paths: list[Path] | None = None) -> list[SuiteSummary]:
    summaries = []
    for origin, root in registry_roots(extra_paths):
        if not root.is_dir():
            continue
        for entry in sorted(root.iterdir(), key=lambda item: item.name):
            if not entry.is_dir():
                continue
            if not (entry / SUITE_FILE_NAME).is_file():
                continue
            summaries.append(summarize_suite_dir(entry, origin))
    id_counts: dict[str, int] = {}
    for summary in summaries:
        id_counts[summary.suite_id] = id_counts.get(summary.suite_id, 0) + 1
    for summary in summaries:
        if id_counts[summary.suite_id] > 1 and summary.error is None:
            summary.error = "suite id is defined in several registry locations"
    return sorted(summaries, key=lambda summary: summary.suite_id)


def summarize_suite_dir(suite_dir: SuiteDir, origin: str) -> SuiteSummary:
    location = suite_location(suite_dir, origin)
    try:
        raw = (suite_dir / SUITE_FILE_NAME).read_text(encoding="utf-8")
    except OSError as exc:
        return SuiteSummary(
            suite_id=suite_dir.name, version=None, name=None, origin=origin,
            location=location, task_count=None, error=f"cannot read suite.yaml: {exc}",
        )
    try:
        definition = parse_suite_definition(raw)
    except SuiteError as exc:
        return SuiteSummary(
            suite_id=suite_dir.name, version=None, name=None, origin=origin,
            location=location, task_count=None, error=str(exc),
        )
    return SuiteSummary(
        suite_id=definition.id, version=definition.version, name=definition.name,
        origin=origin, location=location, task_count=len(definition.tasks), error=None,
    )


def suite_location(suite_dir: SuiteDir, origin: str) -> str:
    if origin == BUILTIN_ORIGIN:
        return f"{BUILTIN_ORIGIN}:{suite_dir.name}"
    return str(suite_dir)


def parse_suite_definition(text: str) -> SuiteDefinition:
    try:
        raw = yaml.safe_load(text)
    except yaml.YAMLError as exc:
        raise SuiteError(f"invalid YAML in suite definition: {exc}") from exc
    if not isinstance(raw, dict):
        raise SuiteError("suite definition must be a YAML mapping")
    try:
        return SuiteDefinition.model_validate(raw)
    except ValidationError as exc:
        raise SuiteError(f"invalid suite definition: {format_validation_error(exc)}") from exc


def load_suite(suite_id: str, extra_paths: list[Path] | None = None) -> LoadedSuite:
    candidates: list[tuple[str, SuiteDir]] = []
    for origin, root in registry_roots(extra_paths):
        if not root.is_dir():
            continue
        suite_dir = root / suite_id
        if suite_dir.is_dir() and (suite_dir / SUITE_FILE_NAME).is_file():
            candidates.append((origin, suite_dir))
    if not candidates:
        available = sorted({
            summary.suite_id for summary in discover_suites(extra_paths)
        })
        hint = f" available suites: {', '.join(available)}" if available else ""
        raise SuiteError(f"no suite '{suite_id}' in the registry.{hint}")
    if len(candidates) > 1:
        locations = ", ".join(suite_location(dir_, origin) for origin, dir_ in candidates)
        raise SuiteError(
            f"suite id '{suite_id}' is defined in several registry locations: {locations}"
        )
    origin, suite_dir = candidates[0]
    return load_suite_dir(suite_dir, origin)


def load_suite_dir(suite_dir: SuiteDir, origin: str = "path") -> LoadedSuite:
    location = suite_location(suite_dir, origin)
    definition = parse_suite_definition((suite_dir / SUITE_FILE_NAME).read_text(encoding="utf-8"))
    if isinstance(suite_dir, Path):
        resolve_rubric_path(definition, suite_dir)
    datasets = {}
    for task in definition.tasks:
        datasets[task.id] = load_task_dataset(suite_dir, task.dataset)
    return LoadedSuite(
        definition=definition, origin=origin, location=location, datasets=datasets
    )


def resolve_rubric_path(definition: SuiteDefinition, suite_dir: Path) -> None:
    judge = definition.evaluators.judge if definition.evaluators else None
    if judge is None or not judge.rubric:
        return
    rubric_path = Path(judge.rubric)
    if not rubric_path.is_absolute():
        rubric_path = suite_dir / rubric_path
    judge.rubric = str(rubric_path.resolve())


def load_task_dataset(suite_dir: SuiteDir, reference: str) -> TaskDataset:
    target = suite_dir / reference
    if isinstance(suite_dir, Path):
        resolved = target.resolve()
        if not resolved.is_relative_to(suite_dir.resolve()):
            raise SuiteError(
                f"dataset reference '{reference}' escapes the suite directory"
            )
        if not resolved.is_file():
            raise SuiteError(f"dataset file not found: {reference}")
        size = resolved.stat().st_size
        if size > MAX_DATASET_BYTES:
            raise SuiteError(
                f"dataset '{reference}' is too large ({size} bytes, "
                f"limit {MAX_DATASET_BYTES})"
            )
    try:
        content = target.read_bytes()
    except OSError as exc:
        raise SuiteError(f"cannot read dataset '{reference}': {exc}") from exc
    if len(content) > MAX_DATASET_BYTES:
        raise SuiteError(
            f"dataset '{reference}' is too large ({len(content)} bytes, "
            f"limit {MAX_DATASET_BYTES})"
        )
    try:
        text = content.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise SuiteError(f"dataset '{reference}' is not valid UTF-8") from exc
    entries = parse_dataset_content(text, reference)
    return TaskDataset(
        reference=reference,
        sha256=hashlib.sha256(content).hexdigest(),
        entries=entries,
    )


def validate_suite(loaded: LoadedSuite) -> list[str]:
    errors: list[str] = []
    definition = loaded.definition
    for task in definition.tasks:
        dataset = loaded.datasets.get(task.id)
        if dataset is None:
            errors.append(f"task '{task.id}': dataset could not be loaded")
            continue
        if task.expected_metric is not None:
            offending = [
                entry.id for entry in dataset.entries
                if entry.metric != task.expected_metric
            ]
            if offending:
                sample = ", ".join(offending[:5])
                suffix = "…" if len(offending) > 5 else ""
                errors.append(
                    f"task '{task.id}' expects metric '{task.expected_metric}' but "
                    f"{len(offending)} entr{'y uses' if len(offending) == 1 else 'ies use'} "
                    f"another metric: {sample}{suffix}"
                )
        metrics = dataset.metric_counts()
        needs_code = "code_tests" in metrics
        needs_judge = "judge" in metrics
        if needs_code and (definition.evaluators is None or definition.evaluators.code is None):
            errors.append(
                f"task '{task.id}' has code_tests entries but the suite does not "
                "configure evaluators.code"
            )
        if needs_judge and (definition.evaluators is None or definition.evaluators.judge is None):
            errors.append(
                f"task '{task.id}' has judge entries but the suite does not "
                "configure evaluators.judge"
            )
    judge = definition.evaluators.judge if definition.evaluators else None
    if judge is not None and judge.rubric:
        if loaded.origin == BUILTIN_ORIGIN:
            errors.append(
                "builtin suites cannot reference a rubric file: use the default rubric "
                "or ship the suite as a local directory"
            )
        else:
            try:
                load_rubric(Path(judge.rubric))
            except Exception as exc:
                errors.append(f"judge rubric cannot be loaded: {exc}")
    return errors
