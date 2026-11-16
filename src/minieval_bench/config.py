from __future__ import annotations

import json
from pathlib import Path

import yaml
from pydantic import ValidationError

from minieval_bench.models import BenchmarkConfig, DatasetEntry

MODEL_FIELDS_WITH_DEFAULTS = ("temperature", "max_tokens", "timeout")


class ConfigError(Exception):
    pass


class DatasetError(Exception):
    pass


def format_validation_error(exc: ValidationError) -> str:
    lines = []
    for error in exc.errors(include_url=False):
        location = ".".join(str(part) for part in error["loc"]) or "config"
        lines.append(f"{location}: {error['msg']}")
    return "; ".join(lines)


def load_config(config_path: Path) -> BenchmarkConfig:
    if not config_path.is_file():
        raise ConfigError(f"config file not found: {config_path}")
    try:
        raw = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    except yaml.YAMLError as exc:
        raise ConfigError(f"invalid YAML in {config_path}: {exc}") from exc

    try:
        config = BenchmarkConfig.model_validate(raw)
    except ValidationError as exc:
        raise ConfigError(
            f"invalid config {config_path}: {format_validation_error(exc)}"
        ) from exc

    names = [model.name for model in config.models]
    duplicates = sorted({name for name in names if names.count(name) > 1})
    if duplicates:
        raise ConfigError(f"duplicate model names: {', '.join(duplicates)}")

    for model in config.models:
        if model.provider == "litellm" and not model.model:
            raise ConfigError(
                f"model '{model.name}' uses provider 'litellm' and requires a model string"
            )

    apply_defaults(config)
    resolve_dataset_path(config, config_path)
    resolve_evaluator_paths(config, config_path)
    return config


def apply_defaults(config: BenchmarkConfig) -> None:
    for model in config.models:
        for field in MODEL_FIELDS_WITH_DEFAULTS:
            if field not in model.model_fields_set and getattr(config.defaults, field) is not None:
                setattr(model, field, getattr(config.defaults, field))
        if model.model is None:
            model.model = model.name


def resolve_dataset_path(config: BenchmarkConfig, config_path: Path) -> None:
    dataset_path = Path(config.dataset)
    if not dataset_path.is_absolute():
        dataset_path = config_path.parent / dataset_path
    config.dataset = str(dataset_path.resolve())


def resolve_evaluator_paths(config: BenchmarkConfig, config_path: Path) -> None:
    judge = config.evaluators.judge if config.evaluators else None
    if judge is None:
        return
    for field in ("rubric", "calibration"):
        path = getattr(judge, field)
        if path and not Path(path).is_absolute():
            setattr(judge, field, str((config_path.parent / path).resolve()))


def load_dataset(dataset_path: Path) -> list[DatasetEntry]:
    if not dataset_path.is_file():
        raise DatasetError(f"dataset file not found: {dataset_path}")
    try:
        content = dataset_path.read_text(encoding="utf-8")
    except UnicodeDecodeError as exc:
        raise DatasetError(f"{dataset_path}: dataset is not valid UTF-8") from exc
    return parse_dataset_content(content, str(dataset_path))


def parse_dataset_content(content: str, source_name: str) -> list[DatasetEntry]:
    entries = []
    seen_ids = set()
    for line_number, line in enumerate(content.splitlines(), 1):
        if not line.strip():
            continue
        try:
            raw_entry = json.loads(line)
        except json.JSONDecodeError as exc:
            raise DatasetError(
                f"{source_name}:{line_number}: invalid JSON ({exc.msg})"
            ) from exc
        try:
            entry = DatasetEntry.model_validate(raw_entry)
        except ValidationError as exc:
            raise DatasetError(
                f"{source_name}:{line_number}: {format_validation_error(exc)}"
            ) from exc
        if entry.id in seen_ids:
            raise DatasetError(
                f"{source_name}:{line_number}: duplicate entry id '{entry.id}'"
            )
        seen_ids.add(entry.id)
        entries.append(entry)

    if not entries:
        raise DatasetError(f"{source_name}: dataset is empty")
    return entries
