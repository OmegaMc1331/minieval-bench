from __future__ import annotations

import hashlib
import json
import re
from pathlib import PurePosixPath

from pydantic import BaseModel, ConfigDict, Field, model_validator

from minieval_bench.models import Defaults, EvaluatorsConfig, MetricName, StatisticsConfig

SUITE_ID_PATTERN = re.compile(r"^[a-z0-9][a-z0-9-]*$")
TASK_ID_PATTERN = re.compile(r"^[a-z0-9][a-z0-9-]*$")
VERSION_PATTERN = re.compile(r"^(0|[1-9]\d*)\.(0|[1-9]\d*)\.(0|[1-9]\d*)$")


class SuiteError(Exception):
    pass


class SuiteTask(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: str = Field(min_length=1)
    dataset: str = Field(min_length=1)
    weight: float = Field(default=1.0, gt=0)
    description: str | None = None
    expected_metric: MetricName | None = None

    @model_validator(mode="after")
    def check_task_id(self) -> SuiteTask:
        if not TASK_ID_PATTERN.match(self.id):
            raise ValueError(
                f"task id '{self.id}' must use lowercase letters, digits and hyphens"
            )
        if not is_safe_dataset_reference(self.dataset):
            raise ValueError(
                f"dataset reference '{self.dataset}' must be a relative .jsonl path "
                "inside the suite directory"
            )
        return self


def is_safe_dataset_reference(reference: str) -> bool:
    path = PurePosixPath(reference)
    if path.is_absolute() or ".." in path.parts:
        return False
    return path.suffix == ".jsonl"


class SuiteDefinition(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: str
    version: str
    name: str = Field(min_length=1)
    description: str = Field(min_length=1)
    license: str = Field(min_length=1)
    source: str | None = None
    author: str | None = None
    defaults: Defaults = Defaults()
    tasks: list[SuiteTask] = Field(min_length=1)
    evaluators: EvaluatorsConfig | None = None
    statistics: StatisticsConfig = StatisticsConfig()

    @model_validator(mode="after")
    def check_definition(self) -> SuiteDefinition:
        if not SUITE_ID_PATTERN.match(self.id):
            raise ValueError(
                f"suite id '{self.id}' must use lowercase letters, digits and hyphens"
            )
        if not VERSION_PATTERN.match(self.version):
            raise ValueError(f"version '{self.version}' must be a semantic version X.Y.Z")
        task_ids = [task.id for task in self.tasks]
        duplicates = sorted({task_id for task_id in task_ids if task_ids.count(task_id) > 1})
        if duplicates:
            raise ValueError(f"duplicate task ids: {', '.join(duplicates)}")
        return self

    def version_tuple(self) -> tuple[int, int, int]:
        major, minor, patch = self.version.split(".")
        return int(major), int(minor), int(patch)

    def definition_payload(self) -> dict:
        return self.model_dump(mode="json", exclude_none=True)

    def fingerprint(self) -> str:
        canonical = json.dumps(
            self.definition_payload(), sort_keys=True, separators=(",", ":")
        )
        return hashlib.sha256(canonical.encode()).hexdigest()


def canonical_fingerprint(payload: dict) -> str:
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode()).hexdigest()
