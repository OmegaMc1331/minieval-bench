from __future__ import annotations

import re
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, ValidationInfo, field_validator

MetricName = Literal["exact_match", "contains", "regex"]


class Defaults(BaseModel):
    model_config = ConfigDict(extra="forbid")

    repeats: int = Field(default=1, ge=1)
    concurrency: int = Field(default=4, ge=1)
    seed: int = 0
    timeout: float = Field(default=60, gt=0)
    max_tokens: int | None = Field(default=None, ge=1)
    temperature: float | None = None


class ModelConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str = Field(min_length=1)
    provider: Literal["litellm", "mock"] = "litellm"
    model: str | None = None
    temperature: float | None = None
    max_tokens: int | None = Field(default=None, ge=1)
    timeout: float | None = Field(default=None, gt=0)
    drop_params: bool = False
    input_cost_per_1m: float | None = Field(default=None, ge=0)
    output_cost_per_1m: float | None = Field(default=None, ge=0)


class BenchmarkConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    benchmark: str = Field(min_length=1)
    dataset: str = Field(min_length=1)
    models: list[ModelConfig] = Field(min_length=1)
    defaults: Defaults = Defaults()


class DatasetEntry(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: str = Field(min_length=1)
    prompt: str = Field(min_length=1)
    metric: MetricName
    reference: str = Field(min_length=1)
    system: str | None = None

    @field_validator("reference")
    @classmethod
    def check_regex_reference(cls, value: str, info: ValidationInfo) -> str:
        if info.data.get("metric") == "regex":
            try:
                re.compile(value)
            except re.error as exc:
                raise ValueError(f"invalid regex reference: {exc}") from exc
        return value


class TrialResult(BaseModel):
    model: str
    entry_id: str
    repeat: int
    ok: bool
    response: str | None
    latency_ms: float
    input_tokens: int | None
    output_tokens: int | None
    cost_usd: float | None
    error_type: str | None
    error_message: str | None
    passed: bool | None


class ModelStats(BaseModel):
    model: str
    requests: int
    succeeded: int
    failed: int
    success_rate: float
    evaluable: int
    passed: int
    score: float | None
    median_latency_ms: float | None
    p95_latency_ms: float | None
    input_tokens: int | None
    output_tokens: int | None
    cost_usd: float | None
    errors: dict[str, int]


class Manifest(BaseModel):
    minieval_version: str
    benchmark: str
    created_at_utc: str
    seed: int
    repeats: int
    concurrency: int
    dataset_path: str
    dataset_sha256: str
    dataset_entries: int
    models: list[ModelConfig]
