from __future__ import annotations

import re
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

MetricName = Literal["exact_match", "contains", "regex", "judge", "json_schema", "code_tests"]
DeterministicMetric = Literal["exact_match", "contains", "regex"]
DETERMINISTIC_METRICS = ("exact_match", "contains", "regex")


class Defaults(BaseModel):
    model_config = ConfigDict(extra="forbid")

    repeats: int = Field(default=1, ge=1)
    concurrency: int = Field(default=4, ge=1)
    seed: int = 0
    timeout: float = Field(default=60, gt=0)
    max_tokens: int | None = Field(default=None, ge=1)
    temperature: float | None = None
    shuffle: bool = False


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


class JudgeConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    provider: Literal["litellm", "mock"] = "litellm"
    model: str | None = None
    temperature: float | None = 0
    max_tokens: int | None = Field(default=None, ge=1)
    timeout: float | None = Field(default=None, gt=0)
    repeats: int = Field(default=1, ge=1)
    rubric: str | None = None
    calibration: str | None = None
    max_calls: int | None = Field(default=None, ge=1)
    budget_usd: float | None = Field(default=None, gt=0)
    input_cost_per_1m: float | None = Field(default=None, ge=0)
    output_cost_per_1m: float | None = Field(default=None, ge=0)


class CodeEvalConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    runtime: Literal["docker"] = "docker"
    image: str = "python:3.12-slim"
    timeout_seconds: float = Field(default=30, gt=0)
    per_test_timeout_seconds: float = Field(default=10, gt=0)
    memory_mb: int = Field(default=256, ge=16)
    cpus: float = Field(default=1.0, gt=0)
    pids_limit: int = Field(default=64, ge=1)
    max_output_bytes: int = Field(default=65536, ge=1024)


class EvaluatorsConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    judge: JudgeConfig | None = None
    code: CodeEvalConfig | None = None


class StatisticsConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    bootstrap_iterations: int = Field(default=1000, ge=100)
    confidence_level: float = Field(default=0.95, gt=0, lt=1)


class BenchmarkConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    benchmark: str = Field(min_length=1)
    dataset: str = Field(min_length=1)
    models: list[ModelConfig] = Field(min_length=1)
    defaults: Defaults = Defaults()
    evaluators: EvaluatorsConfig | None = None
    statistics: StatisticsConfig = StatisticsConfig()


class DatasetEntry(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: str = Field(min_length=1)
    prompt: str = Field(min_length=1)
    metric: MetricName
    reference: str | None = None
    system: str | None = None
    json_schema: dict | None = None
    tests: str | None = None

    @model_validator(mode="after")
    def check_metric_requirements(self) -> DatasetEntry:
        if self.metric in DETERMINISTIC_METRICS:
            if not self.reference:
                raise ValueError(f"metric '{self.metric}' requires a reference")
            if self.metric == "regex":
                try:
                    re.compile(self.reference)
                except re.error as exc:
                    raise ValueError(f"invalid regex reference: {exc}") from exc
        if self.metric == "json_schema":
            if self.json_schema is None:
                raise ValueError("metric 'json_schema' requires a schema")
            from minieval_bench.evaluation.structured import check_schema

            check_schema(self.json_schema)
        if self.metric == "code_tests":
            if not self.tests:
                raise ValueError("metric 'code_tests' requires a tests module")
            try:
                compile(self.tests, f"{self.id}:tests", "exec")
            except SyntaxError as exc:
                raise ValueError(f"invalid tests module: {exc}") from exc
        return self


class EvaluationOutcome(BaseModel):
    evaluator: str
    status: Literal["completed", "failed", "unavailable"]
    passed: bool | None = None
    score: float | None = None
    verdict: str | None = None
    details: dict | None = None
    error_message: str | None = None
    input_tokens: int | None = None
    output_tokens: int | None = None
    cost_usd: float | None = None


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
    queue_ms: float | None = None
    plan_index: int | None = None
    evaluation: EvaluationOutcome | None = None
    generation_ms: float | None = None
    evaluation_ms: float | None = None
    total_ms: float | None = None


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
    median_evaluation_ms: float | None = None
    median_total_ms: float | None = None
    input_tokens: int | None
    output_tokens: int | None
    cost_usd: float | None
    errors: dict[str, int]
    answer_accuracy: float | None = None
    evaluation_coverage: float | None = None
    overall_pass_rate: float | None = None
    incorrect: int | None = None
    median_queue_ms: float | None = None
    input_tokens_reported: int | None = None
    output_tokens_reported: int | None = None
    cost_reported: int | None = None
    evaluator_failed: int = 0
    evaluator_unavailable: int = 0
    evaluation_input_tokens: int | None = None
    evaluation_output_tokens: int | None = None
    evaluation_cost_usd: float | None = None
    evaluation_cost_reported: int | None = None

    @model_validator(mode="before")
    @classmethod
    def derive_legacy_fields(cls, data: Any) -> Any:
        if not isinstance(data, dict):
            return data
        if data.get("answer_accuracy") is None and data.get("score") is not None:
            data["answer_accuracy"] = data["score"]
        requests = data.get("requests")
        if requests:
            if data.get("evaluation_coverage") is None and data.get("evaluable") is not None:
                data["evaluation_coverage"] = data["evaluable"] / requests
            if data.get("overall_pass_rate") is None and data.get("passed") is not None:
                data["overall_pass_rate"] = data["passed"] / requests
        succeeded = data.get("succeeded")
        passed = data.get("passed")
        if data.get("incorrect") is None and succeeded is not None and passed is not None:
            data["incorrect"] = succeeded - passed
        return data


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
    duration_seconds: float | None = None
    trial_seeds: list[int] | None = None
    shuffle: bool | None = None
    evaluators: dict | None = None
    statistics: dict | None = None
