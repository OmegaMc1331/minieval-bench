from __future__ import annotations

import json
from pathlib import Path

from minieval_bench.evaluation.statistics import (
    binary_pass_value,
    judge_score_value,
    paired_comparison,
)
from minieval_bench.models import StatisticsConfig
from minieval_bench.reports import load_results

INDICATORS = [
    ("answer_accuracy", "Answer accuracy (deterministic)", "rate"),
    ("json_schema_conformance", "JSON Schema conformance", "rate"),
    ("code_eval_pass_rate", "Code evaluation pass rate", "rate"),
    ("judge_score_mean", "Judge score (mean)", "score"),
    ("overall_pass_rate", "Overall pass rate", "rate"),
    ("success_rate", "Request success rate", "rate"),
    ("evaluation_coverage", "Evaluation coverage", "rate"),
    ("median_latency_ms", "Median generation latency (ms)", "ms"),
    ("p95_latency_ms", "P95 generation latency (ms)", "ms"),
    ("median_evaluation_ms", "Median evaluation latency (ms)", "ms"),
    ("median_total_ms", "Median total latency (ms)", "ms"),
    ("input_tokens", "Input tokens", "int"),
    ("output_tokens", "Output tokens", "int"),
    ("cost_usd", "Estimated cost (USD)", "cost"),
]

CONFIG_FIELDS = ("provider", "model", "temperature", "max_tokens", "timeout", "drop_params")

PAIRED_METRICS = [
    ("pass rate (conditional on evaluation)", binary_pass_value),
    ("judge score", judge_score_value),
]


def load_run(run_dir: Path) -> dict:
    manifest, stats, trials = load_results(run_dir)
    return {
        "manifest": manifest,
        "stats_by_model": {item.model: item for item in stats},
        "trials": trials,
        "path": str(run_dir),
    }


def build_comparison_payload(run_dirs: list[Path]) -> dict:
    runs = [load_run(run_dir) for run_dir in run_dirs]
    labels = unique_labels(runs)

    common_models = sorted(
        set.intersection(*(set(run["stats_by_model"]) for run in runs))
    )

    return {
        "kind": "compare",
        "runs": [
            {
                "label": label,
                "benchmark": run["manifest"].benchmark,
                "created_at_utc": run["manifest"].created_at_utc,
                "minieval_version": run["manifest"].minieval_version,
                "seed": run["manifest"].seed,
                "repeats": run["manifest"].repeats,
                "concurrency": run["manifest"].concurrency,
                "shuffle": run["manifest"].shuffle,
                "dataset_sha256": run["manifest"].dataset_sha256,
                "dataset_entries": run["manifest"].dataset_entries,
                "path": run["path"],
                "models": [model.model_dump(mode="json") for model in run["manifest"].models],
                "judge": run["manifest"].judge,
            }
            for label, run in zip(labels, runs, strict=True)
        ],
        "warnings": comparability_warnings(runs, common_models),
        "paired": paired_analyses(runs),
        "commonModels": [
            {
                "model": name,
                "configConsistent": model_config_consistent(name, runs),
                "configFields": config_differences(name, runs),
                "indicators": [
                    {
                        "key": key,
                        "label": label,
                        "format": value_format,
                        "values": [run_value(run, name, key) for run in runs],
                        "deltaAbs": delta_absolute(runs, name, key),
                        "deltaRel": delta_relative(runs, name, key),
                    }
                    for key, label, value_format in INDICATORS
                ],
            }
            for name in common_models
        ],
        "modelsOnlyIn": models_only_in(runs, labels),
    }


def unique_labels(runs: list[dict]) -> list[str]:
    labels = [run["manifest"].benchmark for run in runs]
    unique = []
    for index, label in enumerate(labels):
        if labels.count(label) > 1:
            unique.append(f"{label} ({index + 1})")
        else:
            unique.append(label)
    return unique


def run_value(run: dict, model: str, key: str) -> float | None:
    stats = run["stats_by_model"].get(model)
    if stats is None:
        return None
    return getattr(stats, key)


def first_and_last(runs: list[dict], model: str, key: str) -> tuple[float | None, float | None]:
    first = run_value(runs[0], model, key)
    last = run_value(runs[-1], model, key)
    return first, last


def delta_absolute(runs: list[dict], model: str, key: str) -> float | None:
    first, last = first_and_last(runs, model, key)
    if first is None or last is None:
        return None
    return last - first


def delta_relative(runs: list[dict], model: str, key: str) -> float | None:
    first, last = first_and_last(runs, model, key)
    if first is None or last is None or first == 0:
        return None
    return (last - first) / abs(first)


def model_config(run: dict, model: str) -> dict | None:
    for item in run["manifest"].models:
        if item.name == model:
            return item.model_dump(mode="json")
    return None


def config_differences(model: str, runs: list[dict]) -> list[str]:
    configs = [model_config(run, model) for run in runs]
    differing = []
    for field in CONFIG_FIELDS:
        values = {config.get(field) if config else None for config in configs}
        if len(values) > 1:
            differing.append(field)
    return differing


def model_config_consistent(model: str, runs: list[dict]) -> bool:
    return not config_differences(model, runs)


def comparability_warnings(runs: list[dict], common_models: list[str]) -> list[str]:
    manifests = [run["manifest"] for run in runs]
    warnings = []

    datasets = {manifest.dataset_sha256 for manifest in manifests}
    if len(datasets) > 1:
        warnings.append(
            "Datasets differ between runs: score differences may reflect the dataset change, "
            "not a change in model quality."
        )

    for field in ("repeats", "seed", "concurrency"):
        values = [getattr(manifest, field) for manifest in manifests]
        if len(set(values)) > 1:
            warnings.append(
                f"Experimental conditions differ: {field} values are {values} across runs."
            )

    versions = {manifest.minieval_version for manifest in manifests}
    if len(versions) > 1:
        warnings.append(
            f"Runs were produced by different MiniEval versions: {sorted(versions)}."
        )

    latency_semantics = {
        version_tuple(manifest.minieval_version) >= (0, 3, 1) for manifest in manifests
    }
    if len(latency_semantics) > 1:
        warnings.append(
            "Latency semantics differ: runs produced before 0.3.1 store a single latency "
            "mixing generation and evaluation."
        )

    judge_blocks = [manifest.judge for manifest in manifests]
    if any(block is not None for block in judge_blocks):
        known_blocks = [block for block in judge_blocks if block]
        if len(known_blocks) != len(judge_blocks):
            warnings.append(
                "Judge evaluation is enabled in some runs only, or its rubric is unknown for "
                "runs produced before 0.3.1: judge scores are not directly comparable."
            )
        else:
            fingerprints = {block.get("rubric_fingerprint") for block in known_blocks}
            if len(fingerprints) > 1:
                warnings.append(
                    "Judge rubrics differ between runs (criteria, scales or weights): "
                    "subjective scores are not directly comparable."
                )
            judge_models = {str(block.get("model")) for block in known_blocks}
            if len(judge_models) > 1:
                warnings.append(
                    f"Judge models differ across runs: {sorted(judge_models)}."
                )

    code_configs = [
        json.dumps(manifest.evaluators.get("code"), sort_keys=True)
        if manifest.evaluators
        else None
        for manifest in manifests
    ]
    if len(set(code_configs)) > 1:
        warnings.append("Code evaluation configuration differs between runs.")

    for model in common_models:
        differing = config_differences(model, runs)
        if differing:
            warnings.append(
                f"Model '{model}' uses a different configuration across runs "
                f"(differing fields: {', '.join(differing)}); "
                "displayed names match but the models may not be identical."
            )

    if not common_models:
        warnings.append("No model name is common to every run: only run conditions are compared.")

    return warnings


def version_tuple(version: str) -> tuple[int, ...]:
    parts = []
    for piece in version.split(".")[:3]:
        try:
            parts.append(int(piece))
        except ValueError:
            parts.append(0)
    return tuple(parts)


def paired_analyses(runs: list[dict]) -> list[dict]:
    if len(runs) != 2:
        return []
    first, second = runs
    if first["manifest"].dataset_sha256 != second["manifest"].dataset_sha256:
        return []
    statistics_config = StatisticsConfig.model_validate(
        first["manifest"].statistics or {}
    )
    analyses = []
    for model in sorted(set(first["stats_by_model"]) & set(second["stats_by_model"])):
        trials_a = [trial for trial in first["trials"] if trial.model == model]
        trials_b = [trial for trial in second["trials"] if trial.model == model]
        for metric, selector in PAIRED_METRICS:
            comparison = paired_comparison(
                trials_a, trials_b, statistics_config, first["manifest"].seed, metric, selector
            )
            if comparison["units"] == 0:
                continue
            analyses.append({"model": model, **comparison})
    return analyses


def models_only_in(runs: list[dict], labels: list[str]) -> dict[str, list[str]]:
    common = set.intersection(*(set(run["stats_by_model"]) for run in runs))
    return {
        label: sorted(set(run["stats_by_model"]) - common)
        for label, run in zip(labels, runs, strict=True)
        if set(run["stats_by_model"]) - common
    }
