from __future__ import annotations

import json
from importlib.resources import files
from pathlib import Path

from minieval_bench.evaluation.statistics import (
    analyze_model_statistics,
    summarize_evaluations,
)
from minieval_bench.models import (
    DatasetEntry,
    Manifest,
    ModelStats,
    StatisticsConfig,
    TrialResult,
)

TEMPLATE_MARKERS = {
    "styles": "/*__STYLES__*/",
    "data": "__DATA__",
    "script": "/*__SCRIPT__*/",
}


def read_template(name: str) -> str:
    return files("minieval_bench").joinpath("templates", name).read_text(encoding="utf-8")


def render_standalone(payload: dict, template_name: str, script_name: str) -> str:
    html = read_template(template_name)
    data = json.dumps(payload, ensure_ascii=False, separators=(",", ":")).replace("</", "<\\/")
    html = html.replace(TEMPLATE_MARKERS["styles"], read_template("styles.css"))
    html = html.replace(TEMPLATE_MARKERS["data"], data)
    return html.replace(TEMPLATE_MARKERS["script"], read_template(script_name))


def build_analysis(
    manifest: Manifest, trials: list[TrialResult]
) -> tuple[list[dict], list[dict]]:
    statistics_config = StatisticsConfig.model_validate(manifest.statistics or {})
    statistics = analyze_model_statistics(trials, statistics_config, manifest.seed)
    evaluator_summaries = summarize_evaluations(trials)
    return statistics, evaluator_summaries


def write_html_report(
    output_path: Path,
    manifest: Manifest,
    stats: list[ModelStats],
    trials: list[TrialResult],
    dataset: list[DatasetEntry] | None = None,
) -> None:
    statistics, evaluator_summaries = build_analysis(manifest, trials)
    payload = build_run_payload(
        manifest, stats, trials, dataset, statistics, evaluator_summaries
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    standalone = render_standalone(payload, "report.html", "dashboard.js")
    output_path.write_text(standalone, encoding="utf-8")


def build_run_payload(
    manifest: Manifest,
    stats: list[ModelStats],
    trials: list[TrialResult],
    dataset: list[DatasetEntry] | None,
    statistics: list[dict] | None = None,
    evaluator_summaries: list[dict] | None = None,
) -> dict:
    models = sorted({trial.model for trial in trials} | {item.model for item in stats})
    entry_ids = sorted({trial.entry_id for trial in trials})
    error_types = sorted({trial.error_type for trial in trials if trial.error_type})
    model_index = {name: index for index, name in enumerate(models)}
    entry_index = {entry_id: index for index, entry_id in enumerate(entry_ids)}
    error_index = {name: index for index, name in enumerate(error_types)}

    compact_trials = [
        [
            model_index[trial.model],
            entry_index[trial.entry_id],
            trial.repeat,
            trial.ok,
            trial.passed,
            round(trial.latency_ms, 3),
            None if trial.queue_ms is None else round(trial.queue_ms, 3),
            trial.input_tokens,
            trial.output_tokens,
            None if trial.cost_usd is None else round(trial.cost_usd, 10),
            None if trial.error_type is None else error_index[trial.error_type],
        ]
        for trial in trials
    ]

    return {
        "kind": "run",
        "manifest": manifest.model_dump(mode="json"),
        "models": models,
        "entryIds": entry_ids,
        "errorTypes": error_types,
        "stats": [item.model_dump(mode="json") for item in stats],
        "trials": compact_trials,
        "responses": [trial.response for trial in trials],
        "errorMessages": [trial.error_message for trial in trials],
        "evaluations": [
            None if trial.evaluation is None else trial.evaluation.model_dump(mode="json")
            for trial in trials
        ],
        "dataset": (
            None if dataset is None else [entry.model_dump(mode="json") for entry in dataset]
        ),
        "statistics": statistics or None,
        "evaluatorSummaries": evaluator_summaries or None,
    }
