from __future__ import annotations

import pytest

from minieval_bench.evaluation.statistics import (
    analyze_model_statistics,
    bootstrap_interval,
    numeric_summary,
    paired_comparison,
    paired_question_pass_rates,
    question_pass_rates,
    summarize_evaluations,
)
from minieval_bench.models import EvaluationOutcome, StatisticsConfig
from test_stats import make_trial

STATS_CONFIG = StatisticsConfig(bootstrap_iterations=200, confidence_level=0.95)


def judged_trial(model: str = "m", entry_id: str = "e1", score: float = 0.8, repeat: int = 0):
    trial = make_trial(model=model, passed=None)
    trial.entry_id = entry_id
    trial.repeat = repeat
    trial.evaluation = EvaluationOutcome(
        evaluator="judge", status="completed", passed=None, score=score
    )
    return trial


def schema_trial(model: str = "m", entry_id: str = "e1", passed: bool = True):
    trial = make_trial(model=model, passed=passed)
    trial.entry_id = entry_id
    trial.evaluation = EvaluationOutcome(
        evaluator="json_schema", status="completed", passed=passed,
        score=1.0 if passed else 0.0,
    )
    return trial


def test_numeric_summary_handles_empty_and_single_values():
    empty = numeric_summary([])
    assert empty["count"] == 0 and empty["mean"] is None

    single = numeric_summary([4.0])
    assert single["mean"] == 4.0
    assert single["std"] is None


def test_numeric_summary_computes_dispersion():
    summary = numeric_summary([1.0, 2.0, 3.0, 4.0])

    assert summary["mean"] == pytest.approx(2.5)
    assert summary["median"] == pytest.approx(2.5)
    assert summary["std"] == pytest.approx(1.290994449)
    assert summary["min"] == 1.0
    assert summary["max"] == 4.0
    assert summary["count"] == 4


def test_bootstrap_interval_requires_two_units():
    interval = bootstrap_interval([0.5], 100, 0.95, seed=1)

    assert interval["low"] is None and interval["high"] is None


def test_bootstrap_interval_is_reproducible():
    values = [0.0, 0.5, 1.0, 0.25, 0.75, 0.5]

    first = bootstrap_interval(values, 200, 0.95, seed=42)
    second = bootstrap_interval(values, 200, 0.95, seed=42)
    other_seed = bootstrap_interval(values, 200, 0.95, seed=43)

    assert first == second
    assert first != other_seed
    assert first["low"] <= first["high"]


def test_question_pass_rates_average_repeats_per_question():
    trials = [
        make_trial(model="m", passed=True),
        make_trial(model="m", passed=True),
    ]
    trials[0].entry_id = "e1"
    trials[1].entry_id = "e1"
    extra = make_trial(model="m", passed=False)
    extra.entry_id = "e2"
    trials.append(extra)

    rates = question_pass_rates(trials)

    assert sorted(rates) == pytest.approx([0.0, 1.0])


def test_question_pass_rates_skip_failed_and_non_binary_trials():
    failed = make_trial(model="m", ok=False, error_type="Boom")
    judged = judged_trial()

    rates = question_pass_rates([failed, judged])

    assert rates == []


def test_analyze_model_statistics_reports_units_and_intervals():
    trials = []
    for index in range(4):
        trial = make_trial(model="m", passed=index % 2 == 0)
        trial.entry_id = f"e{index}"
        trials.append(trial)

    analysis = analyze_model_statistics(trials, STATS_CONFIG, seed=7)

    assert len(analysis) == 1
    entry = analysis[0]
    assert entry["model"] == "m"
    assert entry["pass_rate"]["summary"]["count"] == 4
    assert entry["pass_rate"]["ci95"]["low"] is not None
    assert entry["latency_ms"]["summary"]["count"] == 4
    assert entry["judge_score"]["summary"]["count"] == 0
    assert entry["judge_score"]["ci95"]["low"] is None
    assert entry["resampling"]["unit"] == "question (repeats averaged first)"


def test_paired_question_pass_rates_uses_common_questions():
    trials_a = []
    trials_b = []
    for index in range(3):
        trial_a = make_trial(model="a", passed=False)
        trial_a.entry_id = f"e{index}"
        trials_a.append(trial_a)
        trial_b = make_trial(model="b", passed=True)
        trial_b.entry_id = f"e{index}"
        trials_b.append(trial_b)
    only_a = make_trial(model="a", passed=True)
    only_a.entry_id = "e9"
    trials_a.append(only_a)

    differences = paired_question_pass_rates(trials_a, trials_b)

    assert differences == pytest.approx([1.0, 1.0, 1.0])


def test_paired_comparison_includes_bootstrap():
    trials_a = []
    trials_b = []
    for index in range(5):
        trial_a = make_trial(model="a", passed=index < 2)
        trial_a.entry_id = f"e{index}"
        trials_a.append(trial_a)
        trial_b = make_trial(model="b", passed=index < 4)
        trial_b.entry_id = f"e{index}"
        trials_b.append(trial_b)

    comparison = paired_comparison(trials_a, trials_b, STATS_CONFIG, seed=3)

    assert comparison["units"] == 5
    assert comparison["differences_summary"]["mean"] == pytest.approx(0.4)
    assert comparison["ci95"]["low"] is not None


def test_summarize_evaluations_separates_evaluator_types():
    trials = [
        make_trial(model="m", passed=True),
        schema_trial(model="m", entry_id="e2", passed=False),
        judged_trial(model="m", entry_id="e3", score=0.9),
    ]

    summaries = {item["evaluator"]: item for item in summarize_evaluations(trials)}

    assert summaries["deterministic"]["trials"] == 1
    assert summaries["deterministic"]["passed"] == 1
    assert summaries["json_schema"]["failed_binary"] == 1
    assert summaries["json_schema"]["mean_score"] == 0.0
    assert summaries["judge"]["passed"] == 0
    assert summaries["judge"]["binary_pass_rate"] is None
    assert summaries["judge"]["mean_score"] == pytest.approx(0.9)


def test_summarize_evaluations_counts_evaluator_failures_and_costs():
    failed = make_trial(model="m", passed=None)
    failed.evaluation = EvaluationOutcome(
        evaluator="judge", status="failed", error_message="bad verdict"
    )
    unavailable = make_trial(model="m", passed=None)
    unavailable.evaluation = EvaluationOutcome(
        evaluator="code_tests", status="unavailable", error_message="no docker"
    )
    priced = judged_trial(entry_id="e2", score=0.5)
    priced.evaluation.cost_usd = 0.001

    summaries = {
        item["evaluator"]: item
        for item in summarize_evaluations([failed, unavailable, priced])
    }

    assert summaries["judge"]["evaluator_failed"] == 1
    assert summaries["judge"]["cost_usd"] == pytest.approx(0.001)
    assert summaries["code_tests"]["evaluator_unavailable"] == 1
