from __future__ import annotations

import pytest

from minieval_bench.models import TrialResult
from minieval_bench.runner import available_total, compute_stats, percentile


def make_trial(
    model: str = "m",
    ok: bool = True,
    latency_ms: float = 10.0,
    passed: bool | None = True,
    input_tokens: int | None = 10,
    output_tokens: int | None = 5,
    cost_usd: float | None = None,
    error_type: str | None = None,
) -> TrialResult:
    return TrialResult(
        model=model,
        entry_id="e1",
        repeat=0,
        ok=ok,
        response="resp" if ok else None,
        latency_ms=latency_ms,
        input_tokens=input_tokens if ok else None,
        output_tokens=output_tokens if ok else None,
        cost_usd=cost_usd if ok else None,
        error_type=error_type,
        error_message="boom" if error_type else None,
        passed=passed if ok else None,
    )


def test_compute_stats_reports_counts_rates_and_all_indicators():
    trials = [
        make_trial(passed=True, latency_ms=100.0),
        make_trial(passed=True, latency_ms=200.0),
        make_trial(passed=False, latency_ms=300.0),
        make_trial(ok=False, error_type="TimeoutError", latency_ms=5000.0),
    ]

    stats = compute_stats(trials)[0]

    assert stats.model == "m"
    assert stats.requests == 4
    assert stats.succeeded == 3
    assert stats.failed == 1
    assert stats.success_rate == pytest.approx(0.75)
    assert stats.evaluable == 3
    assert stats.passed == 2
    assert stats.incorrect == 1
    assert stats.score == pytest.approx(2 / 3)
    assert stats.answer_accuracy == pytest.approx(2 / 3)
    assert stats.evaluation_coverage == pytest.approx(0.75)
    assert stats.overall_pass_rate == pytest.approx(0.5)
    assert stats.errors == {"TimeoutError": 1}


def test_compute_stats_marks_unavailable_values_when_no_request_succeeds():
    stats = compute_stats([make_trial(ok=False, error_type="APIError", latency_ms=1.0)])[0]

    assert stats.score is None
    assert stats.answer_accuracy is None
    assert stats.median_latency_ms is None
    assert stats.p95_latency_ms is None
    assert stats.input_tokens is None
    assert stats.input_tokens_reported == 0
    assert stats.output_tokens is None
    assert stats.cost_usd is None
    assert stats.cost_reported == 0
    assert stats.success_rate == 0.0
    assert stats.evaluation_coverage == 0.0
    assert stats.overall_pass_rate == 0.0
    assert stats.failed == 1


def test_compute_stats_reports_full_token_coverage():
    trials = [make_trial(input_tokens=3, output_tokens=2) for _ in range(4)]

    stats = compute_stats(trials)[0]

    assert stats.input_tokens == 12
    assert stats.input_tokens_reported == 4
    assert stats.output_tokens == 8
    assert stats.output_tokens_reported == 4


def test_compute_stats_identifies_partial_totals():
    trials = [make_trial(input_tokens=1) for _ in range(10)]
    trials[7].input_tokens = None
    trials[8].input_tokens = None
    trials[9].input_tokens = None

    stats = compute_stats(trials)[0]

    assert stats.input_tokens == 7
    assert stats.input_tokens_reported == 7
    assert stats.succeeded == 10


def test_compute_stats_distinguishes_zero_consumption_from_unavailable():
    trials = [make_trial(input_tokens=0, output_tokens=0) for _ in range(3)]

    stats = compute_stats(trials)[0]

    assert stats.input_tokens == 0
    assert stats.input_tokens_reported == 3
    assert stats.output_tokens == 0
    assert stats.output_tokens_reported == 3

    missing = compute_stats([make_trial(input_tokens=None, output_tokens=None)])[0]
    assert missing.input_tokens is None
    assert missing.input_tokens_reported == 0


def test_compute_stats_tokens_without_known_pricing():
    trials = [make_trial(input_tokens=10, output_tokens=5) for _ in range(3)]

    stats = compute_stats(trials)[0]

    assert stats.input_tokens == 30
    assert stats.cost_usd is None
    assert stats.cost_reported == 0


def test_compute_stats_sums_available_costs_with_coverage():
    trials = [make_trial(cost_usd=0.001), make_trial(cost_usd=0.002), make_trial(cost_usd=None)]

    stats = compute_stats(trials)[0]

    assert stats.cost_usd == pytest.approx(0.003)
    assert stats.cost_reported == 2
    assert stats.succeeded == 3


def test_compute_stats_overall_pass_rate_counts_failures_as_not_passed():
    trials = [
        make_trial(passed=True),
        make_trial(passed=False),
        make_trial(ok=False, error_type="RateLimitError"),
    ]

    stats = compute_stats(trials)[0]

    assert stats.overall_pass_rate == pytest.approx(1 / 3)
    assert stats.answer_accuracy == pytest.approx(1 / 2)
    assert stats.success_rate == pytest.approx(2 / 3)


def test_compute_stats_latency_aggregates_over_successful_requests_only():
    trials = [
        make_trial(latency_ms=10.0),
        make_trial(latency_ms=20.0),
        make_trial(latency_ms=30.0),
        make_trial(ok=False, latency_ms=9999.0, error_type="Boom"),
    ]

    stats = compute_stats(trials)[0]

    assert stats.median_latency_ms == pytest.approx(20.0)
    assert stats.p95_latency_ms == pytest.approx(30.0)


def test_compute_stats_groups_and_sorts_by_model():
    trials = [make_trial(model="b"), make_trial(model="a")]

    stats = compute_stats(trials)

    assert [item.model for item in stats] == ["a", "b"]


def test_percentile_uses_nearest_rank():
    assert percentile([1.0], 95) == 1.0
    assert percentile([10, 20, 30, 40], 95) == 40
    assert percentile([10, 20, 30, 40], 50) == 20
    assert percentile(list(range(1, 101)), 95) == 95


def test_available_total_returns_sum_and_coverage():
    assert available_total([1, 2, None, 3]) == (6, 3)
    assert available_total([None, None]) == (None, 0)
    assert available_total([0, 0]) == (0, 2)
