from __future__ import annotations

import pytest

from minieval_bench.models import ModelConfig
from minieval_bench.providers import MockProvider, cost_from_pricing


def mock_model(**overrides) -> ModelConfig:
    return ModelConfig(name="mock-a", provider="mock", **overrides)


def test_mock_provider_is_deterministic_for_same_inputs():
    provider = MockProvider(mock_model(), seed=7)

    first = provider.complete("same prompt", "same system")
    second = provider.complete("same prompt", "same system")

    assert first.text == second.text
    assert first.input_tokens == second.input_tokens


def test_mock_provider_changes_with_seed():
    first = MockProvider(mock_model(), seed=1).complete("prompt", None)
    second = MockProvider(mock_model(), seed=2).complete("prompt", None)

    assert first.text != second.text


def test_mock_provider_response_shape():
    response = MockProvider(mock_model(), seed=1).complete("one two three", "be brief")

    assert response.text.startswith("mock-")
    assert len(response.text) == len("mock-") + 8
    assert response.input_tokens == 5
    assert response.output_tokens == 4
    assert response.cost_usd is None


def test_mock_provider_reports_cost_when_priced():
    model = mock_model(input_cost_per_1m=0.5, output_cost_per_1m=1.5)
    response = MockProvider(model, seed=1).complete("one two three four five", None)

    assert response.cost_usd == pytest.approx((5 * 0.5 + 4 * 1.5) / 1_000_000)


def test_cost_from_pricing_requires_both_prices():
    model = mock_model(input_cost_per_1m=1.0)

    assert cost_from_pricing(model, 100, 100) is None


def test_cost_from_pricing_requires_token_counts():
    model = mock_model(input_cost_per_1m=1.0, output_cost_per_1m=1.0)

    assert cost_from_pricing(model, None, 100) is None
    assert cost_from_pricing(model, 100, None) is None
    assert cost_from_pricing(model, 100, 100) == pytest.approx(200 / 1_000_000)
