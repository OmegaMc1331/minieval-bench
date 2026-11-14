from __future__ import annotations

import sys

import pytest

from minieval_bench.models import ModelConfig
from minieval_bench.providers import (
    LitellmProvider,
    MockProvider,
    cost_from_pricing,
    is_transient_error,
    retry_delay_seconds,
)


def mock_model(**overrides) -> ModelConfig:
    return ModelConfig(name="mock-a", provider="mock", **overrides)


def test_mock_provider_is_deterministic_for_same_inputs():
    provider = MockProvider(mock_model())

    first = provider.complete("same prompt", "same system", 7)
    second = provider.complete("same prompt", "same system", 7)

    assert first.text == second.text
    assert first.input_tokens == second.input_tokens


def test_mock_provider_changes_with_seed():
    provider = MockProvider(mock_model())

    first = provider.complete("prompt", None, 1)
    second = provider.complete("prompt", None, 2)

    assert first.text != second.text


def test_mock_provider_response_shape():
    response = MockProvider(mock_model()).complete("one two three", "be brief", 1)

    assert response.text.startswith("mock-")
    assert len(response.text) == len("mock-") + 8
    assert response.input_tokens == 5
    assert response.output_tokens == 4
    assert response.cost_usd is None


def test_mock_provider_reports_cost_when_priced():
    model = mock_model(input_cost_per_1m=0.5, output_cost_per_1m=1.5)
    response = MockProvider(model).complete("one two three four five", None, 1)

    assert response.cost_usd == pytest.approx((5 * 0.5 + 4 * 1.5) / 1_000_000)


def test_cost_from_pricing_requires_both_prices():
    model = mock_model(input_cost_per_1m=1.0)

    assert cost_from_pricing(model, 100, 100) is None


def test_cost_from_pricing_requires_token_counts():
    model = mock_model(input_cost_per_1m=1.0, output_cost_per_1m=1.0)

    assert cost_from_pricing(model, None, 100) is None
    assert cost_from_pricing(model, 100, None) is None
    assert cost_from_pricing(model, 100, 100) == pytest.approx(200 / 1_000_000)


class FakeAPIError(Exception):
    def __init__(self, status_code: int) -> None:
        super().__init__(f"http {status_code}")
        self.status_code = status_code


def test_is_transient_error_accepts_rate_limit_and_server_errors():
    assert is_transient_error(FakeAPIError(429))
    assert is_transient_error(FakeAPIError(500))
    assert is_transient_error(FakeAPIError(503))


def test_is_transient_error_rejects_auth_and_client_errors():
    assert not is_transient_error(FakeAPIError(401))
    assert not is_transient_error(FakeAPIError(403))
    assert not is_transient_error(FakeAPIError(400))
    assert not is_transient_error(FakeAPIError(404))


def test_is_transient_error_detects_timeout_and_connection_errors():
    assert is_transient_error(TimeoutError("too slow"))
    assert is_transient_error(ConnectionError("refused"))


def test_retry_delay_uses_exponential_backoff():
    assert retry_delay_seconds(FakeAPIError(500), 0) == pytest.approx(0.5)
    assert retry_delay_seconds(FakeAPIError(500), 1) == pytest.approx(1.0)


def test_retry_delay_honors_bounded_retry_after():
    class RetryAfterError(Exception):
        retry_after = 5

    assert retry_delay_seconds(RetryAfterError(), 0) == pytest.approx(5.0)


def test_retry_delay_ignores_retry_after_above_limit():
    class LongRetryAfterError(Exception):
        retry_after = 120

    assert retry_delay_seconds(LongRetryAfterError(), 0) == pytest.approx(0.5)


def test_litellm_provider_retries_transient_errors(monkeypatch):
    calls = {"count": 0}

    class FakeResponse:
        choices = [type("Choice", (), {"message": type("Message", (), {"content": "done"})()})()]
        usage = None

    class FakeLitellm:
        @staticmethod
        def completion(model, messages, **kwargs):
            calls["count"] += 1
            if calls["count"] < 3:
                raise FakeAPIError(503)
            return FakeResponse()

        @staticmethod
        def completion_cost(response):
            raise ValueError("no pricing")

    monkeypatch.setitem(sys.modules, "litellm", FakeLitellm)
    provider = LitellmProvider(ModelConfig(name="svc", model="openai/x"))
    monkeypatch.setattr("time.sleep", lambda seconds: None)

    response = provider.complete("prompt", None, 1)

    assert response.text == "done"
    assert calls["count"] == 3


def test_litellm_provider_does_not_retry_auth_errors(monkeypatch):
    calls = {"count": 0}

    class FakeLitellm:
        @staticmethod
        def completion(model, messages, **kwargs):
            calls["count"] += 1
            raise FakeAPIError(401)

    monkeypatch.setitem(sys.modules, "litellm", FakeLitellm)
    provider = LitellmProvider(ModelConfig(name="svc", model="openai/x"))

    with pytest.raises(Exception, match="http 401"):
        provider.complete("prompt", None, 1)

    assert calls["count"] == 1
