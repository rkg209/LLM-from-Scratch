"""frontier.py's pure logic (cost estimation, provider dispatch, the stub client) is
fully testable without network access or the google-genai package."""

from __future__ import annotations

import pytest
from ch2_adaptation.baseline import Usage
from ch2_adaptation.frontier import (
    GeminiClient,
    StubFrontierClient,
    estimate_cost_usd,
    make_client,
)


def test_estimate_cost_usd_raises_if_prices_unset(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("PRICE_PER_1K_INPUT_USD", raising=False)
    monkeypatch.delenv("PRICE_PER_1K_OUTPUT_USD", raising=False)
    with pytest.raises(RuntimeError, match="PRICE_PER_1K"):
        estimate_cost_usd(100, 50)


def test_estimate_cost_usd_never_hardcodes_a_price(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("PRICE_PER_1K_INPUT_USD", "0.001")
    monkeypatch.setenv("PRICE_PER_1K_OUTPUT_USD", "0.002")

    cost = estimate_cost_usd(input_tokens=2000, output_tokens=1000)

    assert cost == pytest.approx(2000 / 1000 * 0.001 + 1000 / 1000 * 0.002)


def test_stub_frontier_client_makes_no_network_call() -> None:
    client = StubFrontierClient()
    outputs, usage = client.generate(["prompt one", "prompt two"])

    assert len(outputs) == 2
    assert usage == Usage(input_tokens=0, output_tokens=0, cost_usd=0.0, tier="free")


def test_stub_frontier_client_output_is_schema_valid_json() -> None:
    import json

    client = StubFrontierClient()
    outputs, _ = client.generate(["prompt"])
    parsed = json.loads(outputs[0])

    assert {"severity", "category", "line", "issue", "suggested_fix"} <= parsed.keys()


def test_make_client_dispatches_stub() -> None:
    client = make_client("stub", "stub-frontier", 0.0, 32)
    assert isinstance(client, StubFrontierClient)


def test_make_client_dispatches_gemini() -> None:
    client = make_client("gemini", "gemini-2.5-flash", 0.0, 256)
    assert isinstance(client, GeminiClient)
    assert client.model_tag == "gemini-2.5-flash"


def test_make_client_rejects_unknown_provider() -> None:
    with pytest.raises(ValueError, match="unknown frontier_provider"):
        make_client("openai", "gpt-4", 0.0, 256)


def test_gemini_client_generate_requires_an_api_key(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    client = GeminiClient("gemini-2.5-flash", 0.0, 256)
    with pytest.raises(RuntimeError, match="GEMINI_API_KEY"):
        client.generate(["prompt"])
