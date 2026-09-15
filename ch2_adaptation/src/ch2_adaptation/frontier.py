"""The frontier API client (AC-7): Gemini 2.5 Flash, free tier, 3-shot.

Heavy imports (`google.genai`) stay function-local inside `GeminiClient.generate` so this
module imports cleanly in the base+dev CI environment, which never installs it.
"""

from __future__ import annotations

import os
from typing import TYPE_CHECKING, Any, Protocol

from ch2_adaptation.baseline import Usage

if TYPE_CHECKING:
    from pydantic import BaseModel


def _to_gemini_response_schema(model: type[BaseModel]) -> dict[str, Any]:
    """Convert a Pydantic model to a schema Gemini's `response_schema` will accept.

    Gemini's structured-output schema is a restricted subset of OpenAPI and does not
    support `additionalProperties` -- the API rejects the request outright with
    `Unknown name "additional_properties"` if it is present. Pydantic's own
    `model_json_schema()` always emits it (as `false`) for a model with
    `extra="forbid"`, which `ReviewOutput` needs for its own local validation. Strip the
    unsupported key here rather than relaxing that validation.
    """
    schema = model.model_json_schema()
    schema.pop("additionalProperties", None)
    return schema


class FrontierClient(Protocol):
    """A frontier API that turns prompts into raw review responses and reports usage."""

    def generate(self, prompts: list[str]) -> tuple[list[str], Usage]: ...


def estimate_cost_usd(input_tokens: int, output_tokens: int) -> float:
    """Cost of one scoring run, from prices read out of the environment.

    Never hardcoded (mirrors the rule C3's budget guard is specified with,
    `planning/03-system-design.md:400-405`): a price baked into source code goes stale
    the moment the provider changes it, silently.
    """
    input_price = os.environ.get("PRICE_PER_1K_INPUT_USD")
    output_price = os.environ.get("PRICE_PER_1K_OUTPUT_USD")
    if input_price is None or output_price is None:
        raise RuntimeError(
            "PRICE_PER_1K_INPUT_USD and PRICE_PER_1K_OUTPUT_USD must both be set "
            "(see .env.example) — cost is never hardcoded in source."
        )
    return (input_tokens / 1000) * float(input_price) + (output_tokens / 1000) * float(output_price)


class GeminiClient:
    """Gemini 2.5 Flash, free tier. Native JSON mode sidesteps the markdown-fence problem
    that the base model is scored strictly against (see prompts.py)."""

    def __init__(self, model_tag: str, temperature: float, max_new_tokens: int) -> None:
        self.model_tag = model_tag
        self.temperature = temperature
        self.max_new_tokens = max_new_tokens

    def generate(self, prompts: list[str]) -> tuple[list[str], Usage]:
        api_key = os.environ.get("GEMINI_API_KEY")
        if not api_key:
            raise RuntimeError("GEMINI_API_KEY must be set to call the Gemini frontier API.")
        # Fail loud on a missing price *before* making any paid-tier-capable call — waiting
        # until token counts come back would let a misconfigured environment silently record
        # a bogus $0.00 if usage_metadata ever comes back None instead of raising.
        estimate_cost_usd(0, 0)

        from google import genai
        from google.genai import types

        from ch2_adaptation.schema import ReviewOutput

        client = genai.Client(api_key=api_key)
        response_schema = _to_gemini_response_schema(ReviewOutput)
        outputs: list[str] = []
        input_tokens = 0
        output_tokens = 0

        for prompt in prompts:
            response = client.models.generate_content(
                model=self.model_tag,
                contents=prompt,
                config=types.GenerateContentConfig(
                    temperature=self.temperature,
                    max_output_tokens=self.max_new_tokens,
                    response_mime_type="application/json",
                    response_schema=response_schema,
                ),
            )
            outputs.append(response.text or "")
            usage = response.usage_metadata
            if usage is not None:
                input_tokens += usage.prompt_token_count or 0
                output_tokens += usage.candidates_token_count or 0

        cost_usd = estimate_cost_usd(input_tokens, output_tokens)
        usage = Usage(
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            cost_usd=cost_usd,
            tier="free",
        )
        return outputs, usage


class StubFrontierClient:
    """Returns canned valid JSON. Makes no network call, needs no API key — what
    baseline_smoke.yaml selects."""

    def generate(self, prompts: list[str]) -> tuple[list[str], Usage]:
        canned = (
            '{"severity": "major", "category": "stub", "line": 1, '
            '"issue": "stub issue", "suggested_fix": "stub fix"}'
        )
        outputs = [canned for _ in prompts]
        usage = Usage(input_tokens=0, output_tokens=0, cost_usd=0.0, tier="free")
        return outputs, usage


def make_client(
    frontier_provider: str, model_tag: str, temperature: float, max_new_tokens: int
) -> FrontierClient:
    if frontier_provider == "stub":
        return StubFrontierClient()
    if frontier_provider == "gemini":
        return GeminiClient(model_tag, temperature, max_new_tokens)
    raise ValueError(f"unknown frontier_provider: {frontier_provider!r}")
