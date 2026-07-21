"""Ch3's zero-shot prompt: binds to the live schema, carries context, no ch2 import."""

from __future__ import annotations

import json

from ch3_operation.prompts import format_zero_shot

from eval.harness import load_schema


def test_system_prompt_embeds_the_live_schema() -> None:
    """Schema edits must not silently desync the served prompt."""
    schema_json = json.dumps(load_schema(), indent=2)
    prompt = format_zero_shot("public void f() {}")

    assert schema_json in prompt


def test_prompt_carries_the_code() -> None:
    code = "public int add(int a, int b) { return a + b; }"

    prompt = format_zero_shot(code)

    assert code in prompt


def test_prompt_carries_optional_context() -> None:
    prompt = format_zero_shot("public void f() {}", context="Spring Boot controller")

    assert "Spring Boot controller" in prompt


def test_empty_context_is_omitted() -> None:
    prompt = format_zero_shot("public void f() {}", context="")

    assert "Context:" not in prompt


def test_context_defaults_to_omitted() -> None:
    prompt = format_zero_shot("public void f() {}")

    assert "Context:" not in prompt


def test_no_markdown_fence_instruction_drift() -> None:
    """Copied verbatim from ch2 — a different prompt per system is not a comparison."""
    prompt = format_zero_shot("public void f() {}")

    assert "Do not wrap it in a markdown code fence" in prompt
