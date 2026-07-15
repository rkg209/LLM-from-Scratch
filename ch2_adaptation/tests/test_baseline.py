"""score_system and build_baselines_doc, driven entirely by fakes -- no torch, no network,
so this runs in the base+dev CI environment (spec C1, AC-3/AC-4)."""

from __future__ import annotations

import json
from dataclasses import replace
from datetime import datetime
from pathlib import Path

import pytest
from ch2_adaptation.baseline import Usage, build_baselines_doc, score_system
from ch2_adaptation.config import load_baseline_config

from eval.harness import EvalResult, write_json_atomic

CONFIGS = Path("ch2_adaptation/configs")

STUB = [
    {"id": "a", "code": "...", "line": 10, "category": "npe", "severity": "major"},
    {"id": "b", "code": "...", "line": 20, "category": "npe", "severity": "major"},
    {"id": "c", "code": "...", "line": 30, "category": "npe", "severity": "major"},
    {"id": "d", "code": "...", "line": 40, "category": "npe", "severity": "major"},
]


def review(line: int) -> str:
    return json.dumps(
        {
            "severity": "major",
            "category": "npe-risk",
            "line": line,
            "issue": "Possible null dereference.",
            "suggested_fix": "Guard with Objects.requireNonNull.",
        }
    )


@pytest.fixture
def stub_path(tmp_path: Path) -> Path:
    path = tmp_path / "stub.jsonl"
    path.write_text("\n".join(json.dumps(record) for record in STUB))
    return path


def test_score_system_produces_exact_rates(stub_path: Path) -> None:
    # 2 caught (exact/near line), 1 valid-but-wrong-line, 1 unparseable.
    canned = [review(10), review(22), review(99), "not json"]

    def fake_generate(prompts: list[str]) -> list[str]:
        assert len(prompts) == len(STUB)
        return canned

    result = score_system(["prompt"] * len(STUB), fake_generate, stub_path)

    assert result.n_samples == 4
    assert result.n_valid == 3
    assert result.n_caught == 2
    assert result.schema_validity_rate == 0.75
    assert result.bug_catch_rate == 0.5


def test_score_system_passes_prompts_through_to_generate(stub_path: Path) -> None:
    seen: list[list[str]] = []

    def fake_generate(prompts: list[str]) -> list[str]:
        seen.append(prompts)
        return [review(record["line"]) for record in STUB]

    prompts = [f"review this: {i}" for i in range(len(STUB))]
    score_system(prompts, fake_generate, stub_path)

    assert seen == [prompts]


def test_build_baselines_doc_has_both_systems_and_a_parseable_timestamp() -> None:
    cfg = load_baseline_config(CONFIGS / "baseline_full.yaml")
    base = EvalResult(0.5, 0.25, 4, 2, 1, [])
    frontier = EvalResult(0.9, 0.8, 4, 4, 3, [])
    usage = Usage(input_tokens=120, output_tokens=40, cost_usd=0.0, tier="free")

    doc = build_baselines_doc(base, frontier, cfg, usage)

    assert "base_model" in doc
    assert "frontier_api" in doc
    assert doc["frontier_api"]["model_tag"] == cfg.frontier_model_tag
    assert doc["frontier_api"]["mode"] == "3-shot"
    assert doc["base_model"]["model_tag"] == cfg.base_model_tag
    assert doc["frontier_api"]["schema_validity_rate"] == 0.9
    assert doc["base_model"]["bug_catch_rate"] == 0.25
    datetime.fromisoformat(doc["timestamp"])


def test_build_baselines_doc_records_frontier_usage_honestly() -> None:
    cfg = load_baseline_config(CONFIGS / "baseline_full.yaml")
    result = EvalResult(1.0, 1.0, 1, 1, 1, [])
    usage = Usage(input_tokens=500, output_tokens=200, cost_usd=0.0, tier="free")

    doc = build_baselines_doc(result, result, cfg, usage)

    assert doc["frontier_api"]["tier"] == "free"
    assert doc["frontier_api"]["input_tokens"] == 500
    assert doc["frontier_api"]["output_tokens"] == 200
    assert doc["frontier_api"]["cost_usd"] == 0.0


def test_build_baselines_doc_records_the_schema_hash() -> None:
    """AC-8 (spec C5): `schema_sha256` is what a later run checks against, to confirm
    eval/schema.json hasn't changed since the baselines were measured."""
    import hashlib

    from eval.harness import DEFAULT_SCHEMA_PATH

    cfg = load_baseline_config(CONFIGS / "baseline_full.yaml")
    result = EvalResult(1.0, 1.0, 1, 1, 1, [])
    usage = Usage(0, 0, 0.0, "free")

    doc = build_baselines_doc(result, result, cfg, usage)

    expected = hashlib.sha256(Path(DEFAULT_SCHEMA_PATH).read_bytes()).hexdigest()
    assert doc["schema_sha256"] == expected
    assert len(doc["schema_sha256"]) == 64


def test_build_baselines_doc_hashes_change_with_stub_config(tmp_path: Path) -> None:
    cfg = load_baseline_config(CONFIGS / "baseline_smoke.yaml")
    other_stub = tmp_path / "other.jsonl"
    other_stub.write_text('{"id": "x", "code": "...", "line": 1}\n')
    cfg_other = replace(cfg, stub_set_path=str(other_stub))

    result = EvalResult(1.0, 1.0, 1, 1, 1, [])
    usage = Usage(0, 0, 0.0, "free")

    doc = build_baselines_doc(result, result, cfg, usage)
    doc_other = build_baselines_doc(result, result, cfg_other, usage)

    assert doc["stub_set_sha256"] != doc_other["stub_set_sha256"]


def test_write_json_atomic_leaves_no_tmp_and_round_trips(tmp_path: Path) -> None:
    path = tmp_path / "results" / "baselines.json"
    write_json_atomic({"a": 1}, path)

    assert not path.with_suffix(".json.tmp").exists()
    assert json.loads(path.read_text()) == {"a": 1}
