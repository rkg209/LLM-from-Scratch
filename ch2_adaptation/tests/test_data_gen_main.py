"""main()'s helper functions -- budget guard, holdout-hash loading, batch building, atomic
JSONL writes. Driven by fixtures/monkeypatch, no network, no torch (spec C3, task 4)."""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import pytest
from ch2_adaptation.config import load_data_gen_config
from ch2_adaptation.data_gen import (
    _build_snippet_batch,
    _check_budget,
    _load_holdout_hashes,
    _load_seed_pool,
    _write_jsonl_atomic,
)

CONFIGS = Path("ch2_adaptation/configs")


def test_load_holdout_hashes_raises_when_missing_and_not_allowed(tmp_path: Path) -> None:
    missing = tmp_path / "no_such_file.txt"
    with pytest.raises(FileNotFoundError, match="frozen"):
        _load_holdout_hashes(allow_missing=False, path=missing)


def test_load_holdout_hashes_returns_empty_set_when_missing_and_allowed(tmp_path: Path) -> None:
    missing = tmp_path / "no_such_file.txt"
    assert _load_holdout_hashes(allow_missing=True, path=missing) == set()


def test_load_holdout_hashes_reads_existing_file(tmp_path: Path) -> None:
    path = tmp_path / "hashes.txt"
    path.write_text("aaa\nbbb\n\n")
    assert _load_holdout_hashes(allow_missing=False, path=path) == {"aaa", "bbb"}


def test_build_snippet_batch_respects_n_snippets_and_variants(tmp_path: Path) -> None:
    pool = [{"code": f"code{i}"} for i in range(10)]
    cfg = load_data_gen_config(CONFIGS / "datagen_smoke.yaml")
    cfg = replace(cfg, n_snippets=3, variants_per_snippet=2)

    batch = _build_snippet_batch(pool, cfg)

    assert len(batch) == 6
    assert batch == ["code0", "code0", "code1", "code1", "code2", "code2"]


def test_check_budget_skips_for_smoke() -> None:
    cfg = load_data_gen_config(CONFIGS / "datagen_smoke.yaml")
    assert cfg.is_smoke is True
    assert _check_budget(["snippet"], cfg) == 0.0


def test_check_budget_raises_when_estimate_exceeds_cap(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("PRICE_PER_1K_INPUT_USD", "1000")
    monkeypatch.setenv("PRICE_PER_1K_OUTPUT_USD", "1000")
    cfg = load_data_gen_config(CONFIGS / "datagen_full.yaml")
    cfg = replace(cfg, max_budget_usd=0.0001)

    with pytest.raises(RuntimeError, match="max_budget_usd"):
        _check_budget(["a" * 1000], cfg)


def test_load_seed_pool_reads_jsonl(tmp_path: Path) -> None:
    path = tmp_path / "pool.jsonl"
    path.write_text('{"code": "a"}\n{"code": "b"}\n')
    pool = _load_seed_pool(path)
    assert pool == [{"code": "a"}, {"code": "b"}]


def test_write_jsonl_atomic_leaves_no_tmp_and_round_trips(tmp_path: Path) -> None:
    path = tmp_path / "out" / "records.jsonl"
    records = [{"code": "a"}, {"code": "b"}]

    _write_jsonl_atomic(records, path)

    assert not path.with_suffix(".jsonl.tmp").exists()
    lines = path.read_text().splitlines()
    assert [json.loads(line) for line in lines] == records
