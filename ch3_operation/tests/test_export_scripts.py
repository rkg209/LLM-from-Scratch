"""O1: the export pipeline's config guards + the three CLI scripts, with subprocess and
llama.cpp filesystem layout faked out -- no real llama.cpp checkout needed in CI.

`scripts/` is not an importable package (see `scripts/README.md`), so each module is loaded
by file path rather than `import scripts.merge_adapter`.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from types import ModuleType

import pytest
from ch3_operation.config import ExportConfig, load_export_config

CONFIGS = Path("ch3_operation/configs")
SCRIPTS = Path("scripts")


def _load_script(name: str) -> ModuleType:
    spec = importlib.util.spec_from_file_location(f"_scripts_{name}", SCRIPTS / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


merge_adapter = _load_script("merge_adapter")
export_gguf = _load_script("export_gguf")
quantize_gguf = _load_script("quantize_gguf")


# --- ExportConfig guards -----------------------------------------------------------------


def test_smoke_export_config_loads() -> None:
    config = load_export_config(CONFIGS / "export_smoke.yaml")
    assert config.is_smoke
    assert config.base_model_tag == "trl-internal-testing/tiny-Qwen2ForCausalLM-2.5"


def test_full_export_config_loads() -> None:
    config = load_export_config(CONFIGS / "export_full.yaml")
    assert not config.is_smoke
    assert config.base_model_tag == "Qwen/Qwen2.5-Coder-1.5B-Instruct"


def test_unknown_quant_type_is_rejected() -> None:
    config = load_export_config(CONFIGS / "export_smoke.yaml")
    with pytest.raises(ValueError, match="quant_type"):
        ExportConfig(**{**vars(config), "quant_type": "Q2_K"})


def test_unrecognized_base_model_tag_is_rejected() -> None:
    config = load_export_config(CONFIGS / "export_smoke.yaml")
    with pytest.raises(ValueError, match="base_model_tag"):
        ExportConfig(**{**vars(config), "base_model_tag": "some/other-model"})


def test_smoke_config_may_not_target_a_published_path() -> None:
    config = load_export_config(CONFIGS / "export_smoke.yaml")
    with pytest.raises(ValueError, match="published path"):
        ExportConfig(**{**vars(config), "merged_dir": "outputs/merged"})


# --- merge_adapter.py ---------------------------------------------------------------------


def test_check_adapter_base_tag_accepts_a_matching_adapter(tmp_path: Path) -> None:
    adapter_dir = tmp_path / "adapter"
    adapter_dir.mkdir()
    (adapter_dir / "adapter_config.json").write_text(
        json.dumps({"base_model_name_or_path": "some/base"})
    )

    merge_adapter.check_adapter_base_tag(adapter_dir, "some/base")


def test_check_adapter_base_tag_rejects_a_mismatched_adapter(tmp_path: Path) -> None:
    adapter_dir = tmp_path / "adapter"
    adapter_dir.mkdir()
    (adapter_dir / "adapter_config.json").write_text(
        json.dumps({"base_model_name_or_path": "wrong/base"})
    )

    with pytest.raises(merge_adapter.BaseModelMismatchError):
        merge_adapter.check_adapter_base_tag(adapter_dir, "some/base")


def test_check_adapter_base_tag_requires_the_config_file(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        merge_adapter.check_adapter_base_tag(tmp_path / "missing", "some/base")


def test_load_sanity_prompts_reads_n_records_as_zero_shot_prompts() -> None:
    prompts = merge_adapter.load_sanity_prompts(
        "ch3_operation/tests/fixtures/sanity_sample.jsonl", n=2
    )

    assert len(prompts) == 2
    assert "public String f(User u)" in prompts[0]


def test_load_sanity_prompts_requires_the_file_to_exist(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        merge_adapter.load_sanity_prompts(tmp_path / "missing.jsonl", n=3)


# --- export_gguf.py -------------------------------------------------------------------


def test_convert_to_f16_gguf_requires_the_converter_script(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="convert_hf_to_gguf"):
        export_gguf.convert_to_f16_gguf(
            tmp_path / "merged", tmp_path / "no-llama-cpp", tmp_path / "out.gguf"
        )


def test_convert_to_f16_gguf_rejects_a_truncated_output(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    llama_cpp_dir = tmp_path / "llama.cpp"
    llama_cpp_dir.mkdir()
    (llama_cpp_dir / "convert_hf_to_gguf.py").write_text("# stub")
    out_path = tmp_path / "out.gguf"

    def fake_run(cmd: list[str], check: bool) -> None:
        out_path.write_bytes(b"too small")  # under _MIN_PLAUSIBLE_GGUF_BYTES

    monkeypatch.setattr(export_gguf.subprocess, "run", fake_run)

    with pytest.raises(RuntimeError, match="implausibly small"):
        export_gguf.convert_to_f16_gguf(tmp_path / "merged", llama_cpp_dir, out_path)


def test_convert_to_f16_gguf_succeeds_when_the_output_looks_real(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    llama_cpp_dir = tmp_path / "llama.cpp"
    llama_cpp_dir.mkdir()
    (llama_cpp_dir / "convert_hf_to_gguf.py").write_text("# stub")
    out_path = tmp_path / "out.gguf"

    def fake_run(cmd: list[str], check: bool) -> None:
        out_path.write_bytes(b"0" * 2048)

    monkeypatch.setattr(export_gguf.subprocess, "run", fake_run)

    result = export_gguf.convert_to_f16_gguf(tmp_path / "merged", llama_cpp_dir, out_path)

    assert result == out_path


# --- quantize_gguf.py -----------------------------------------------------------------


def test_find_quantize_binary_requires_a_built_llama_cpp(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="llama-quantize"):
        quantize_gguf._find_quantize_binary(tmp_path)


def test_quantize_rejects_a_truncated_output(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    llama_cpp_dir = tmp_path / "llama.cpp"
    llama_cpp_dir.mkdir()
    (llama_cpp_dir / "llama-quantize").write_text("# stub")
    out_path = tmp_path / "out-Q4_K_M.gguf"

    def fake_run(cmd: list[str], check: bool) -> None:
        out_path.write_bytes(b"tiny")

    monkeypatch.setattr(quantize_gguf.subprocess, "run", fake_run)

    with pytest.raises(RuntimeError, match="implausibly small"):
        quantize_gguf.quantize(tmp_path / "f16.gguf", out_path, "Q4_K_M", llama_cpp_dir)


def test_quantize_succeeds_when_the_output_looks_real(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    llama_cpp_dir = tmp_path / "llama.cpp"
    llama_cpp_dir.mkdir()
    (llama_cpp_dir / "llama-quantize").write_text("# stub")
    out_path = tmp_path / "out-Q4_K_M.gguf"

    def fake_run(cmd: list[str], check: bool) -> None:
        out_path.write_bytes(b"0" * 2048)

    monkeypatch.setattr(quantize_gguf.subprocess, "run", fake_run)

    result = quantize_gguf.quantize(tmp_path / "f16.gguf", out_path, "Q4_K_M", llama_cpp_dir)

    assert result == out_path
