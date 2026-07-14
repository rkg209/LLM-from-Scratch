# scripts/ — standalone model-preparation helpers

Not an importable package. One-shot scripts run by hand between chapters, not imported at runtime.

| Script | Purpose | Spec |
|---|---|---|
| `merge_adapter.py` | `peft.merge_and_unload()` → `outputs/merged/` | O1 |
| `export_gguf.py` | llama.cpp `convert_hf_to_gguf.py` → `outputs/model.gguf` | O1 |
| `quantize_gguf.py` | llama.cpp quantize → `outputs/model-Q4_K_M.gguf` | O1 |

None of these scripts exist yet — O1 owns writing them. This directory is reserved by F1 so O1 has a place to land.
