"""Static checks on docker/Dockerfile that need no Docker daemon.

The real build/run test (`test_docker.py`) mounts the model into /models, so it can never see
the first-boot path where the entrypoint downloads the GGUF itself -- which is the only path a
deployed Space takes. These pin the two things that path needs.
"""

from __future__ import annotations

from pathlib import Path

DOCKERFILE = Path("docker/Dockerfile").read_text()
FINAL_STAGE = DOCKERFILE.split("AS final", 1)[1]


def test_runtime_user_is_uid_1000_for_hf_spaces() -> None:
    assert "--uid 1000" in FINAL_STAGE
    assert "USER reviewer" in FINAL_STAGE


def test_model_dir_exists_and_belongs_to_the_runtime_user() -> None:
    model_dir = "/models"
    assert "ENV MODEL_PATH=/models/model.gguf" in FINAL_STAGE
    assert f"mkdir -p {model_dir}" in FINAL_STAGE
    assert f"chown reviewer:reviewer {model_dir}" in FINAL_STAGE
    assert FINAL_STAGE.index(f"chown reviewer:reviewer {model_dir}") < FINAL_STAGE.index(
        "USER reviewer"
    )
