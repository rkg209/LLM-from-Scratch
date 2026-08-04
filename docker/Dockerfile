# Multi-stage CPU-only image for ch3_operation (spec O4). No CUDA anywhere (CON-12, AC-5):
# llama-cpp-python's CPU wheel is compiled once in the builder stage; the compiler toolchain
# used to build it does not survive into the final image (AC-7).
#
# Build (amd64 is not optional, AC-8 -- HF Spaces will not start an arm64-only image built
# on an Apple Silicon laptop):
#   docker buildx build --platform linux/amd64 -f docker/Dockerfile -t reviewer:local .

FROM python:3.12-slim AS builder

RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        cmake \
        git \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /build

RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Only the ch3 extra + its own dependency closure is compiled -- torch/transformers (ch1/ch2)
# never enter this image; the served artifact is a GGUF, not a fine-tune checkpoint.
COPY pyproject.toml /build/pyproject.toml
COPY eval /build/eval
COPY monitoring /build/monitoring
COPY ch3_operation /build/ch3_operation
RUN pip install --no-cache-dir /build[ch3]

FROM python:3.12-slim AS final

RUN groupadd --system reviewer && useradd --system --gid reviewer --create-home reviewer

COPY --from=builder /opt/venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

WORKDIR /app
COPY eval /app/eval
COPY monitoring /app/monitoring
COPY ch3_operation /app/ch3_operation
COPY docker/entrypoint.sh /app/entrypoint.sh
RUN chmod +x /app/entrypoint.sh && chown -R reviewer:reviewer /app

USER reviewer

ENV CONFIG_PATH=/app/ch3_operation/configs/full.yaml
ENV MODEL_PATH=/models/model.gguf
ENV PORT=8000

EXPOSE 8000

ENTRYPOINT ["/app/entrypoint.sh"]
