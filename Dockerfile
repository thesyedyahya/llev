FROM python:3.12-slim AS build
RUN apt-get update && apt-get install -y --no-install-recommends build-essential cmake git \
    && rm -rf /var/lib/apt/lists/*
WORKDIR /app
COPY pyproject.toml ./
# Default: tuned for the build machine's CPU (AVX2/AVX-512 on EPYC). If you build on the
# same server that runs it, this is the fastest option. Building on one CPU type and running on
# another? --build-arg LLAMA_CMAKE_ARGS="-DGGML_NATIVE=OFF -DGGML_AVX2=ON -DGGML_FMA=ON -DGGML_F16C=ON"
# Never ship GGML_NATIVE=OFF alone: it drops AVX2 and inference gets ~10x slower.
ARG LLAMA_CMAKE_ARGS=""
ENV CMAKE_ARGS="${LLAMA_CMAKE_ARGS}"
RUN pip install --no-cache-dir --prefix=/install \
    "llama-cpp-python>=0.3.0" "fastapi>=0.115" "uvicorn[standard]>=0.30" \
    "pydantic>=2.7" "pydantic-settings>=2.3" "numpy>=1.26" "huggingface_hub>=0.24"

FROM python:3.12-slim
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 curl \
    && rm -rf /var/lib/apt/lists/*
COPY --from=build /install /usr/local
WORKDIR /app
COPY llev ./llev
COPY scripts ./scripts
ENV LLEV_MODELS_DIR=/models LLEV_DATA_DIR=/data LLEV_PORT=8088
VOLUME ["/models", "/data"]
EXPOSE 8088
HEALTHCHECK --interval=30s --timeout=5s --start-period=120s CMD curl -fs http://127.0.0.1:8088/health || exit 1
# Download the configured model(s) on first boot, then serve.
CMD ["sh", "-c", "python scripts/download_model.py $LLEV_MODEL ${LLEV_ESCALATE_MODEL:-} && python -m llev"]
