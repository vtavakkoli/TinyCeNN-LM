ARG PYTORCH_IMAGE=pytorch/pytorch:2.14.0-cuda12.6-cudnn9-runtime
FROM ${PYTORCH_IMAGE}

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    HF_HOME=/cache/huggingface \
    TRANSFORMERS_CACHE=/cache/huggingface/transformers \
    HF_DATASETS_CACHE=/cache/huggingface/datasets \
    TOKENIZERS_PARALLELISM=true \
    PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

WORKDIR /workspace

COPY pyproject.toml README.md LICENSE ./
COPY src ./src
COPY scripts ./scripts
COPY tests ./tests

RUN python -m pip install --upgrade pip setuptools wheel \
    && python -m pip install --no-cache-dir -e . \
    && python -m compileall -q src scripts

RUN mkdir -p /workspace/checkpoints /workspace/runs /cache/huggingface

CMD ["python", "scripts/train_adapter.py", "--help"]
