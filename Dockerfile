# Base ships torch 2.1.0 built against CUDA 11.8.
#
# REPRODUCIBILITY NOTE: requirements.txt pins torch==2.13.0, the version all
# benchmark numbers were produced on. This image therefore runs a DIFFERENT
# torch than the reference environment. Installing the pinned torch here would
# overwrite the base image's CUDA-matched build with a generic wheel carrying
# its own CUDA runtime -- gigabytes of bloat and a real chance GPU support
# breaks silently. Excluding torch from the pip step is the lesser evil; the
# correct long-term fix is a base image matching the pinned version.
FROM pytorch/pytorch:2.1.0-cuda11.8-cudnn8-runtime

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    HF_HOME=/root/.cache/huggingface

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
# Install everything EXCEPT torch (provided by the base image, CUDA-matched).
RUN grep -v '^torch==' requirements.txt > /tmp/req-docker.txt \
    && pip install --no-cache-dir -r /tmp/req-docker.txt \
    && python -c "import torch; print('torch', torch.__version__, 'cuda', torch.version.cuda)"

COPY src/ ./src/

# Fine-tuned CNI model (ADR-002). Not on the HF hub, so it must be baked in;
# .dockerignore un-excludes this one path. Without it the extractor degrades
# to base IndicNER (F1 0.186 vs 0.758) and logs a warning.
COPY data/models/indicner-cni-ft/ ./data/models/indicner-cni-ft/

# Fails fast at build time rather than at 3am in the cluster.
RUN python -c "\
import sys; sys.path.insert(0,'.');\
from src.layer3_native_nlp.gazetteer import CNI_GAZETTEER;\
print('gazetteer terms:', len(CNI_GAZETTEER))"

CMD ["python", "src/layer3_native_nlp/entity_extractor.py"]
