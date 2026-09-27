# CPU-friendly base image; swap FROM to a CUDA base (e.g.
# nvidia/cuda:12.1.0-runtime-ubuntu22.04 + python3 install) if you have a GPU
# and want training/inference inside the container to use it.
FROM python:3.10-slim

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

WORKDIR /app

# libgl1/libglib2 are required by opencv's imshow/video decode paths, even
# in a headless container (kept from the original repository's Dockerfile,
# which needed them for the same reason).
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        libgl1-mesa-glx \
        libglib2.0-0 \
        ffmpeg \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .
RUN pip install --no-cache-dir -e .

# Default: print shape trace as a container smoke test. Override with
# `docker run <image> python scripts/train_finetune.py --config configs/finetune.yaml`
# etc. for real use; realtime_demo.py additionally needs the host's camera
# device and an X11/display forward, which `docker run` does not set up by
# default.
CMD ["python", "scripts/inspect_shapes.py", "--T", "32", "--H", "64", "--W", "64"]
