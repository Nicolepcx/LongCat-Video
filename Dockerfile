# ============================================================================
# LongCat-Video Avatar – RunPod Serverless Worker
# ============================================================================
# Build:
#   docker build -t longcat-video-avatar .
#
# Run locally (test):
#   docker run --gpus all -v /path/to/weights:/runpod-volume/weights \
#       -p 8000:8000 longcat-video-avatar
#
# Model weights are NOT baked into the image – mount them at runtime from
# a RunPod Network Volume at /runpod-volume/weights.
# ============================================================================

FROM nvidia/cuda:12.4.1-devel-ubuntu22.04

# Prevent interactive prompts during apt-get
ENV DEBIAN_FRONTEND=noninteractive

# ------------------------------------------------------------------
# System dependencies  (Python 3.11 via deadsnakes PPA)
# ------------------------------------------------------------------
RUN apt-get update && apt-get install -y --no-install-recommends \
        software-properties-common \
    && add-apt-repository ppa:deadsnakes/ppa -y \
    && apt-get update && apt-get install -y --no-install-recommends \
        ffmpeg \
        libsndfile1 \
        libgl1-mesa-glx \
        libglib2.0-0 \
        git \
        wget \
        curl \
        python3.11 \
        python3.11-dev \
        python3.11-distutils \
        build-essential \
        ninja-build \
    && ln -sf /usr/bin/python3.11 /usr/bin/python \
    && ln -sf /usr/bin/python3.11 /usr/bin/python3 \
    && curl -sS https://bootstrap.pypa.io/get-pip.py | python3.11 \
    && rm -rf /var/lib/apt/lists/*

# Upgrade pip
RUN python -m pip install --no-cache-dir --upgrade pip setuptools wheel packaging

# ------------------------------------------------------------------
# Python dependencies (installed before copying code for layer caching)
# ------------------------------------------------------------------
WORKDIR /app

COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# flash-attn needs special treatment (no build isolation)
RUN pip install --no-cache-dir flash-attn --no-build-isolation

# ------------------------------------------------------------------
# Copy project code
# ------------------------------------------------------------------
COPY . /app

# ------------------------------------------------------------------
# Runtime configuration
# ------------------------------------------------------------------
ENV WEIGHTS_DIR=/runpod-volume/weights
ENV OUTPUT_DIR=/tmp/outputs
ENV PYTHONUNBUFFERED=1

# Expose default port for local testing
EXPOSE 8000

CMD ["python", "-u", "rp_handler.py"]
