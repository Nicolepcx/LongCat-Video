#!/usr/bin/env bash
set -euo pipefail

# Usage:
#   bash export_runpod_runtime.sh myrepo/longcat-runtime:runpod-h100-$(date +%Y%m%d)
#
# Optional:
#   REF_MP4=/path/to/known_good.mp4 bash export_runpod_runtime.sh myrepo/...
#   MUX_CMD_FILE=/path/to/mux_command.txt bash export_runpod_runtime.sh myrepo/...
#
# Output dir:
#   /workspace/runtime-export/

IMAGE_TAG="${1:-myrepo/longcat-runtime:runpod-h100-latest}"
OUT_DIR="${OUT_DIR:-/workspace/runtime-export}"
REF_MP4="${REF_MP4:-}"
MUX_CMD_FILE="${MUX_CMD_FILE:-}"

mkdir -p "$OUT_DIR"

echo "== Collecting runtime metadata =="
python3 -V | tee "$OUT_DIR/python_version.txt"
which python3 | tee "$OUT_DIR/python_path.txt" || true

nvidia-smi | tee "$OUT_DIR/nvidia_smi.txt"

# Torch and determinism knobs that can affect denoise or alignment
python3 - <<'PY' | tee "$OUT_DIR/torch_env.txt"
import os, torch
print("torch", torch.__version__)
print("cuda", torch.version.cuda)
print("cudnn", torch.backends.cudnn.version())
print("allow_tf32_matmul", torch.backends.cuda.matmul.allow_tf32)
print("allow_tf32_cudnn", torch.backends.cudnn.allow_tf32)
print("cudnn_benchmark", torch.backends.cudnn.benchmark)
print("cudnn_deterministic", torch.backends.cudnn.deterministic)
print("CUBLAS_WORKSPACE_CONFIG", os.environ.get("CUBLAS_WORKSPACE_CONFIG"))
PY

# FFmpeg details (version alone is not enough)
ffmpeg -version | tee "$OUT_DIR/ffmpeg_version_full.txt" || true
ffmpeg -buildconf > "$OUT_DIR/ffmpeg_buildconf.txt" || true
command -v ffprobe >/dev/null 2>&1 && ffprobe -version > "$OUT_DIR/ffprobe_version.txt" || true

# apt package versions that matter (video/audio plumbing)
dpkg-query -W -f='${binary:Package}=${Version}\n' \
  ffmpeg libsndfile1 sox \
  > "$OUT_DIR/apt-lock-av.txt" || true

# full pip state (for completeness)
python3 -m pip freeze --all | sed '/^-e /d' > "$OUT_DIR/requirements-lock-full.txt"
python3 -m pip list --format=json > "$OUT_DIR/pip_list.json"

# filtered runtime lock (keep your set but write it deterministically)
python3 - <<'PY'
from pathlib import Path

keep = {
 "torch","torchvision","torchaudio","triton","flash-attn",
 "transformers","tokenizers","huggingface_hub","sentencepiece","tiktoken","protobuf","regex",
 "diffusers","accelerate","safetensors","peft","einops","ftfy","loguru","psutil",
 "numpy","scipy","scikit-learn","scikit-image","sympy",
 "av","opencv-python","imageio","imageio-ffmpeg","pillow",
 "librosa","soundfile","soxr","pyloudnorm","audio-separator","onnx","onnxruntime","onnx2torch","pydub",
 "ffmpeg-python","moviepy","decord",
 "cffi","chardet","tzdata","nvidia-ml-py"
}

lines = Path("/workspace/runtime-export/requirements-lock-full.txt").read_text().splitlines()
out = []
for line in lines:
    if "==" not in line:
        continue
    name = line.split("==",1)[0].strip().lower()
    if name in keep:
        out.append(line.strip())

Path("/workspace/runtime-export/requirements-lock-runtime.txt").write_text(
    "\n".join(sorted(set(out))) + "\n"
)
print(f"Wrote runtime lock with {len(set(out))} packages")
PY

# Stream timing metadata for a known-good reference output
if [ -n "$REF_MP4" ] && [ -f "$REF_MP4" ]; then
  echo "== Capturing ffprobe on reference mp4: $REF_MP4 =="
  ffprobe -hide_banner -show_streams -show_format "$REF_MP4" \
    > "$OUT_DIR/ffprobe_reference_good.txt" || true
fi

# Save mux command or postprocess command if you have it
if [ -n "$MUX_CMD_FILE" ] && [ -f "$MUX_CMD_FILE" ]; then
  cp "$MUX_CMD_FILE" "$OUT_DIR/mux_command.txt"
fi

echo "== Writing Dockerfile scaffold =="
cat > "$OUT_DIR/Dockerfile.runtime" <<'DOCKER'
# Replace base with the exact working image you used on RunPod
FROM runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04

WORKDIR /root/LongCat-Video

# Copy lockfiles + project
COPY requirements-lock-runtime.txt /tmp/requirements-lock-runtime.txt
COPY . /root/LongCat-Video

# Install system deps
RUN apt-get update && apt-get install -y \
    ffmpeg sox libsndfile1 git \
  && rm -rf /var/lib/apt/lists/*

# Install exact runtime deps
RUN python3 -m pip install -U pip packaging ninja && \
    python3 -m pip install -r /tmp/requirements-lock-runtime.txt

# Optional sanity checks
RUN python3 - <<'PY'
import torch
print("torch", torch.__version__)
print("cuda", torch.version.cuda)
PY
DOCKER

cat > "$OUT_DIR/BUILD_AND_PUSH.md" <<EOF
# Build and push
cd /workspace/LongCat-Video
cp /workspace/runtime-export/requirements-lock-runtime.txt .
cp /workspace/runtime-export/Dockerfile.runtime .

docker build -f Dockerfile.runtime -t ${IMAGE_TAG} .
docker push ${IMAGE_TAG}
EOF

echo "== Export complete =="
echo "Output: $OUT_DIR"
echo "Next: read $OUT_DIR/BUILD_AND_PUSH.md"
