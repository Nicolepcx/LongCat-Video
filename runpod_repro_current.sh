#!/usr/bin/env bash
set -euo pipefail

# Reproducible RunPod direct-pod runner using CURRENT repo code.
# Purpose: reproduce quality on a plain GPU pod with minimal moving parts.
#
# Usage on pod:
#   cd /workspace/LongCat-Video
#   bash runpod_repro_current.sh
#
# Optional env vars:
#   REPO_DIR=/workspace/LongCat-Video
#   WEIGHTS_ROOT=/workspace/weights
#   INPUT_AUDIO=assets/avatar/single/Nicole.wav
#   INPUT_IMAGE=assets/avatar/single/tabby.png
#   INPUT_PROMPT="an animated cat talking naturally, looking at the camera"
#   INPUT_JSON=/tmp/tabby_test.json
#   FORCE_SYNC=0            # set 1 to hard reset repo to origin/main
#   SKIP_SETUP_POD=0        # set 1 if setup_pod.sh already completed

REPO_DIR="${REPO_DIR:-/workspace/LongCat-Video}"
WEIGHTS_ROOT="${WEIGHTS_ROOT:-/workspace/weights}"
INPUT_AUDIO="${INPUT_AUDIO:-assets/avatar/single/Nicole.wav}"
INPUT_IMAGE="${INPUT_IMAGE:-assets/avatar/single/tabby.png}"
INPUT_PROMPT="${INPUT_PROMPT:-an animated cat talking naturally, looking at the camera}"
INPUT_JSON="${INPUT_JSON:-/tmp/tabby_test.json}"
FORCE_SYNC="${FORCE_SYNC:-0}"
SKIP_SETUP_POD="${SKIP_SETUP_POD:-0}"

cd "$REPO_DIR"

echo "== [1/7] Repo sync =="
if [[ "$FORCE_SYNC" == "1" ]]; then
  echo "FORCE_SYNC=1 -> hard syncing to origin/main"
  git fetch origin
  git reset --hard origin/main
  git clean -fd
else
  git pull --ff-only || true
fi

echo "== [2/7] Optional base setup =="
if [[ "$SKIP_SETUP_POD" != "1" ]]; then
  # setup_pod can fail on some images due to distutils-installed packages (e.g., blinker).
  # Continue with deterministic recovery below.
  bash setup_pod.sh || true
fi

echo "== [3/7] Dependency recovery / pinning =="
# Remove accidentally installed incompatible stacks (e.g., transformers>=5, numpy 2.x).
pip uninstall -y transformers tokenizers huggingface-hub typer typer-slim rich annotated-doc markdown-it-py || true
pip uninstall -y numpy onnxruntime audio-separator || true

python3 -m pip install -U pip setuptools wheel packaging --break-system-packages || true

# Install project dependencies while ignoring known distutils conflict package.
PIP_BREAK_SYSTEM_PACKAGES=1 pip install -r requirements.txt --ignore-installed blinker

# Re-pin critical known-good versions for this project.
pip install --force-reinstall \
  "numpy==1.26.4" \
  "transformers==4.46.3" \
  "diffusers==0.32.2" \
  "huggingface_hub<1.0" \
  "tokenizers<0.21,>=0.20" \
  "onnxruntime==1.16.3" \
  "audio-separator==0.30.2" \
  "sentencepiece" \
  "protobuf"

echo "== [4/7] Version sanity =="
python3 - <<'PY'
import numpy, transformers, diffusers, onnxruntime
print("numpy:", numpy.__version__)
print("transformers:", transformers.__version__)
print("diffusers:", diffusers.__version__)
print("onnxruntime:", onnxruntime.__version__)
PY

echo "== [5/7] Runtime env =="
export BASE_MODEL_DIR="${BASE_MODEL_DIR:-$WEIGHTS_ROOT/LongCat-Video}"
export AVATAR_WEIGHTS_DIR="${AVATAR_WEIGHTS_DIR:-$WEIGHTS_ROOT/LongCat-Video-Avatar}"
export RANK=0
export WORLD_SIZE=1
export LOCAL_RANK=0
export MASTER_ADDR=127.0.0.1
export MASTER_PORT=29500

echo "BASE_MODEL_DIR=$BASE_MODEL_DIR"
echo "AVATAR_WEIGHTS_DIR=$AVATAR_WEIGHTS_DIR"

echo "== [6/7] Build input JSON =="
cat > "$INPUT_JSON" <<JSON
{
  "prompt": "${INPUT_PROMPT}",
  "cond_image": "${INPUT_IMAGE}",
  "cond_audio": {
    "person1": "${INPUT_AUDIO}"
  }
}
JSON

echo "Wrote $INPUT_JSON"

echo "== [7/7] Run inference (direct script path) =="
python run_demo_avatar_single_audio_to_video.py \
  --input_json "$INPUT_JSON" \
  --base_model_dir "$BASE_MODEL_DIR" \
  --checkpoint_dir "$AVATAR_WEIGHTS_DIR" \
  --resolution 480p \
  --num_inference_steps 50 \
  --text_guidance_scale 4.0 \
  --audio_guidance_scale 4.0 \
  --num_segments 1

