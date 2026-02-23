#!/usr/bin/env bash
# ──────────────────────────────────────────────────────────────
#  LongCat-Video · RunPod Pod Setup
#  Run once on a fresh pod to install all deps & verify weights.
#
#  Usage:
#    cd /workspace/LongCat-Video
#    chmod +x setup_pod.sh
#    ./setup_pod.sh
#
#  Optional env vars:
#    WEIGHTS_DIR   – where model weights live (default: /workspace/weights)
#    HF_TOKEN      – Hugging Face token for faster downloads
# ──────────────────────────────────────────────────────────────
set -euo pipefail

WEIGHTS_DIR="${WEIGHTS_DIR:-/workspace/weights}"

echo "============================================"
echo " LongCat-Video · Pod Setup"
echo "============================================"
echo " Weights dir : $WEIGHTS_DIR"
echo " Date        : $(date)"
echo ""

# ── 1. System dependencies ──────────────────────
echo "[1/6] Installing system dependencies..."
apt-get update -qq
apt-get install -y -qq ffmpeg libsndfile1 git > /dev/null 2>&1
echo "  ✓ ffmpeg $(ffmpeg -version 2>&1 | head -1 | awk '{print $3}')"
echo "  ✓ libsndfile1"

# ── 2. Python dependencies ──────────────────────
echo "[2/6] Upgrading pip & build tools..."
pip install -q --upgrade pip packaging ninja

echo "[3/6] Installing Python requirements..."
pip install -q -r requirements.txt
# Extra deps that may not be in requirements.txt
pip install -q librosa soundfile pyloudnorm audio-separator onnxruntime \
    accelerate peft safetensors imageio imageio-ffmpeg
echo "  ✓ Python requirements installed"

# ── 3. flash-attn ────────────────────────────────
echo "[4/6] Setting up flash-attn..."
TORCH_VER=$(python3 -c "import torch; print(torch.__version__)")
TORCH_ABI=$(python3 -c "import torch; print(torch._C._GLIBCXX_USE_CXX11_ABI)")
echo "  PyTorch : $TORCH_VER"
echo "  CXX11_ABI: $TORCH_ABI"

# Remove any pre-installed (potentially broken) flash-attn
pip uninstall flash-attn -y 2>/dev/null || true
rm -rf /usr/local/lib/python3.11/dist-packages/flash_attn* 2>/dev/null || true

FLASH_OK=false

if command -v nvcc &> /dev/null; then
    NVCC_VER=$(nvcc --version 2>/dev/null | grep release | awk '{print $6}' || echo "unknown")
    echo "  nvcc    : $NVCC_VER"
    echo "  Building flash-attn from source (this takes ~15-20 min)..."
    if pip install flash-attn --no-build-isolation --no-cache-dir 2>&1 | tail -3; then
        # Verify the .so actually loads without ABI errors
        if python3 -c "from flash_attn import flash_attn_func; print('  ✓ flash-attn compiled and verified')" 2>/dev/null; then
            FLASH_OK=true
        fi
    fi
fi

if [ "$FLASH_OK" = false ]; then
    echo "  ⚠ flash-attn not available — cleaning up broken artifacts"
    pip uninstall flash-attn -y 2>/dev/null || true
    rm -rf /usr/local/lib/python3.11/dist-packages/flash_attn* 2>/dev/null || true
    echo "  ✓ Will use PyTorch SDPA fallback (same performance on A100)"
fi

# ── 4. Download weights (if missing) ────────────
echo "[5/6] Checking model weights..."
NEED_DOWNLOAD=false

if [ ! -d "$WEIGHTS_DIR/LongCat-Video/dit" ]; then
    echo "  ✗ LongCat-Video weights not found"
    NEED_DOWNLOAD=true
else
    echo "  ✓ LongCat-Video        : $(du -sh "$WEIGHTS_DIR/LongCat-Video" | cut -f1)"
fi

if [ ! -d "$WEIGHTS_DIR/LongCat-Video-Avatar/avatar_single" ]; then
    echo "  ✗ LongCat-Video-Avatar weights not found"
    NEED_DOWNLOAD=true
else
    echo "  ✓ LongCat-Video-Avatar : $(du -sh "$WEIGHTS_DIR/LongCat-Video-Avatar" | cut -f1)"
fi

if [ "$NEED_DOWNLOAD" = true ]; then
    echo ""
    echo "  Downloading missing weights..."
    echo "  (Set HF_TOKEN env var for faster authenticated downloads)"

    # Point HF cache to a temp dir on the volume to avoid filling container disk
    export HF_HOME="${WEIGHTS_DIR}/../hf_cache"
    mkdir -p "$HF_HOME"

    if [ ! -d "$WEIGHTS_DIR/LongCat-Video/dit" ]; then
        echo "  → Downloading meituan-longcat/LongCat-Video (~80 GB)..."
        python3 -c "
from huggingface_hub import snapshot_download
snapshot_download(
    repo_id='meituan-longcat/LongCat-Video',
    local_dir='$WEIGHTS_DIR/LongCat-Video',
)
print('  ✓ LongCat-Video downloaded')
"
        # Clean intermediate cache between downloads to save disk
        rm -rf "$HF_HOME"
        mkdir -p "$HF_HOME"
    fi

    if [ ! -d "$WEIGHTS_DIR/LongCat-Video-Avatar/avatar_single" ]; then
        echo "  → Downloading meituan-longcat/LongCat-Video-Avatar (~120 GB)..."
        python3 -c "
from huggingface_hub import snapshot_download
snapshot_download(
    repo_id='meituan-longcat/LongCat-Video-Avatar',
    local_dir='$WEIGHTS_DIR/LongCat-Video-Avatar',
)
print('  ✓ LongCat-Video-Avatar downloaded')
"
    fi

    # Final cache cleanup
    rm -rf "$HF_HOME"
fi

# ── 5. Smoke test ────────────────────────────────
echo "[6/6] Smoke test..."
python3 -c "
import torch
gpu = torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'N/A'
print(f'  PyTorch {torch.__version__}, GPU: {gpu}')
try:
    from flash_attn import flash_attn_func
    print('  Attention backend : flash-attn')
except (ImportError, RuntimeError):
    print('  Attention backend : PyTorch SDPA (fallback)')
from longcat_video.pipeline_longcat_video_avatar import LongCatVideoAvatarPipeline
print('  Pipeline import   : ✓')
import subprocess
subprocess.run(['ffmpeg', '-version'], capture_output=True, check=True)
print('  ffmpeg            : ✓')
"

echo ""
echo "============================================"
echo " ✅ Setup complete!"
echo ""
echo " Run inference:"
echo ""
echo "   torchrun --nproc_per_node=1 run_demo_avatar_single_audio_to_video.py \\"
echo "     --base_model_dir $WEIGHTS_DIR/LongCat-Video \\"
echo "     --checkpoint_dir $WEIGHTS_DIR/LongCat-Video-Avatar \\"
echo "     --input_json assets/avatar/single_example_1.json"
echo ""
echo "============================================"
