# Deploying LongCat-Video Avatar

Three deployment options:

1. **RunPod Pod** (recommended) — persistent GPU instance with network volume for weights
2. **Modal Endpoint** — serverless GPU endpoint with persistent Modal Volume
3. **Gradio UI on DigitalOcean** — lightweight password-protected frontend

---

## Architecture

```
┌────────────────────────────────┐
│  Gradio UI  (DigitalOcean)     │  ← $5/mo, password-protected
│  Upload audio + image          │
└───────────────┬────────────────┘
                │  base64 payload via RunPod API
                ▼
┌────────────────────────────────┐      ┌─────────────────────────────┐
│  RunPod Pod / Modal Endpoint   │◄────►│  Persistent Volume (250 GB) │
│  A100 80 GB GPU                │      │  /workspace/weights/        │
│  Returns video as base64       │      │  ├── LongCat-Video/         │
│                                │      │  └── LongCat-Video-Avatar/  │
└────────────────────────────────┘      └─────────────────────────────┘
```

---

## Part 1: RunPod Setup

### 1.1 — Create a Network Volume

1. **RunPod Console → Storage → Network Volumes → Create**
2. Name: `longcat-weights`
3. Region: **same region you'll use for the pod** (e.g. `US-TX-3`)
4. Size: **250 GB** (weights are ~200 GB total)

### 1.2 — Download weights to the volume (one-time)

Spin up a temporary pod to download models:

1. **RunPod Console → Pods → Deploy**
2. Pick any GPU (cheapest available — it's just downloading)
3. **Attach volume** `longcat-weights` → mount path: `/workspace`
4. Template: any PyTorch template (e.g. `RunPod PyTorch 2.x`)
5. Open a terminal

> **Important:** On RunPod Pods the network volume mounts at `/workspace`.
> Set `HF_HOME` to the volume so the download cache doesn't fill the container disk.

```bash
# Point HF cache to the volume (avoids filling the 20 GB container disk)
export HF_HOME=/workspace/hf_cache
mkdir -p /workspace/weights

# Optional: set for faster authenticated downloads
# export HF_TOKEN="hf_xxxxxxxx"

pip install -q huggingface_hub

# Download LongCat-Video (~80 GB)
python3 -c "
from huggingface_hub import snapshot_download
snapshot_download('meituan-longcat/LongCat-Video',
                  local_dir='/workspace/weights/LongCat-Video')
print('Done: LongCat-Video')
"

# Clean cache between downloads to save disk space
rm -rf /workspace/hf_cache

# Download LongCat-Video-Avatar (~120 GB)
export HF_HOME=/workspace/hf_cache
python3 -c "
from huggingface_hub import snapshot_download
snapshot_download('meituan-longcat/LongCat-Video-Avatar',
                  local_dir='/workspace/weights/LongCat-Video-Avatar')
print('Done: LongCat-Video-Avatar')
"

# Final cleanup & verify
rm -rf /workspace/hf_cache
du -sh /workspace/weights/*
```

Expected output:
```
78G   /workspace/weights/LongCat-Video
120G  /workspace/weights/LongCat-Video-Avatar
```

**Terminate the temporary pod** — the volume persists.

### 1.3 — Run on a Pod

1. **RunPod Console → Pods → Deploy**
2. GPU: **A100 80 GB** (minimum 48 GB VRAM)
3. **Attach volume** `longcat-weights` → mount path: `/workspace`
4. Template: `RunPod PyTorch 2.x` (or any CUDA 12.x template)
5. Open a terminal

```bash
# Clone the repo
git clone https://github.com/Nicolepcx/LongCat-Video.git /workspace/LongCat-Video
cd /workspace/LongCat-Video

# Run the setup script (installs deps, flash-attn, verifies weights)
chmod +x setup_pod.sh
./setup_pod.sh

# Run inference with the example files
torchrun --nproc_per_node=1 run_demo_avatar_single_audio_to_video.py \
  --base_model_dir /workspace/weights/LongCat-Video \
  --checkpoint_dir /workspace/weights/LongCat-Video-Avatar \
  --input_json assets/avatar/single_example_1.json
```

The setup script handles:
- System deps (`ffmpeg`, `libsndfile1`)
- Python requirements
- `flash-attn` compilation (with automatic SDPA fallback if build fails)
- Weight verification
- Import smoke test

### 1.4 — Using custom audio/image

Create a JSON config file (e.g. `my_input.json`):

```json
{
    "prompt": "A woman speaking naturally with expressive gestures",
    "cond_image": "path/to/face.png",
    "cond_audio": {
        "person1": "path/to/speech.wav"
    }
}
```

Then run:
```bash
torchrun --nproc_per_node=1 run_demo_avatar_single_audio_to_video.py \
  --base_model_dir /workspace/weights/LongCat-Video \
  --checkpoint_dir /workspace/weights/LongCat-Video-Avatar \
  --input_json my_input.json \
  --stage_1 ai2v
```

### 1.5 — RunPod Serverless (alternative)

For scale-to-zero behavior, use the serverless endpoint:

1. **RunPod Console → Serverless → New Endpoint**
2. **Connect GitHub** → select `Nicolepcx/LongCat-Video`
3. RunPod detects `Dockerfile` + `rp_handler.py` automatically
4. Configure:

| Setting | Value |
|---------|-------|
| **GPU** | A100 80 GB |
| **Network Volume** | `longcat-weights` |
| **Min Workers** | `0` (scale to zero) |
| **Max Workers** | `1` |
| **Idle Timeout** | `300` s |
| **Execution Timeout** | `900` s |
| **Flash Boot** | Enabled |

5. Environment variables:

| Variable | Value |
|----------|-------|
| `WEIGHTS_DIR` | `/runpod-volume/weights` |

> **Note:** Serverless mounts volumes at `/runpod-volume`, not `/workspace`.

6. Click **Deploy**

Test with:

```bash
# Write payload to file (base64 strings are too long for shell args)
cat > /tmp/test_payload.json << 'EOF'
{
  "input": {
    "audio_base64": "<base64-encoded-audio>",
    "image_base64": "<base64-encoded-image>",
    "prompt": "A person speaking naturally",
    "stage": "ai2v"
  }
}
EOF

curl -s -X POST "https://api.runpod.ai/v2/${ENDPOINT_ID}/run" \
  -H "Authorization: Bearer ${RUNPOD_API_KEY}" \
  -H "Content-Type: application/json" \
  -d @/tmp/test_payload.json
```

> **Warning:** Serverless cold starts are **very slow** for this model (10-40 min)
> due to the large weight loading time. A persistent Pod with `setup_pod.sh` is
> recommended for development and regular use.

---

## Part 2: Modal Deployment

Modal is a strong alternative to RunPod Serverless for this project: persistent
volume support, clean Python DX, and scale-to-zero endpoints.

### 2.1 — Install Modal CLI (local machine)

```bash
pip install -r requirements-modal.txt
modal setup
```

### 2.2 — Create Modal volume

```bash
modal volume create longcat-weights
```

If you use a different name, set:

```bash
export MODAL_WEIGHTS_VOLUME="your-volume-name"
```

### 2.3 — One-time weight download to Modal Volume

```bash
cd LongCat-Video
modal run modal_app.py --download --hf-token hf_xxxxxxxx
```

This populates:

```
/weights/weights/LongCat-Video
/weights/weights/LongCat-Video-Avatar
```

### 2.4 — Deploy Modal endpoint

```bash
modal deploy modal_app.py
```

Deployment output includes a public URL for `inference_endpoint`, typically:

```text
https://<workspace>--longcat-video-avatar-inference-endpoint.modal.run
```

### 2.5 — Test Modal endpoint

```bash
AUDIO_B64=$(base64 -i assets/avatar/single/man.mp3)
IMAGE_B64=$(base64 -i assets/avatar/single/man.png)

curl -X POST "https://<workspace>--longcat-video-avatar-inference-endpoint.modal.run" \
  -H "Content-Type: application/json" \
  -d "{
    \"audio_base64\": \"${AUDIO_B64}\",
    \"image_base64\": \"${IMAGE_B64}\",
    \"prompt\": \"A person speaking naturally\",
    \"stage\": \"ai2v\",
    \"resolution\": \"480p\"
  }"
```

Response:

```json
{ "video_base64": "<base64-encoded MP4>" }
```

---

## Part 3: Gradio UI on DigitalOcean

The Gradio app (`app.py`) is a lightweight frontend — no GPU needed. It base64-encodes
your audio/image and sends them to the RunPod endpoint.

### 3.1 — Deploy on DigitalOcean App Platform

1. **DigitalOcean Console → App Platform → Create App**
2. **Source:** GitHub → select `Nicolepcx/LongCat-Video`
3. **Dockerfile path:** `Dockerfile.app`
4. **Instance type:** Basic ($5/mo)
5. **Environment variables:**

| Variable | Value |
|----------|-------|
| `RUNPOD_API_KEY` | `rp_xxxxxxxx` (RunPod backend only) |
| `RUNPOD_ENDPOINT_ID` | your endpoint ID (RunPod backend only) |
| `INFERENCE_API_URL` | `https://<workspace>--longcat-video-avatar-inference-endpoint.modal.run` (Modal backend only) |
| `GRADIO_USERNAME` | your login username |
| `GRADIO_PASSWORD` | a strong password |
| `PORT` | `7860` |

6. Deploy

### 3.2 — Test locally (optional)

```bash
pip install gradio requests
# RunPod backend:
# export RUNPOD_API_KEY="rp_xxxxxxxx"
# export RUNPOD_ENDPOINT_ID="your-endpoint-id"
# OR Modal backend:
# export INFERENCE_API_URL="https://<workspace>--longcat-video-avatar-inference-endpoint.modal.run"
export GRADIO_PASSWORD="testpass"
python app.py
# Open http://localhost:7860
```

---

## Expected Performance

| Phase | Duration | Notes |
|-------|----------|-------|
| `setup_pod.sh` (first run) | ~20-30 min | Mostly flash-attn compilation |
| `setup_pod.sh` (re-run) | ~2 min | Deps already cached |
| Model loading | ~1-2 min | Weights loaded from network volume |
| Audio separation | ~1 min | Vocal extraction with ONNX |
| Denoising (50 steps, 480p) | ~15-17 min | ~20s per step on A100 80 GB |
| **Total inference** | **~17-20 min** | Single segment |

---

## Cost Estimates

### Monthly (idle)

| Component | Cost |
|-----------|------|
| RunPod Pod (stopped) | $0.00 |
| Network Volume (250 GB) | ~$17.50 |
| DigitalOcean App (Basic) | ~$5.00 |
| **Total idle** | **~$22.50/mo** |

### Per video (A100 80 GB @ ~$1.10/hr)

| Phase | Duration | Cost |
|-------|----------|------|
| Inference (1 segment, 480p) | ~17-20 min | ~$0.31-0.37 |

---

## API Reference (Serverless)

### Input (`job["input"]`)

| Field | Type | Required | Default |
|-------|------|----------|---------|
| `audio_base64` | string | **yes** | — |
| `image_base64` | string | ai2v only | — |
| `prompt` | string | no | "A person speaking naturally" |
| `negative_prompt` | string | no | (built-in) |
| `stage` | string | no | `"ai2v"` |
| `resolution` | string | no | `"480p"` |
| `num_inference_steps` | int | no | `50` |
| `text_guidance_scale` | float | no | `4.0` |
| `audio_guidance_scale` | float | no | `4.0` |
| `num_segments` | int | no | `1` |
| `seed` | int | no | `42` |

### Output

```json
{ "video_base64": "<base64-encoded MP4>" }
```

---

## Troubleshooting

| Problem | Fix |
|---------|-----|
| `flash_attn` ABI mismatch / undefined symbol | Run `setup_pod.sh` — it auto-detects and falls back to PyTorch SDPA |
| `ffmpeg: not found` | Run `apt-get install -y ffmpeg` or re-run `setup_pod.sh` |
| `No space left on device` during weight download | Set `HF_HOME` to the volume; download models one at a time |
| Volume not mounted | Pods use `/workspace`, Serverless uses `/runpod-volume` |
| Serverless cold start > 30 min | Use a Pod instead; Serverless is impractical for this model size |
| OOM | Use A100 80 GB; lower resolution to 480p; set segments to 1 |

---

## Files

| File | What it does |
|------|-------------|
| `setup_pod.sh` | One-command pod setup (deps, flash-attn, verification) |
| `rp_handler.py` | RunPod serverless worker (GPU) |
| `Dockerfile` | Docker image for the GPU worker |
| `app.py` | Gradio UI with password auth (no GPU) |
| `Dockerfile.app` | Docker image for the Gradio UI |
| `requirements.txt` | Python deps for the GPU worker |
| `requirements-app.txt` | Python deps for the Gradio UI |
| `download_weights.py` | Programmatic weight download helper |
| `.dockerignore` | Keeps Docker images lean |
