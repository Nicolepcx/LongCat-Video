# Deploying LongCat-Video Avatar

Two things to deploy:

1. **RunPod Serverless endpoint** — the GPU worker that runs inference (scales to zero)
2. **Gradio UI on DigitalOcean** — password-protected frontend to upload audio + image

No local Docker builds needed. RunPod builds from GitHub, DO builds from `Dockerfile.app`.

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
│  RunPod Serverless Endpoint    │◄────►│  Network Volume             │
│  Built from GitHub repo        │      │  /runpod-volume/weights/    │
│  Scales 0 → N GPUs             │      │  ├── LongCat-Video/        │
│  Returns video as base64       │      │  └── LongCat-Video-Avatar/ │
└────────────────────────────────┘      └─────────────────────────────┘
```

---

## Part 1: RunPod Serverless Endpoint

### 1.1 — Push repo to GitHub

Make sure the repo is pushed to GitHub with all our changes — especially:
- `rp_handler.py` (with `runpod.serverless.start()` at module level)
- `Dockerfile` (CUDA base image + all deps)
- `requirements.txt`
- `download_weights.py`

### 1.2 — Create a Network Volume

1. **RunPod Console → Storage → Network Volumes → Create**
2. Name: `longcat-weights`
3. Region: **same region you'll use for the endpoint**
4. Size: **100 GB**

### 1.3 — Download weights to the volume (one-time)

Spin up a temporary pod to download the HuggingFace models:

1. **RunPod Console → Pods → Deploy**
2. Pick any GPU (cheapest available is fine — it's just downloading)
3. Attach volume `longcat-weights` (mounts at `/runpod-volume`)
4. Open a terminal

```bash
pip install huggingface_hub

python -c "
from huggingface_hub import snapshot_download
snapshot_download('meituan-longcat/LongCat-Video',
                  local_dir='/runpod-volume/weights/LongCat-Video',
                  local_dir_use_symlinks=False)
snapshot_download('meituan-longcat/LongCat-Video-Avatar',
                  local_dir='/runpod-volume/weights/LongCat-Video-Avatar',
                  local_dir_use_symlinks=False)
print('Done!')
"
```

Or clone the repo and use the helper script:

```bash
git clone https://github.com/Nicolepcx/LongCat-Video.git /tmp/repo
cd /tmp/repo
python download_weights.py --output_dir /runpod-volume/weights
```

Verify the structure:

```
/runpod-volume/weights/
├── LongCat-Video/
│   ├── tokenizer/
│   ├── text_encoder/
│   ├── vae/
│   └── scheduler/
└── LongCat-Video-Avatar/
    ├── avatar_single/
    ├── chinese-wav2vec2-base/
    └── vocal_separator/Kim_Vocal_2.onnx
```

**Terminate the temporary pod** — the volume persists.

### 1.4 — Create the Serverless Endpoint from GitHub

1. **RunPod Console → Serverless → New Endpoint**
2. Click **"Connect GitHub"** → authorize → select `Nicolepcx/LongCat-Video`
3. RunPod will detect the `Dockerfile` and `runpod.serverless.start()` in `rp_handler.py`
4. Configure the endpoint:

| Setting | Value |
|---------|-------|
| **Endpoint name** | `LongCat-Video` |
| **GPU** | A100 80 GB (or A40 / L40S — 48 GB min) |
| **Network Volume** | `longcat-weights` |
| **Min Workers** | `0` (scale to zero) |
| **Max Workers** | `1` |
| **Idle Timeout** | `300` s |
| **Execution Timeout** | `900` s |
| **Flash Boot** | Enabled |
| **Container Disk** | `20` GB |
| **Container start command** | `python -u rp_handler.py` |

5. Environment variables:

| Variable | Value |
|----------|-------|
| `WEIGHTS_DIR` | `/runpod-volume/weights` |

6. Click **Deploy**

RunPod builds the Docker image from your repo in the cloud — no local build needed.
Note the **Endpoint ID** (e.g. `abc123xyz`) once it's created.

### 1.5 — Test the endpoint

```bash
export RUNPOD_API_KEY="rp_xxxxxxxx"
export ENDPOINT_ID="abc123xyz"

# Encode test files
AUDIO_B64=$(base64 -i test_audio.wav)
IMAGE_B64=$(base64 -i test_face.png)

# Submit
curl -s -X POST "https://api.runpod.ai/v2/${ENDPOINT_ID}/run" \
  -H "Authorization: Bearer ${RUNPOD_API_KEY}" \
  -H "Content-Type: application/json" \
  -d "{
    \"input\": {
      \"audio_base64\": \"${AUDIO_B64}\",
      \"image_base64\": \"${IMAGE_B64}\",
      \"prompt\": \"A person speaking naturally\",
      \"stage\": \"ai2v\"
    }
  }"
```

Poll with:

```bash
JOB_ID="your-job-id-here"
curl -s "https://api.runpod.ai/v2/${ENDPOINT_ID}/status/${JOB_ID}" \
  -H "Authorization: Bearer ${RUNPOD_API_KEY}"
```

---

## Part 2: Gradio UI on DigitalOcean

The Gradio app (`app.py`) is a lightweight frontend — no GPU, no torch. It uses
`Dockerfile.app` which only installs `gradio` and `requests`.

### 2.1 — Deploy on DigitalOcean App Platform

1. **DigitalOcean Console → App Platform → Create App**
2. **Source:** GitHub → select `Nicolepcx/LongCat-Video`
3. **Dockerfile path:** `Dockerfile.app`
4. **Instance type:** Basic ($5/mo)
5. **Environment variables:**

| Variable | Value |
|----------|-------|
| `RUNPOD_API_KEY` | `rp_xxxxxxxx` |
| `RUNPOD_ENDPOINT_ID` | `abc123xyz` (from step 1.4) |
| `GRADIO_USERNAME` | your login username |
| `GRADIO_PASSWORD` | a strong password |
| `PORT` | `7860` |

6. Deploy

You'll get a URL like `https://your-app-xxxxx.ondigitalocean.app`.
Login with the username/password you set → upload audio + image → get video.

### 2.2 — Test locally first (optional)

```bash
pip install gradio requests

export RUNPOD_API_KEY="rp_xxxxxxxx"
export RUNPOD_ENDPOINT_ID="abc123xyz"
export GRADIO_PASSWORD="testpass"

python app.py
# Open http://localhost:7860
```

---

## How It All Fits Together

```
You (browser)
    │
    │  Login with username/password
    ▼
Gradio on DO ($5/mo)
    │
    │  1. Reads your audio + image files
    │  2. Base64-encodes them
    │  3. POSTs to RunPod API
    │  4. Polls for result
    │  5. Decodes video, shows in browser
    ▼
RunPod Serverless (pay-per-second)
    │
    │  - Scales from 0 when job arrives (~2-3 min cold start)
    │  - Loads weights from Network Volume
    │  - Runs inference (~3-10 min)
    │  - Returns video as base64
    │  - Scales back to 0 after idle timeout
    ▼
Network Volume ($7/mo for 100 GB)
    │
    └── Stores model weights permanently
```

### Monthly cost when not in use

| Component | Cost |
|-----------|------|
| RunPod workers (scaled to zero) | $0.00 |
| Network Volume (100 GB) | ~$7.00 |
| DigitalOcean App (Basic) | ~$5.00 |
| **Total idle** | **~$12/mo** |

### Cost per video

| Phase | Duration | Cost (A100 80GB) |
|-------|----------|-------------------|
| Cold start (if scaled to zero) | ~2-3 min | ~$0.14-0.22 |
| Inference (1 segment, 480p) | ~3-5 min | ~$0.22-0.36 |
| **Total per video** | ~5-8 min | **~$0.36-0.58** |

---

## API Reference

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
| RunPod says "Could not find runpod.serverless.start()" | Make sure `rp_handler.py` has `runpod.serverless.start()` at module level (not inside `if __name__`) and the code is pushed to the default branch |
| Cold start too slow | Enable Flash Boot; verify volume is in the same region |
| OOM | Use A100 80 GB; lower resolution to 480p; set segments to 1 |
| "No vocal detected" | Audio may not contain speech — try 16 kHz mono WAV |
| Gradio shows "app is unprotected" | Set `GRADIO_PASSWORD` environment variable |

---

## Files

| File | What it does |
|------|-------------|
| `rp_handler.py` | RunPod serverless worker (GPU) |
| `Dockerfile` | Docker image for the GPU worker (built by RunPod from GitHub) |
| `app.py` | Gradio UI with password auth (no GPU) |
| `Dockerfile.app` | Docker image for the Gradio UI (for DigitalOcean) |
| `requirements.txt` | Python deps for the GPU worker |
| `requirements-app.txt` | Python deps for the Gradio UI (`gradio`, `requests`) |
| `download_weights.py` | One-time weight download to network volume |
| `.dockerignore` | Keeps Docker images lean |
