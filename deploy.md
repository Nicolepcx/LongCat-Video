# Deploying LongCat-Video Avatar on RunPod (Serverless)

End-to-end guide: build the Docker image, set up model weights on a RunPod
Network Volume, create a serverless endpoint that scales to zero, and run
the Gradio UI to upload audio + image and get a video back.

---

## Architecture

```
┌───────────────────────────┐
│   Gradio UI  (app.py)     │   ← No GPU — runs locally or on a $5/mo host
│   Upload audio + image    │
└────────────┬──────────────┘
             │  audio & image sent as base64
             │  via RunPod REST API
             ▼
┌───────────────────────────┐       ┌──────────────────────────────┐
│  RunPod Serverless        │◄─────►│  Network Volume              │
│  Endpoint                 │       │  /runpod-volume/weights/      │
│  (scales 0 → N GPUs)     │       │   ├── LongCat-Video/         │
│                           │       │   └── LongCat-Video-Avatar/  │
│  Returns video as base64  │       └──────────────────────────────┘
└───────────────────────────┘
```

**No S3, no file hosting** — audio and image files are encoded as base64
and sent directly in the job payload. The generated video comes back as
base64 in the response.

---

## Prerequisites

| Tool | Purpose |
|------|---------|
| [Docker](https://docs.docker.com/get-docker/) | Build the worker image |
| [RunPod account](https://runpod.io) | Serverless GPU + network volume |
| Docker Hub account (or GHCR / ECR) | Push the built image |
| HuggingFace account | Download model weights |

Create a RunPod API key at **RunPod Console → Settings → API Keys**.

---

## Step 1 — Build & Push the Docker Image

```bash
# Build (flash-attn compilation takes ~10-20 min)
docker build -t <your-dockerhub-user>/longcat-video-avatar:latest .

# Push
docker push <your-dockerhub-user>/longcat-video-avatar:latest
```

**Optional local test** (needs an NVIDIA GPU + downloaded weights):

```bash
docker run --gpus all \
  -v /path/to/weights:/runpod-volume/weights \
  -p 8000:8000 \
  <your-dockerhub-user>/longcat-video-avatar:latest
```

---

## Step 2 — Create a RunPod Network Volume

1. **RunPod Console → Storage → Network Volumes → Create**
2. Name: `longcat-weights`
3. Region: **same region** as the endpoint you'll create
4. Size: **100 GB** (weights ≈ 60 GB + headroom)

---

## Step 3 — Download Weights onto the Volume

Spin up a temporary pod with the volume attached:

1. **RunPod Console → Pods → Deploy**
2. Pick any GPU (or even CPU — it's just downloading)
3. Attach volume `longcat-weights` (mounts at `/runpod-volume`)
4. Open a terminal in the pod

```bash
pip install huggingface_hub

# Option A: clone repo and use the helper script
git clone https://github.com/<your-user>/LongCat-Video.git /tmp/repo
cd /tmp/repo
python download_weights.py --output_dir /runpod-volume/weights

# Option B: quick inline download
python -c "
from huggingface_hub import snapshot_download
snapshot_download('meituan-longcat/LongCat-Video',
                  local_dir='/runpod-volume/weights/LongCat-Video',
                  local_dir_use_symlinks=False)
snapshot_download('meituan-longcat/LongCat-Video-Avatar',
                  local_dir='/runpod-volume/weights/LongCat-Video-Avatar',
                  local_dir_use_symlinks=False)
"
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

**Terminate the temporary pod** — the volume persists on its own.

---

## Step 4 — Create the Serverless Endpoint

1. **RunPod Console → Serverless → Endpoints → New Endpoint**
2. Configure:

| Setting | Value |
|---------|-------|
| **Name** | `longcat-avatar` |
| **Docker Image** | `<your-dockerhub-user>/longcat-video-avatar:latest` |
| **GPU Type** | A100 80 GB (recommended) or A40 / L40S (48 GB min) |
| **Network Volume** | `longcat-weights` |
| **Min Workers** | `0` ← scale to zero |
| **Max Workers** | `1` (or more for concurrency) |
| **Idle Timeout** | `300` s (5 min) |
| **Execution Timeout** | `900` s (15 min) |
| **Flash Boot** | Enabled |

3. Click **Create Endpoint**
4. Note down the **Endpoint ID** (e.g. `abc123xyz`)

---

## Step 5 — Run the Gradio UI

The Gradio app (`app.py`) needs only `gradio` and `requests` — no GPU,
no torch, no CUDA. Run it anywhere.

```bash
pip install gradio requests

export RUNPOD_API_KEY="rp_xxxxxxxx"
export RUNPOD_ENDPOINT_ID="abc123xyz"

python app.py
```

Open **http://localhost:7860** → upload audio + image → click Generate → get video.

### Deploy the UI on a cheap host

The Gradio app is stateless and tiny. Any of these work:

| Platform | Cost | Run command |
|----------|------|-------------|
| **Local** | free | `python app.py` |
| **DigitalOcean App Platform** | ~$5/mo | Set env vars, run `python app.py` |
| **Railway** | ~$5/mo | Same |
| **Fly.io** | ~$3/mo | Same |
| **Hugging Face Spaces** | free | Push repo, set secrets |

Set `RUNPOD_API_KEY` and `RUNPOD_ENDPOINT_ID` as environment variables
on whichever platform you choose.

---

## Step 6 — Test via curl / Python (without Gradio)

### curl

```bash
export RUNPOD_API_KEY="rp_xxxxxxxx"
export ENDPOINT_ID="abc123xyz"

# Encode files
AUDIO_B64=$(base64 -i speech.wav)
IMAGE_B64=$(base64 -i face.png)

# Submit job
JOB_ID=$(curl -s -X POST "https://api.runpod.ai/v2/${ENDPOINT_ID}/run" \
  -H "Authorization: Bearer ${RUNPOD_API_KEY}" \
  -H "Content-Type: application/json" \
  -d "{
    \"input\": {
      \"audio_base64\": \"${AUDIO_B64}\",
      \"image_base64\": \"${IMAGE_B64}\",
      \"prompt\": \"A person speaking naturally\",
      \"stage\": \"ai2v\"
    }
  }" | python -c "import sys,json; print(json.load(sys.stdin)['id'])")

echo "Job submitted: ${JOB_ID}"

# Poll until done
while true; do
  STATUS=$(curl -s "https://api.runpod.ai/v2/${ENDPOINT_ID}/status/${JOB_ID}" \
    -H "Authorization: Bearer ${RUNPOD_API_KEY}" \
    | python -c "import sys,json; print(json.load(sys.stdin)['status'])")
  echo "Status: ${STATUS}"
  [ "$STATUS" = "COMPLETED" ] && break
  [ "$STATUS" = "FAILED" ] && echo "FAILED" && exit 1
  sleep 5
done

# Download result
curl -s "https://api.runpod.ai/v2/${ENDPOINT_ID}/status/${JOB_ID}" \
  -H "Authorization: Bearer ${RUNPOD_API_KEY}" \
  | python -c "
import sys, json, base64
data = json.load(sys.stdin)
video = base64.b64decode(data['output']['video_base64'])
with open('output.mp4', 'wb') as f:
    f.write(video)
print(f'Saved output.mp4 ({len(video):,} bytes)')
"
```

### Python

```python
import base64, time, requests

API_KEY = "rp_xxxxxxxx"
ENDPOINT_ID = "abc123xyz"
BASE = f"https://api.runpod.ai/v2/{ENDPOINT_ID}"
HEADERS = {"Authorization": f"Bearer {API_KEY}", "Content-Type": "application/json"}

# Encode files
with open("speech.wav", "rb") as f:
    audio_b64 = base64.b64encode(f.read()).decode()
with open("face.png", "rb") as f:
    image_b64 = base64.b64encode(f.read()).decode()

# Submit
resp = requests.post(f"{BASE}/run", headers=HEADERS, json={"input": {
    "audio_base64": audio_b64,
    "image_base64": image_b64,
    "prompt": "A person speaking naturally",
    "stage": "ai2v",
}})
job_id = resp.json()["id"]
print(f"Job: {job_id}")

# Poll
while True:
    r = requests.get(f"{BASE}/status/{job_id}", headers=HEADERS).json()
    print(f"  {r['status']}")
    if r["status"] == "COMPLETED":
        break
    if r["status"] == "FAILED":
        raise RuntimeError(r)
    time.sleep(5)

# Save video
video = base64.b64decode(r["output"]["video_base64"])
with open("output.mp4", "wb") as f:
    f.write(video)
print(f"Saved output.mp4 ({len(video):,} bytes)")
```

---

## How Scale-to-Zero Works

```
Job arrives    Worker starts    Inference        Worker idle      Worker stops
               (cold ~2-3 min)  (~3-10 min)      (5 min timeout)
    │              │                │                │                │
    ▼              ▼                ▼                ▼                ▼
   QUEUE ──────► RUNNING ───────► IDLE ──────────► ZERO ────────────►
    $0/sec       $X.XX/sec       $X.XX/sec         $0/sec
```

- **Min workers = 0** → no cost when nobody is using it
- **Idle timeout = 5 min** → worker stays warm for quick follow-ups
- **Flash Boot** → container image is cached, so cold starts are as fast as
  possible (model loading from network volume dominates at ~2-3 min)
- **Network volume** → weights are pre-loaded, not downloaded each time

**Cost per video** (A100 80 GB ≈ $0.0012/sec):

| Phase | Duration | Cost |
|-------|----------|------|
| Cold start (model loading) | ~2-3 min | ~$0.14-0.22 |
| Inference (1 segment, 480p) | ~3-5 min | ~$0.22-0.36 |
| **Total** | **~5-8 min** | **~$0.36-0.58** |
| When idle | — | **$0.00** |

Network volume storage: ~$0.07/GB/month × 100 GB = ~$7/month.

---

## API Reference

### Input (`job["input"]`)

| Field | Type | Required | Default | Description |
|-------|------|----------|---------|-------------|
| `audio_base64` | string | **yes** | — | Base64-encoded audio file (.wav / .mp3) |
| `image_base64` | string | ai2v only | — | Base64-encoded reference image |
| `prompt` | string | no | "A person speaking naturally" | Text prompt |
| `negative_prompt` | string | no | (built-in) | Negative prompt |
| `stage` | string | no | `"ai2v"` | `"ai2v"` or `"at2v"` |
| `resolution` | string | no | `"480p"` | `"480p"` or `"720p"` |
| `num_inference_steps` | int | no | `50` | Denoising steps |
| `text_guidance_scale` | float | no | `4.0` | Text conditioning strength |
| `audio_guidance_scale` | float | no | `4.0` | Audio conditioning strength |
| `num_segments` | int | no | `1` | More segments = longer video |
| `seed` | int | no | `42` | Random seed |

### Output

```json
{ "video_base64": "<base64-encoded MP4>" }
```

On error:

```json
{ "error": "description of what went wrong" }
```

---

## Troubleshooting

| Problem | Fix |
|---------|-----|
| Cold start too slow | Enable **Flash Boot**; check volume is in same region |
| OOM | Use A100 80 GB; lower resolution to 480p; reduce segments |
| "No vocal detected" | Audio may not contain speech — convert to 16 kHz mono WAV first |
| Worker keeps restarting | Check logs for import errors; verify Docker build completed |
| Payload too large | Audio files >10 MB may hit RunPod limits — trim or compress first |

---

## File Overview

| File | Description |
|------|-------------|
| `rp_handler.py` | RunPod serverless worker (GPU container entry point) |
| `app.py` | Gradio UI (lightweight, no GPU needed) |
| `Dockerfile` | Worker container image |
| `download_weights.py` | One-time weight download to network volume |
| `requirements.txt` | Python dependencies for the worker |
| `.dockerignore` | Keeps Docker image lean |
