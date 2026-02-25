#!/usr/bin/env bash
set -euo pipefail

# Quick test client for Modal endpoint.
#
# Usage:
#   export INFERENCE_API_URL="https://<workspace>--longcat-video-avatar-inference-endpoint.modal.run"
#   ./modal_test.sh
#
# Optional env vars:
#   AUDIO_FILE            (default: assets/avatar/single/man.mp3)
#   IMAGE_FILE            (default: assets/avatar/single/man.png)
#   PROMPT                (default: "A person speaking naturally")
#   STAGE                 (default: ai2v)
#   RESOLUTION            (default: 480p)
#   NUM_INFERENCE_STEPS   (default: 50)
#   TEXT_GUIDANCE_SCALE   (default: 4.0)
#   AUDIO_GUIDANCE_SCALE  (default: 4.0)
#   NUM_SEGMENTS          (default: 1)
#   SEED                  (default: 42)

INFERENCE_API_URL="${INFERENCE_API_URL:-}"
AUDIO_FILE="${AUDIO_FILE:-assets/avatar/single/man.mp3}"
IMAGE_FILE="${IMAGE_FILE:-assets/avatar/single/man.png}"
PROMPT="${PROMPT:-A person speaking naturally}"
STAGE="${STAGE:-ai2v}"
RESOLUTION="${RESOLUTION:-480p}"
NUM_INFERENCE_STEPS="${NUM_INFERENCE_STEPS:-50}"
TEXT_GUIDANCE_SCALE="${TEXT_GUIDANCE_SCALE:-4.0}"
AUDIO_GUIDANCE_SCALE="${AUDIO_GUIDANCE_SCALE:-4.0}"
NUM_SEGMENTS="${NUM_SEGMENTS:-1}"
SEED="${SEED:-42}"

if [[ -z "$INFERENCE_API_URL" ]]; then
  echo "ERROR: INFERENCE_API_URL is not set"
  echo "Example:"
  echo '  export INFERENCE_API_URL="https://<workspace>--longcat-video-avatar-inference-endpoint.modal.run"'
  exit 1
fi

if [[ ! -f "$AUDIO_FILE" ]]; then
  echo "ERROR: AUDIO_FILE not found: $AUDIO_FILE"
  exit 1
fi

if [[ "$STAGE" == "ai2v" && ! -f "$IMAGE_FILE" ]]; then
  echo "ERROR: IMAGE_FILE not found (required for ai2v): $IMAGE_FILE"
  exit 1
fi

if ! command -v python3 >/dev/null 2>&1; then
  echo "ERROR: python3 is required"
  exit 1
fi

timestamp="$(date +%Y%m%d_%H%M%S)"
payload_file="/tmp/modal_payload_${timestamp}.json"
response_file="/tmp/modal_response_${timestamp}.json"
output_video="modal_output_${timestamp}.mp4"

echo "Preparing payload..."
python3 - <<'PY' "$payload_file" "$AUDIO_FILE" "$IMAGE_FILE" "$PROMPT" "$STAGE" "$RESOLUTION" "$NUM_INFERENCE_STEPS" "$TEXT_GUIDANCE_SCALE" "$AUDIO_GUIDANCE_SCALE" "$NUM_SEGMENTS" "$SEED"
import base64
import json
import sys
from pathlib import Path

(
    payload_path,
    audio_path,
    image_path,
    prompt,
    stage,
    resolution,
    num_inference_steps,
    text_guidance_scale,
    audio_guidance_scale,
    num_segments,
    seed,
) = sys.argv[1:]

def b64(path: str) -> str:
    return base64.b64encode(Path(path).read_bytes()).decode("utf-8")

payload = {
    "audio_base64": b64(audio_path),
    "prompt": prompt,
    "stage": stage,
    "resolution": resolution,
    "num_inference_steps": int(num_inference_steps),
    "text_guidance_scale": float(text_guidance_scale),
    "audio_guidance_scale": float(audio_guidance_scale),
    "num_segments": int(num_segments),
    "seed": int(seed),
}

if stage == "ai2v":
    payload["image_base64"] = b64(image_path)

Path(payload_path).write_text(json.dumps(payload), encoding="utf-8")
print(f"Payload written: {payload_path}")
PY

echo "Sending request to Modal endpoint..."
curl -sS -X POST "$INFERENCE_API_URL" \
  -H "Content-Type: application/json" \
  -d @"$payload_file" \
  -o "$response_file"

echo "Decoding response..."
python3 - <<'PY' "$response_file" "$output_video"
import base64
import json
import sys
from pathlib import Path

response_path, output_video = sys.argv[1:]
data = json.loads(Path(response_path).read_text(encoding="utf-8"))

if "error" in data:
    raise SystemExit(f"Endpoint error: {data['error']}")

video_b64 = data.get("video_base64")
if not video_b64:
    raise SystemExit(f"No video_base64 in response. Keys: {list(data.keys())}")

Path(output_video).write_bytes(base64.b64decode(video_b64))
print(f"Saved video: {output_video}")
PY

echo "Done: $output_video"
