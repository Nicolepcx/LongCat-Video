"""
Gradio UI for LongCat-Video Avatar.

Upload an audio file and a reference image → get a talking-head video back.
Files are sent directly to RunPod as base64 — no S3, no file hosting needed.

Usage
-----
RunPod backend:
    export RUNPOD_API_KEY="rp_xxxxxxxx"
    export RUNPOD_ENDPOINT_ID="your-endpoint-id"
    python app.py

Modal backend:
    export INFERENCE_API_URL="https://<workspace>--<app>-inference-endpoint.modal.run"
    python app.py
"""

import os
import time
import base64
import tempfile
import requests
import gradio as gr

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
RUNPOD_API_KEY = os.environ.get("RUNPOD_API_KEY", "")
RUNPOD_ENDPOINT_ID = os.environ.get("RUNPOD_ENDPOINT_ID", "")
INFERENCE_API_URL = os.environ.get("INFERENCE_API_URL", "").strip()
POLL_INTERVAL = 5  # seconds between status checks


def _api_url(path: str = "") -> str:
    return f"https://api.runpod.ai/v2/{RUNPOD_ENDPOINT_ID}{path}"


def _headers() -> dict:
    return {
        "Authorization": f"Bearer {RUNPOD_API_KEY}",
        "Content-Type": "application/json",
    }


def _file_to_base64(filepath: str) -> str:
    """Read a local file and return its contents as a base64 string."""
    with open(filepath, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


# ---------------------------------------------------------------------------
# Main generation function
# ---------------------------------------------------------------------------

def generate_video(
    audio_file,
    image_file,
    prompt: str,
    stage: str,
    resolution: str,
    num_inference_steps: int,
    text_guidance_scale: float,
    audio_guidance_scale: float,
    num_segments: int,
    seed: int,
    progress=gr.Progress(track_tqdm=True),
):
    """Encode files as base64, submit RunPod job, poll, return video."""

    # --- Validate ---
    use_modal = bool(INFERENCE_API_URL)
    if not use_modal:
        if not RUNPOD_API_KEY:
            raise gr.Error("Set RUNPOD_API_KEY (RunPod mode) or INFERENCE_API_URL (Modal mode)")
        if not RUNPOD_ENDPOINT_ID:
            raise gr.Error("Set RUNPOD_ENDPOINT_ID (RunPod mode) or INFERENCE_API_URL (Modal mode)")
    if audio_file is None:
        raise gr.Error("Please upload an audio file")
    if stage == "ai2v" and image_file is None:
        raise gr.Error("ai2v mode requires a reference image")

    # --- Encode files ---
    progress(0.05, desc="Encoding files …")
    payload = {
        "audio_base64": _file_to_base64(audio_file),
        "prompt": prompt or "A person speaking naturally",
        "stage": stage,
        "resolution": resolution,
        "num_inference_steps": int(num_inference_steps),
        "text_guidance_scale": float(text_guidance_scale),
        "audio_guidance_scale": float(audio_guidance_scale),
        "num_segments": int(num_segments),
        "seed": int(seed),
    }
    if image_file is not None:
        payload["image_base64"] = _file_to_base64(image_file)

    # --- Submit / Execute job ---
    if use_modal:
        progress(0.10, desc="Submitting job to Modal endpoint …")
        r = requests.post(
            INFERENCE_API_URL,
            json=payload,
            timeout=3600,  # modal call is synchronous
        )
        r.raise_for_status()
        output = r.json()
        if "error" in output:
            raise gr.Error(f"Inference error: {output['error']}")
    else:
        progress(0.10, desc="Submitting job to RunPod …")
        resp = requests.post(
            _api_url("/run"),
            headers=_headers(),
            json={"input": payload},
            timeout=60,
        )
        resp.raise_for_status()
        job_id = resp.json()["id"]

        # --- Poll for result ---
        progress(0.15, desc=f"Job {job_id} queued — waiting for GPU worker …")
        timeout = 900  # 15 min
        start = time.time()

        while time.time() - start < timeout:
            r = requests.get(
                _api_url(f"/status/{job_id}"),
                headers=_headers(),
                timeout=30,
            )
            r.raise_for_status()
            data = r.json()
            status = data.get("status")

            if status == "COMPLETED":
                output = data.get("output", {})
                if "error" in output:
                    raise gr.Error(f"Inference error: {output['error']}")
                break

            elif status == "FAILED":
                error_msg = data.get("output", {}).get("error", "Unknown error")
                raise gr.Error(f"Job failed: {error_msg}")

            elif status == "IN_PROGRESS":
                elapsed = time.time() - start
                frac = min(0.15 + elapsed / timeout * 0.75, 0.90)
                progress(frac, desc="Generating video … (this takes a few minutes)")

            else:  # IN_QUEUE, etc.
                elapsed = time.time() - start
                frac = min(0.15 + elapsed / timeout * 0.30, 0.40)
                progress(frac, desc=f"Status: {status} — waiting for worker …")

            time.sleep(POLL_INTERVAL)
        else:
            raise gr.Error(f"Job timed out after {timeout}s")

    # --- Decode video ---
    progress(0.95, desc="Decoding video …")
    video_b64 = output.get("video_base64")
    if not video_b64:
        raise gr.Error("No video returned from endpoint")

    video_bytes = base64.b64decode(video_b64)
    tmp = tempfile.NamedTemporaryFile(suffix=".mp4", delete=False)
    tmp.write(video_bytes)
    tmp.close()

    progress(1.0, desc="Done!")
    return tmp.name


# ---------------------------------------------------------------------------
# Gradio UI
# ---------------------------------------------------------------------------

def build_ui():
    with gr.Blocks(
        title="LongCat Video Avatar",
        theme=gr.themes.Soft(),
    ) as demo:
        gr.Markdown(
            """
            # LongCat Video Avatar
            Upload an **audio clip** and a **reference image** to generate a talking-head video.
            The model runs on a RunPod GPU — this UI is just a lightweight frontend.
            """
        )

        with gr.Row():
            with gr.Column(scale=1):
                audio_input = gr.Audio(
                    label="Audio file",
                    type="filepath",
                    sources=["upload", "microphone"],
                )
                image_input = gr.Image(
                    label="Reference image (required for ai2v)",
                    type="filepath",
                )
                prompt_input = gr.Textbox(
                    label="Prompt",
                    value="A person speaking naturally with clear lip movements",
                    lines=2,
                )

                with gr.Accordion("Advanced settings", open=False):
                    stage_input = gr.Dropdown(
                        choices=["ai2v", "at2v"],
                        value="ai2v",
                        label="Stage (ai2v = image+audio → video, at2v = audio+text → video)",
                    )
                    resolution_input = gr.Dropdown(
                        choices=["480p", "720p"],
                        value="480p",
                        label="Resolution",
                    )
                    steps_input = gr.Slider(
                        minimum=10, maximum=100, value=50, step=5,
                        label="Inference steps",
                    )
                    text_cfg = gr.Slider(
                        minimum=1.0, maximum=10.0, value=4.0, step=0.5,
                        label="Text guidance scale",
                    )
                    audio_cfg = gr.Slider(
                        minimum=1.0, maximum=10.0, value=4.0, step=0.5,
                        label="Audio guidance scale",
                    )
                    segments_input = gr.Slider(
                        minimum=1, maximum=10, value=1, step=1,
                        label="Number of segments (longer video)",
                    )
                    seed_input = gr.Number(
                        value=42, label="Seed", precision=0,
                    )

                generate_btn = gr.Button("Generate Video", variant="primary", size="lg")

            with gr.Column(scale=1):
                video_output = gr.Video(label="Generated Video")

                gr.Markdown(
                    """
                    ### How it works
                    1. Your audio & image are sent directly to the inference API (as base64)
                    2. A GPU worker picks up the job (cold start if scaled to zero)
                    3. Inference runs (~10-20 min depending on settings)
                    4. The video is sent back to your browser

                    **Tip:** For RunPod mode set `RUNPOD_API_KEY` + `RUNPOD_ENDPOINT_ID`.
                    For Modal mode set `INFERENCE_API_URL`.
                    """
                )

        generate_btn.click(
            fn=generate_video,
            inputs=[
                audio_input, image_input, prompt_input,
                stage_input, resolution_input, steps_input,
                text_cfg, audio_cfg, segments_input, seed_input,
            ],
            outputs=video_output,
        )

    return demo


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    # Password protection via environment variables
    username = os.environ.get("GRADIO_USERNAME", "admin")
    password = os.environ.get("GRADIO_PASSWORD")

    auth = (username, password) if password else None
    if not auth:
        print("WARNING: No GRADIO_PASSWORD set — app is unprotected!")

    demo = build_ui()
    demo.launch(
        server_name="0.0.0.0",
        server_port=int(os.environ.get("PORT", 7860)),
        share=False,
        auth=auth,
    )
