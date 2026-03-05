"""
Modal deployment for LongCat-Video Avatar.

This module provides:
1) `download_weights` to populate a persistent Modal Volume once
2) `generate` GPU method to run inference
3) `inference_endpoint` HTTP endpoint accepting base64 audio/image payloads

API payload (JSON):
{
  "audio_base64": "...",                # required
  "image_base64": "...",                # required when stage == "ai2v"
  "prompt": "A person speaking naturally",
  "negative_prompt": null,
  "stage": "ai2v",                      # "ai2v" | "at2v"
  "resolution": "480p",                 # "480p" | "720p"
  "num_inference_steps": 50,
  "text_guidance_scale": 4.0,
  "audio_guidance_scale": 4.0,
  "num_segments": 1,
  "seed": 42
}
"""

from __future__ import annotations

import base64
import datetime
import os
import random
import shutil
import subprocess
import tempfile
import time
from collections import deque
from pathlib import Path

import modal


APP_NAME = "longcat-video-avatar"
VOLUME_NAME = os.environ.get("MODAL_WEIGHTS_VOLUME", "longcat-weights")
VOLUME_MOUNT_PATH = "/weights"
WEIGHTS_DIR = f"{VOLUME_MOUNT_PATH}/weights"
OUTPUT_DIR = "/tmp/outputs"


def _image() -> modal.Image:
    # Strict parity path: mirror the working RunPod container/setup sequence.
    return (
        modal.Image.from_registry("runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04")
        .add_local_file("requirements.txt", "/tmp/requirements.txt", copy=True)
        .add_local_file("requirements_avatar.txt", "/tmp/requirements_avatar.txt", copy=True)
        .run_commands(
            # Follow notebook-proven install flow, avoiding streamlit/blinker conflicts.
            "pip install -U pip",
            "pip install torch==2.6.0+cu124 torchvision==0.21.0+cu124 torchaudio==2.6.0 --index-url https://download.pytorch.org/whl/cu124",
            "pip install -U packaging ninja",
            "pip install flash_attn==2.7.4.post1 --no-build-isolation",
            "pip install -r /tmp/requirements_avatar.txt",
            # Notebook overrides that matched your successful RunPod behavior.
            "pip install -U diffusers==0.32.2 transformers==4.46.3 accelerate safetensors peft einops librosa soundfile pyloudnorm audio-separator onnxruntime imageio imageio-ffmpeg av opencv-python loguru ftfy psutil numpy==1.26.4",
            "pip install 'fastapi[standard]>=0.115.0'",
            # Ensure runtime deps used by audio-separator path are present.
            "pip install onnx2torch pydub sentencepiece protobuf tiktoken regex",
        )
        .workdir("/root/LongCat-Video")
        # Add project source last so Modal can mount local files at container startup.
        .add_local_dir(".", remote_path="/root/LongCat-Video")
    )


app = modal.App(APP_NAME)
image = _image()
weights_volume = modal.Volume.from_name(VOLUME_NAME, create_if_missing=True)


def _generate_random_uid() -> str:
    ts = str(int(time.time()))[-6:]
    rnd = str(random.randint(100000, 999999))
    return ts + rnd


@app.function(
    image=image,
    volumes={VOLUME_MOUNT_PATH: weights_volume},
    timeout=60 * 60 * 8,  # 8 hours — large downloads can be slow on Modal CPU containers
)
def download_weights(hf_token: str | None = None, force: bool = False) -> dict:
    """
    One-time weight download into Modal Volume.

    Downloads each model separately and commits after each one so that
    partial progress is preserved if the container is interrupted.

    Usage:
      modal run modal_app.py::download_weights --hf-token YOUR_TOKEN
    """
    from huggingface_hub import snapshot_download

    os.makedirs(WEIGHTS_DIR, exist_ok=True)
    os.environ["HF_HOME"] = f"{VOLUME_MOUNT_PATH}/hf_cache"
    os.makedirs(os.environ["HF_HOME"], exist_ok=True)

    if hf_token:
        os.environ["HF_TOKEN"] = hf_token

    base_dir = f"{WEIGHTS_DIR}/LongCat-Video"
    avatar_dir = f"{WEIGHTS_DIR}/LongCat-Video-Avatar"

    # Thorough completeness checks (not just one directory)
    def _base_model_complete():
        """Check that all critical subdirs of LongCat-Video have real files."""
        checks = [
            f"{base_dir}/dit/diffusion_pytorch_model-00001-of-00006.safetensors",
            f"{base_dir}/vae/diffusion_pytorch_model.safetensors",
            f"{base_dir}/scheduler/scheduler_config.json",
        ]
        # tokenizer: look for any .model file (spiece.model / tokenizer.model)
        tok_dir = f"{base_dir}/tokenizer"
        tok_ok = False
        if os.path.isdir(tok_dir):
            for f in os.listdir(tok_dir):
                if f.endswith(".model") or f == "tokenizer.json":
                    tok_ok = True
                    break
        # text_encoder: look for at least one safetensors shard
        te_dir = f"{base_dir}/text_encoder"
        te_ok = False
        if os.path.isdir(te_dir):
            for f in os.listdir(te_dir):
                if f.endswith(".safetensors"):
                    te_ok = True
                    break
        files_ok = all(os.path.exists(c) for c in checks)
        return files_ok and tok_ok and te_ok

    def _avatar_model_complete():
        checks = [
            f"{avatar_dir}/avatar_single/diffusion_pytorch_model-00001-of-00006.safetensors",
            f"{avatar_dir}/chinese-wav2vec2-base/pytorch_model.bin",
        ]
        return all(os.path.exists(c) for c in checks)

    if force and os.path.exists(base_dir):
        shutil.rmtree(base_dir, ignore_errors=True)
    if force and os.path.exists(avatar_dir):
        shutil.rmtree(avatar_dir, ignore_errors=True)

    # --- Model 1: LongCat-Video (~83 GB) ---
    if _base_model_complete() and not force:
        print("✓ LongCat-Video already complete — skipping")
    else:
        print("Downloading LongCat-Video (~83 GB) ...")
        if os.path.exists(base_dir):
            print("  (Removing incomplete previous download)")
            shutil.rmtree(base_dir, ignore_errors=True)
        snapshot_download(
            repo_id="meituan-longcat/LongCat-Video",
            local_dir=base_dir,
            token=hf_token,
        )
        shutil.rmtree(os.environ["HF_HOME"], ignore_errors=True)
        os.makedirs(os.environ["HF_HOME"], exist_ok=True)
        print("Committing LongCat-Video to volume ...")
        weights_volume.commit()
        print("✓ LongCat-Video saved to volume")

    # --- Model 2: LongCat-Video-Avatar (~120 GB) ---
    if _avatar_model_complete() and not force:
        print("✓ LongCat-Video-Avatar already complete — skipping")
    else:
        print("Downloading LongCat-Video-Avatar (~120 GB) ...")
        if os.path.exists(avatar_dir):
            print("  (Removing incomplete previous download)")
            shutil.rmtree(avatar_dir, ignore_errors=True)
        snapshot_download(
            repo_id="meituan-longcat/LongCat-Video-Avatar",
            local_dir=avatar_dir,
            token=hf_token,
        )
        shutil.rmtree(os.environ["HF_HOME"], ignore_errors=True)
        print("Committing LongCat-Video-Avatar to volume ...")
        weights_volume.commit()
        print("✓ LongCat-Video-Avatar saved to volume")

    # Final cleanup of any remaining cache
    hf_cache = os.environ.get("HF_HOME", "")
    if hf_cache and os.path.exists(hf_cache):
        shutil.rmtree(hf_cache, ignore_errors=True)
        weights_volume.commit()

    return {
        "weights_dir": WEIGHTS_DIR,
        "base_model_size": os.popen(f"du -sh {base_dir} | cut -f1").read().strip(),
        "avatar_model_size": os.popen(f"du -sh {avatar_dir} | cut -f1").read().strip(),
    }


@app.cls(
    image=image,
    gpu="A100-80GB",
    memory=262144,  # 256 GB system RAM — needed to load ~200GB of weights
    timeout=60 * 60,
    scaledown_window=60,  # scale down GPU quickly after each request
    volumes={VOLUME_MOUNT_PATH: weights_volume},
)
class AvatarInference:
    @modal.enter()
    def load(self):
        self.base_model_dir = os.path.join(WEIGHTS_DIR, "LongCat-Video")
        self.avatar_weights_dir = os.path.join(WEIGHTS_DIR, "LongCat-Video-Avatar")
        self.script_path = "/root/LongCat-Video/run_demo_avatar_single_audio_to_video.py"
        if not os.path.exists(self.base_model_dir) or not os.path.exists(self.avatar_weights_dir):
            raise RuntimeError(f"Missing weights in {WEIGHTS_DIR}. Run download_weights first.")
        if not os.path.exists(self.script_path):
            raise RuntimeError(f"Missing script: {self.script_path}")
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

    def _generate_impl(self, payload: dict) -> dict:
        req_tmp_dir = None
        out_dir = None
        video_path = None
        try:
            if "audio_base64" not in payload:
                return {"error": "Missing required field: audio_base64"}

            stage = payload.get("stage", "ai2v")
            if stage == "ai2v" and not payload.get("image_base64"):
                return {"error": "ai2v stage requires image_base64"}

            uid = _generate_random_uid()
            req_tmp_dir = tempfile.mkdtemp(prefix=f"modal_req_{uid}_")
            out_dir = os.path.join(OUTPUT_DIR, uid)
            os.makedirs(out_dir, exist_ok=True)

            audio_ext = payload.get("audio_ext") or ".wav"
            if not audio_ext.startswith("."):
                audio_ext = f".{audio_ext}"
            audio_path = os.path.join(req_tmp_dir, f"input_audio{audio_ext}")
            with open(audio_path, "wb") as f:
                f.write(base64.b64decode(payload["audio_base64"]))

            image_path = None
            if payload.get("image_base64"):
                image_path = os.path.join(req_tmp_dir, "input_image.png")
                with open(image_path, "wb") as f:
                    f.write(base64.b64decode(payload["image_base64"]))

            stage_1 = payload.get("stage", "ai2v")
            if stage_1 not in {"ai2v", "at2v"}:
                return {"error": f"Unsupported stage: {stage_1}"}

            prompt = payload.get("prompt", "A person speaking naturally")
            input_json_data = {
                "prompt": prompt,
                "cond_audio": {"person1": audio_path},
            }
            if stage_1 == "ai2v":
                if not image_path:
                    return {"error": "ai2v stage requires image_base64"}
                input_json_data["cond_image"] = image_path

            input_json_path = os.path.join(req_tmp_dir, "input.json")
            with open(input_json_path, "w", encoding="utf-8") as f:
                import json
                json.dump(input_json_data, f)

            env = os.environ.copy()
            env["RANK"] = "0"
            env["WORLD_SIZE"] = "1"
            env["LOCAL_RANK"] = "0"
            env["MASTER_ADDR"] = "127.0.0.1"
            env["MASTER_PORT"] = str(29500 + random.randint(0, 1000))
            env["BASE_MODEL_DIR"] = self.base_model_dir
            env["AVATAR_WEIGHTS_DIR"] = self.avatar_weights_dir
            env.setdefault("TOKENIZERS_PARALLELISM", "false")
            env.setdefault("PYTHONUNBUFFERED", "1")

            cmd = [
                "python",
                self.script_path,
                "--input_json", input_json_path,
                "--output_dir", out_dir,
                "--resolution", payload.get("resolution", "480p"),
                "--num_inference_steps", str(int(payload.get("num_inference_steps", 50))),
                "--text_guidance_scale", str(float(payload.get("text_guidance_scale", 4.0))),
                "--audio_guidance_scale", str(float(payload.get("audio_guidance_scale", 4.0))),
                "--num_segments", str(max(1, int(payload.get("num_segments", 1)))),
                "--stage_1", stage_1,
                "--context_parallel_size", "1",
                "--base_model_dir", self.base_model_dir,
                "--checkpoint_dir", self.avatar_weights_dir,
            ]

            log_tail = deque(maxlen=120)
            proc = subprocess.Popen(
                cmd,
                env=env,
                cwd="/root/LongCat-Video",
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
            )
            start_t = time.time()
            try:
                assert proc.stdout is not None
                for line in proc.stdout:
                    print(line.rstrip(), flush=True)
                    log_tail.append(line.rstrip())
                    if time.time() - start_t > 60 * 60 * 3:
                        proc.kill()
                        return {"error": "Script timed out after 3 hours"}
                rc = proc.wait()
            except Exception:
                proc.kill()
                rc = proc.wait()

            if rc != 0:
                tail_out = "\n".join(log_tail)
                return {
                    "error": (
                        f"Script failed with code {rc}\n"
                        f"--- combined log tail ---\n{tail_out}"
                    )
                }

            num_segments = max(1, int(payload.get("num_segments", 1)))
            if num_segments > 1:
                video_path = os.path.join(out_dir, f"video_continue_{num_segments}.mp4")
            elif stage_1 == "at2v":
                video_path = os.path.join(out_dir, "at2v_demo_1.mp4")
            else:
                video_path = os.path.join(out_dir, "ai2v_demo_1.mp4")

            if not os.path.exists(video_path):
                candidates = sorted(Path(out_dir).glob("*.mp4"), key=lambda p: p.stat().st_mtime)
                if not candidates:
                    return {"error": f"No output mp4 found in {out_dir}"}
                video_path = str(candidates[-1])

            with open(video_path, "rb") as f:
                video_b64 = base64.b64encode(f.read()).decode("utf-8")
            return {"video_base64": video_b64}

        except Exception as exc:
            return {"error": str(exc)}
        finally:
            if req_tmp_dir and os.path.exists(req_tmp_dir):
                shutil.rmtree(req_tmp_dir, ignore_errors=True)
            if video_path and os.path.exists(video_path):
                shutil.rmtree(os.path.dirname(video_path), ignore_errors=True)


    @modal.method()
    def generate(self, payload: dict) -> dict:
        return self._generate_impl(payload)

    @modal.fastapi_endpoint(method="POST")
    def inference_endpoint(self, payload: dict):
        """
        HTTP endpoint to call from Gradio/frontend.
        Runs directly on the warm GPU class instance.
        """
        return self._generate_impl(payload)


@app.local_entrypoint()
def main(
    download: bool = False,
    hf_token: str = "",
    force_download: bool = False,
):
    if download:
        result = download_weights.remote(hf_token=hf_token or None, force=force_download)
        print(result)
    else:
        print("Deploy with: modal deploy modal_app.py")
        print("Run one-time download with:")
        print("  modal run modal_app.py --download --hf-token <hf_token>")
