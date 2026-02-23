"""
RunPod Serverless Handler for LongCat-Video Avatar inference.

This module is the container entry-point for RunPod serverless workers.
On cold start it loads all model weights from the path given by the
WEIGHTS_DIR environment variable (default: /runpod-volume/weights) and
keeps them in GPU memory. Each incoming job is processed by the handler()
function which accepts audio + image inputs and returns the generated video.

Environment variables
---------------------
WEIGHTS_DIR          Root directory that contains both LongCat-Video/ and
                     LongCat-Video-Avatar/ sub-folders.  (default: /runpod-volume/weights)
OUTPUT_DIR           Where generated videos are written.  (default: /tmp/outputs)
"""

import os
import io
import sys
import json
import math
import time
import base64
import random
import shutil
import tempfile
import datetime
import traceback
from pathlib import Path

import runpod

import numpy as np
import PIL.Image
import torch
import torch.distributed as dist

# ---------------------------------------------------------------------------
# Resolve weight paths from env
# ---------------------------------------------------------------------------
WEIGHTS_DIR = os.environ.get("WEIGHTS_DIR", "/runpod-volume/weights")
BASE_MODEL_DIR = os.path.join(WEIGHTS_DIR, "LongCat-Video")
AVATAR_WEIGHTS_DIR = os.path.join(WEIGHTS_DIR, "LongCat-Video-Avatar")
OUTPUT_DIR = os.environ.get("OUTPUT_DIR", "/tmp/outputs")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ---------------------------------------------------------------------------
# Bootstrap torch.distributed for single-GPU (avoids needing torchrun)
# ---------------------------------------------------------------------------
os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
os.environ.setdefault("MASTER_PORT", "29500")
os.environ.setdefault("RANK", "0")
os.environ.setdefault("WORLD_SIZE", "1")
os.environ.setdefault("LOCAL_RANK", "0")

if not dist.is_initialized():
    dist.init_process_group(
        backend="nccl",
        timeout=datetime.timedelta(seconds=3600),
    )

LOCAL_RANK = int(os.environ.get("LOCAL_RANK", 0))
torch.cuda.set_device(LOCAL_RANK)

# ---------------------------------------------------------------------------
# Imports that depend on torch / project modules
# ---------------------------------------------------------------------------
import librosa
from transformers import AutoTokenizer, UMT5EncoderModel, Wav2Vec2FeatureExtractor
from diffusers.utils import load_image
from audio_separator.separator import Separator

from longcat_video.pipeline_longcat_video_avatar import LongCatVideoAvatarPipeline
from longcat_video.modules.scheduling_flow_match_euler_discrete import FlowMatchEulerDiscreteScheduler
from longcat_video.modules.autoencoder_kl_wan import AutoencoderKLWan
from longcat_video.modules.avatar.longcat_video_dit_avatar import LongCatVideoAvatarTransformer3DModel
from longcat_video.context_parallel import context_parallel_util
from longcat_video.audio_process.wav2vec2 import Wav2Vec2ModelWrapper
from longcat_video.audio_process.torch_utils import save_video_ffmpeg

# ---------------------------------------------------------------------------
# Helper utilities
# ---------------------------------------------------------------------------

def torch_gc():
    torch.cuda.empty_cache()
    torch.cuda.ipc_collect()


def generate_random_uid():
    ts = str(int(time.time()))[-6:]
    rnd = str(random.randint(100000, 999999))
    return ts + rnd


def extract_vocal_from_speech(source_path, target_path, vocal_separator, audio_output_dir_temp):
    outputs = vocal_separator.separate(source_path)
    if len(outputs) <= 0:
        print("Audio separate failed. Using raw audio.")
        return None
    default_vocal_path = audio_output_dir_temp / "vocals" / outputs[0]
    default_vocal_path = default_vocal_path.resolve().as_posix()
    shutil.move(default_vocal_path, target_path)
    return target_path


# ---------------------------------------------------------------------------
# Model loading (runs once on cold start)
# ---------------------------------------------------------------------------
print("[rp_handler] Loading models …")
_load_start = time.time()

context_parallel_util.init_context_parallel(
    context_parallel_size=1,
    global_rank=dist.get_rank(),
    world_size=dist.get_world_size(),
)
cp_split_hw = context_parallel_util.get_optimal_split(1)

tokenizer = AutoTokenizer.from_pretrained(
    BASE_MODEL_DIR, subfolder="tokenizer", torch_dtype=torch.bfloat16
)
text_encoder = UMT5EncoderModel.from_pretrained(
    BASE_MODEL_DIR, subfolder="text_encoder", torch_dtype=torch.bfloat16
)
vae = AutoencoderKLWan.from_pretrained(
    BASE_MODEL_DIR, subfolder="vae", torch_dtype=torch.bfloat16
)
scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(
    BASE_MODEL_DIR, subfolder="scheduler", torch_dtype=torch.bfloat16
)
dit = LongCatVideoAvatarTransformer3DModel.from_pretrained(
    AVATAR_WEIGHTS_DIR, subfolder="avatar_single",
    cp_split_hw=cp_split_hw, torch_dtype=torch.bfloat16,
)

wav2vec_path = os.path.join(AVATAR_WEIGHTS_DIR, "chinese-wav2vec2-base")
audio_encoder = Wav2Vec2ModelWrapper(wav2vec_path).to(LOCAL_RANK)
audio_encoder.feature_extractor._freeze_parameters()
wav2vec_feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(
    wav2vec_path, local_files_only=True
)

# Vocal separator
vocal_separator_path = os.path.join(AVATAR_WEIGHTS_DIR, "vocal_separator/Kim_Vocal_2.onnx")
audio_output_dir_temp = Path("/tmp/audio_temp")
audio_output_dir_temp.mkdir(parents=True, exist_ok=True)
vocal_separator = Separator(
    output_dir=audio_output_dir_temp / "vocals",
    output_single_stem="vocals",
    model_file_dir=os.path.dirname(vocal_separator_path),
)
vocal_separator.load_model(os.path.basename(vocal_separator_path))

# Build pipeline
pipe = LongCatVideoAvatarPipeline(
    tokenizer=tokenizer,
    text_encoder=text_encoder,
    vae=vae,
    scheduler=scheduler,
    dit=dit,
    audio_encoder=audio_encoder,
    wav2vec_feature_extractor=wav2vec_feature_extractor,
)
pipe.to(LOCAL_RANK)

print(f"[rp_handler] Models loaded in {time.time() - _load_start:.1f}s")


# ---------------------------------------------------------------------------
# Inference logic
# ---------------------------------------------------------------------------

def run_inference(
    audio_path: str,
    image_path: str | None,
    prompt: str,
    negative_prompt: str | None = None,
    stage: str = "ai2v",
    resolution: str = "480p",
    num_inference_steps: int = 50,
    text_guidance_scale: float = 4.0,
    audio_guidance_scale: float = 4.0,
    num_segments: int = 1,
    seed: int = 42,
) -> str:
    """Run avatar inference and return path to the generated .mp4 file."""

    if negative_prompt is None:
        negative_prompt = (
            "Close-up, Bright tones, overexposed, static, blurred details, subtitles, "
            "style, works, paintings, images, static, overall gray, worst quality, low quality, "
            "JPEG compression residue, ugly, incomplete, extra fingers, poorly drawn hands, "
            "poorly drawn faces, deformed, disfigured, misshapen limbs, fused fingers, "
            "still picture, messy background, three legs, many people in the background, "
            "walking backwards"
        )

    save_fps = 16
    num_frames = 93
    num_cond_frames = 13
    audio_stride = 2

    if resolution == "480p":
        height, width = 480, 832
    elif resolution == "720p":
        height, width = 768, 1280
    else:
        height, width = 480, 832

    num_segments = max(1, num_segments)

    generator = torch.Generator(device=LOCAL_RANK)
    generator.manual_seed(seed)

    # --- vocal extraction & audio embedding --------------------------------
    temp_vocal_path = extract_vocal_from_speech(
        audio_path,
        f"/tmp/temp_speech_{generate_random_uid()}_vocal.wav",
        vocal_separator,
        audio_output_dir_temp,
    )
    assert temp_vocal_path is not None and os.path.exists(temp_vocal_path), "No vocal detected"

    generate_duration = (
        num_frames / save_fps
        + (num_segments - 1) * (num_frames - num_cond_frames) / save_fps
    )
    speech_array, sr = librosa.load(temp_vocal_path, sr=16000)
    source_duration = len(speech_array) / sr
    added_samples = math.ceil((generate_duration - source_duration) * sr)
    if added_samples > 0:
        speech_array = np.append(speech_array, [0.0] * added_samples)

    full_audio_emb = pipe.get_audio_embedding(
        speech_array, fps=save_fps * audio_stride, device=LOCAL_RANK, sample_rate=sr
    )
    if torch.isnan(full_audio_emb).any():
        raise ValueError("Broken audio embedding with NaN values")

    if os.path.exists(temp_vocal_path):
        os.remove(temp_vocal_path)

    # first clip audio embedding
    indices = torch.arange(2 * 2 + 1) - 2
    audio_start_idx = 0
    audio_end_idx = audio_start_idx + audio_stride * num_frames
    center_indices = (
        torch.arange(audio_start_idx, audio_end_idx, audio_stride).unsqueeze(1)
        + indices.unsqueeze(0)
    )
    center_indices = torch.clamp(center_indices, min=0, max=full_audio_emb.shape[0] - 1)
    audio_emb = full_audio_emb[center_indices][None, ...].to(LOCAL_RANK)

    # --- stage 1 generation ------------------------------------------------
    uid = generate_random_uid()
    out_dir = os.path.join(OUTPUT_DIR, uid)
    os.makedirs(out_dir, exist_ok=True)

    if stage == "at2v":
        output_tuple = pipe.generate_at2v(
            prompt=prompt,
            negative_prompt=negative_prompt,
            height=height,
            width=width,
            num_frames=num_frames,
            num_inference_steps=num_inference_steps,
            text_guidance_scale=text_guidance_scale,
            audio_guidance_scale=audio_guidance_scale,
            generator=generator,
            output_type="both",
            audio_emb=audio_emb,
        )
        output, latent = output_tuple
        output = output[0]
        video = [(output[i] * 255).astype(np.uint8) for i in range(output.shape[0])]
        video = [PIL.Image.fromarray(img) for img in video]

        output_tensor = torch.from_numpy(np.array(video))
        save_video_ffmpeg(
            output_tensor,
            os.path.join(out_dir, "at2v_demo_1"),
            audio_path,
            fps=save_fps,
            quality=5,
        )
        del output
        torch_gc()

    elif stage == "ai2v":
        assert image_path is not None, "ai2v stage requires an image"
        image = load_image(image_path)
        output_tuple = pipe.generate_ai2v(
            image=image,
            prompt=prompt,
            negative_prompt=negative_prompt,
            resolution=resolution,
            num_frames=num_frames,
            num_inference_steps=num_inference_steps,
            text_guidance_scale=text_guidance_scale,
            audio_guidance_scale=audio_guidance_scale,
            output_type="both",
            generator=generator,
            audio_emb=audio_emb,
        )
        output, latent = output_tuple
        output = output[0]
        video = [(output[i] * 255).astype(np.uint8) for i in range(output.shape[0])]
        video = [PIL.Image.fromarray(img) for img in video]

        output_tensor = torch.from_numpy(np.array(video))
        save_video_ffmpeg(
            output_tensor,
            os.path.join(out_dir, "ai2v_demo_1"),
            audio_path,
            fps=save_fps,
            quality=5,
        )
        del output
        torch_gc()
    else:
        raise ValueError(f"Unsupported stage: {stage}")

    # --- long video continuation -------------------------------------------
    if num_segments > 1:
        ref_img_index = 10
        mask_frame_range = 3
        w_, h_ = video[0].size
        current_video = video
        ref_latent = latent[:, :, :1].clone()
        all_generated_frames = list(video)

        for seg_idx in range(1, num_segments):
            print(f"Generating segment {seg_idx + 1}/{num_segments}...")
            audio_start_idx += audio_stride * (num_frames - num_cond_frames)
            audio_end_idx = audio_start_idx + audio_stride * num_frames
            center_indices = (
                torch.arange(audio_start_idx, audio_end_idx, audio_stride).unsqueeze(1)
                + indices.unsqueeze(0)
            )
            center_indices = torch.clamp(
                center_indices, min=0, max=full_audio_emb.shape[0] - 1
            )
            audio_emb = full_audio_emb[center_indices][None, ...].to(LOCAL_RANK)

            output_tuple = pipe.generate_avc(
                video=current_video,
                video_latent=latent,
                prompt=prompt,
                negative_prompt=negative_prompt,
                height=h_,
                width=w_,
                num_frames=num_frames,
                num_cond_frames=num_cond_frames,
                num_inference_steps=num_inference_steps,
                text_guidance_scale=text_guidance_scale,
                audio_guidance_scale=audio_guidance_scale,
                generator=generator,
                output_type="both",
                use_kv_cache=True,
                offload_kv_cache=False,
                enhance_hf=True,
                audio_emb=audio_emb,
                ref_latent=ref_latent,
                ref_img_index=ref_img_index,
                mask_frame_range=mask_frame_range,
            )
            output, latent = output_tuple
            output = output[0]
            new_video = [
                (output[i] * 255).astype(np.uint8) for i in range(output.shape[0])
            ]
            new_video = [PIL.Image.fromarray(img) for img in new_video]
            del output

            all_generated_frames.extend(new_video[num_cond_frames:])
            current_video = new_video

            output_tensor = torch.from_numpy(np.array(all_generated_frames))
            save_video_ffmpeg(
                output_tensor,
                os.path.join(out_dir, f"video_continue_{seg_idx + 1}"),
                audio_path,
                fps=save_fps,
                quality=5,
            )
            del output_tensor

    # Return path of the final video
    # For single segment: at2v_demo_1.mp4 or ai2v_demo_1.mp4
    # For multi-segment: video_continue_N.mp4
    if num_segments > 1:
        final_video = os.path.join(out_dir, f"video_continue_{num_segments}.mp4")
    elif stage == "at2v":
        final_video = os.path.join(out_dir, "at2v_demo_1.mp4")
    else:
        final_video = os.path.join(out_dir, "ai2v_demo_1.mp4")

    return final_video


# ---------------------------------------------------------------------------
# RunPod handler
# ---------------------------------------------------------------------------

def _resolve_input_file(inp: dict, b64_key: str, ext: str) -> str | None:
    """Decode a base64-encoded file from the job payload and write it to a
    temp file.  Returns the path to the temp file, or None if the key is
    absent."""
    data = inp.get(b64_key)
    if not data:
        return None
    raw = base64.b64decode(data)
    tmp = tempfile.NamedTemporaryFile(suffix=ext, delete=False)
    tmp.write(raw)
    tmp.close()
    return tmp.name


def handler(job):
    """
    RunPod serverless handler.

    Files are sent as base64-encoded strings directly in the payload — no
    external URLs or file hosts needed.

    Expected job["input"] schema::

        {
            "audio_base64": str        -- base64-encoded audio bytes (.wav)
            "image_base64": str | None -- base64-encoded image bytes (required for ai2v)
            "prompt":       str        -- text prompt
            "negative_prompt": str | None
            "stage":        "ai2v" | "at2v"   (default "ai2v")
            "resolution":   "480p" | "720p"   (default "480p")
            "num_inference_steps": int        (default 50)
            "text_guidance_scale": float      (default 4.0)
            "audio_guidance_scale": float     (default 4.0)
            "num_segments": int               (default 1)
            "seed": int                       (default 42)
        }

    Returns::

        {"video_base64": str}   -- base64-encoded .mp4 bytes
    """
    audio_tmp_path = None
    image_tmp_path = None
    video_path = None

    try:
        inp = job["input"]

        # --- Validate required fields ---
        if "audio_base64" not in inp:
            return {"error": "Missing required field: audio_base64"}
        stage = inp.get("stage", "ai2v")
        if stage == "ai2v" and not inp.get("image_base64"):
            return {"error": "ai2v stage requires image_base64"}

        # --- Decode input files from base64 ---
        print("[handler] Decoding input files …")
        audio_tmp_path = _resolve_input_file(inp, "audio_base64", ".wav")
        image_tmp_path = _resolve_input_file(inp, "image_base64", ".png")

        # --- Run inference ---
        print("[handler] Starting inference …")
        t0 = time.time()
        video_path = run_inference(
            audio_path=audio_tmp_path,
            image_path=image_tmp_path,
            prompt=inp.get("prompt", "A person speaking naturally"),
            negative_prompt=inp.get("negative_prompt"),
            stage=stage,
            resolution=inp.get("resolution", "480p"),
            num_inference_steps=inp.get("num_inference_steps", 50),
            text_guidance_scale=inp.get("text_guidance_scale", 4.0),
            audio_guidance_scale=inp.get("audio_guidance_scale", 4.0),
            num_segments=inp.get("num_segments", 1),
            seed=inp.get("seed", 42),
        )
        print(f"[handler] Inference completed in {time.time() - t0:.1f}s")

        # --- Encode result as base64 ---
        with open(video_path, "rb") as f:
            video_b64 = base64.b64encode(f.read()).decode("utf-8")

        return {"video_base64": video_b64}

    except Exception as exc:
        traceback.print_exc()
        return {"error": str(exc)}

    finally:
        # Cleanup all temp files
        for p in [audio_tmp_path, image_tmp_path]:
            if p and os.path.exists(p):
                os.unlink(p)
        # Cleanup output directory for this job
        if video_path and os.path.exists(video_path):
            out_dir = os.path.dirname(video_path)
            shutil.rmtree(out_dir, ignore_errors=True)


# ---------------------------------------------------------------------------
# Entry point — must be at module level for RunPod's GitHub scanner to detect
# ---------------------------------------------------------------------------
runpod.serverless.start({"handler": handler})
