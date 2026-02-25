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
import math
import os
import random
import shutil
import tempfile
import time
from pathlib import Path

import modal


APP_NAME = "longcat-video-avatar"
VOLUME_NAME = os.environ.get("MODAL_WEIGHTS_VOLUME", "longcat-weights")
VOLUME_MOUNT_PATH = "/weights"
WEIGHTS_DIR = f"{VOLUME_MOUNT_PATH}/weights"
OUTPUT_DIR = "/tmp/outputs"


def _image() -> modal.Image:
    # Use CUDA base image and install Python + project dependencies.
    return (
        modal.Image.from_registry("nvidia/cuda:12.4.1-devel-ubuntu22.04", add_python="3.11")
        .apt_install("ffmpeg", "libsndfile1", "git")
        .add_local_file("requirements.txt", "/tmp/requirements.txt")
        .run_commands(
            "pip install -U pip packaging ninja",
            "pip install -r /tmp/requirements.txt",
            # Try flash-attn, but don't fail build if unavailable.
            "pip install --no-build-isolation --no-cache-dir flash-attn || true",
        )
        .add_local_dir(".", remote_path="/root/LongCat-Video")
        .workdir("/root/LongCat-Video")
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
    timeout=60 * 60 * 4,
)
def download_weights(hf_token: str | None = None, force: bool = False) -> dict:
    """
    One-time weight download into Modal Volume.

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

    if force and os.path.exists(base_dir):
        shutil.rmtree(base_dir, ignore_errors=True)
    if force and os.path.exists(avatar_dir):
        shutil.rmtree(avatar_dir, ignore_errors=True)

    if not os.path.exists(f"{base_dir}/dit"):
        snapshot_download(
            repo_id="meituan-longcat/LongCat-Video",
            local_dir=base_dir,
            token=hf_token,
        )
        shutil.rmtree(os.environ["HF_HOME"], ignore_errors=True)
        os.makedirs(os.environ["HF_HOME"], exist_ok=True)

    if not os.path.exists(f"{avatar_dir}/avatar_single"):
        snapshot_download(
            repo_id="meituan-longcat/LongCat-Video-Avatar",
            local_dir=avatar_dir,
            token=hf_token,
        )

    shutil.rmtree(os.environ["HF_HOME"], ignore_errors=True)
    weights_volume.commit()

    return {
        "weights_dir": WEIGHTS_DIR,
        "base_model_size": os.popen(f"du -sh {base_dir} | cut -f1").read().strip(),
        "avatar_model_size": os.popen(f"du -sh {avatar_dir} | cut -f1").read().strip(),
    }


@app.cls(
    image=image,
    gpu="A100-80GB",
    timeout=60 * 60,
    scaledown_window=60 * 5,
    volumes={VOLUME_MOUNT_PATH: weights_volume},
)
class AvatarInference:
    @modal.enter()
    def load(self):
        import librosa
        import numpy as np
        import PIL.Image
        import torch
        import torch.distributed as dist
        from audio_separator.separator import Separator
        from diffusers.utils import load_image
        from transformers import AutoTokenizer, UMT5EncoderModel, Wav2Vec2FeatureExtractor

        from longcat_video.audio_process.torch_utils import save_video_ffmpeg
        from longcat_video.audio_process.wav2vec2 import Wav2Vec2ModelWrapper
        from longcat_video.context_parallel import context_parallel_util
        from longcat_video.modules.autoencoder_kl_wan import AutoencoderKLWan
        from longcat_video.modules.avatar.longcat_video_dit_avatar import LongCatVideoAvatarTransformer3DModel
        from longcat_video.modules.scheduling_flow_match_euler_discrete import FlowMatchEulerDiscreteScheduler
        from longcat_video.pipeline_longcat_video_avatar import LongCatVideoAvatarPipeline

        self.np = np
        self.PIL = PIL
        self.torch = torch
        self.librosa = librosa
        self.load_image = load_image
        self.save_video_ffmpeg = save_video_ffmpeg

        base_model_dir = os.path.join(WEIGHTS_DIR, "LongCat-Video")
        avatar_weights_dir = os.path.join(WEIGHTS_DIR, "LongCat-Video-Avatar")

        if not os.path.exists(base_model_dir) or not os.path.exists(avatar_weights_dir):
            raise RuntimeError(
                f"Missing weights in {WEIGHTS_DIR}. Run download_weights first."
            )

        os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
        os.environ.setdefault("MASTER_PORT", "29500")
        os.environ.setdefault("RANK", "0")
        os.environ.setdefault("WORLD_SIZE", "1")
        os.environ.setdefault("LOCAL_RANK", "0")

        try:
            if not dist.is_initialized():
                dist.init_process_group(
                    backend="nccl",
                    timeout=datetime.timedelta(seconds=3600),
                )
        except Exception:
            if not dist.is_initialized():
                dist.init_process_group(
                    backend="gloo",
                    timeout=datetime.timedelta(seconds=3600),
                )

        self.local_rank = int(os.environ.get("LOCAL_RANK", "0"))
        if torch.cuda.is_available():
            torch.cuda.set_device(self.local_rank)

        context_parallel_util.init_context_parallel(
            context_parallel_size=1,
            global_rank=dist.get_rank(),
            world_size=dist.get_world_size(),
        )
        cp_split_hw = context_parallel_util.get_optimal_split(1)

        tokenizer = AutoTokenizer.from_pretrained(
            base_model_dir, subfolder="tokenizer", torch_dtype=torch.bfloat16
        )
        text_encoder = UMT5EncoderModel.from_pretrained(
            base_model_dir, subfolder="text_encoder", torch_dtype=torch.bfloat16
        )
        vae = AutoencoderKLWan.from_pretrained(
            base_model_dir, subfolder="vae", torch_dtype=torch.bfloat16
        )
        scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(
            base_model_dir, subfolder="scheduler", torch_dtype=torch.bfloat16
        )
        dit = LongCatVideoAvatarTransformer3DModel.from_pretrained(
            avatar_weights_dir,
            subfolder="avatar_single",
            cp_split_hw=cp_split_hw,
            torch_dtype=torch.bfloat16,
        )

        wav2vec_path = os.path.join(avatar_weights_dir, "chinese-wav2vec2-base")
        audio_encoder = Wav2Vec2ModelWrapper(wav2vec_path).to(self.local_rank)
        audio_encoder.feature_extractor._freeze_parameters()
        wav2vec_feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(
            wav2vec_path, local_files_only=True
        )

        vocal_separator_path = os.path.join(
            avatar_weights_dir, "vocal_separator/Kim_Vocal_2.onnx"
        )
        self.audio_output_dir_temp = Path("/tmp/audio_temp")
        self.audio_output_dir_temp.mkdir(parents=True, exist_ok=True)
        self.vocal_separator = Separator(
            output_dir=self.audio_output_dir_temp / "vocals",
            output_single_stem="vocals",
            model_file_dir=os.path.dirname(vocal_separator_path),
        )
        self.vocal_separator.load_model(os.path.basename(vocal_separator_path))

        self.pipe = LongCatVideoAvatarPipeline(
            tokenizer=tokenizer,
            text_encoder=text_encoder,
            vae=vae,
            scheduler=scheduler,
            dit=dit,
            audio_encoder=audio_encoder,
            wav2vec_feature_extractor=wav2vec_feature_extractor,
        )
        self.pipe.to(self.local_rank)

        os.makedirs(OUTPUT_DIR, exist_ok=True)

    def _torch_gc(self):
        self.torch.cuda.empty_cache()
        self.torch.cuda.ipc_collect()

    def _extract_vocal_from_speech(self, source_path: str, target_path: str):
        outputs = self.vocal_separator.separate(source_path)
        if len(outputs) <= 0:
            return None
        default_vocal_path = self.audio_output_dir_temp / "vocals" / outputs[0]
        default_vocal_path = default_vocal_path.resolve().as_posix()
        shutil.move(default_vocal_path, target_path)
        return target_path

    @modal.method()
    def generate(self, payload: dict) -> dict:
        audio_tmp_path = None
        image_tmp_path = None
        video_path = None
        try:
            if "audio_base64" not in payload:
                return {"error": "Missing required field: audio_base64"}

            stage = payload.get("stage", "ai2v")
            if stage == "ai2v" and not payload.get("image_base64"):
                return {"error": "ai2v stage requires image_base64"}

            # Decode request files
            raw_audio = base64.b64decode(payload["audio_base64"])
            audio_tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
            audio_tmp.write(raw_audio)
            audio_tmp.close()
            audio_tmp_path = audio_tmp.name

            if payload.get("image_base64"):
                raw_image = base64.b64decode(payload["image_base64"])
                image_tmp = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
                image_tmp.write(raw_image)
                image_tmp.close()
                image_tmp_path = image_tmp.name

            prompt = payload.get("prompt", "A person speaking naturally")
            negative_prompt = payload.get("negative_prompt") or (
                "Close-up, Bright tones, overexposed, static, blurred details, subtitles, "
                "style, works, paintings, images, static, overall gray, worst quality, "
                "low quality, JPEG compression residue, ugly, incomplete, extra fingers, "
                "poorly drawn hands, poorly drawn faces, deformed, disfigured, misshapen limbs, "
                "fused fingers, still picture, messy background, three legs, many people in the "
                "background, walking backwards"
            )

            resolution = payload.get("resolution", "480p")
            num_inference_steps = int(payload.get("num_inference_steps", 50))
            text_guidance_scale = float(payload.get("text_guidance_scale", 4.0))
            audio_guidance_scale = float(payload.get("audio_guidance_scale", 4.0))
            num_segments = max(1, int(payload.get("num_segments", 1)))
            seed = int(payload.get("seed", 42))

            save_fps = 16
            num_frames = 93
            num_cond_frames = 13
            audio_stride = 2
            if resolution == "720p":
                height, width = 768, 1280
            else:
                height, width = 480, 832

            generator = self.torch.Generator(device=self.local_rank)
            generator.manual_seed(seed)

            temp_vocal_path = self._extract_vocal_from_speech(
                audio_tmp_path,
                f"/tmp/temp_speech_{_generate_random_uid()}_vocal.wav",
            )
            if temp_vocal_path is None or not os.path.exists(temp_vocal_path):
                return {"error": "No vocal detected in provided audio"}

            generate_duration = (
                num_frames / save_fps
                + (num_segments - 1) * (num_frames - num_cond_frames) / save_fps
            )
            speech_array, sr = self.librosa.load(temp_vocal_path, sr=16000)
            source_duration = len(speech_array) / sr
            added_samples = math.ceil((generate_duration - source_duration) * sr)
            if added_samples > 0:
                speech_array = self.np.append(speech_array, [0.0] * added_samples)

            full_audio_emb = self.pipe.get_audio_embedding(
                speech_array, fps=save_fps * audio_stride, device=self.local_rank, sample_rate=sr
            )
            if self.torch.isnan(full_audio_emb).any():
                return {"error": "Broken audio embedding with NaN values"}

            if os.path.exists(temp_vocal_path):
                os.remove(temp_vocal_path)

            indices = self.torch.arange(2 * 2 + 1) - 2
            audio_start_idx = 0
            audio_end_idx = audio_start_idx + audio_stride * num_frames
            center_indices = (
                self.torch.arange(audio_start_idx, audio_end_idx, audio_stride).unsqueeze(1)
                + indices.unsqueeze(0)
            )
            center_indices = self.torch.clamp(
                center_indices, min=0, max=full_audio_emb.shape[0] - 1
            )
            audio_emb = full_audio_emb[center_indices][None, ...].to(self.local_rank)

            uid = _generate_random_uid()
            out_dir = os.path.join(OUTPUT_DIR, uid)
            os.makedirs(out_dir, exist_ok=True)

            if stage == "at2v":
                output_tuple = self.pipe.generate_at2v(
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
                video = [(output[i] * 255).astype(self.np.uint8) for i in range(output.shape[0])]
                video = [self.PIL.Image.fromarray(img) for img in video]
                output_tensor = self.torch.from_numpy(self.np.array(video))
                self.save_video_ffmpeg(
                    output_tensor, os.path.join(out_dir, "at2v_demo_1"), audio_tmp_path, fps=save_fps, quality=5
                )
                del output
                self._torch_gc()
            elif stage == "ai2v":
                image = self.load_image(image_tmp_path)
                output_tuple = self.pipe.generate_ai2v(
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
                video = [(output[i] * 255).astype(self.np.uint8) for i in range(output.shape[0])]
                video = [self.PIL.Image.fromarray(img) for img in video]
                output_tensor = self.torch.from_numpy(self.np.array(video))
                self.save_video_ffmpeg(
                    output_tensor, os.path.join(out_dir, "ai2v_demo_1"), audio_tmp_path, fps=save_fps, quality=5
                )
                del output
                self._torch_gc()
            else:
                return {"error": f"Unsupported stage: {stage}"}

            if num_segments > 1:
                ref_img_index = 10
                mask_frame_range = 3
                w_, h_ = video[0].size
                current_video = video
                ref_latent = latent[:, :, :1].clone()
                all_generated_frames = list(video)

                for seg_idx in range(1, num_segments):
                    audio_start_idx += audio_stride * (num_frames - num_cond_frames)
                    audio_end_idx = audio_start_idx + audio_stride * num_frames
                    center_indices = (
                        self.torch.arange(audio_start_idx, audio_end_idx, audio_stride).unsqueeze(1)
                        + indices.unsqueeze(0)
                    )
                    center_indices = self.torch.clamp(
                        center_indices, min=0, max=full_audio_emb.shape[0] - 1
                    )
                    audio_emb = full_audio_emb[center_indices][None, ...].to(self.local_rank)

                    output_tuple = self.pipe.generate_avc(
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
                    new_video = [(output[i] * 255).astype(self.np.uint8) for i in range(output.shape[0])]
                    new_video = [self.PIL.Image.fromarray(img) for img in new_video]
                    del output

                    all_generated_frames.extend(new_video[num_cond_frames:])
                    current_video = new_video
                    output_tensor = self.torch.from_numpy(self.np.array(all_generated_frames))
                    self.save_video_ffmpeg(
                        output_tensor,
                        os.path.join(out_dir, f"video_continue_{seg_idx + 1}"),
                        audio_tmp_path,
                        fps=save_fps,
                        quality=5,
                    )
                    del output_tensor

            if num_segments > 1:
                video_path = os.path.join(out_dir, f"video_continue_{num_segments}.mp4")
            elif stage == "at2v":
                video_path = os.path.join(out_dir, "at2v_demo_1.mp4")
            else:
                video_path = os.path.join(out_dir, "ai2v_demo_1.mp4")

            with open(video_path, "rb") as f:
                video_b64 = base64.b64encode(f.read()).decode("utf-8")
            return {"video_base64": video_b64}

        except Exception as exc:
            return {"error": str(exc)}
        finally:
            for p in [audio_tmp_path, image_tmp_path]:
                if p and os.path.exists(p):
                    os.unlink(p)
            if video_path and os.path.exists(video_path):
                shutil.rmtree(os.path.dirname(video_path), ignore_errors=True)


@app.function(timeout=60 * 60)
@modal.fastapi_endpoint(method="POST")
def inference_endpoint(payload: dict):
    """
    HTTP endpoint to call from Gradio/frontend.
    """
    return AvatarInference().generate.remote(payload)


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
