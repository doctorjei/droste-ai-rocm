#!/usr/bin/env bash
# get_hunyuan15.sh — download ComfyUI HunyuanVideo 1.5 (T2V & I2V) models into the
# shared HF cache (resume-friendly). The startup model scanner links cached models
# into ComfyUI/models.
set -euo pipefail

export HF_HUB_ENABLE_HF_TRANSFER=1
HF="/opt/venv/bin/hf"

# Repositories
REPO_MAIN="Comfy-Org/HunyuanVideo_1.5_repackaged"
REPO_VISION="Comfy-Org/sigclip_vision_384"

dl () {
  local repo="$1"
  local remote="$2"
  echo "↓ Fetching $repo :: $remote"
  "$HF" download "$repo" "$remote" --repo-type model
}

usage() {
  cat <<'USAGE'
Usage: get_hunyuan15.sh <target>

Targets:
  common     Text Encoders, VAE, CLIP Vision (Shared dependencies)
             - text_encoders/qwen_2.5_vl_7b_fp8_scaled.safetensors
             - text_encoders/byt5_small_glyphxl_fp16.safetensors
             - vae/hunyuanvideo15_vae_fp16.safetensors
             - clip_vision/sigclip_vision_patch14_384.safetensors (Only for I2V)

  720p-t2v   Text-to-Video Model (FP16)
             - diffusion_models/hunyuanvideo1.5_720p_t2v_fp16.safetensors

  720p-i2v   Image-to-Video Model (FP16)
             - diffusion_models/hunyuanvideo1.5_720p_i2v_fp16.safetensors

  upscale    Upscaling Models (1080p SR + Latent Upsampler)
             - diffusion_models/hunyuanvideo1.5_1080p_sr_distilled_fp16.safetensors
             - latent_upscale_models/hunyuanvideo15_latent_upsampler_1080p.safetensors

  lora       HunyuanVideo 1.5 LoRAs
             - loras/hunyuanvideo1.5_t2v_480p_lightx2v_4step_lora_rank_32_bf16.safetensors

  all        Download EVERYTHING (T2V, I2V, Upscale, LoRA, Common)

Notes:
- Downloads land in the shared HF cache (~/.cache/huggingface) and RESUME automatically.
- Models appear in ComfyUI after a container restart (the model scanner runs at start),
  or immediately after running: model_scanner.py sync
USAGE
}

case "${1:-}" in
  common)
    echo "==> Text Encoders, VAE, & CLIP Vision"
    dl "$REPO_MAIN" "split_files/text_encoders/qwen_2.5_vl_7b_fp8_scaled.safetensors"
    dl "$REPO_MAIN" "split_files/text_encoders/byt5_small_glyphxl_fp16.safetensors"
    dl "$REPO_MAIN" "split_files/vae/hunyuanvideo15_vae_fp16.safetensors"
    dl "$REPO_VISION" "sigclip_vision_patch14_384.safetensors"
    ;;

  720p-t2v)
    echo "==> 720p Text-to-Video Model"
    dl "$REPO_MAIN" "split_files/diffusion_models/hunyuanvideo1.5_720p_t2v_fp16.safetensors"
    ;;

  720p-i2v)
    echo "==> 720p Image-to-Video Model"
    dl "$REPO_MAIN" "split_files/diffusion_models/hunyuanvideo1.5_720p_i2v_fp16.safetensors"
    ;;

  upscale)
    echo "==> 1080p Upscaling Models"
    dl "$REPO_MAIN" "split_files/diffusion_models/hunyuanvideo1.5_1080p_sr_distilled_fp16.safetensors"
    dl "$REPO_MAIN" "split_files/latent_upscale_models/hunyuanvideo15_latent_upsampler_1080p.safetensors"
    ;;

  lora)
    echo "==> HunyuanVideo 1.5 LoRAs"
    dl "$REPO_MAIN" "split_files/loras/hunyuanvideo1.5_t2v_480p_lightx2v_4step_lora_rank_32_bf16.safetensors"
    ;;

  all)
    echo "==> Downloading Full Suite (T2V + I2V + Upscale + LoRA)..."
    "$0" common
    "$0" 720p-t2v
    "$0" 720p-i2v
    "$0" upscale
    "$0" lora
    ;;

  ""|-h|--help|help)
    usage
    exit 0
    ;;
  *)
    echo "Unknown target: $1" >&2
    usage
    exit 1
    ;;
esac

echo "✓ Done. Files are in the shared HF cache (~/.cache/huggingface)."
echo "  They appear in ComfyUI after a container restart (the model scanner runs at start),"
echo "  or immediately after running: model_scanner.py sync"
