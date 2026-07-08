#!/usr/bin/env bash
# get_wan22.sh — download Wan 2.2 models into the shared HF cache (resume-friendly).
# The startup model scanner links cached models into ComfyUI/models.
set -euo pipefail

export HF_HUB_ENABLE_HF_TRANSFER=1
HF="/opt/venv/bin/hf"

# Repositories
REPO_22="Comfy-Org/Wan_2.2_ComfyUI_Repackaged"
REPO_21="Comfy-Org/Wan_2.1_ComfyUI_repackaged"
REPO_LORA="lightx2v/Wan2.2-Lightning"

PRECISION="fp8"
if [[ "${2:-}" == "fp16" ]]; then
  PRECISION="fp16"
fi

dl () {
  local repo="$1"
  local remote="$2"
  echo "↓ Fetching $repo :: $remote"
  "$HF" download "$repo" "$remote" --repo-type model
}

usage() {
  cat <<'USAGE'
Usage: get_wan22.sh <target> [fp16]

Targets:
  common     Text encoder + VAEs
  14b-t2v    14B T2V diffusion models (Defaults to FP8, use 'fp16' as 2nd arg for FP16)
  14b-i2v    14B I2V diffusion models (Defaults to FP8, use 'fp16' as 2nd arg for FP16)
  lora       Wan2.2 Lightning LoRAs

Notes:
- Downloads land in the shared HF cache (~/.cache/huggingface) and RESUME automatically.
- Models appear in ComfyUI after a container restart (the model scanner runs at start),
  or immediately after running: model_scanner.py sync
USAGE
}

case "${1:-}" in
  common)
    echo "==> text encoder + VAEs"
    if [[ "$PRECISION" == "fp16" ]]; then
         # Use fp16 text encoder if available or if standard
         dl "$REPO_22" "split_files/text_encoders/umt5_xxl_fp16.safetensors"
    else
         dl "$REPO_21" "split_files/text_encoders/umt5_xxl_fp8_e4m3fn_scaled.safetensors"
    fi
    dl "$REPO_22" "split_files/vae/wan_2.1_vae.safetensors"
    ;;
  14b-t2v)
    echo "==> 14B Text→Video ($PRECISION)"
    if [[ "$PRECISION" == "fp16" ]]; then
        dl "$REPO_22" "split_files/diffusion_models/wan2.2_t2v_high_noise_14B_fp16.safetensors"
        dl "$REPO_22" "split_files/diffusion_models/wan2.2_t2v_low_noise_14B_fp16.safetensors"
    else
        dl "$REPO_22" "split_files/diffusion_models/wan2.2_t2v_high_noise_14B_fp8_scaled.safetensors"
        dl "$REPO_22" "split_files/diffusion_models/wan2.2_t2v_low_noise_14B_fp8_scaled.safetensors"
    fi
    ;;
  14b-i2v)
    echo "==> 14B Image→Video ($PRECISION)"
    if [[ "$PRECISION" == "fp16" ]]; then
        dl "$REPO_22" "split_files/diffusion_models/wan2.2_i2v_high_noise_14B_fp16.safetensors"
        dl "$REPO_22" "split_files/diffusion_models/wan2.2_i2v_low_noise_14B_fp16.safetensors"
    else
        dl "$REPO_22" "split_files/diffusion_models/wan2.2_i2v_high_noise_14B_fp8_scaled.safetensors"
        dl "$REPO_22" "split_files/diffusion_models/wan2.2_i2v_low_noise_14B_fp8_scaled.safetensors"
    fi
    ;;
  lora)
    echo "==> Wan2.2 Lightning LoRAs (Seko V2)"
    dl "$REPO_LORA" "Wan2.2-T2V-A14B-4steps-lora-rank64-Seko-V2.0/high_noise_model.safetensors"
    dl "$REPO_LORA" "Wan2.2-T2V-A14B-4steps-lora-rank64-Seko-V2.0/low_noise_model.safetensors"
    dl "$REPO_LORA" "Wan2.2-I2V-A14B-4steps-lora-rank64-Seko-V1/high_noise_model.safetensors"
    dl "$REPO_LORA" "Wan2.2-I2V-A14B-4steps-lora-rank64-Seko-V1/low_noise_model.safetensors"
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
