#!/usr/bin/env bash
# get_qwen_image.sh — download Qwen Image / Qwen Image Edit models into the shared
# HF cache (resume-friendly). The startup model scanner links cached models into
# ComfyUI/models.
set -euo pipefail

export HF_HUB_ENABLE_HF_TRANSFER=1
HF="/opt/venv/bin/hf"

dl() {
  local repo="$1"
  local remote="$2"
  echo "↓ Fetching $repo :: $remote"
  "$HF" download "$repo" "$remote" --repo-type model
}

echo "Which Qwen variant do you want to download?"
echo "  1) Qwen-Image 2512 (20B text-to-image)"
echo "  2) Qwen-Image-Edit 2511 (image editing)"
echo "  3) Qwen-Image-Lightning LoRA (4-steps)"
echo "  4) Qwen-Image-Edit-Lightning LoRA (4-steps, bf16)"

# Check if an argument is provided
if [ -n "${1:-}" ]; then
  choice="$1"
else
  read -rp "Enter 1, 2, 3 or 4: " choice
fi

PRECISION="fp8"
if [[ "${2:-}" == "bf16" ]]; then
  PRECISION="bf16"
fi

case "$choice" in
  1)
    REPO="Comfy-Org/Qwen-Image_ComfyUI"
    echo "==> Downloading Qwen-Image 2512 (20B) - $PRECISION"
    if [[ "$PRECISION" == "bf16" ]]; then
         dl "$REPO" "split_files/diffusion_models/qwen_image_2512_bf16.safetensors"
    else
         dl "$REPO" "split_files/diffusion_models/qwen_image_2512_fp8_e4m3fn.safetensors"
    fi
    dl "$REPO" "split_files/text_encoders/qwen_2.5_vl_7b_fp8_scaled.safetensors"
    dl "$REPO" "split_files/vae/qwen_image_vae.safetensors"
    ;;
  2)
    REPO="Comfy-Org/Qwen-Image-Edit_ComfyUI"
    echo "==> Downloading Qwen-Image-Edit - $PRECISION"
    # Requires text encoder + VAE from Qwen-Image
    BASE="Comfy-Org/Qwen-Image_ComfyUI"
    dl "$BASE" "split_files/text_encoders/qwen_2.5_vl_7b_fp8_scaled.safetensors"
    dl "$BASE" "split_files/vae/qwen_image_vae.safetensors"

    if [[ "$PRECISION" == "bf16" ]]; then
        dl "$REPO" "split_files/diffusion_models/qwen_image_edit_2511_bf16.safetensors"
    else
        dl "$REPO" "split_files/diffusion_models/qwen_image_edit_2511_fp8mixed.safetensors"
    fi
    ;;
  3)
    REPO="lightx2v/Qwen-Image-2512-Lightning"
    echo "==> Downloading Qwen-Image-2512-Lightning LoRA"
    dl "$REPO" "Qwen-Image-2512-Lightning-4steps-V1.0-bf16.safetensors"
    ;;
  4)
    REPO="lightx2v/Qwen-Image-Edit-2511-Lightning"
    echo "==> Downloading Qwen-Image-Edit-Lightning LoRA"
    dl "$REPO" "Qwen-Image-Edit-2511-Lightning-4steps-V1.0-bf16.safetensors"
    ;;
  *)
    echo "Invalid choice. Exiting."
    exit 1
    ;;
esac

echo "✓ Done. Files are in the shared HF cache (~/.cache/huggingface)."
echo "  They appear in ComfyUI after a container restart (the model scanner runs at start),"
echo "  or immediately after running: model_scanner.py sync"
