#!/usr/bin/env bash
# get_ltx2.sh — download LTX-2 models into the shared HF cache (resume-friendly).
# The startup model scanner links cached models into ComfyUI/models.
set -euo pipefail

export HF_HUB_ENABLE_HF_TRANSFER=1
HF="/opt/venv/bin/hf"

dl () {
  local repo="$1"
  local remote="$2"
  echo "↓ Fetching $repo :: $remote"
  "$HF" download "$repo" "$remote" --repo-type model
}

usage() {
  cat <<'USAGE'
Usage: get_ltx2.sh <target> [variant]

Targets:
  common       Text encoder (Gemma 3) + Spatial Upscaler
  checkpoint   LTX-2 19B Checkpoint (Default: BF16. Use 'fp8' as 2nd arg for FP8)
  lora         Distilled LoRA + Camera Control LoRA

Notes:
- Downloads land in the shared HF cache (~/.cache/huggingface) and RESUME automatically.
- Models appear in ComfyUI after a container restart (the model scanner runs at start),
  or immediately after running: model_scanner.py sync
USAGE
}

case "${1:-}" in
  common)
    echo "==> Text Encoder + Spatial Upscaler"
    # Text Encoder: Gemma 3 12B IT FP4 Mixed
    dl "Comfy-Org/ltx-2" "split_files/text_encoders/gemma_3_12B_it_fp4_mixed.safetensors"

    # Spatial Upscaler x2
    dl "Lightricks/LTX-2" "ltx-2-spatial-upscaler-x2-1.0.safetensors"
    ;;

  checkpoint)
    VARIANT="${2:-bf16}"
    echo "==> LTX-2 19B Checkpoint ($VARIANT)"

    if [[ "$VARIANT" == "fp8" ]]; then
        dl "Lightricks/LTX-2" "ltx-2-19b-dev-fp8.safetensors"
    else
        # Default / BF16
        dl "Lightricks/LTX-2" "ltx-2-19b-dev.safetensors"
    fi
    ;;

  lora)
    echo "==> LTX-2 LoRAs"
    # Distilled LoRA
    dl "Lightricks/LTX-2" "ltx-2-19b-distilled-lora-384.safetensors"

    # Camera Control LoRA
    # User link: https://huggingface.co/Lightricks/LTX-2-19b-LoRA-Camera-Control-Dolly-Left
    dl "Lightricks/LTX-2-19b-LoRA-Camera-Control-Dolly-Left" "ltx-2-19b-lora-camera-control-dolly-left.safetensors"
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
