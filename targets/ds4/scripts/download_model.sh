#!/bin/sh
# download_model.sh — DeepSeek V4 GGUF downloader (droste cache-native rework).
#
# Reworked from upstream kyuz0/ds4 @00e64ea download_model.sh:
#   * Downloads INTO the shared HF cache (`hf download` WITHOUT --local-dir) —
#     single-copy store shared across ports/boxes — then prints the absolute
#     snapshot path to use as DS4_DROSTE_MODEL in /opt/data/ds4.env.
#   * DS4_GGUF_DIR stays as an explicit flat-dir override (--local-dir into your
#     own writable bind) for users who want plain files instead of the cache.
#   * The curl fallback and the ./ds4flash.gguf symlink are gone: the hf CLI is
#     baked into this image (resumes natively), and the model path is config-
#     driven (ds4.env), not CWD-relative.
set -e

REPO="antirez/deepseek-v4-gguf"
Q2_IMATRIX_FILE="DeepSeek-V4-Flash-IQ2XXS-w2Q2K-AProjQ8-SExpQ8-OutQ8-chat-v2-imatrix.gguf"
Q4_IMATRIX_FILE="DeepSeek-V4-Flash-Q4KExperts-F16HC-F16Compressor-F16Indexer-Q8Attn-Q8Shared-Q8Out-chat-v2-imatrix.gguf"
Q2_Q4_IMATRIX_FILE="DeepSeek-V4-Flash-Layers37-42Q4KExperts-OtherExpertLayersIQ2XXSGateUp-Q2KDown-AProjQ8-SExpQ8-OutQ8-chat-v2-imatrix-fixed.gguf"
PRO_Q2_IMATRIX_FILE="DeepSeek-V4-Pro-IQ2XXS-w2Q2K-AProjQ8-SExpQ8-OutQ8-Instruct-imatrix.gguf"
PRO_Q4_LAYERS00_30_FILE="DeepSeek-V4-Pro-Q4K-Layers00-30.gguf"
PRO_Q4_LAYERS31_OUTPUT_FILE="DeepSeek-V4-Pro-Q4K-Layers-31-output.gguf"
MTP_FILE="DeepSeek-V4-Flash-MTP-Q4K-Q8_0-F32.gguf"

OUT_DIR=${DS4_GGUF_DIR:-}
TOKEN=${HF_TOKEN:-}

usage() {
    cat <<EOF
DeepSeek V4 GGUF downloader (droste, cache-native)

Usage:
  download_model.sh q2-imatrix [--token TOKEN]
  download_model.sh q2-q4-imatrix [--token TOKEN]
  download_model.sh q4-imatrix [--token TOKEN]
  download_model.sh pro-q2-imatrix [--token TOKEN]
  download_model.sh pro-q4-layers00-30 [--token TOKEN]
  download_model.sh pro-q4-layers31-output [--token TOKEN]
  download_model.sh pro-q4-split [--token TOKEN]
  download_model.sh mtp [--token TOKEN]

Targets:

  q2-imatrix
       2-bit routed experts, about 81 GB on disk.
       Recommended model for 96 and 128 GB RAM machines.

  q2-q4-imatrix
       Mixed Flash quant: mostly q2 routed experts, with the last 6 layers
       using q4 routed experts. About 98 GB on disk. Good for higher
       quality inference for 128 GB machines.

  q4-imatrix
       4-bit routed experts, about 153 GB on disk.
       Recommended model for machines with 256 GB RAM or more.

  pro-q2-imatrix
       DeepSeek V4 PRO q2 imatrix quant, as a single GGUF file. About 430 GB
       on disk; intended for 512 GB RAM machines.

  pro-q4-layers00-30
       First half of the DeepSeek V4 PRO Q4 routed-expert quant, layers 0..30.
       Use on the coordinator in a two-machine distributed run. About 426 GB.

  pro-q4-layers31-output
       Second half of the DeepSeek V4 PRO Q4 routed-expert quant, layers
       31..output. Use on the worker in a two-machine distributed run.
       About 412 GB.

  pro-q4-split
       Downloads both PRO Q4 split files. About 838 GB total.

  mtp  Optional speculative decoding component, about 3.5 GB on disk.
       It is useful with q2-imatrix, q2-q4-imatrix, and q4-imatrix, but must be
       enabled explicitly (DS4_DROSTE_MTP in /opt/data/ds4.env, or --mtp).

Options:
  --token TOKEN  Hugging Face token. Otherwise HF_TOKEN or the local HF token
                 cache (~/.cache/huggingface/token) is used if present.

Environment:
  DS4_GGUF_DIR   OPTIONAL flat-directory override. When set, files are
                 downloaded as plain files into this directory (use your own
                 writable bind). When unset (default), downloads go into the
                 shared HF cache (~/.cache/huggingface) — single copy, shared
                 across containers, resumable.

After a download the script prints the absolute path of each file — set it in
/opt/data/ds4.env:
  DS4_DROSTE_MODEL=<path>        (main model)
  DS4_DROSTE_MTP=<path>          (mtp component, optional)
If a download stops, run the same command again to resume it.
EOF
}

if [ $# -eq 0 ]; then
    usage
    exit 1
fi

MODEL=$1
shift
MODEL_FILES=

case "$MODEL" in
    q2-imatrix) MODEL_FILES=$Q2_IMATRIX_FILE ;;
    q2-q4-imatrix) MODEL_FILES=$Q2_Q4_IMATRIX_FILE ;;
    q4-imatrix) MODEL_FILES=$Q4_IMATRIX_FILE ;;
    pro-q2-imatrix) MODEL_FILES=$PRO_Q2_IMATRIX_FILE ;;
    pro-q4-layers00-30) MODEL_FILES=$PRO_Q4_LAYERS00_30_FILE ;;
    pro-q4-layers31-output) MODEL_FILES=$PRO_Q4_LAYERS31_OUTPUT_FILE ;;
    pro-q4-split)
        MODEL_FILES="$PRO_Q4_LAYERS00_30_FILE $PRO_Q4_LAYERS31_OUTPUT_FILE"
        ;;
    mtp) MODEL_FILES=$MTP_FILE ;;
    -h|--help|help)
        usage
        exit 0
        ;;
    *)
        echo "Unknown model: $MODEL" >&2
        echo >&2
        usage >&2
        exit 1
        ;;
esac

while [ $# -gt 0 ]; do
    case "$1" in
        --token)
            shift
            if [ $# -eq 0 ]; then
                echo "Missing value after --token" >&2
                exit 1
            fi
            TOKEN=$1
            ;;
        *)
            echo "Unknown option: $1" >&2
            exit 1
            ;;
    esac
    shift
done

# hf reads its own cached login too; an explicit token (flag/env/cache file)
# simply takes precedence.
if [ -z "$TOKEN" ] && [ -s "$HOME/.cache/huggingface/token" ]; then
    TOKEN=$(cat "$HOME/.cache/huggingface/token")
fi

if ! command -v hf >/dev/null 2>&1; then
    echo "This script requires the Hugging Face CLI (baked into this image)." >&2
    echo "If it is missing, install it with:" >&2
    echo "  pip install -U 'huggingface_hub[hf_xet]'" >&2
    exit 1
fi

# download_one <file> — download into the HF cache (default) or DS4_GGUF_DIR
# (flat override); prints the absolute local path on the last stdout line.
download_one() {
    file=$1

    echo "Downloading $file" >&2
    echo "from https://huggingface.co/$REPO" >&2
    if [ -n "$OUT_DIR" ]; then
        echo "into flat directory $OUT_DIR (DS4_GGUF_DIR override)" >&2
        mkdir -p "$OUT_DIR"
        set -- "$REPO" "$file" --repo-type model --local-dir "$OUT_DIR"
    else
        echo "into the shared HF cache (~/.cache/huggingface)" >&2
        set -- "$REPO" "$file" --repo-type model
    fi
    if [ -n "$TOKEN" ]; then
        set -- "$@" --token "$TOKEN"
    fi
    echo "If the download stops, run the same command again to resume it." >&2

    # `hf download` prints the local path (cache snapshot path, or the
    # --local-dir path) as its final stdout line.
    path=$(hf download "$@" | tail -n 1)
    if [ -z "$path" ] || [ ! -s "$path" ]; then
        echo "Download finished but the expected file is missing: '$path'" >&2
        exit 1
    fi
    printf '%s\n' "$path"
}

PATHS=
for file in $MODEL_FILES; do
    p=$(download_one "$file")
    PATHS="$PATHS $p"
    echo
    echo "Downloaded: $p"
done

echo
if [ "$MODEL" = "mtp" ]; then
    echo "MTP is an optional component for q2-imatrix, q2-q4-imatrix, and q4-imatrix."
    echo "Enable it in /opt/data/ds4.env:"
    for p in $PATHS; do
        echo "  DS4_DROSTE_MTP=$p"
    done
    echo "  DS4_DROSTE_MTP_DRAFT=2"
elif [ "$MODEL" = "pro-q4-layers00-30" ] || [ "$MODEL" = "pro-q4-layers31-output" ] || [ "$MODEL" = "pro-q4-split" ]; then
    echo "Downloaded PRO Q4 distributed split file(s). Use them with --layers"
    echo "(DS4_DROSTE_EXTRA_ARGS), for example coordinator layers 0:30 and"
    echo "worker layers 31:output. File paths:"
    for p in $PATHS; do
        echo "  $p"
    done
else
    echo "Point ds4-server at the model in /opt/data/ds4.env:"
    for p in $PATHS; do
        echo "  DS4_DROSTE_MODEL=$p"
    done
fi

echo
echo "Done."
