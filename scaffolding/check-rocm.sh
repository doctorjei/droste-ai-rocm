#!/usr/bin/env bash
# check-rocm.sh — runtime validation sweep for the droste *-halo (gfx1151 / Strix Halo) images.
#
# CI proves these images BUILD and ahead-of-time COMPILE for gfx1151 on x86 (no GPU).
# This script proves they RUN on real hardware — the one thing CI cannot do. Run it ON a
# gfx1151 host (Strix Halo) that exposes /dev/kfd + /dev/dri (or a rootful LXC on such a host).
#
# It sweeps only the RUNNABLE images. The four *-build carriers are FROM scratch (they carry
# only /artifacts for the matching runtime to COPY --from) and cannot run anything, so they are
# skipped by design. build-base is a compile toolchain, not a runtime, and is skipped too.
#
# Image names follow the ${SERIES}-<tier>-${ARCH} scheme (default droste-<tier>-halo), matching
# the Containerfile ARG convention. Flip --arch when a future gfx target ships.
#
# Checks are two tiers:
#   CORE  — deterministic, must pass: GPU enumerates as gfx1151; torch sees the GPU.
#   APP   — per-toolbox smoke: the app's binary runs / imports. These probe each tool's CLI;
#           if a tool's flags differ from what's assumed here, adjust the SMOKE commands below
#           (they are the first thing to tune on a real run — see the README).
set -euo pipefail

# ── Config (env or flags) ────────────────────────────────────────────────────
REGISTRY="${REGISTRY:-ghcr.io}"
OWNER="${OWNER:-doctorjei}"
SERIES="${SERIES:-droste}"            # image name: <REGISTRY>/<OWNER>/<SERIES>-<tier>-<ARCH>
ARCH="${ARCH:-halo}"                  # hardware line (halo = Strix Halo / gfx1151)
TAG="${TAG:-latest}"
RUNTIME="${RUNTIME:-podman}"          # podman | docker
PULL=0                                # --pull: pull each image before checking
GFX="${GFX:-gfx1151}"

usage() {
  cat <<EOF
check-rocm.sh — runtime validation sweep for the droste *-halo (gfx1151) images.

USAGE:
  ./check-rocm.sh [options]

OPTIONS:
  --tag <tag>         Image tag to check (default: ${TAG}; e.g. a commit sha for a pinned run)
  --owner <owner>     GHCR owner/org (default: ${OWNER})
  --registry <reg>    Registry host (default: ${REGISTRY})
  --series <name>     Image series prefix (default: ${SERIES})
  --arch <name>       Hardware-line suffix (default: ${ARCH})
  --runtime <r>       Container runtime: podman or docker (default: ${RUNTIME})
  --pull              Pull each image before checking
  -h, --help          Show this help

REQUIREMENTS:
  A gfx1151 GPU host exposing /dev/kfd and /dev/dri, and ${RUNTIME}.
  Rootless podman ROCm access also needs the invoking user in the render/video groups;
  this script adds --group-add keep-groups + --security-opt seccomp=unconfined for podman.

EXIT: non-zero if any check fails; a summary is printed at the end.
EOF
}

# ── Arg parsing ──────────────────────────────────────────────────────────────
while [[ $# -gt 0 ]]; do
  case "$1" in
    --tag)      TAG="$2"; shift 2 ;;
    --owner)    OWNER="$2"; shift 2 ;;
    --registry) REGISTRY="$2"; shift 2 ;;
    --series)   SERIES="$2"; shift 2 ;;
    --arch)     ARCH="$2"; shift 2 ;;
    --runtime)  RUNTIME="$2"; shift 2 ;;
    --pull)     PULL=1; shift ;;
    -h|--help)  usage; exit 0 ;;
    *) echo "Unknown option: $1" >&2; usage >&2; exit 2 ;;
  esac
done

# ── Prerequisites ────────────────────────────────────────────────────────────
command -v "$RUNTIME" >/dev/null 2>&1 || { echo "ERROR: '$RUNTIME' not found on PATH." >&2; exit 2; }
[[ -e /dev/kfd ]] || echo "WARNING: /dev/kfd missing — this host has no AMD GPU compute node; checks will fail." >&2
[[ -e /dev/dri ]] || echo "WARNING: /dev/dri missing — no render node; checks will fail." >&2

DEVICE_ARGS=(--rm --device /dev/kfd --device /dev/dri)
if [[ "$RUNTIME" == "podman" ]]; then
  DEVICE_ARGS+=(--group-add keep-groups --security-opt seccomp=unconfined)
fi

img() { echo "${REGISTRY}/${OWNER}/${SERIES}-$1-${ARCH}:${TAG}"; }

PASS=0 ; FAIL=0 ; FAILED_NAMES=""

# run_check <label> <tier> <shell-command-run-via-bash-lc>   (tier → <SERIES>-<tier>-<ARCH>)
run_check() {
  local label="$1" tier="$2" cmd="$3" image out rc
  image="$(img "$tier")"
  printf '── %-28s %s\n' "$label" "$image"
  if [[ "$PULL" -eq 1 ]]; then "$RUNTIME" pull -q "$image" >/dev/null 2>&1 || true; fi
  set +e
  # ALLOW_EPHEMERAL: the shared entrypoint hard-errors on unbound critical
  # mounts before running the probe command; smoke probes are deliberately
  # bind-less, so downgrade the criticals to warnings.
  out="$("$RUNTIME" run -e ALLOW_EPHEMERAL=1 "${DEVICE_ARGS[@]}" "$image" bash -lc "$cmd" 2>&1)"
  rc=$?
  set -e
  if [[ $rc -eq 0 ]]; then
    PASS=$((PASS + 1))
    printf '   ✅ PASS  %s\n' "$(echo "$out" | tail -1)"
  else
    FAIL=$((FAIL + 1))
    FAILED_NAMES="${FAILED_NAMES} ${label}"
    printf '   ❌ FAIL (rc=%s)\n' "$rc"
    echo "$out" | sed 's/^/      /' | tail -8
  fi
}

echo "=== ${SERIES}-*-${ARCH} runtime validation — tag ${TAG}, runtime ${RUNTIME} ==="
echo

# ── CORE: GPU enumerates + torch sees it (deterministic) ─────────────────────
run_check "gpu-enumerate (base)" runtime-base \
  "rocminfo | grep -i -m1 '${GFX}'"

# torch.cuda on every torch-carrying tier
TORCH_PROBE='python -c "import torch; ok=torch.cuda.is_available(); print((\"gpu=\"+torch.cuda.get_device_name(0)) if ok else \"NO GPU VISIBLE\"); import sys; sys.exit(0 if ok else 1)"'
run_check "torch.cuda (comfyui)"      comfyui      "$TORCH_PROBE"
run_check "torch.cuda (vllm)"         vllm         "$TORCH_PROBE"
run_check "torch.cuda (finetuning)"   finetuning   "$TORCH_PROBE"

# ── APP smoke (per-toolbox; tune the commands here if a tool's CLI differs) ───
# llama.cpp turboquant: llama-server ships in /usr/local/bin.
run_check "llama-server --version" llama \
  "llama-server --version"

# ds4: confirm the binary is present + its shared libs (hip/rocblas/hipblaslt) all resolve.
# (ds4's inference CLI needs a model to do more; library resolution is the meaningful GPU-stack probe.)
run_check "ds4 binary + ldd" ds4 \
  'B="$(command -v ds4-bench || command -v ds4-server || command -v ds4)"; echo "bin=$B"; ldd "$B" | grep -q "not found" && { ldd "$B" | grep "not found"; exit 1; }; echo "libs resolve"'

# vLLM: import the package + confirm torch GPU in the same interpreter.
run_check "import vllm + torch.cuda" vllm \
  'python -c "import vllm, torch; print(\"vllm\", vllm.__version__, \"cuda\", torch.cuda.is_available()); import sys; sys.exit(0 if torch.cuda.is_available() else 1)"'

# bitsandbytes (finetuning): imports its gfx1151 .so against the runtime.
run_check "import bitsandbytes" finetuning \
  'python -c "import bitsandbytes as b; print(\"bitsandbytes\", getattr(b,\"__version__\",\"?\"))"'

# ── Summary ──────────────────────────────────────────────────────────────────
echo
echo "=== summary: ${PASS} passed, ${FAIL} failed ==="
if [[ $FAIL -gt 0 ]]; then
  echo "   failed:${FAILED_NAMES}"
  echo "   (CORE failures = GPU/toolchain issue; APP-only failures may just be a CLI-flag mismatch — see comments.)"
  exit 1
fi
echo "   all runtime checks passed — gfx1151 images run on this host."
