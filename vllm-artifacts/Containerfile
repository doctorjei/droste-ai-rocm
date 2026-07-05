# vllm-artifacts: compile the heavy ROCm/gfx1151 wheels for the vLLM toolbox, then ship
# them alone from `scratch`. HEAVIEST port — builds flash-attention (ROCm fork), aiter
# (amd_aiter*.whl), and vLLM itself from source against the pinned TheRock torch. The final
# stage carries ONLY /artifacts (wheels + the pure-python FP8 kernel tree); vllm-runtime
# does `COPY --from` of these. See rocm/rocm-build-base for the toolchain we build against.
#
# Translated from droste-ai-rocm/upstream/vllm/Dockerfile (single-stage Fedora toolbox by
# kyuz0). Toolbox submodule provenance (droste-ai-rocm): 6446b9595273f289e11586c3c7d3e1e6f2945888
#
# KEY DEVIATION vs upstream: upstream installs a Fedora ROCm-SDK TARBALL via
# scripts/install_rocm_sdk.sh (into /opt/rocm) — DROPPED here. rocm-build-base already
# provides the pip TheRock ROCm SDK (rocm-sdk-devel) at ROCM_PATH=/opt/rocm, with the ROCm
# clang under /opt/rocm/lib/llvm/bin (NOT the Fedora /opt/rocm/llvm/bin). We build against that.
ARG BASE_IMAGE=localhost/rocm-build-base
FROM ${BASE_IMAGE} AS build
# Local podman builds need the localhost/ prefix (default above); CI can override:
#   --build-arg BASE_IMAGE=ghcr.io/<owner>/rocm-build-base:<tag>

LABEL description="vllm-artifacts: flash-attention + aiter + vLLM wheels for gfx1151 (TheRock torch)"

# --- The unified pin (rocm-pin.env). ARG names MUST match the env keys the wrapper passes. ---
ARG ROCM_INDEX_URL=https://rocm.nightlies.amd.com/v2/gfx1151/
ARG TORCH_VERSION=2.9.1+rocm7.13.0a20260501
ARG GFX_TARGET=gfx1151

# --- Clone pins. VLLM_REF is pinned to v0.16.0 — the newest vLLM stable tag that targets
# torch 2.9.1 (its requirements/cuda.txt: torch==2.9.1). v0.16.1rc0+ bump to torch 2.10.0 and
# add the csrc/libtorch_stable extension, which needs torch/csrc/stable/device.h (torch-2.10 ABI)
# and fails to compile against our pinned torch 2.9.1. flash-attention still floats main_perf —
# FLAG: pin FLASH_ATTENTION_REF to a ~Feb-2026 (v0.16.0-era) sha on-host for reproducibility (a
# flash-attn sha transitively pins its aiter + composable_kernel submodules via the gitlink).
# FP8 kernels are pinned to upstream's default. ---
ARG FLASH_ATTENTION_REPO=https://github.com/ROCm/flash-attention.git
ARG FLASH_ATTENTION_BRANCH=main_perf
# FLASH_ATTENTION_REF resolved from branch main_perf (pinned 2026-07-05)
ARG FLASH_ATTENTION_REF=3f94643fb41bcedded28c85185a8e11d42ef1592
ARG VLLM_REPO=https://github.com/vllm-project/vllm.git
ARG VLLM_REF=v0.16.0
ARG FP8_KERNELS_REPO=https://github.com/leonyurko/vllm-fp8-strix-halo-kernel-support.git
ARG FP8_KERNELS_REF=50424f5525b8382353551e3301d0da56eca0be2b

# Torch (pinned TheRock nightly) into the venv the base already built. vLLM + flash-attn +
# aiter all compile their C++/HIP extensions against this torch's headers/ABI. FLAG: the pin's
# TORCHVISION_VERSION/TORCHAUDIO_VERSION are unset — vLLM multimodal paths may want torchvision;
# add them here once locked on the same +rocm date.
RUN pip install --index-url ${ROCM_INDEX_URL} "torch==${TORCH_VERSION}"

# Python build backends for the extension builds (mirrors upstream). setuptools<80 avoids the
# vllm/flash-attn setup.py breakage on the newer editable-install API.
RUN pip install --upgrade \
      cmake ninja packaging wheel numpy "setuptools-scm>=8" "setuptools<80.0.0" \
      scikit-build-core pybind11 numba scipy

# ---------------------------------------------------------------------------------------------
# flash-attention (ROCm fork) + aiter
# ---------------------------------------------------------------------------------------------
# Upstream installs flash-attn in-place; we instead build BOTH aiter and flash-attn as WHEELS
# into /artifacts/wheels. aiter must be built+installed FIRST (flash-attn's setup.py builds
# against it) and its bundled ck_tile headers patched for RDNA3.5 (gfx1151) scalar fallbacks
# before flash-attn compiles. The Fedora lib/ vs lib64/ site-packages merge from upstream is
# DROPPED — Debian venvs have a single lib/pythonX.Y/site-packages (no lib64 split).
ENV FLASH_ATTENTION_TRITON_AMD_ENABLE=TRUE \
    LD_LIBRARY_PATH=/opt/rocm/lib:/opt/rocm/lib64:$LD_LIBRARY_PATH

COPY patch_aiter_headers.py /opt/patch_aiter_headers.py

# Steps: clone flash-attn -> init aiter + composable_kernel submodules -> build the aiter wheel
# into /artifacts/wheels and install it -> patch installed aiter ck_tile headers for gfx1151
# (needed by the flash-attn build AND by aiter's runtime JIT; vllm-runtime re-patches its own
# copy) -> neutralize flash-attn setup.py's aiter-submodule build subprocess (aiter already
# built) -> build the flash-attn wheel (upstream pip-installs; we ship the wheel instead).
RUN mkdir -p /artifacts/wheels \
    && git clone "${FLASH_ATTENTION_REPO}" /opt/flash-attention \
    && cd /opt/flash-attention \
    && if [ -n "${FLASH_ATTENTION_REF}" ]; then git checkout "${FLASH_ATTENTION_REF}"; \
       else git checkout "${FLASH_ATTENTION_BRANCH}"; fi \
    && git submodule update --init third_party/aiter \
    && cd third_party/aiter \
    && git submodule update --init 3rdparty/composable_kernel \
    && export CK_DIR="$(pwd)/3rdparty/composable_kernel" \
    && python -m pip wheel --no-build-isolation --no-deps -w /artifacts/wheels -v . \
    && python -m pip install --force-reinstall /artifacts/wheels/amd_aiter*.whl \
    && python /opt/patch_aiter_headers.py \
    && cd /opt/flash-attention \
    && python -c "import re; f=open('setup.py','r'); t=f.read(); f.close(); t=re.sub(r'subprocess\.run\([\s\S]*?third_party/aiter[\s\S]*?check=True,\s*\)', 'pass # patched', t); f=open('setup.py','w'); f.write(t)" \
    && python -m pip wheel --no-build-isolation --no-deps -w /artifacts/wheels -v .

# ---------------------------------------------------------------------------------------------
# vLLM
# ---------------------------------------------------------------------------------------------
# Rust toolchain for vLLM's PyO3/_rust_*.so parser extensions (setuptools-rust backend).
# Fedora `dnf install rust cargo` -> Debian rustc/cargo. Kept after the flash-attn/aiter layers
# so those stay cacheable. python3.13-dev supplies Python.h + the cp313 dev components CMake
# FindPython(Development.Module/SABIModule) needs to configure vLLM's C++/HIP extensions against
# the venv interpreter (rocm-build-base ships no python dev headers; only vLLM compiles _C here).
# libdrm-dev: torch's LoadHIP.cmake runs pkg_check_modules(libdrm) via rocm_smi-config.cmake
# when vLLM does find_package(Torch); needs libdrm.pc + headers (rocm-build-base ships neither).
RUN apt-get -qq update \
    && apt-get -qq install -y --no-install-recommends rustc cargo python3.13-dev libdrm-dev \
    && rm -rf /var/lib/apt/lists/* && apt-get clean \
    && pip install "setuptools-rust>=1.9.0"

# Clone + patch vLLM. patch_strix.py (amdsmi stub, forced gfx1151, aiter/MoE/rmsnorm gating,
# clang-safe spinloop include) + patch_fp8_kernels.py (opt-in FP8 Triton dequant-GEMM shim).
COPY patch_strix.py       /opt/patch_strix.py
COPY patch_fp8_kernels.py /opt/patch_fp8_kernels.py
RUN git clone "${VLLM_REPO}" /opt/vllm \
    && cd /opt/vllm \
    && if [ -n "${VLLM_REF}" ]; then git checkout "${VLLM_REF}"; fi \
    && cp /opt/patch_strix.py /opt/patch_fp8_kernels.py . \
    && python patch_strix.py \
    && python patch_fp8_kernels.py

# Build the vLLM wheel with the ROCm clang host compiler (ABI-aligns vLLM's C++ extensions with
# torch — avoids the GCC-host segfault). NOTE the SDK layout difference: pip TheRock ships clang
# under /opt/rocm/lib/llvm/bin (Fedora tarball used /opt/rocm/llvm/bin).
ENV ROCM_HOME=/opt/rocm \
    HIP_PATH=/opt/rocm \
    HIP_PLATFORM=amd \
    VLLM_TARGET_DEVICE=rocm \
    PYTORCH_ROCM_ARCH=gfx1151 \
    HIP_ARCHITECTURES=gfx1151 \
    AMDGPU_TARGETS=gfx1151 \
    MAX_JOBS=4 \
    CC=/opt/rocm/lib/llvm/bin/clang \
    CXX=/opt/rocm/lib/llvm/bin/clang++
RUN cd /opt/vllm \
    && export HIP_DEVICE_LIB_PATH="$(find /opt/rocm -type d -name bitcode -print -quit)" \
    && echo "Compiling vLLM with bitcode: ${HIP_DEVICE_LIB_PATH}, clang: ${CC}" \
    && export CMAKE_ARGS="-DROCM_PATH=/opt/rocm -DHIP_PATH=/opt/rocm -DAMDGPU_TARGETS=gfx1151 -DHIP_ARCHITECTURES=gfx1151" \
    && python -m pip wheel --no-build-isolation --no-deps -w /artifacts/wheels -v .

# Pure-python FP8 Triton kernels (leonyurko). NOT a wheel — the modules live on PYTHONPATH at
# serve time (patch_fp8_kernels.py's shim does `from fp8_triton import fp8_gemm`, opt-in via
# VLLM_STRIX_FP8_TRITON=1). Carried as a source tree; vllm-runtime COPY's it to /opt/fp8.
RUN git clone "${FP8_KERNELS_REPO}" /artifacts/fp8 \
    && cd /artifacts/fp8 && git checkout "${FP8_KERNELS_REF}" \
    && rm -rf /artifacts/fp8/.git

RUN echo "=== /artifacts/wheels ===" && ls -1 /artifacts/wheels

# --- Final carrier: ONLY the wheels + FP8 kernel tree. No base, no torch, no ROCm SDK. ---
FROM scratch
COPY --from=build /artifacts /artifacts
