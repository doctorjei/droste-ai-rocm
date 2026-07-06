# vllm-runtime: the shippable vLLM toolbox for Strix Halo / gfx1151. FROM rocm-runtime-base
# (TheRock runtime kernels, /opt/venv, ROCM_PATH=/opt/rocm) + the pinned TheRock torch, then
# `COPY --from` the flash-attention/aiter/vLLM wheels compiled in vllm-artifacts and pip-install
# them into the base venv. No compilers, no ROCm -dev — pure runtime.
#
# Translated from droste-ai-rocm/upstream/vllm/Dockerfile (kyuz0). Toolbox submodule provenance
# (droste-ai-rocm): 6446b9595273f289e11586c3c7d3e1e6f2945888
ARG BASE_IMAGE=localhost/rocm-runtime-base
ARG ARTIFACTS_IMAGE=localhost/vllm-artifacts
# CI can override both, e.g. --build-arg ARTIFACTS_IMAGE=ghcr.io/<owner>/vllm-artifacts:<tag>
FROM ${ARTIFACTS_IMAGE} AS artifacts
FROM ${BASE_IMAGE}

LABEL description="vllm-runtime: vLLM toolbox for gfx1151 (TheRock torch + prebuilt wheels)"

# --- The unified pin (rocm-pin.env). ARG names MUST match the env keys the wrapper passes. ---
ARG ROCM_INDEX_URL=https://rocm.nightlies.amd.com/v2/gfx1151/
ARG TORCH_VERSION=2.9.1+rocm7.13.0a20260501
ARG GFX_TARGET=gfx1151

# Torch (pinned TheRock nightly) into the base venv — must match the torch the wheels were
# compiled against (same pin). Installed first so the vLLM wheel's torch requirement is already
# satisfied and pip does NOT pull a PyPI/CUDA torch over it. FLAG: if current vLLM main pins an
# exact torch that 2.9.1 doesn't satisfy, pip will try to replace it — reconcile on-host.
RUN pip install --index-url ${ROCM_INDEX_URL} "torch==${TORCH_VERSION}"

# Runtime libs: libnuma (vLLM numa lookup on `import vllm`) + libgomp1 — torch links
# libgomp.so.1 (OpenMP), which the lean rocm-runtime-base does NOT carry, so `import torch`
# (and thus `import vllm`) fails without it. Verified on gfx1151 hardware 2026-07-06.
RUN apt-get -qq update \
    && apt-get -qq install -y --no-install-recommends libnuma1 numactl libgomp1 \
    && rm -rf /var/lib/apt/lists/* && apt-get clean

# Prebuilt wheels + the pure-python FP8 kernel tree from vllm-artifacts.
COPY --from=artifacts /artifacts/wheels /tmp/wheels
COPY --from=artifacts /artifacts/fp8    /opt/fp8
COPY patch_aiter_headers.py /opt/patch_aiter_headers.py

# Install order matters:
#  1. aiter + flash-attn wheels (--no-deps: their deps are torch/triton, already present).
#  2. re-apply patch_aiter_headers.py to the freshly-installed aiter — its ck_tile headers are
#     used by aiter's RUNTIME JIT on gfx1151 (the artifacts stage patched its own copy; the
#     shipped wheel is pre-patch, so patch it again here).
#  3. vLLM wheel WITH deps so pip resolves transformers/etc. from PyPI (torch already satisfied).
#     Use the legacy resolver for this RUN: vLLM pins `torch==2.9.1` while the installed wheel is
#     `2.9.1+rocm7.13.0a20260501`, and the new resolvelib backtracker explores an exploding search
#     space over that (+ the large dep graph) and dies with RecursionError in _has_route_to_root
#     (survives even a 100k recursion limit — it's pathological backtracking, not mere depth). The
#     legacy resolver takes the first satisfying candidate (our torch satisfies ==2.9.1) and skips
#     the backtracking entirely. Scoped to this RUN via the env export.
#  4. ray (upstream installs it explicitly; used by the cluster launcher paths).
RUN export PIP_USE_DEPRECATED=legacy-resolver \
    && pip install --no-deps /tmp/wheels/amd_aiter*.whl /tmp/wheels/flash_attn*.whl \
    && python /opt/patch_aiter_headers.py \
    && pip install /tmp/wheels/vllm*.whl \
    && pip install ray \
    && rm -rf /tmp/wheels /root/.cache/pip

# Runtime shell env + banner (upstream ships these in /etc/profile.d). 01-rocm-env-for-triton.sh
# sets the gfx1151/Triton/vLLM serve-time env; 99-toolbox-banner.sh prints the toolbox banner;
# zz-venv-last.sh keeps /opt/venv/bin first on PATH under distrobox user dotfiles.
COPY 01-rocm-env-for-triton.sh /etc/profile.d/01-rocm-env-for-triton.sh
COPY 99-toolbox-banner.sh      /etc/profile.d/99-toolbox-banner.sh
COPY zz-venv-last.sh           /etc/profile.d/zz-venv-last.sh
RUN chmod 0644 /etc/profile.d/01-rocm-env-for-triton.sh \
      /etc/profile.d/99-toolbox-banner.sh /etc/profile.d/zz-venv-last.sh

# The FP8 shim (patch_fp8_kernels.py, baked into the vLLM wheel) imports fp8_triton from here at
# serve time when VLLM_STRIX_FP8_TRITON=1. Also mirror the Triton/vLLM env into the image env so
# non-login shells (podman exec, distrobox) get it without sourcing /etc/profile.d.
ENV PYTHONPATH=/opt/fp8 \
    FLASH_ATTENTION_TRITON_AMD_ENABLE=TRUE \
    VLLM_TARGET_DEVICE=rocm \
    TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL=1 \
    MIOPEN_FIND_MODE=FAST \
    VLLM_DISABLE_COMPILE_CACHE=1 \
    PYTHONNOUSERSITE=1

CMD ["/bin/bash"]
