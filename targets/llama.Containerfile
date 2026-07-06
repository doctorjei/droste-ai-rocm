# llama-runtime: thin gfx1151 llama.cpp toolbox on rocm-runtime-base. COPY --from the
# llama-artifacts scratch carrier (bins + libllama*.so + vram helper) onto the ROCm runtime
# kernels the base already carries — NO ROCm re-adds here (rocm-sdk-libraries-gfx1151 with
# hip-runtime/rocblas/hipblas/hipblaslt is inherited). The base already writes a real
# /etc/profile.d/rocm.sh, so the upstream empty-profile bug does not apply.
ARG BASE_IMAGE=localhost/rocm-runtime-base
ARG ARTIFACTS_IMAGE=localhost/llama-artifacts
# Local podman builds need a localhost/ prefix (already the defaults above); CI can override:
#   --build-arg BASE_IMAGE=ghcr.io/<owner>/rocm-runtime-base:<tag>
#   --build-arg ARTIFACTS_IMAGE=ghcr.io/<owner>/llama-artifacts:<tag>
FROM ${ARTIFACTS_IMAGE} AS artifacts

FROM ${BASE_IMAGE}

LABEL description="llama-runtime: gfx1151 llama.cpp (turboquant) toolbox on rocm-runtime-base"

# libgomp1: llama-server links libgomp.so.1 (OpenMP); the lean rocm-runtime-base doesn't carry
# it, so the binary fails at load with "libgomp.so.1: cannot open shared object file". Verified
# on gfx1151 hardware 2026-07-06.
RUN apt-get -qq update \
    && apt-get -qq install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/* && apt-get clean

# Drop llama.cpp's compiled outputs onto the runtime: bins → /usr/local/bin, libllama/libggml
# → /usr/local/lib64, vram helper → /usr/local/bin (executable). /usr/local/lib{,64} are already
# on the base's ld path via /etc/ld.so.conf.d (rocm.conf) — add local.conf + ldconfig to be sure.
COPY --from=artifacts /artifacts/bin/   /usr/local/bin/
COPY --from=artifacts /artifacts/lib64/ /usr/local/lib64/
COPY --from=artifacts /artifacts/share/gguf-vram-estimator.py /usr/local/bin/gguf-vram-estimator.py

RUN chmod +x /usr/local/bin/gguf-vram-estimator.py \
    && printf '%s\n' /usr/local/lib /usr/local/lib64 > /etc/ld.so.conf.d/local.conf \
    && ldconfig

# Interactive toolbox: default to a shell (distrobox injects the host user at run time).
CMD ["/bin/bash"]
