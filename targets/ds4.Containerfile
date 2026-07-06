# ds4-runtime: the shippable ds4 toolbox. FROM rocm-runtime-base (canopy + de-divert + the pinned
# TheRock gfx1151 runtime kernels in /opt/venv), plus ds4's compiled outputs COPY'd in from the
# ds4-artifacts carrier and only the app-level runtime ds4 itself needs. FIRST of the five ROCm
# toolbox ports — sets the runtime pattern the other four copy.
# The ROCm runtime .so (hip runtime, rocblas/hipblas/hipblaslt/MIOpen) already live in the base's
# rocm-sdk-libraries-gfx1151 wheel, so — unlike the Fedora source — this port re-adds NO ROCm libs
# (hipblaslt included). It only layers ds4's binaries + the huggingface CLI + the ds4 cockpit TUI.
ARG BASE_IMAGE=localhost/rocm-runtime-base
ARG ARTIFACTS_IMAGE=localhost/ds4-artifacts
# Local podman builds need a localhost/ prefix (already the defaults above); CI can override:
#   --build-arg BASE_IMAGE=ghcr.io/<owner>/rocm-runtime-base:<tag>
#   --build-arg ARTIFACTS_IMAGE=ghcr.io/<owner>/ds4-artifacts:<tag>
FROM ${ARTIFACTS_IMAGE} AS artifacts

FROM ${BASE_IMAGE}

LABEL description="ds4-runtime: ds4 (gfx1151) inference toolbox + cockpit TUI on rocm-runtime-base"

# --- Pinned cockpit ref (reproducibility). Default = strix-halo-ds4-toolbox submodule HEAD
#     (`git -C upstream/ds4 rev-parse HEAD`); the cockpit pip package is the repo's subdirectory. ---
ARG COCKPIT_REF=84a580d8c8b03602143a9f2a14183f3b45e3bbc1
ARG COCKPIT_REPO=https://github.com/kyuz0/strix-halo-ds4-toolbox.git

# --- Pull in ds4's own build outputs from the ds4-artifacts carrier. ---
# The seam: bin -> /usr/local/bin, lib64 -> /usr/local/lib64. share is carried for pattern parity
# (empty for ds4 today). The carrier image is the ARTIFACTS_IMAGE ARG above (overridable in CI).
COPY --from=artifacts /artifacts/bin/   /usr/local/bin/
COPY --from=artifacts /artifacts/lib64/ /usr/local/lib64/
COPY --from=artifacts /artifacts/share/ /usr/local/share/

# Make ds4's shared libs resolvable without touching env (mirrors the Fedora runtime's local.conf).
# The base already wires /opt/rocm/lib{,64}; this adds the COPY'd /usr/local/lib{,64}.
RUN printf '/usr/local/lib\n/usr/local/lib64\n' > /etc/ld.so.conf.d/ds4-local.conf \
    && ldconfig

# --- App-level Python runtime (into the base's /opt/venv — VIRTUAL_ENV/PATH are set by the base). ---
# huggingface CLI for model downloads; the hf_xet extra flips on HF_XET_HIGH_PERFORMANCE=1. python3-pip
# is NOT re-added — rocm-runtime-base already installs it, and `pip` here is the venv pip (PEP668-safe).
RUN pip install --no-cache-dir 'huggingface_hub[hf_xet]'

# --- ds4 cockpit TUI, from the PINNED git ref, isolated via pipx --global. ---
# git is a genuinely-missing runtime dep here (canopy/rocm-runtime-base ship none) and is required to
# resolve the git+https spec — added minimally. pipx is pip-installed into the base venv, then invoked
# with --global so the cockpit gets its OWN isolated venv at /opt/pipx and its launcher at
# /usr/local/bin/ds4-cockpit — both container-owned (NOT the distrobox-shared host ~/.local, which
# PYTHONNOUSERSITE below also guards against). This mirrors droste's kento `pipx install --global`.
RUN apt-get -qq update \
    && apt-get -qq install -y --no-install-recommends git \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean
RUN pip install --no-cache-dir pipx \
    && pipx install --global \
       "git+${COCKPIT_REPO}@${COCKPIT_REF}#subdirectory=ds4-strix-halo-cockpit"

# Distrobox/Toolbox shares the host home directory. Prevent the host's ~/.local/lib/python* from
# shadowing container-installed packages (e.g. the hf CLI importing a host-side broken huggingface_hub).
ENV PYTHONNOUSERSITE=1

# Interactive toolbox entrypoint (matches the Fedora runtime stage).
CMD ["/bin/bash"]
