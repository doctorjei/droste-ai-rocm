# rocm-runtime-base: TheRock gfx1151 ROCm RUNTIME kernels on canopy (no init).
# Root of the unified ROCm lineage — replaces amd-runtime-base + therock-torch-base.
# ONE pinned TheRock nightly (rocm-pin.env) feeds every image: the runtime kernels
# (rocm-sdk-libraries-gfx1151) are pip-installed into a venv; NO apt ROCm repo, NO S3
# tarball. rocm-build-base builds FROM this, so the de-divert runs exactly once and any
# compiled artifacts ABI-match the runtime libs shipped here.
FROM ghcr.io/doctorjei/gemet/canopy:1.7.3

LABEL description="rocm-runtime-base: TheRock gfx1151 ROCm runtime kernels (canopy + de-divert, pip venv)"

# Canopy routes coreutils/etc. to busybox via .distrib dpkg diversions; pip/apt postinsts
# need real GNU binaries. Remove the .distrib diversions, then rehydrate GNU coreutils
# BEFORE any other install. (Lifted verbatim from amd-runtime-base / droste-seed.) Inherited
# by rocm-build-base, so the de-divert happens exactly once for the whole ROCm lineage.
RUN while IFS= read -r orig && IFS= read -r div && IFS= read -r owner; do \
        case "$div" in \
            *.distrib) dpkg-divert --quiet --no-rename --remove "$orig" ;; \
        esac; \
    done < /var/lib/dpkg/diversions

RUN apt-get -qq update \
    && apt-get -qq install -y --no-install-recommends \
       coreutils grep sed findutils gzip diffutils \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

# --- The unified pin (rocm-pin.env). A build wrapper sources rocm-pin.env and passes each
# key as --build-arg; these ARG names MUST match the env keys. ---
ARG ROCM_INDEX_URL=https://rocm.nightlies.amd.com/v2/gfx1151/
ARG ROCM_SDK_LIBRARIES_VERSION=7.13.0a20260501
ARG GFX_TARGET=gfx1151

# Re-add exactly what canopy purges (per the canopy PURGELIST) plus the venv toolchain.
# Canopy KEEPS bash/coreutils/libstdc++6/libgcc-s1/libgomp1/ca-certificates/python3 (3.13) —
# do NOT re-add those. It PURGES sudo/procps/libnss-myhostname: sudo+procps for interactive
# use, libnss-myhostname for distrobox host-name resolution. radeontop is a Debian-main GPU
# monitor (not in any gemet base). python3-venv/python3-pip are NOT present in canopy and are
# needed to build the /opt/venv and pip-install the ROCm wheels.
RUN apt-get -qq update \
    && apt-get -qq install -y --no-install-recommends \
       sudo procps libnss-myhostname radeontop \
       python3-venv python3-pip \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

# Python venv with the TheRock gfx1151 RUNTIME kernels. Debian 13 is PEP668
# externally-managed, so we install into a venv (preferred, matches therock-torch-base)
# rather than --break-system-packages. Install the `rocm` meta with the [libraries] extra:
# `rocm[libraries]` == rocm (provides the `rocm-sdk` CLI + rocm_sdk module) + rocm-sdk-core
# (rocminfo/rocm-smi/hipcc console scripts + the HIP runtime .so / _rocm_sdk_core tree) +
# rocm-sdk-libraries-gfx1151 (rocBLAS/hipBLASLt/MIOpen in its own _rocm_sdk_libraries_gfx1151
# tree). Installing the meta (not the bare leaves) is what makes `rocm-sdk path` work below.
RUN python3 -m venv /opt/venv
ENV VIRTUAL_ENV=/opt/venv PATH=/opt/venv/bin:$PATH PIP_NO_CACHE_DIR=1
RUN pip install --upgrade pip setuptools wheel \
    && pip install --index-url ${ROCM_INDEX_URL} \
       "rocm[libraries]==${ROCM_SDK_LIBRARIES_VERSION}"

# Pin a stable /opt/rocm -> the ROCm core tree. NOTE: `rocm-sdk path --root` CANNOT be used here
# — it routes through the devel module and errors without rocm[devel] (which this runtime image
# does not carry). The core tree is deterministically <venv-site-packages>/_rocm_sdk_core, so
# derive it directly (no CLI, no devel). ld.so wiring lets the runtime .so resolve for non-venv
# procs: emit /opt/rocm/lib{,64} FIRST, then every sibling _rocm_sdk_*/lib tree (so rocBLAS/
# hipBLASLt/MIOpen from the split libraries wheel resolve too), then ldconfig.
RUN ln -sfn "$(python3 -c "import sysconfig,os; print(os.path.join(sysconfig.get_paths()['purelib'],'_rocm_sdk_core'))")" /opt/rocm \
    && printf '/opt/rocm/lib\n/opt/rocm/lib64\n' > /etc/ld.so.conf.d/rocm.conf \
    && python3 -c "import sysconfig,glob,os; sp=sysconfig.get_paths()['purelib']; print('\n'.join(glob.glob(os.path.join(sp,'_rocm_sdk_*','lib'))))" >> /etc/ld.so.conf.d/rocm.conf \
    && ldconfig

ENV ROCM_PATH=/opt/rocm HIP_PATH=/opt/rocm \
    LD_LIBRARY_PATH=/opt/rocm/lib:/opt/rocm/lib64 \
    PATH=/opt/venv/bin:/opt/rocm/bin:$PATH

# Real /etc/profile.d/rocm.sh (the old llama Containerfile wrote EMPTY here — bug). Activates
# the venv and exports ROCM_PATH from the stable /opt/rocm symlink (built above), then the
# HIP/PATH/LD env for interactive shells.
RUN printf '%s\n' \
    'source /opt/venv/bin/activate' \
    'ROCM_PATH=/opt/rocm' \
    'export ROCM_PATH HIP_PATH="$ROCM_PATH"' \
    'export PATH="$ROCM_PATH/bin:/opt/venv/bin:$PATH"' \
    'export LD_LIBRARY_PATH="$ROCM_PATH/lib:$ROCM_PATH/lib64:$LD_LIBRARY_PATH"' \
    > /etc/profile.d/rocm.sh

# No CMD.
