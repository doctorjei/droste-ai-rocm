#!/usr/bin/env bash
# droste-init-hook.sh — DISTROBOX-lane resolver invocation, for distrobox.ini
# init_hooks (see targets/<port>/distrobox.ini for per-port examples).
#
# distrobox/toolbx replace pid1, so the image ENTRYPOINT never runs there; this
# wrapper is the distrobox counterpart: it sources the same shared resolver +
# per-port build-spec and runs resolve::apply_spec with DROSTE_LANE=distrobox.
# Since lane unification the hook performs the SAME mounts as the server lane —
# overlays (venv/custom-node uppers on /opt/data), surfaces, cache binds — so
# container-lifecycle events never destroy in-box state (the founding
# requirement). Order is apply_spec's: ensure_data → surfaces/overlays/caches →
# CRITICAL binds (checked AFTER the mounts; declare them as volume= lines in
# distrobox.ini — the HF cache is satisfied by the auto-bound real home) →
# OPTIONAL marker → templates.yaml seeding → ENV_FILE source → PRE_LAUNCH.
# No exec — the service is started by the user (or not at all; distrobox is
# the interactive lane). In-box mounting needs CAP_SYS_ADMIN + /dev/fuse:
# additional_flags="--cap-add sys_admin --device /dev/fuse" in the ini.
# Idempotent: init_hooks run on every container start; every resolver mount
# skips when its exact target is already a mountpoint.
set -euo pipefail

export DROSTE_LANE=distrobox

# init_hooks run as root (HOME=/root), but the spec's $HOME-relative paths must
# resolve to the DISTROBOX USER's home (the host-home bind — that is what makes
# e.g. the HF-cache CRITICAL read as bound). Derive user + home from the first
# regular user distrobox created (uid >= 1000); DROSTE_USER / DROSTE_USER_HOME
# override. Both are EXPORTED for the resolver's lane deviations: it remaps
# /root/-prefixed SURFACE/CACHE dests to $DROSTE_USER_HOME and chowns the dirs
# it creates to $DROSTE_USER (this hook runs as root; the box user is not).
if [ -z "${DROSTE_USER:-}" ]; then
    DROSTE_USER=$(awk -F: '$3 >= 1000 && $3 < 65534 { print $1; exit }' /etc/passwd)
fi
if [ -z "${DROSTE_USER_HOME:-}" ]; then
    DROSTE_USER_HOME=$(awk -F: '$3 >= 1000 && $3 < 65534 { print $6; exit }' /etc/passwd)
fi
export DROSTE_USER DROSTE_USER_HOME
if [ -n "${DROSTE_USER_HOME:-}" ]; then
    export HOME="$DROSTE_USER_HOME"
fi

RESOLVE_DIR=${RESOLVE_DIR:-/opt/resources/resolve}
# shellcheck source=/dev/null
source "$RESOLVE_DIR/droste-resolve.sh"

SPEC=${DROSTE_BUILD_SPEC:-/opt/resources/build-spec}
if [ ! -f "$SPEC" ]; then
    resolve::err "build-spec not found at $SPEC"
    exit 1
fi

# Row defaults BEFORE sourcing the spec (set -u safety; spec may omit any row).
SERVICE=()
ENV_FILE=""
OVERLAYS=()
SURFACES=()
CRITICAL=()
OPTIONAL=()
CACHES=()
PRE_LAUNCH=""

# shellcheck source=/dev/null
source "$SPEC"

resolve::apply_spec
