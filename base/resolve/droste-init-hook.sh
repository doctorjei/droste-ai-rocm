#!/usr/bin/env bash
# droste-init-hook.sh — DISTROBOX-lane resolver invocation, for distrobox.ini
# init_hooks (see targets/<port>/distrobox.ini for per-port examples).
#
# distrobox/toolbx replace pid1, so the image ENTRYPOINT never runs there; this
# wrapper is the distrobox counterpart: it sources the same shared resolver +
# per-port build-spec and runs resolve::apply_spec with DROSTE_LANE=distrobox —
# the mount primitives (overlay/surface/cache) become no-ops ($HOME is the
# persistent host home), while the non-mount steps still run: /opt/data check,
# CRITICAL binds (declare them as volume= lines in distrobox.ini), OPTIONAL
# marker, templates.yaml seeding, ENV_FILE source, PRE_LAUNCH. No exec — the
# service is started by the user (or not at all; distrobox is the interactive
# lane). Idempotent: init_hooks run on every cold start.
set -euo pipefail

export DROSTE_LANE=distrobox

# init_hooks run as root (HOME=/root), but the spec's $HOME-relative paths must
# resolve to the DISTROBOX USER's home (the host-home bind — that is what makes
# e.g. the HF-cache CRITICAL read as bound). Derive it from the first regular
# user distrobox created (uid >= 1000); DROSTE_USER_HOME overrides.
if [ -z "${DROSTE_USER_HOME:-}" ]; then
    DROSTE_USER_HOME=$(awk -F: '$3 >= 1000 && $3 < 65534 { print $6; exit }' /etc/passwd)
fi
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
