#!/usr/bin/env bash
# droste-entrypoint.sh — shared SERVER-lane ENTRYPOINT for all 5 ports.
#
# Ports opt in with `ENTRYPOINT ["droste-entrypoint.sh"]` (on PATH) or the absolute
# path. It sources the per-port /opt/resources/build-spec, runs the resolver in the
# EXACT design order (resolve::apply_spec), then execs the service.
#
# This entrypoint only ever runs in the SERVER lane — distrobox/toolbx replace pid1 and
# bypass it, calling the resolver library directly from init_hooks (lane=distrobox).
#
# Standard entrypoint convention: if the user supplies a command
# (`podman run IMAGE bash`), exec THAT after resolving; otherwise exec SERVICE.
# NOTE: the CRITICAL checks still run first, so a quick UNBOUND shell needs
# `-e ALLOW_EPHEMERAL=1` — without it, no binds means a hard-error before exec.
set -euo pipefail

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

export DROSTE_LANE=server
resolve::apply_spec

# User-supplied command wins (keeps `podman run -it IMAGE bash` working).
if [ "$#" -gt 0 ]; then
    exec "$@"
fi

if [ ${#SERVICE[@]} -eq 0 ]; then
    resolve::err "no SERVICE defined in $SPEC and no command was given"
    exit 1
fi

exec "${SERVICE[@]}"
