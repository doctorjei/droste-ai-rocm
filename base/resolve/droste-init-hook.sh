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

# Group membership for the writable baked venv (fix: distrobox-lane pip installs).
# The image makes /opt/venv (+ custom_nodes) group-writable under group `droste`;
# the box user must belong to that group to write after overlay copy-up. Rootless
# userns blocks runtime setgid(), so membership must be granted at LOGIN — this hook
# runs as root before distrobox's login shell, so usermod here takes effect there.
# Best-effort: never abort startup if the group is absent (older image) or usermod fails.
if [ -n "${DROSTE_USER:-}" ] && getent group droste >/dev/null 2>&1; then
    usermod -aG droste "$DROSTE_USER" 2>/dev/null \
        || printf 'droste-init-hook: WARN: could not add %s to group droste\n' "$DROSTE_USER" >&2
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

# Surface resolver diagnostics. distrobox shows only a generic "An error occurred"
# for a failed init hook, hiding the resolver's actionable CRITICAL message. That
# failure often arrives as an internal `exit 1` from resolve::critical (sourced,
# runs in THIS shell), so we catch it with an EXIT trap — not `|| rc=$?`, which the
# direct exit bypasses — and write stderr to a log SYNCHRONOUSLY (a backgrounded
# tee could be killed mid-flush by that exit, truncating the log).
RESOLVE_LOG="${DROSTE_DATA_DIR:-/opt/data}/.droste-resolve.log"
# Fall back to /tmp if the data dir isn't writable (ro bind, missing, etc.) so the
# redirect itself can never abort the hook.
if ! ( : >>"$RESOLVE_LOG" ) 2>/dev/null; then
    RESOLVE_LOG="/tmp/droste-resolve.log"
fi
# On ANY non-zero exit (including resolve::critical's internal exit 1) dump the log
# to stderr with a pointer; on success this is a no-op.
trap 'ec=$?; if [ "$ec" -ne 0 ]; then { printf "droste-init-hook: resolver FAILED (exit %s). Detail (also saved to %s):\n" "$ec" "$RESOLVE_LOG"; tail -n 30 "$RESOLVE_LOG" 2>/dev/null; } >&2; fi' EXIT
resolve::apply_spec 2>"$RESOLVE_LOG"
# Success path: surface the resolver's own INFO/WARN lines (fuse fallback, etc.) too.
cat "$RESOLVE_LOG" >&2
