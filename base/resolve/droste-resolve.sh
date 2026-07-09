#!/usr/bin/env bash
# droste-resolve.sh — shared runtime resolver library (SOURCED, not executed).
#
# Baked into the runtime base image; all 5 ports (comfyui/finetuning/vllm/llama/ds4)
# inherit ONE copy. The per-port `build-spec` file declares WHICH primitives to apply;
# this library IS the primitives. See build-spec.example for the contract.
#
# Two lanes (DROSTE_LANE):
#   server    (default) — image ENTRYPOINT runs as root; overlays/surfaces/caches are
#                         mounted to redirect app writes onto the /opt/data volume.
#   distrobox           — distrobox init_hooks source this lib, set DROSTE_LANE=distrobox,
#                         and call the SAME primitives. Since lane unification the mounts
#                         run in-box too: container-lifecycle events must never destroy
#                         user state (venv/custom-node installs land on /opt/data, not in
#                         the container layer). Needs CAP_SYS_ADMIN in the box — ini:
#                         additional_flags="--cap-add sys_admin --device /dev/fuse".
#                         Deliberate lane DEVIATIONS, each commented at its site:
#                           - /root/-prefixed SURFACE/CACHE dests remap to the box user's
#                             home (_lane_dest — server runs as root, the box user doesn't);
#                           - dirs the resolver creates are chowned to the box user
#                             (_mkuserdir / copy-mode chown — the hook runs as root);
#                           - every mount skips if its target is already a mountpoint
#                             (_is_mountpoint — init_hooks re-run on every start);
#                           - the HF cache is NEVER resolver-mounted in either lane (it is
#                             a CRITICAL user bind; in-box the auto-bound real home
#                             satisfies it natively).
#
# Sourced by a caller that has already set `set -euo pipefail`; kept in effect here so a
# failing primitive aborts container startup loudly.
set -euo pipefail

# ── Config (override via env before sourcing) ───────────────────────────────
: "${DROSTE_LANE:=server}"
: "${DROSTE_DATA_DIR:=/opt/data}"
: "${DROSTE_CACHES_DIR:=/opt/caches}"   # optional SHARED compute-cache volume (see cache_bind)
: "${RESOLVE_TEMPLATES_DIR:=/opt/resources/templates}"
: "${RESOLVE_APPLY_TEMPLATES:=/opt/resources/resolve/apply_templates.py}"
: "${DROSTE_OVERLAY_MODE:=auto}"   # auto (kernel→fuse→copy) | kernel | fuse | copy (non-auto = forced, no fallback)
# DROSTE_RESOLVE_DRYRUN (bare name) — echo mount commands instead of running them.
# ALLOW_EPHEMERAL     (bare name) — downgrade a CRITICAL unbound hard-error to a warning.

# ── Messaging ───────────────────────────────────────────────────────────────
resolve::info() { printf 'droste-resolve: INFO: %s\n' "$*" >&2; }
resolve::warn() { printf 'droste-resolve: WARN: %s\n' "$*" >&2; }
resolve::err()  { printf 'droste-resolve: ERROR: %s\n' "$*" >&2; }

# ── Mount helpers ───────────────────────────────────────────────────────────
# _domount — run a mount command, or echo it under DROSTE_RESOLVE_DRYRUN (CI/tests
# have no mount privileges here). Real mounts happen only on a capable host.
resolve::_domount() {
    if [ -n "${DROSTE_RESOLVE_DRYRUN:-}" ]; then
        printf 'droste-resolve: [dryrun] %s\n' "$*" >&2
        return 0
    fi
    "$@"
}

# _try_mount — run a mount-ish command, capturing its stderr into RESOLVE_MOUNT_ERR for
# diagnostics; returns the command's status. Dryrun echoes (like _domount) and succeeds.
RESOLVE_MOUNT_ERR=""
resolve::_try_mount() {
    RESOLVE_MOUNT_ERR=""
    if [ -n "${DROSTE_RESOLVE_DRYRUN:-}" ]; then
        printf 'droste-resolve: [dryrun] %s\n' "$*" >&2
        return 0
    fi
    local rc=0
    RESOLVE_MOUNT_ERR=$("$@" 2>&1) || rc=$?
    return $rc
}

# Classify a failed mount's stderr (RESOLVE_MOUNT_ERR).
# EPERM shape: rootless podman strips CAP_SYS_ADMIN, so EVERY in-container mount
# (overlay AND plain bind) fails "permission denied" without --cap-add sys_admin.
resolve::_mount_err_is_eperm() {
    local e=${RESOLVE_MOUNT_ERR,,}
    case "$e" in
        *"permission denied"*|*"operation not permitted"*) return 0 ;;
    esac
    return 1
}
# FEATURE shape: kernel overlayfs requires the upperdir fs to support O_TMPFILE +
# RENAME_WHITEOUT; ecryptfs/NFS/virtiofs-class filesystems under /opt/data don't,
# and mount reports the generic "wrong fs type, bad option, bad superblock" line.
resolve::_mount_err_is_feature() {
    local e=${RESOLVE_MOUNT_ERR,,}
    case "$e" in
        *"wrong fs type"*|*"bad option"*|*"bad superblock"*) return 0 ;;
    esac
    return 1
}

# _err_need_cap — the CAP_SYS_ADMIN diagnostic shared by every mount path.
resolve::_err_need_cap() {
    resolve::err "$1: mount failed with a permission error — the resolver needs CAP_SYS_ADMIN to mount inside the container."
    resolve::err "  server lane: add '--cap-add sys_admin' to the run command; distrobox lane: additional_flags=\"--cap-add sys_admin --device /dev/fuse\" in distrobox.ini."
    resolve::err "  (Rootless podman: the capability is namespaced to the container's user namespace — it grants no host privilege.)"
}

# _bind — plain bind mount with the EPERM diagnostic (surfaces, caches, copy mode).
resolve::_bind() {
    local src=$1 dest=$2
    if resolve::_try_mount mount --bind "$src" "$dest"; then
        return 0
    fi
    if resolve::_mount_err_is_eperm; then
        resolve::_err_need_cap "bind $src -> $dest"
        return 1
    fi
    resolve::err "bind mount $src -> $dest failed: ${RESOLVE_MOUNT_ERR:-unknown error}"
    return 1
}

# _mountinfo — the mountinfo source. DROSTE_RESOLVE_MOUNTINFO overrides it (tests point
# it at a fixture); defaults to the live /proc/self/mountinfo.
resolve::_mountinfo() {
    printf '%s' "${DROSTE_RESOLVE_MOUNTINFO:-/proc/self/mountinfo}"
}

# is_bound — lane-agnostic bind detection via ANCESTOR-WALK. A `-v` bind (server) and a
# distrobox `volume=` bind look identical here; so does an ancestor bind (e.g. the whole
# distrobox `$HOME`). We find the LONGEST mountinfo mount-point (field 5) that is a
# whole-path-component prefix of <target> (so /opt/data does NOT match /opt/database):
#   longest prefix is the container rootfs "/"  → UNBOUND (nothing covers it)
#   any deeper mount covers the path            → BOUND
# This makes a critical path UNDER an ancestor bind (distrobox $HOME/.cache/huggingface)
# read as bound, while a bare container-rootfs path reads as unbound. Paths with spaces
# are not supported (none of ours have them).
resolve::is_bound() {
    local target=$1
    [ "$target" != "/" ] && target=${target%/}
    awk -v t="$target" '
        function is_prefix(mp, tgt) {
            if (mp == tgt) return 1
            if (mp == "/") return 1
            if (substr(tgt, 1, length(mp) + 1) == mp "/") return 1
            return 0
        }
        {
            mp = $5
            if (is_prefix(mp, t) && length(mp) > blen) { blen = length(mp); best = mp }
        }
        END { if (best == "" || best == "/") exit 1; exit 0 }
    ' "$(resolve::_mountinfo)"
}

# _anon_volume — is <target> mounted on a container-runtime ANONYMOUS volume?
# `VOLUME /opt/data` makes the runtime auto-mount one whenever the user binds
# nothing, so is_bound alone can't catch a forgotten -v. Anonymous podman/docker
# volumes materialize in the volume store as .../volumes/<64-hex-name>/_data
# (named volumes carry the human-readable name there instead). Check the exact
# mountinfo line for <target>: field 4 (root) AND the post-separator source field.
resolve::_anon_volume() {
    local target=$1 fields root src
    fields=$(awk -v t="$target" '
        $5 == t {
            src = "?"
            for (i = 7; i <= NF; i++) if ($i == "-") { src = $(i + 2); break }
            print $4
            print src
            exit
        }
    ' "$(resolve::_mountinfo)")
    [ -n "$fields" ] || return 1
    root=${fields%%$'\n'*}
    src=${fields#*$'\n'}
    local re='/volumes/[0-9a-f]{64}/_data$'
    [[ $root =~ $re || $src =~ $re ]]
}

# _is_mountpoint — is <target> ITSELF a mountpoint (exact mountinfo field-5 match)?
# DISTINCT SEMANTICS from is_bound: is_bound's ancestor-walk answers "is this path
# covered by any bind" (right for criticals — the whole distrobox $HOME bind counts);
# here we answer "did something already mount exactly HERE", the re-entry guard for
# init_hooks, which re-run on every container start within the same boot — mounting
# again would stack a second layer over our own earlier mount.
resolve::_is_mountpoint() {
    local target=$1
    [ "$target" != "/" ] && target=${target%/}
    awk -v t="$target" '$5 == t { found = 1; exit } END { exit !found }' \
        "$(resolve::_mountinfo)"
}

# _lane_dest — remap a spec dest for the current lane. WHY: build-spec dest paths are
# written for the SERVER lane, which runs as root ($HOME=/root) — caches target
# /root/.cache/*, ds4's surface targets /root/.ds4. In the distrobox lane the box user
# is non-root and the app runs in THEIR home, so a literal /root/ prefix would park
# state where the box user never looks. Remap the /root prefix to the box user's home
# (DROSTE_USER_HOME, exported by droste-init-hook.sh; fall back to /root = unchanged).
# NOTE: the hook also re-exports HOME before sourcing the spec, so $HOME-relative spec
# paths normally arrive already user-homed — this catches literal /root strings and
# the failed-user-derivation case.
resolve::_lane_dest() {
    local dest=$1
    if [ "$DROSTE_LANE" = distrobox ]; then
        case "$dest" in
            /root/*) dest="${DROSTE_USER_HOME:-/root}${dest#/root}" ;;
        esac
    fi
    printf '%s' "$dest"
}

# _mkuserdir — mkdir -p with the distrobox OWNERSHIP deviation. WHY: init_hooks run as
# root but the box user is non-root; dirs the resolver creates (overlay upper/work,
# surface/cache src, dests inside the auto-bound REAL home) must be writable by — and
# on host-visible paths, owned by — the box user, or overlay copy-ups and cache writes
# fail EACCES (and root-owned litter lands in the host home). Chowns the leaf plus any
# path components created THIS run, NON-recursively: pre-existing content (root-baked
# lowers, files a server-lane run left in an upper) is deliberately never touched —
# reads work, and new writes land in user-owned dirs. Server lane: plain mkdir -p.
resolve::_mkuserdir() {
    local dir=$1
    if [ "$DROSTE_LANE" != distrobox ] || [ -z "${DROSTE_USER:-}" ]; then
        mkdir -p "$dir"
        return 0
    fi
    local created=() d
    d=$(dirname "$dir")
    while [ ! -e "$d" ] && [ "$d" != / ]; do
        created+=("$d")
        d=$(dirname "$d")
    done
    mkdir -p "$dir"
    local m
    for m in "$dir" ${created[@]+"${created[@]}"}; do
        resolve::_domount chown "$DROSTE_USER:" "$m"
    done
}

# ── Primitive: overlay (BOTH lanes since lane unification) ──────────────────
# entry form: <upper>:<lower>  (upper = /opt/data side, lower = baked app dir)
# Mounts a writable layer OVER the baked lower. Strategy = DROSTE_OVERLAY_MODE:
#   kernel — mount -t overlay: lowerdir=<lower>, upperdir=<upper>,
#            workdir=<dirname upper>/.work/<basename upper> (sibling of upper, same FS).
#   fuse   — fuse-overlayfs, same dirs. Works without overlay-upper fs features AND
#            without CAP_SYS_ADMIN (FUSE mounts are userns-permitted); slower I/O.
#   copy   — LAST RESORT: cp -a the baked lower to $DROSTE_DATA_DIR/copy/<name> once,
#            then bind that copy over the lower. Content frozen at first-copy time.
#   auto   — kernel → fuse → copy, falling back on FEATURE failures only. EPERM
#            ("permission denied") means the container lacks CAP_SYS_ADMIN — fuse would
#            still mount, but the plain binds (surfaces/caches) need the cap anyway, so
#            the run is broken regardless → fail fast with the cap diagnostic instead.
resolve::overlay() {
    local upper=$1 lower=$2
    if [ ! -d "$lower" ]; then
        resolve::warn "overlay lower '$lower' does not exist; skipping"
        return 0
    fi
    # IDEMPOTENCY: init_hooks re-run on every container start (same boot); if the
    # lower already IS a mountpoint (our earlier overlay/copy-bind, or a user's own
    # bind over it), mounting again would stack a second layer. Exact-mountpoint
    # check — NOT is_bound's ancestor-walk, which would false-positive on any
    # ancestor bind (e.g. the whole distrobox $HOME).
    if resolve::_is_mountpoint "$lower"; then
        resolve::info "overlay $lower: already mounted — skipping (re-entrant init hook)"
        return 0
    fi
    local mode=$DROSTE_OVERLAY_MODE
    case "$mode" in
        auto|kernel|fuse|copy) ;;
        *) resolve::err "unknown DROSTE_OVERLAY_MODE '$mode' (want auto|kernel|fuse|copy)"; return 1 ;;
    esac
    local name work
    name=$(basename "$upper")
    work="$(dirname "$upper")/.work/$name"
    # OWNERSHIP wrinkle (distrobox): _mkuserdir chowns upper/work to the box user so
    # copy-ups/writes performed as that user succeed. The root-baked LOWER stays
    # root-owned on purpose — reads work, writes land in the user-owned upper.
    resolve::_mkuserdir "$upper"
    resolve::_mkuserdir "$work"

    if [ -n "${DROSTE_RESOLVE_DRYRUN:-}" ]; then
        local desc=$mode
        [ "$mode" = auto ] && desc='auto(kernel→fuse→copy)'
        resolve::info "[dryrun] overlay $lower: mode=$desc"
    fi

    # KERNEL attempt (auto's first choice; forced by mode=kernel).
    if [ "$mode" = auto ] || [ "$mode" = kernel ]; then
        if resolve::_try_mount mount -t overlay overlay \
               -o "lowerdir=$lower,upperdir=$upper,workdir=$work" "$lower"; then
            return 0
        fi
        if resolve::_mount_err_is_eperm; then
            resolve::_err_need_cap "overlay $lower"
            return 1
        fi
        if [ "$mode" = kernel ]; then
            resolve::err "kernel overlay for $lower failed (DROSTE_OVERLAY_MODE=kernel, no fallback): ${RESOLVE_MOUNT_ERR:-unknown error}"
            return 1
        fi
        if resolve::_mount_err_is_feature; then
            resolve::warn "kernel overlay for $lower rejected: the $DROSTE_DATA_DIR filesystem lacks the overlay-upper features O_TMPFILE/RENAME_WHITEOUT (common on ecryptfs, NFS, virtiofs) — trying fuse-overlayfs."
        else
            resolve::warn "kernel overlay for $lower failed (${RESOLVE_MOUNT_ERR:-unknown error}) — trying fuse-overlayfs."
        fi
    fi

    # FUSE attempt (auto's second choice; forced by mode=fuse). Dryrun skips the
    # environment probe so the forced mode still reports coherently on any host.
    if [ "$mode" = auto ] || [ "$mode" = fuse ]; then
        local fuse_ok=""
        if command -v fuse-overlayfs >/dev/null 2>&1 && [ -e /dev/fuse ]; then
            fuse_ok=1
        fi
        [ -n "${DROSTE_RESOLVE_DRYRUN:-}" ] && fuse_ok=1
        if [ -n "$fuse_ok" ]; then
            if resolve::_try_mount fuse-overlayfs \
                   -o "lowerdir=$lower,upperdir=$upper,workdir=$work" "$lower"; then
                resolve::warn "using fuse-overlayfs for $lower (userspace overlay; slower I/O; kernel overlay unavailable on this $DROSTE_DATA_DIR filesystem)."
                return 0
            fi
            if [ "$mode" = fuse ]; then
                resolve::err "fuse-overlayfs for $lower failed (DROSTE_OVERLAY_MODE=fuse, no fallback): ${RESOLVE_MOUNT_ERR:-unknown error}"
                return 1
            fi
            resolve::warn "fuse-overlayfs for $lower failed (${RESOLVE_MOUNT_ERR:-unknown error}) — falling back to copy mode."
        else
            if [ "$mode" = fuse ]; then
                resolve::err "DROSTE_OVERLAY_MODE=fuse but fuse-overlayfs or /dev/fuse is unavailable in this container."
                return 1
            fi
            resolve::warn "fuse-overlayfs unavailable (binary or /dev/fuse missing) — falling back to copy mode for $lower."
        fi
    fi

    # COPY (auto's last resort; forced by mode=copy).
    resolve::_overlay_copy "$name" "$lower" "$upper"
}

# _overlay_copy — overlay substitute of last resort: materialize a one-time writable
# copy of the baked lower under $DROSTE_DATA_DIR/copy/<name>, bind it over the lower.
# The copy is made only when the dir is ABSENT — deleting it forces a fresh copy.
# ATOMIC: staged into a "$copydir.tmp" sibling and mv'd into place last, so an
# interrupted cp leaves only a .tmp dir (removed and redone on the next run) and
# an existing $copydir is always a reliable COMPLETED-copy marker.
resolve::_overlay_copy() {
    local name=$1 lower=$2 upper=$3
    local copydir="$DROSTE_DATA_DIR/copy/$name" tmpdir
    tmpdir="$copydir.tmp"
    if [ ! -d "$copydir" ]; then
        resolve::_domount rm -rf "$tmpdir"
        resolve::_domount mkdir -p "$tmpdir"
        resolve::_domount cp -a "$lower/." "$tmpdir/"
        resolve::_domount mv "$tmpdir" "$copydir"
        # OWNERSHIP (distrobox): recursive chown is safe ONLY here — every file in a
        # freshly-created copydir was cloned from the root-baked lower THIS run, and
        # copy mode has no upper to absorb writes, so the box user needs write access
        # throughout. Pre-existing copydirs are left untouched (may hold user content
        # with deliberate ownership/modes).
        if [ "$DROSTE_LANE" = distrobox ] && [ -n "${DROSTE_USER:-}" ]; then
            resolve::_domount chown -R "$DROSTE_USER:" "$copydir"
        fi
    fi
    resolve::warn "copy-mode engaged for $lower:"
    resolve::warn "  - baked image content is FROZEN at first-copy time — image updates will NOT appear here;"
    resolve::warn "  - disk cost is duplicated under $copydir;"
    resolve::warn "  - delete $copydir to force a fresh copy from the current image."
    if [ -d "$upper" ] && [ -n "$(ls -A "$upper" 2>/dev/null)" ]; then
        resolve::warn "  - overlay upper $upper is non-empty — deltas from previous overlay-mode runs are NOT visible in copy mode."
    fi
    resolve::_bind "$copydir" "$lower"
}

# ── Primitive: surface (plain bind, BOTH lanes since lane unification) ──────
# entry form: <src>:<dest>  (src = /opt/data side, dest = app-side path)
# Binds a /opt/data subpath onto the path the tool expects; creates src if absent.
# Distrobox deviations: /root/ dest remap (_lane_dest), box-user chown (_mkuserdir),
# exact-mountpoint re-entry skip (_is_mountpoint) — rationale at each helper.
resolve::surface() {
    local src=$1 dest
    dest=$(resolve::_lane_dest "$2")
    # IDEMPOTENCY: init_hooks re-run per container start — skip if something (our
    # earlier run, or a user bind) is already mounted exactly on the dest.
    if resolve::_is_mountpoint "$dest"; then
        resolve::info "surface $dest: already mounted — skipping (re-entrant init hook)"
        return 0
    fi
    resolve::_mkuserdir "$src"
    # dest may sit inside the auto-bound REAL home in the distrobox lane — create
    # it via _mkuserdir so a dir we make there isn't left root-owned on the host.
    [ -d "$dest" ] || resolve::_mkuserdir "$dest"
    resolve::_bind "$src" "$dest"
}

# ── Shared compute-cache volume ($DROSTE_CACHES_DIR, both lanes) ────────────
# OPTIONAL cross-container cache store: bind ONE host dir (default host path
# ~/droste/caches — a default only, any path works) onto /opt/caches and every
# CACHES row whose src lives under $DROSTE_DATA_DIR/cache/ is rewritten to source
# from the shared dir instead, so all droste boxes share one MIOpen/Triton/torch/vLLM
# kernel-cache store. Unbound => graceful degrade to today's per-box
# $DROSTE_DATA_DIR/cache/ sources, announced ONCE per start (below). Deliberately
# NO `VOLUME /opt/caches` directive in any Containerfile: unbound must NOT spawn
# an anonymous volume. Dests are never touched — only the src side moves.
# Detection = is_bound on $DROSTE_CACHES_DIR ITSELF (a user bind exactly there, or
# an ancestor bind covering it, both mean the dir persists — exact-mountpoint
# semantics would wrongly reject the ancestor case).
# MIOpen caveat (commented once, here): the miopen-db row carries MIOpen's
# machine-written tuning DB, so sharing means one DB for all boxes — two boxes doing HEAVY
# tuning concurrently will contend on its locks (POSIX locks work fine across
# bind mounts; contention serializes tuning, it does not corrupt). Fine for the
# normal one-box-hot-at-a-time pattern; remove the /opt/caches bind from a box's
# ini/run flags to give it private caches again.
RESOLVE_SHARED_CACHES=""   # memo: "" = unchecked, y = bound, n = unbound (info shown)
resolve::_shared_caches() {
    if [ -z "$RESOLVE_SHARED_CACHES" ]; then
        if resolve::is_bound "$DROSTE_CACHES_DIR"; then
            RESOLVE_SHARED_CACHES=y
        else
            RESOLVE_SHARED_CACHES=n
            resolve::info "$DROSTE_CACHES_DIR is not bound — compute caches stay per-box under $DROSTE_DATA_DIR/cache/. Bind one shared host dir (e.g. server: -v ~/droste/caches:$DROSTE_CACHES_DIR; distrobox ini: volume=\"~/droste/caches:$DROSTE_CACHES_DIR\") to share kernel caches across all droste boxes."
        fi
    fi
    [ "$RESOLVE_SHARED_CACHES" = y ]
}

# ── Primitive: cache_bind (BOTH lanes since lane unification) ───────────────
# Structurally identical to surface plus the shared-cache src rewrite above; kept a
# SEPARATE primitive (per design) so cache behaviour can diverge without touching
# surfaces (it now does — surfaces are state and never rewrite to /opt/caches).
# DELIBERATE DEVIATION (both lanes): the HF cache is NEVER a cache_bind — it is a
# CRITICAL user bind (server: -v flag; distrobox: the auto-bound real home already
# satisfies it), so the resolver mounts nothing for it in either lane.
resolve::cache_bind() {
    local src=$1 dest
    dest=$(resolve::_lane_dest "$2")
    # Shared-cache src rewrite (policy + MIOpen caveat: block above). Only srcs
    # under $DROSTE_DATA_DIR/cache/ qualify — session state elsewhere on the data
    # volume (llama slots, ds4 kv-disk) is not a CACHES row and never rewrites.
    case "$src" in
        "$DROSTE_DATA_DIR/cache/"*)
            if resolve::_shared_caches; then
                src="$DROSTE_CACHES_DIR/${src#"$DROSTE_DATA_DIR/cache/"}"
            fi
            ;;
    esac
    # IDEMPOTENCY: same re-entry guard as surface (init_hooks re-run per start).
    if resolve::_is_mountpoint "$dest"; then
        resolve::info "cache $dest: already mounted — skipping (re-entrant init hook)"
        return 0
    fi
    # OWNERSHIP on the shared dir: _mkuserdir chowns only the per-cache leaf dirs
    # it creates ($DROSTE_CACHES_DIR/<name>) — the shared root belongs to the
    # user's bind and is never touched.
    resolve::_mkuserdir "$src"
    # dest may sit inside the auto-bound REAL home in the distrobox lane — create
    # it via _mkuserdir so a dir we make there isn't left root-owned on the host.
    [ -d "$dest" ] || resolve::_mkuserdir "$dest"
    resolve::_bind "$src" "$dest"
}

# ── Primitive: critical (both lanes) ────────────────────────────────────────
# entry form: <label>:<path>
# Hard-error if the path is not bound, naming the label + an example flag — UNLESS
# ALLOW_EPHEMERAL is set (bare name), then warn and continue (data will NOT persist).
resolve::critical() {
    local label=$1 path=$2
    if resolve::is_bound "$path"; then
        return 0
    fi
    if [ -n "${ALLOW_EPHEMERAL:-}" ]; then
        resolve::warn "CRITICAL '$label' ($path) is not bound; ALLOW_EPHEMERAL is set — running ephemeral, data will NOT persist."
        return 0
    fi
    resolve::err "CRITICAL '$label' ($path) is not bound — refusing to start."
    resolve::err "  bind it, e.g.  -v /host/$label:$path   (distrobox: volume=/host/$label:$path)"
    resolve::err "  or set ALLOW_EPHEMERAL=1 to run ephemerally (data will NOT persist)."
    exit 1
}

# ── Primitive: optional (both lanes) — tell-once-don't-nag ──────────────────
# entry form: <label>:<path>[:<marker>]   (marker defaults to <path>/.droste-informed)
# Populated (any non-marker entry present)          → silent (in use).
# Unpopulated (empty or marker-only) + marker absent → INFO once + create the marker.
# Unpopulated + marker present                       → silent (already informed).
# The marker's EXISTENCE signifies "the user has been told" — so it is created here,
# never baked. Its body comes from a template file (RESOLVE_TEMPLATES_DIR/<basename>)
# if present, else a built-in one-liner.
resolve::_populated() {
    local dir=$1 marker_base=$2 e
    [ -d "$dir" ] || return 1
    local entries=()
    shopt -s nullglob dotglob
    entries=("$dir"/*)
    shopt -u nullglob dotglob
    for e in ${entries[@]+"${entries[@]}"}; do
        [ "$(basename "$e")" = "$marker_base" ] && continue
        return 0
    done
    return 1
}

resolve::_write_marker() {
    local marker=$1 marker_base template
    marker_base=$(basename "$marker")
    template="$RESOLVE_TEMPLATES_DIR/$marker_base"
    mkdir -p "$(dirname "$marker")"
    if [ -f "$template" ]; then
        cp "$template" "$marker"
    else
        printf '%s\n' \
            "# Nothing is bound here yet." \
            "# Bind a directory of local models onto $(dirname "$marker") to make them available," \
            "# e.g.  -v /host/models:$(dirname "$marker")" \
            "# (This file was created to note you have been informed; deleting it re-shows the notice.)" \
            > "$marker"
    fi
}

resolve::optional() {
    local label=$1 path=$2 marker=${3:-}
    [ -n "$marker" ] || marker="$path/.droste-informed"
    local marker_base
    marker_base=$(basename "$marker")
    mkdir -p "$path"
    if resolve::_populated "$path" "$marker_base"; then
        return 0
    fi
    if [ -e "$marker" ]; then
        return 0
    fi
    resolve::info "nothing bound to $path ($label) — only already-present models are available. See $marker for how to add local models."
    # Marker write is BEST-EFFORT: on a READ-ONLY bind (e.g. -v ~/models:/opt/models:ro,
    # which the docs recommend) the write fails — unguarded under set -e that would
    # abort startup. Failure just means the INFO above repeats every start.
    if ! resolve::_write_marker "$marker" 2>/dev/null; then
        resolve::info "could not write marker $marker (read-only bind?) — this notice will repeat every start."
    fi
}

# ── /opt/data handling (both lanes) ─────────────────────────────────────────
# The Containerfile-level `VOLUME /opt/data` gives auto-anonymous-volume behaviour; from
# inside all we can do is warn if the user did not bind it (state won't survive recreate).
# Because of that VOLUME directive the dir is virtually ALWAYS a mount — the real
# forgot-to-bind signal is the ANONYMOUS-volume shape (_anon_volume), warned on below.
resolve::ensure_data() {
    local dir=${1:-$DROSTE_DATA_DIR}
    mkdir -p "$dir"
    if ! resolve::is_bound "$dir"; then
        resolve::warn "$dir is not a bound volume — using an image-provided (anonymous) volume; bind it with -v <host>:$dir to persist across container recreation."
    elif resolve::_anon_volume "$dir"; then
        resolve::warn "$dir state is on an ANONYMOUS volume (auto-created by the image's VOLUME directive) — it will NOT survive container removal. Bind a host dir (-v /host/data:$dir) or a NAMED volume (-v mydata:$dir) to persist. (This is a warning only — ALLOW_EPHEMERAL is NOT needed for it.)"
    fi
}

# ── Template seeding (both lanes) ───────────────────────────────────────────
# Runs AFTER mounts so seeds land on the mounted destinations. No-op if no manifest.
resolve::apply_templates() {
    local tdir=${1:-$RESOLVE_TEMPLATES_DIR}
    [ -f "$tdir/templates.yaml" ] || return 0
    python3 "$RESOLVE_APPLY_TEMPLATES" "$tdir"
}

# ── Orchestration ───────────────────────────────────────────────────────────
# apply_spec — the EXACT design order, lane-aware, no exec. Consumes the row
# arrays/vars sourced from build-spec (SERVICE/ENV_FILE/OVERLAYS/SURFACES/CRITICAL/
# OPTIONAL/CACHES/PRE_LAUNCH). Called by the server entrypoint and by distrobox
# init_hooks alike; the entrypoint execs SERVICE afterwards, init_hooks do not.
resolve::apply_spec() {
    local entry rest label path marker

    # 1) ensure /opt/data (auto-vol + warn)
    resolve::ensure_data "$DROSTE_DATA_DIR"

    # 2) SURFACES + OVERLAYS + CACHES (BOTH lanes since lane unification —
    #    distrobox mounts in-box via init_hooks; lane deviations live in the
    #    primitives themselves)
    for entry in ${SURFACES[@]+"${SURFACES[@]}"}; do
        resolve::surface "${entry%%:*}" "${entry#*:}"
    done
    for entry in ${OVERLAYS[@]+"${OVERLAYS[@]}"}; do
        resolve::overlay "${entry%%:*}" "${entry#*:}"
    done
    for entry in ${CACHES[@]+"${CACHES[@]}"}; do
        resolve::cache_bind "${entry%%:*}" "${entry#*:}"
    done

    # 3) CRITICAL (hard-error + ALLOW_EPHEMERAL escape)
    for entry in ${CRITICAL[@]+"${CRITICAL[@]}"}; do
        resolve::critical "${entry%%:*}" "${entry#*:}"
    done

    # 4) OPTIONAL (info + marker)
    for entry in ${OPTIONAL[@]+"${OPTIONAL[@]}"}; do
        label=${entry%%:*}
        rest=${entry#*:}
        if [ "$rest" = "${rest#*:}" ]; then
            path=$rest
            marker=""
        else
            path=${rest%%:*}
            marker=${rest#*:}
        fi
        resolve::optional "$label" "$path" "$marker"
    done

    # 5) templates.yaml seeding (AFTER mounts)
    resolve::apply_templates "$RESOLVE_TEMPLATES_DIR"

    # 6) ENV_FILE source (generate-if-absent handled by templates' if_missing above)
    # set -a exports every var the file assigns, so plain VAR= lines reach the
    # service across the exec (llama-server reads LLAMA_ARG_* from its environment).
    if [ -n "${ENV_FILE:-}" ] && [ -f "$ENV_FILE" ]; then
        set -a
        # shellcheck disable=SC1090
        source "$ENV_FILE"
        set +a
    fi

    # 7) PRE_LAUNCH function (defined in build-spec)
    if [ -n "${PRE_LAUNCH:-}" ]; then
        "$PRE_LAUNCH"
    fi
}
