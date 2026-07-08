#!/usr/bin/env bash
# gen_llama_env.sh — BUILD-TIME generator for the llama.env TEMPLATE.
#
# Runs as a RUN step in Container.llama (after llama-server is in place) and emits
# /opt/resources/templates/llama.env — the template that templates.yaml seeds to
# /opt/data/llama.env (if_missing) at container start.
#
# Enumeration strategy (design: complete, drift-free, no hand-maintenance):
#   1. PRIMARY: parse `llama-server --help`. Every flag that has a native env var
#      carries an `(env: LLAMA_ARG_X)` annotation; the flag's `(default: …)` text
#      supplies the commented default value.
#   2. FALLBACK (if --help won't run in the build env, e.g. GPU-probing aborts, or
#      its output carries no env annotations): string-scan the llama-server binary
#      (+ its local libs) for LLAMA_ARG_[A-Z0-9_]+ literals — the arg table stores
#      the env names as plain strings. Names only; defaults left empty.
# Either way the ACTIVE vars are VERIFIED against the enumerated table — a rename
# upstream fails the IMAGE BUILD loudly (verify-at-build by design).
#
# Usage: gen_llama_env.sh [output-path]   (env override: LLAMA_SERVER_BIN)
set -euo pipefail

OUT=${1:-/opt/resources/templates/llama.env}
SERVER=${LLAMA_SERVER_BIN:-llama-server}

die() { printf 'gen_llama_env: ERROR: %s\n' "$*" >&2; exit 1; }
note() { printf 'gen_llama_env: %s\n' "$*" >&2; }

# Our active (uncommented) values. Expressed as env lines, NOT hardcoded flags,
# because CLI flags override env in llama.cpp — env lines keep user edits winning.
ACTIVE_HOST=0.0.0.0
ACTIVE_PORT=8080
# Slot save/restore: the pinned fork ships --slot-save-path with NO env
# annotation, so there is no LLAMA_ARG_SLOT_SAVE_PATH to set here — the flag is
# added by the entrypoint's launch line instead (targets/llama/build-spec,
# llama_pre_launch). This path only feeds the explanatory comment block below.
SLOTS_DIR=/opt/data/cache/slots
# Vars that MUST exist in the pinned llama-server's arg table (build fails if not).
REQUIRED_VARS=(LLAMA_ARG_HOST LLAMA_ARG_PORT LLAMA_ARG_MODEL)
# Vars excluded from the generic commented list (they get dedicated blocks above
# it). LLAMA_ARG_SLOT_SAVE_PATH is deliberately NOT excluded: absent from the
# current pin, but if a future pin gains it, it should flow through as an
# ordinary enumerated (commented) flag — never REQUIRED.
SPECIAL_VARS="LLAMA_ARG_HOST LLAMA_ARG_PORT LLAMA_ARG_MODEL"

TAB=$(printf '\t')

# ── 1) enumerate env-having flags → table of "NAME<TAB>DEFAULT" lines ─────────
# timeout: --help may probe for a GPU and hang on a GPU-less builder; a timeout
# (exit 124) is treated like any other --help failure → string-scan fallback.
mode=help
help_text=$(timeout 60 "$SERVER" --help 2>&1) || mode=scan
if [ "$mode" = help ] && ! grep -q '(env: LLAMA_ARG_' <<<"$help_text"; then
    mode=scan
fi

if [ "$mode" = help ]; then
    table=$(awk '
        {
            line = $0
            # A new option entry (first non-space char is "-") resets the pending
            # default; wrapped description lines are indented and rarely dash-led.
            if (line ~ /^[[:space:]]*-/) { def = "" }
            if (match(line, /\(default: [^)]*\)/)) {
                def = substr(line, RSTART + 10, RLENGTH - 11)
            }
            while (match(line, /\(env: LLAMA_ARG_[A-Z0-9_]+\)/)) {
                name = substr(line, RSTART + 6, RLENGTH - 7)
                printf "%s\t%s\n", name, def
                line = substr(line, RSTART + RLENGTH)
            }
        }
    ' <<<"$help_text")
else
    note "'$SERVER --help' unusable here — falling back to binary string-scan (names only, no defaults)"
    bin_path=$(command -v "$SERVER") || die "cannot locate '$SERVER' for the fallback scan"
    table=$( { grep -haoE 'LLAMA_ARG_[A-Z0-9_]+' "$bin_path" /usr/local/lib64/lib*.so* 2>/dev/null || true; } \
             | sed "s/\$/${TAB}/" )
fi

# de-duplicate by name (keep first default), sort for a stable template
table=$(printf '%s\n' "$table" | awk -F'\t' 'NF && !seen[$1]++' | sort -t"$TAB" -k1,1)
[ -n "$table" ] || die "no LLAMA_ARG_* env-having flags enumerated from '$SERVER' (mode: $mode)"

count=$(printf '%s\n' "$table" | wc -l)
note "enumerated $count env-having flags from '$SERVER' (mode: $mode)"

# ── 2) VERIFY the active/required vars exist in the enumerated table ──────────
names=$(printf '%s\n' "$table" | cut -f1)
for v in "${REQUIRED_VARS[@]}"; do
    grep -qx "$v" <<<"$names" \
        || die "required env var '$v' NOT in the pinned llama-server's arg table — upstream rename? Fix the active lines (or the launch flags) before shipping."
done
note "verified active vars: ${REQUIRED_VARS[*]}"

# ── 3) emit the template ──────────────────────────────────────────────────────
mkdir -p "$(dirname "$OUT")"
{
    printf '%s\n' \
        "# llama.env — llama-server configuration (this file IS the config surface)." \
        "#" \
        "# GENERATED AT IMAGE BUILD from the pinned llama-server's argument table" \
        "# (mode: $mode; $count env-having flags). Seeded to /opt/data/llama.env on" \
        "# first start and NEVER overwritten — your edits here win." \
        "#" \
        "# llama-server reads LLAMA_ARG_* env vars natively, one per flag. CLI flags" \
        "# would override env, so our defaults are env LINES: uncomment a line to set" \
        "# it. Values are used as-is (no shell quoting/expansion beyond this file" \
        "# being sourced)." \
        "#" \
        "# NOTE: LLAMA_CACHE is deliberately NOT listed. Unset, llama-server shares" \
        "# the standard HF cache (~/.cache/huggingface/hub) with the other ports;" \
        "# setting it would re-separate llama's downloads. Leave it unset." \
        ""
    printf '%s\n' \
        "# ── active defaults (droste) ─────────────────────────────────────────────────" \
        "LLAMA_ARG_HOST=$ACTIVE_HOST" \
        "LLAMA_ARG_PORT=$ACTIVE_PORT" \
        "" \
        "# ── slot save/restore ────────────────────────────────────────────────────────" \
        "# Slot save/restore is enabled via the launch flag --slot-save-path $SLOTS_DIR," \
        "# added by the entrypoint's launch line (no env line needed here). To change" \
        "# the location, put your own '--slot-save-path <dir>' in LLAMA_EXTRA_ARGS" \
        "# below — later flags win in llama-server's parser." \
        ""
    printf '%s\n' \
        "# ── model ────────────────────────────────────────────────────────────────────" \
        "# LLAMA_ARG_MODEL= # REQUIRED — server won't start until set (absolute path, or use -hf via LLAMA_EXTRA_ARGS)" \
        "#   -hf downloads land in the shared HF cache (~/.cache/huggingface)." \
        "#   Local GGUFs: bind your collection read-only at /opt/models and point here." \
        ""
    printf '%s\n' \
        "# ── extra args ───────────────────────────────────────────────────────────────" \
        "# Catch-all for flags WITHOUT a native env var (and anything else), appended" \
        "# to the llama-server command line. Quote the WHOLE value (this file is" \
        "# sourced bash); it is then whitespace-split into separate args — individual" \
        "# args cannot themselves contain spaces. Example:" \
        "#   LLAMA_EXTRA_ARGS=\"-hf org/repo:Q4_K_M --jinja\"" \
        "# LLAMA_EXTRA_ARGS=" \
        ""
    printf '%s\n' \
        "# ── all env-having flags of the pinned llama-server (uncomment to set) ───────"
    while IFS="$TAB" read -r name def; do
        case " $SPECIAL_VARS " in *" $name "*) continue ;; esac
        # A simple single-token default becomes the value; prose defaults
        # ("4096, 0 = loaded from model") go into a trailing comment instead,
        # so an uncommented line is always a valid assignment.
        if [ -n "$def" ] && [[ "$def" == *[[:space:],]* ]]; then
            printf '# %s=  # default: %s\n' "$name" "$def"
        else
            printf '# %s=%s\n' "$name" "$def"
        fi
    done <<<"$table"
} > "$OUT"

note "wrote $OUT"
