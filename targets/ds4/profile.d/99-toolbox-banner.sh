#!/usr/bin/env bash
# Lightweight banner with machine/GPU and ROCm version (DS4 edition)
# Same info/format as the other port banners.

# Only show for interactive shells
case $- in *i*) ;; *) return 0 ;; esac

oem_info() {
  local v="" m="" d lv lm
  for d in /sys/class/dmi/id /sys/devices/virtual/dmi/id; do
    [[ -r "$d/sys_vendor" ]] && v=$(<"$d/sys_vendor")
    [[ -r "$d/product_name" ]] && m=$(<"$d/product_name")
    [[ -n "$v" || -n "$m" ]] && break
  done
  # ARM/SBC fallback
  if [[ -z "$v" && -z "$m" && -r /proc/device-tree/model ]]; then
    tr -d '\0' </proc/device-tree/model
    return
  fi
  lv=$(printf '%s' "$v" | tr '[:upper:]' '[:lower:]')
  lm=$(printf '%s' "$m" | tr '[:upper:]' '[:lower:]')
  if [[ -n "$m" && "$lm" == "$lv "* ]]; then
    printf '%s\n' "$m"
  else
    printf '%s %s\n' "${v:-Unknown}" "${m:-Unknown}"
  fi
}

# Reject empty / placeholder GPU names so the ladder keeps falling through.
_gpu_ok() {
  local n
  n=$(printf '%s' "$1" | sed -e 's/^[[:space:]]*//' -e 's/[[:space:]]*$//')
  [[ -z "$n" ]] && return 1
  case "$(printf '%s' "$n" | tr '[:upper:]' '[:lower:]')" in
    n/a|na|none|null|unknown|"not supported"|"unknown amd gpu"|"amd gpu") return 1 ;;
  esac
  return 0
}

# Resolve a friendly GPU name. Runs at every login: every probe is guarded by
# command -v, silenced, and (where a probe could hang) bounded by `timeout`, so
# a missing/wedged tool can never error out or stall the login shell.
# ROCm CLI tools ship in /opt/venv/bin, which is NOT on PATH at profile.d time
# (zz-venv-last.sh only prepends it via PROMPT_COMMAND, after banners source).
# Resolve by PATH first, then that known location — like rocm_version()'s
# absolute python path — so rocminfo/rocm-smi are found at banner time.
_rocm_tool() { command -v "$1" 2>/dev/null || { [[ -x "/opt/venv/bin/$1" ]] && printf '/opt/venv/bin/%s\n' "$1"; }; }

gpu_name() {
  local name="" cand="" gfx="" rinfo="" TO="" rbin="" sbin=""
  # Bound probes that can hang (rocminfo/rocm-smi enumerate hardware). If the
  # `timeout` binary is absent we just run the command directly.
  command -v timeout >/dev/null 2>&1 && TO="timeout 3"

  # rocminfo: capture once, then parse for both a friendly name and the gfx
  # target. APUs like Strix Halo populate "Marketing Name" even when other
  # sources are blank, so it leads the ladder.
  if rbin=$(_rocm_tool rocminfo); then
    rinfo=$($TO "$rbin" 2>/dev/null)
    # (1) First GPU agent's Marketing Name. Device Type appears after the Name/
    # Marketing lines within an agent block, so buffer then emit at Device Type.
    cand=$(printf '%s\n' "$rinfo" | awk '
      /^[[:space:]]*Marketing Name:[[:space:]]/ { m=$0; sub(/^[[:space:]]*Marketing Name:[[:space:]]*/,"",m) }
      /^[[:space:]]*Device Type:[[:space:]]/ {
        d=$0; sub(/^[[:space:]]*Device Type:[[:space:]]*/,"",d)
        if (d ~ /GPU/) { print m; exit }
      }')
    _gpu_ok "$cand" && name="$cand"
    # gfx target (e.g. gfx1151) of the first GPU agent — kept for the fallback.
    gfx=$(printf '%s\n' "$rinfo" | awk '
      /^[[:space:]]*Name:[[:space:]]/ { n=$0; sub(/^[[:space:]]*Name:[[:space:]]*/,"",n) }
      /^[[:space:]]*Device Type:[[:space:]]/ {
        d=$0; sub(/^[[:space:]]*Device Type:[[:space:]]*/,"",d)
        if (d ~ /GPU/) { print n; exit }
      }' | grep -oiE 'gfx[0-9a-f]+' | head -n1)
  fi

  # (2) rocm-smi --showproductname. Column/CSV layout varies by ROCm version,
  # so scan several likely value fields and take the first non-placeholder.
  if [[ -z "$name" ]] && sbin=$(_rocm_tool rocm-smi); then
    cand=$($TO "$sbin" --showproductname 2>/dev/null \
      | grep -iE 'Card Series|Card Model|Product Name|Device Name|Market Name' \
      | sed -E 's/.*:[[:space:]]*//' | head -n1)
    _gpu_ok "$cand" && name="$cand"
    # CSV form: header row of column names, then per-GPU value rows.
    if [[ -z "$name" ]]; then
      cand=$($TO "$sbin" --showproductname --csv 2>/dev/null \
        | awk -F, 'NR>1 && NF>1 { for (i=2;i<=NF;i++) if ($i!="" && $i!="N/A") { print $i; exit } }')
      _gpu_ok "$cand" && name="$cand"
    fi
  fi

  # (3) amdgpu sysfs — product_name is populated on some boards/APUs.
  if [[ -z "$name" ]]; then
    local f
    for f in /sys/class/drm/card*/device/product_name; do
      [[ -r "$f" ]] || continue
      cand=$(<"$f")
      if _gpu_ok "$cand"; then name="$cand"; break; fi
    done
  fi

  # (4) lspci fallback for the display/VGA controller description.
  if [[ -z "$name" ]] && command -v lspci >/dev/null 2>&1; then
    cand=$(lspci 2>/dev/null | grep -iE 'vga|display|3d controller' \
      | grep -iE 'amd|ati|radeon' | head -n1 | sed -E 's/.*: //')
    _gpu_ok "$cand" && name="$cand"
  fi

  # (5) No friendly name, but ROCm clearly sees a GPU → show the gfx target;
  #     far more useful than a generic "Unknown".
  [[ -z "$name" && -n "$gfx" ]] && name="AMD GPU ($gfx)"

  # trim leading/trailing spaces and squeeze multiple spaces to one
  name=$(printf '%s' "$name" | sed -e 's/^[[:space:]]\+//' -e 's/[[:space:]]\+$//' -e 's/[[:space:]]\{2,\}/ /g')
  # (6) Absolute last resort.
  printf '%s\n' "${name:-AMD GPU (gfx target unknown)}"
}

rocm_version() {
  local PY="/opt/venv/bin/python"
  [[ -x "$PY" ]] || PY="python"
  "$PY" - <<'PY' 2>/dev/null || true
try:
    import importlib.metadata as im
    try:
        print(im.version("_rocm_sdk_core"))
    except Exception:
        print(im.version("rocm"))
except Exception:
    print("")
PY
}

MACHINE="$(oem_info)"
GPU="$(gpu_name)"
ROCM_VER="$(rocm_version)"

echo
cat <<'ASCII'
              ╔═╤═╤════╗ 🭺🭺🭺🭺🭺🭺🭺🭺🭺🭺🭺🭺🭺🭺🭺🭺🭺🭺🭺🭺🭺🭺🭺🭺🭺
              ╟─┘■│    ║  █🮂🮂🭕🭏            🭋
              ╟───┘ ██ ║  █   █ 🭩🬂🭗🭄🮂🭏 🭄🮀🭧🭢🬨🬂🭗🭂🮀🭍
              ║        ║  █  🭊🭠 🭞  🭕▂🭠 ▄ 🭨🭬🭦🭩🭛🭓🬭🬽
              ╚════════╝ `🮃🮃🮃🭘🭷🭷🭷🭷🭷🭷🭷🭷🭷🭣🬂🭘🭷🭷🭷🭷🭷🭷🭷🭷
                  DwarfStar 4: Interactive Box

ASCII
echo
printf 'AMD Ryzen AI Max Strix Halo: DS4 Toolbox (gfx1151, ROCm via TheRock)\n'
[[ -n "$ROCM_VER" ]] && printf 'ROCm nightly: %s\n' "$ROCM_VER"
echo
printf 'Machine: %s\n' "$MACHINE"
printf 'GPU    : %s\n\n' "$GPU"
printf 'Image : ghcr.io/doctorjei/droste-ds4-halo\n'
printf 'Repo  : https://github.com/doctorjei/droste-ai-halo\n\n'
printf 'Included:\n'
printf '  - %-18s → %s\n' "ds4-server" "runs by default via the image entrypoint (port 8000)"
printf '  - %-18s → %s\n' "config" "/opt/data/ds4.env (DS4_DROSTE_* + native DS4_* vars)"
printf '  - %-18s → %s\n' "get a model" "download_model.sh q2-imatrix  (easy way; prints the path for DS4_DROSTE_MODEL)"
printf '  - %-18s → %s\n' "ds4-cockpit" "TUI: model manager + server runner"
printf '  - %-18s → %s\n' "ds4 / ds4-bench" "interactive CLI / benchmark"
printf '  - %-18s → %s\n' "API test" "curl localhost:8000/v1/chat/completions"
echo
printf 'SSH tip: ssh -L 8000:localhost:8000 user@host\n\n'

unset PROMPT_COMMAND
PS1='\u@\h:\w\$ '
