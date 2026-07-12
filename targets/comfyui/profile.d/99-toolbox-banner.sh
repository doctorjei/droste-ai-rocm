#!/usr/bin/env bash
# Lightweight banner with machine/GPU and ROCm nightly version

# Load ROCm env quietly if present
[[ -f /etc/profile.d/01-rocm-envs.sh ]] && . /etc/profile.d/01-rocm-envs.sh

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
gpu_name() {
  local name="" cand="" gfx="" rinfo="" TO=""
  # Bound probes that can hang (rocminfo/rocm-smi enumerate hardware). If the
  # `timeout` binary is absent we just run the command directly.
  command -v timeout >/dev/null 2>&1 && TO="timeout 3"

  # rocminfo: capture once, then parse for both a friendly name and the gfx
  # target. APUs like Strix Halo populate "Marketing Name" even when other
  # sources are blank, so it leads the ladder.
  if command -v rocminfo >/dev/null 2>&1; then
    rinfo=$($TO rocminfo 2>/dev/null)
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
  if [[ -z "$name" ]] && command -v rocm-smi >/dev/null 2>&1; then
    cand=$($TO rocm-smi --showproductname 2>/dev/null \
      | grep -iE 'Card Series|Card Model|Product Name|Device Name|Market Name' \
      | sed -E 's/.*:[[:space:]]*//' | head -n1)
    _gpu_ok "$cand" && name="$cand"
    # CSV form: header row of column names, then per-GPU value rows.
    if [[ -z "$name" ]]; then
      cand=$($TO rocm-smi --showproductname --csv 2>/dev/null \
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
        print(im.version('_rocm_sdk_core'))
    except Exception:
        print(im.version('rocm'))
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
                   COMFY UI - INTERACTIVE BOX

ASCII
echo
printf 'AMD Ryzen AI Max Strix Halo: Image & Video Toolbox (gfx1151, ROCm via TheRock)\n'
[[ -n "$ROCM_VER" ]] && printf 'ROCm nightly: %s\n' "$ROCM_VER"
echo
printf 'Machine: %s\n' "$MACHINE"
printf 'GPU    : %s\n\n' "$GPU"
printf 'Image : ghcr.io/doctorjei/droste-comfyui-halo\n'
printf 'Repo  : https://github.com/doctorjei/droste-ai-rocm\n\n'
printf 'ComfyUI server: http://localhost:8188\n'
printf '  - Run as a container → the server starts automatically (image entrypoint).\n'
printf '  - In a distrobox/toolbox shell nothing autostarts → use: start_comfy_ui\n'
echo
printf 'Model downloaders (shared HF cache; scanner links them in at start):\n'
printf '  get_wan22.sh · get_qwen_image.sh · get_hunyuan15.sh · get_ltx2.sh\n\n'
printf 'SSH tip: ssh -L 8188:localhost:8188 user@host\n\n'

# Launcher (flags match the container SERVICE line). A function, not an alias:
# the extra-model-paths config is only seeded where an init hook ran (distrobox);
# plain toolbox has no /opt/data/extra_model_paths.yaml, and ComfyUI's unguarded
# open() would crash on the missing file — pass the flag only if the file exists.
start_comfy_ui() {
  local extra=()
  [[ -f /opt/data/extra_model_paths.yaml ]] \
    && extra=( --extra-model-paths-config /opt/data/extra_model_paths.yaml )
  cd /opt/ComfyUI && python main.py --listen 0.0.0.0 --port 8188 \
    --disable-mmap --gpu-only --disable-smart-memory --cache-none --bf16-vae \
    "${extra[@]}"
}
