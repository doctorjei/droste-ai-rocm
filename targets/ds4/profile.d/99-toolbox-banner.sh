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

gpu_name() {
  local name=""
  if command -v rocm-smi >/dev/null 2>&1; then
    name=$(rocm-smi --showproductname --csv 2>/dev/null | tail -n1 | cut -d, -f2)
    [[ -z "$name" ]] && name=$(rocm-smi --showproductname 2>/dev/null | grep -m1 -E 'Product Name|Card series' | sed 's/.*: //')
  fi
  if [[ -z "$name" ]] && command -v rocminfo >/dev/null 2>&1; then
    name=$(rocminfo 2>/dev/null | awk -F': ' '/^[[:space:]]*Name:/{print $2; exit}')
  fi
  if [[ -z "$name" ]] && command -v lspci >/dev/null 2>&1; then
    name=$(lspci -nn 2>/dev/null | grep -Ei 'vga|display|gpu' | grep -i amd | head -n1 | cut -d: -f3-)
  fi
  # trim
  name=$(printf '%s' "$name" | sed -e 's/^[[:space:]]\+//' -e 's/[[:space:]]\+$//' -e 's/[[:space:]]\{2,\}/ /g')
  printf '%s\n' "${name:-Unknown AMD GPU}"
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
███████╗████████╗██████╗ ██╗██╗  ██╗      ██╗  ██╗ █████╗ ██╗      ██████╗
██╔════╝╚══██╔══╝██╔══██╗██║╚██╗██╔╝      ██║  ██║██╔══██╗██║     ██╔═══██╗
███████╗   ██║   ██████╔╝██║ ╚███╔╝       ███████║███████║██║     ██║   ██║
╚════██║   ██║   ██╔══██╗██║ ██╔██╗       ██╔══██║██╔══██║██║     ██║   ██║
███████║   ██║   ██║  ██║██║██╔╝ ██╗      ██║  ██║██║  ██║███████╗╚██████╔╝
╚══════╝   ╚═╝   ╚═╝  ╚═╝╚═╝╚═╝  ╚═╝      ╚═╝  ╚═╝╚═╝  ╚═╝╚══════╝ ╚═════╝

                                D S 4
ASCII
echo
printf 'AMD STRIX HALO — DS4 Toolbox (gfx1151, ROCm via TheRock)\n'
[[ -n "$ROCM_VER" ]] && printf 'ROCm nightly: %s\n' "$ROCM_VER"
echo
printf 'Machine: %s\n' "$MACHINE"
printf 'GPU    : %s\n\n' "$GPU"
printf 'Repo   : https://github.com/kyuz0/ds4 (DeepSeek V4 Flash engine)\n'
printf 'Image  : ghcr.io/doctorjei/droste-ds4-halo\n\n'
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
