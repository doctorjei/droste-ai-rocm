#!/usr/bin/env bash
# Lightweight banner with machine/GPU and ROCm nightly version

# Load ROCm env quietly if present
[[ -f /etc/profile.d/01-rocm-env-for-triton.sh ]] && . /etc/profile.d/01-rocm-env-for-triton.sh

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
  # trim leading/trailing spaces and squeeze multiple spaces to one
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
███████╗████████╗██████╗ ██╗██╗  ██╗      ██╗  ██╗ █████╗ ██╗      ██████╗ 
██╔════╝╚══██╔══╝██╔══██╗██║╚██╗██╔╝      ██║  ██║██╔══██╗██║     ██╔═══██╗
███████╗   ██║   ██████╔╝██║ ╚███╔╝       ███████║███████║██║     ██║   ██║
╚════██║   ██║   ██╔══██╗██║ ██╔██╗       ██╔══██║██╔══██║██║     ██║   ██║
███████║   ██║   ██║  ██║██║██╔╝ ██╗      ██║  ██║██║  ██║███████╗╚██████╔╝
╚══════╝   ╚═╝   ╚═╝  ╚═╝╚═╝╚═╝  ╚═╝      ╚═╝  ╚═╝╚═╝  ╚═╝╚══════╝ ╚═════╝ 

                        L L M   F I N E - T U N I N G                        

ASCII
echo
printf 'AMD STRIX HALO — LLM Finetuning (gfx1151, ROCm via TheRock)\n'
[[ -n "$ROCM_VER" ]] && printf 'ROCm nightly: %s\n' "$ROCM_VER"
echo
printf 'Machine: %s\n' "$MACHINE"
printf 'GPU    : %s\n\n' "$GPU"
printf 'Ported from: https://github.com/kyuz0/amd-strix-halo-llm-finetuning\n'
printf 'Image      : ghcr.io/doctorjei/droste-finetuning-halo\n\n'
printf 'Server mode (image default): the entrypoint starts JupyterLab on http://localhost:8888\n'
printf '  (auth token in the container log; set JUPYTER_TOKEN to choose your own)\n'
printf 'In this shell (distrobox/toolbox lane), start it yourself:\n'
printf '  jupyter lab --ip 0.0.0.0 --port 8888 --notebook-dir=/opt/workspace --no-browser\n\n'
printf 'Workspace  : /opt/workspace — your bind; starter notebooks seed in when it is empty\n'
printf '             (pristine copies live in /opt/resources/templates/workspace)\n'
printf 'Helpers    : train.py · start-finetuning-cluster.py · benchmark_configs.py ·\n'
printf '             measure_unsloth_memory.py — read-only in /opt/resources/scripts, on PATH\n\n'
printf 'SSH tip: ssh -L 8888:localhost:8888 user@host\n\n'
