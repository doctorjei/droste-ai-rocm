# droste-ai-rocm

Fedora → Debian/gemet port of the [Strix Halo ROCm toolboxes](https://github.com/kyuz0)
(kyuz0's llama / ds4 / comfyui / vllm / finetuning images). These are **gemet-derived**
(Debian 13 / trixie) OCI images for AMD **Strix Halo** APUs, targeting native **gfx1151**.

A branch of the [droste](https://github.com/doctorjei/droste) project (droste-core is the
central branch) under the kento → gemet → * umbrella; consumes the same gemet bases.

## Unified ROCm pin

Everything builds against **one** pinned TheRock nightly, installed via pip `rocm-sdk-*`
wheels from the gfx1151 per-arch index — **no apt ROCm repo, no S3 tarball**. The single
source of truth is [`rocm-version.env`](rocm-version.env):

| Piece | Pin |
|---|---|
| Index | `https://rocm.nightlies.amd.com/v2/gfx1151/` |
| ROCm SDK (`rocm-sdk-devel` / `-libraries-gfx1151`) | `7.13.0a20260501` |
| torch / torchvision / torchaudio | `2.9.1` / `0.24.0` / `2.9.0` (`+rocm7.13.0a20260501`, cp313) |
| Target | `gfx1151` only |

**Why nightly + why this date:** gfx1151 torch exists *only* as TheRock nightly wheels
(no stable/official gfx1151 torch until ROCm 8.0, ~mid-2026). torch is the binding
constraint — the newest Linux + Python-3.13 torch wheel is `7.13.0a20260501`, and both
`rocm-sdk-*` packages exist at that same date, so the whole set is ABI-consistent. This
is arch-specific, so it is also far leaner than the all-arch apt ROCm stack.

## Topology

Two bases feed everything; **torch is a shared layer added where needed**
(`droste-torch-base-halo`), not a base fork.

```
canopy ─ droste-runtime-base-halo   (de-divert + rocm-sdk-libraries-gfx1151 runtime kernels, venv)
           ├─ droste-llama-halo           ← COPY --from droste-llama-build-halo       (no torch)
           ├─ droste-ds4-halo             ← COPY --from droste-ds4-build-halo         (no torch)
           └─ droste-torch-base-halo  (+ shared torch wheel, installed once)
                 ├─ droste-comfyui-halo         (+ torchvision/audio; single interactive image, Triton JIT)
                 ├─ droste-vllm-halo            ← COPY --from droste-vllm-build-halo
                 └─ droste-finetuning-halo      ← COPY --from droste-finetuning-build-halo

droste-runtime-base-halo ─ droste-build-base-halo  (+ rocm-sdk-devel compilers + host toolchain)
           ├─ droste-llama-build-halo / droste-ds4-build-halo        [scratch: /artifacts/{bin,lib64,share}]
           └─ droste-vllm-build-halo / droste-finetuning-build-halo  [scratch: /artifacts/wheels]
```

torch is pip-installed once in `droste-torch-base-halo` and shared by comfyui/vllm/
finetuning (one stored layer instead of three identical copies). llama/ds4 stay
torch-free on the runtime base.

**Artifact-carrier pattern:** heavy compiles happen in `droste-build-base-halo`; outputs
are captured in minimal `FROM scratch` `-build` carriers (holding only `/artifacts`); thin
runtimes `COPY --from` them onto `droste-runtime-base-halo`. Shipped runtimes carry no
SDK/toolchain.

## Images

Published as `ghcr.io/doctorjei/droste-<name>-halo`. Containerfiles are named
`Container.<name>` under `base/`, `scaffolding/`, and `targets/`.

| Image | Containerfile | Base | Notes |
|---|---|---|---|
| `droste-runtime-base-halo` | `base/Container.runtime` | `gemet/canopy` | ROCm runtime kernels (pip), de-divert, venv `/opt/venv` |
| `droste-build-base-halo` | `base/Container.build` | `droste-runtime-base-halo` | + `rocm-sdk-devel` (hipcc/clang) + host toolchain |
| `droste-torch-base-halo` | `base/Container.torch` | `droste-runtime-base-halo` | + shared `torch` wheel (installed once; comfyui/vllm/finetuning build FROM this) |
| `droste-llama-build-halo` | `scaffolding/Container.llama-build` | build base | llama.cpp turboquant fork, gfx1151 HIP build [scratch carrier] |
| `droste-llama-halo` | `targets/Container.llama` | runtime base | llama runtime (no torch); `COPY --from` build carrier |
| `droste-ds4-build-halo` | `scaffolding/Container.ds4-build` | build base | ds4 + rocWMMA build [scratch carrier] |
| `droste-ds4-halo` | `targets/Container.ds4` | runtime base | ds4 runtime (no torch); cockpit via pipx |
| `droste-comfyui-halo` | `targets/Container.comfyui` | torch base | single image; +torchvision/audio; keeps compilers for Triton JIT at runtime |
| `droste-vllm-build-halo` | `scaffolding/Container.vllm-build` | build base | flash-attn + aiter + vllm wheels [scratch carrier] |
| `droste-vllm-halo` | `targets/Container.vllm` | torch base | vllm runtime (torch from base) |
| `droste-finetuning-build-halo` | `scaffolding/Container.finetuning-build` | build base | bitsandbytes + custom RCCL wheels [scratch carrier] |
| `droste-finetuning-halo` | `targets/Container.finetuning` | torch base | HF/unsloth stack (torch from base) |

The `-build` carriers are `FROM scratch` images holding only `/artifacts`.

`scaffolding/_fedora-src/` holds the original Fedora Containerfiles as a translation
reference (not built).

## Running

Every port image is a **server by default**: a shared entrypoint (baked in the
runtime base) reads the port's `/opt/resources/build-spec`, surfaces persistent
state from the `/opt/data` volume at the paths the tools expect (overlay/bind
mounts — every tool runs on its DEFAULTS, zero destination env vars), checks the
critical binds, seeds first-run content, then execs the service. A user command
still wins (`podman run IMAGE bash` gets a shell). The critical-bind checks run
first even then, so a quick shell with no binds needs `-e ALLOW_EPHEMERAL=1`.
Full contract + rationale: [BUILD_NOTES](BUILD_NOTES.md).

| Image | Service | Port | Config file (seeded if missing, on `/opt/data`) |
|---|---|---|---|
| comfyui | ComfyUI web UI | 8188 | `extra_model_paths.yaml` |
| finetuning | JupyterLab | 8888 | — (token auth; see container log) |
| vllm | `vllm serve --config` | 8000 | `vllm_config.yaml` — set `model:` |
| llama | `llama-server` | 8080 | `llama.env` — set `LLAMA_ARG_MODEL` |
| ds4 | `ds4-server` | 8000 | `ds4.env` — set `DS4_DROSTE_MODEL` |

Mount contract (all ports):

- **`/opt/data`** — the one container-specific volume (venv overlay upper, compute
  caches, the seeded config file, comfyui's model tree). Unbound → anonymous
  volume + a warning.
- **Critical binds** — hard-error at start unless bound; `ALLOW_EPHEMERAL=1`
  downgrades that to a warning. Always the **HF cache** (`~/.cache/huggingface` —
  the SINGLE model store, shared across all five ports; bind the same host dir
  everywhere and any model one tool downloads is available to the rest); plus
  comfyui `input`/`output` and finetuning `workspace` (irreplaceable user work).
- **`/opt/models`** — optional read-only local model collection (comfyui scanner
  source #2; the llama/ds4/vllm config model path may point here). Unbound →
  one-time INFO + marker file, never an error.

```bash
podman run -d -p 8188:8188 --device /dev/kfd --device /dev/dri \
  -v ~/droste/comfyui/data:/opt/data \
  -v ~/droste/comfyui/input:/opt/ComfyUI/input \
  -v ~/droste/comfyui/output:/opt/ComfyUI/output \
  -v ~/.cache/huggingface:/root/.cache/huggingface \
  ghcr.io/doctorjei/droste-comfyui-halo:latest
```

comfyui additionally runs a pre-launch **model scanner**: it classifies everything
in the HF cache (+ `/opt/models`) and maintains a ComfyUI-friendly symlink tree
(`/opt/data/model-tree`, surfaced at `/opt/ComfyUI/models`) — models any port
pulls into the shared cache appear in ComfyUI's pickers automatically.

**distrobox lane:** the same images double as `$HOME`-native interactive
toolboxes — `distrobox assemble create --file targets/<port>/distrobox.ini`. The
ini declares the volume binds plus an init hook that runs the same resolver in
distrobox mode (no internal mounts; the auto-bound host home persists the HF
cache and dotfile state natively).

## Building

ROCm/HIP is **ahead-of-time cross-compiled** — images build on any x86 host (no GPU).
Only runtime checks (`rocminfo`, `torch.cuda`, inference) need a real gfx1151 device.

CI (`.github/workflows/build-halo.yml`) builds the two bases, runs a
`hipcc --offload-arch=gfx1151` + `find_package(hip)` probe (the go/no-go that pip
`rocm-sdk-devel` compiles HIP for gfx1151), then builds all five ports — one isolated
job per port (artifacts → runtime). All jobs are green; every gfx1151 HIP compile
(rocWMMA, llama.cpp, vLLM, RCCL, bitsandbytes, aiter/flash-attn) succeeds on x86.

App-source clones are pinned to the SHAs that built green (per-image `ARG *_REF`);
override with `--build-arg <NAME>_REF=<sha>` to bump.

## Runtime validation

CI proves the images build + AOT-compile; it cannot prove they **run** (no GPU). On a
gfx1151 host that exposes `/dev/kfd` + `/dev/dri`, run the sweep:

```bash
scaffolding/check-rocm.sh              # checks :latest via podman
scaffolding/check-rocm.sh --tag <sha> --runtime docker --pull
scaffolding/check-rocm.sh --help       # all options
```

It skips the `*-build` carriers (they are `FROM scratch` — nothing to run) and checks the
runnable tiers in two tiers: **CORE** (deterministic — GPU enumerates as `gfx1151`; `torch.cuda`
sees it on comfyui/vllm/finetuning) and **APP** (per-toolbox smoke: `llama-server --version`,
ds4 binary+`ldd`, `import vllm`, `import bitsandbytes`). Exits non-zero on any failure. The
per-toolbox smoke commands are the first thing to adjust if a tool's CLI differs — see the
comments in `check-rocm.sh`.
