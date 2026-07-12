#!/usr/bin/env python3
"""model-scanner: HF-cache -> ComfyUI symlink-tree scanner for the droste comfyui container.

Runs at every container start (from the entrypoint, before ComfyUI launches). Walks the
shared HuggingFace hub cache plus an optional local models dir, CLASSIFIES every weight
file it finds, and maintains a ComfyUI-friendly symlink tree (default
/opt/data/model-tree, bind-mounted onto /opt/ComfyUI/models). Links point at the resolved
blob (realpath) so they survive HF repo revision bumps. Steady state (nothing changed)
does no heavy I/O: known identities skip classification entirely and go straight to a
cheap link-verify.

The registry is a classified INVENTORY of the model sources; the ComfyUI link tree is
just one consumer of it. EVERYTHING identifiable gets a real classification -- including
things ComfyUI cannot load (HF-format LLM repos -> `llm`, plain-LLM GGUFs -> `gguf-llm`,
CTranslate2 models -> `ctranslate2`, split-GGUF parts, sharded components) -- those are
recorded with links:[] and never linked. `unclassified` is reserved STRICTLY for
genuinely unknown files, so the UNCLASSIFIED report is a true heuristics-gap list.

Usage:
    model_scanner.py sync   [-n|--dry-run] [--no-prune] [--hardlink] [path overrides]
    model_scanner.py status [path overrides]
    (no verb defaults to `sync`)

Path overrides: --cache-dir, --models-dir, --tree, --registry.

DESIGN NOTES / REGISTRY SCHEMA
==============================
One YAML store (default /opt/data/model-registry.yaml) plays TWO roles:

1. Classification cache -- keyed by CONTENT IDENTITY so we never re-inspect a file we
   have already seen:
     * HF blobs:      "hf:<blob-filename>" -- the blob filename IS its sha256 (no re-hash)
     * local files:   "local:<relpath>|<size>|<mtime_ns>" -- cheap stat key; a touched or
       rewritten file gets a NEW identity (old one prunes, new one reclassifies), so
       multi-GB files are never fully hashed.
     * diffusers repo units: "diffusers:<org>/<repo>@<revision>" -- a REPO-LEVEL entry
       (the unit is a snapshot dir, not a blob); re-detected each run (one stat), so it
       carries no classification cost.

2. Ownership ledger -- the safety invariant:
     * a tree-relative link path recorded under some entry's `links` == OURS
       (we may replace or prune it);
     * any path in the tree NOT in the registry == the USER'S -> never clobbered,
       never pruned. Registry lost => worst case orphaned symlinks, never destroyed
       user files.

Schema (version 2):

    version: 2            # registry format version (2: repo units, members, sharded)
    heuristics: 2         # classification-heuristics version; mismatch => reclassify all
                          #   (also migrates v1 trees: old links are still owned, so the
                          #    prune/relink pass renames them cleanly)
    entries:
      "hf:<sha256>":                       # weight-file entry
        origin: hf                         # or "local"
        category: vae                      # ComfyUI category, an inventory category
                                           #   (llm, gguf-llm, ctranslate2, gguf-split,
                                           #    diffusers), or "unclassified"
        sharded: true                      # OPTIONAL; multi-part file: classified but
                                           #   never linked (absent = single-file)
        source: /abs/path/to/blobs/<sha>   # resolved source at last sync (informational)
        display: org/repo/vae/diffusion_pytorch_model.safetensors   # provenance
        links:                             # tree-relative links WE own (ownership!)
          - vae/FLUX.1-Fill-dev--vae.safetensors   # [] for inventory-only/unclassified
      "diffusers:org/repo@<rev>":          # repo-level diffusers unit
        origin: hf
        category: diffusers
        source: /abs/path/to/snapshots/<rev>
        display: org/repo@<rev8>
        links: []                          # NEVER dir-linked (DiffusersLoader is niche)
        members:                           # identities of its component weight files
          - "hf:<sha256>"                  #   (repo <-> component relationship)

Only categories in the ComfyUI category set are linkable, and sharded files are never
linked even when their role is known. Entries whose identity vanished from the sources
are dropped (and their symlinks pruned) on sync unless --no-prune, in which case the
entries are carried forward so ownership of the now-broken links is retained for a
later prune.

LINK NAMING -- generic-filename rule: when a to-be-linked basename is generic
(model.safetensors, diffusion_pytorch_model.safetensors, pytorch_model.bin, model.bin,
fp16/bf16 variants, ...) the link name is ALWAYS derived from provenance:
`<repo-name>--<subdir>` (e.g. vae/FLUX.1-Fill-dev--vae.safetensors) or
`<repo-name>--<stem>` at repo root -- a meaningless plain name is never squatted.
Non-generic basenames keep their plain name; a provenance prefix is added only on
collision.

Classification ladder (cheapest first, stop at first confident match; on conflict or
no signal -> "unclassified", which is recorded but NEVER linked):
  0. diffusers component role: files inside a repo with a root model_index.json inherit
     their component subdir's role (vae -> vae, text_encoder* -> text_encoders,
     transformer/unet/prior -> diffusion_models, image_encoder -> clip_vision, ...)
  1. path segments (snapshot-relative dirs, e.g. Comfy-Org `split_files/<category>/`,
     or category-named subdirs under the local models dir)
  2. filename keywords (lora, vae, controlnet, t5/clip_l/clip_g, esrgan/4x-upscalers,
     yolo detectors -> ultralytics/{bbox,segm}, sam -> sams, face-parsing, pose, ...)
  3. CTranslate2 layout: `model.bin` + a sibling vocabulary file -> ctranslate2
  4. safetensors header (8-byte length + JSON header ONLY -- tensors never read):
     tensor-name prefixes + `__metadata__` modelspec.architecture
  5. sibling config.json (HF-format repos): `architectures` -> category;
     LLM architectures (…ForCausalLM etc.) -> `llm` (vllm's models; inventory-only)
  6. GGUF metadata (`general.architecture`): diffusion archs -> diffusion_models,
     clip/t5 -> text_encoders, LLM archs -> `gguf-llm` (llama/ds4's models;
     inventory-only); a split part whose architecture is unreadable -> `gguf-split`

The linking core (ensure_link/prune, ownership checks) is classification-agnostic so a
later manifest-driven explicit layer (see hf-comfy-link seed) can share it.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import struct
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

try:
    import yaml
except ImportError:  # pragma: no cover
    sys.exit("model-scanner: pyyaml is required (the comfyui image bakes it)")

# --------------------------------------------------------------------------- constants

REGISTRY_VERSION = 2
HEURISTICS_VERSION = 3

DEFAULT_CACHE_DIR = "~/.cache/huggingface/hub"
DEFAULT_MODELS_DIR = "/opt/models"
DEFAULT_TREE = "/opt/data/model-tree"
DEFAULT_REGISTRY = "/opt/data/model-registry.yaml"

WEIGHT_EXTS = {".safetensors", ".gguf", ".pth", ".ckpt", ".pt", ".bin"}

UNCLASSIFIED = "unclassified"

# ComfyUI category dirs (folder_paths.py) == the LINKABLE categories.
CATEGORIES = {
    "checkpoints", "diffusion_models", "text_encoders", "vae", "loras",
    "controlnet", "clip_vision", "upscale_models", "latent_upscale_models",
    "embeddings", "style_models", "photomaker", "gligen", "hypernetworks",
    "vae_approx", "audio_encoders", "model_patches", "diffusers",
    "background_removal", "detection", "frame_interpolation",
    "geometry_estimation", "optical_flow",
}

# Custom-node detector/aux dirs that popular ComfyUI packs scan (Impact-Pack,
# ReActor/facexlib, ControlNet-aux). Not core folder_paths categories, but the tree
# link mechanics are identical, so they are added to the LINKABLE set. Nested paths
# (ultralytics/bbox) work directly: plan_links just joins "<cat>/<link_name>".
DETECTOR_CATEGORIES = {
    "ultralytics/bbox",   # Impact-Pack UltralyticsDetectorProvider: detection/bbox
    "ultralytics/segm",   # Impact-Pack UltralyticsDetectorProvider: segmentation
    "sams",               # Impact-Pack SAMLoader: Segment-Anything checkpoints
    "facedetection",      # ReActor / facerestore facexlib weights (face parsing)
    "controlnet_aux",     # ControlNet-aux annotator / pose-estimator weights
}
CATEGORIES |= DETECTOR_CATEGORIES

# Inventory-only categories: real classifications that ComfyUI cannot load.
# Recorded with links:[] so later features are a linking rule, never a re-scan.
INVENTORY_CATEGORIES = {"llm", "gguf-llm", "ctranslate2", "gguf-split", "diffusers"}

# path-segment -> category (step 1). Includes aliases and common singular forms.
SEGMENT_MAP = {c: c for c in CATEGORIES}
SEGMENT_MAP.update({
    "unet": "diffusion_models",          # legacy alias
    "clip": "text_encoders",             # legacy alias
    "diffusion_model": "diffusion_models",
    "text_encoder": "text_encoders",
    "lora": "loras",
    "checkpoint": "checkpoints",
    "embedding": "embeddings",
    "controlnets": "controlnet",
    "upscaler": "upscale_models",
    "upscalers": "upscale_models",
    "upscale": "upscale_models",
    "hypernetwork": "hypernetworks",
})

# diffusers component subdir -> role (step 0; only inside a model_index.json repo)
DIFFUSERS_COMPONENT_MAP = {
    "vae": "vae", "vae_decoder": "vae", "vae_encoder": "vae",
    "text_encoder": "text_encoders", "text_encoder_2": "text_encoders",
    "text_encoder_3": "text_encoders",
    "transformer": "diffusion_models", "unet": "diffusion_models",
    "prior": "diffusion_models",
    "image_encoder": "clip_vision",
    "controlnet": "controlnet",
}

# GGUF general.architecture -> category (step 6)
GGUF_DIFFUSION_ARCHS = {
    "flux", "sd1", "sd2", "sd3", "sdxl", "stable_diffusion", "sd",
    "wan", "qwen_image", "ltxv", "hunyuan", "hunyuan_video", "hidream",
    "cosmos", "lumina2", "aura", "auraflow", "pixart", "chroma", "omnigen",
}
GGUF_TEXT_ARCHS = {"clip", "t5", "t5encoder", "umt5", "byt5", "bert"}
GGUF_LLM_ARCHS = {
    "llama", "llama4", "qwen2", "qwen2moe", "qwen2vl", "qwen3", "qwen3moe",
    "deepseek", "deepseek2", "deepseek3", "gemma", "gemma2", "gemma3",
    "mistral", "mixtral", "phi2", "phi3", "phimoe", "gpt2", "gptneox",
    "falcon", "starcoder", "starcoder2", "command-r", "cohere", "cohere2",
    "olmo", "olmo2", "granite", "granitemoe", "internlm2", "stablelm",
    "mamba", "rwkv6", "glm4", "chatglm", "minicpm", "minicpm3", "nemotron",
    "exaone", "baichuan", "orion", "plamo", "smollm3",
}

# config.json architectures -> category (step 5)
CONFIG_TEXT_ARCHS = {
    "CLIPModel", "CLIPTextModel", "CLIPTextModelWithProjection",
    "T5EncoderModel", "UMT5EncoderModel", "MT5EncoderModel", "T5Model",
}
CONFIG_VISION_ARCHS = {"CLIPVisionModel", "CLIPVisionModelWithProjection",
                       "SiglipVisionModel", "SiglipModel"}
# HF-format LLM repos are vllm's problem, not comfyui's: inventory-only `llm`
CONFIG_LLM_SUFFIXES = ("ForCausalLM", "ForConditionalGeneration",
                       "LMHeadModel", "ForSeq2SeqLM")

# CTranslate2 model dirs: model.bin + one of these signature vocabulary files (step 3)
CT2_SIBLINGS = ("vocabulary.txt", "vocabulary.json",
                "shared_vocabulary.txt", "shared_vocabulary.json")

# multi-part weight files: model-00001-of-00002.safetensors, *-00001-of-00003.gguf, ...
SHARDED_RE = re.compile(r"-\d{4,6}-of-\d{4,6}\.[A-Za-z0-9]+$")

# ESRGAN-style upscaler scale-factor tag on an otherwise arch-unknown .pth: a leading
# "4x"/"2x"/"8x"/"16x" or a trailing "x4"/"x2"/... (e.g. 4x-ClearRealityV1, RealESRGAN_x4).
# Matched on the normalized stem (non-alnum -> "_"), only AFTER lora/vae/controlnet/text
# checks have excluded genuine models, so a bare scale tag is a strong upscaler signal.
UPSCALE_SCALE_RE = re.compile(
    r"(?:^|_)(?:2|4|8|16)x(?=$|_|[a-z])|(?:^|_)x(?:2|4|8|16)(?=$|_|[a-z])")

# generic basenames that must never be used as link names (provenance rule)
GENERIC_STEM_RE = re.compile(
    r"^(model|pytorch_model|diffusion_pytorch_model|tf_model|flax_model|"
    r"adapter_model)([._-](fp16|fp32|bf16))?$")


# --------------------------------------------------------------------------- logging

def log(msg: str) -> None:
    print(msg, flush=True)


# --------------------------------------------------------------------------- sources

@dataclass
class SourceFile:
    identity: str        # content identity (registry key)
    path: Path           # resolved real path (blob / real file) -- link target
    link_name: str       # preferred basename for the tree link (provenance-resolved)
    display: str         # human-readable provenance
    rel_dir_parts: tuple # lowercase dir segments for step-1 classification
    config_dir: Path     # dir to check for sibling config.json / CT2 vocabularies
    origin: str          # "hf" | "local"
    sharded: bool = False        # multi-part file: classified but never linked
    component: str | None = None  # diffusers component subdir (repo has model_index.json)


@dataclass
class RepoUnit:
    """A repo-level diffusers unit (snapshot dir with a root model_index.json)."""
    identity: str        # "diffusers:<org>/<repo>@<revision>"
    display: str
    source: str          # snapshot dir
    members: list        # identities of its component weight files


def _sanitize(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "-", name).strip("-")


def preferred_link_name(repo_tail: str | None, rel: Path) -> str:
    """Generic basenames ALWAYS get a provenance-derived name; others keep theirs."""
    if not GENERIC_STEM_RE.match(rel.stem.lower()):
        return rel.name
    parts = []
    if repo_tail:
        parts.append(repo_tail)
    dirs = rel.parts[:-1]
    if dirs:
        parts.append("-".join(dirs))
    if not parts:
        return rel.name  # local file at models root: no provenance to derive from
    if len(parts) < 2:
        parts.append(rel.stem)  # keep the stem when only one provenance element
    return _sanitize("--".join(parts)) + rel.suffix


def scan_hf_cache(cache_dir: Path) -> tuple[list[SourceFile], list[RepoUnit]]:
    """Walk the standard hub layout: models--org--repo/snapshots/<rev>/<path> -> blobs/<sha>.

    Dedupes files by blob identity (the same blob referenced from several revisions or
    several repos is scanned once). Snapshots with a root model_index.json additionally
    yield a repo-level diffusers RepoUnit listing their member identities."""
    sources: dict[str, SourceFile] = {}
    units: list[RepoUnit] = []
    if not cache_dir.is_dir():
        return [], []
    for repo_dir in sorted(p for p in cache_dir.iterdir()
                           if p.is_dir() and p.name.startswith("models--")):
        repo_display = repo_dir.name[len("models--"):].replace("--", "/", 1)
        repo_tail = repo_display.split("/", 1)[-1]
        snaps = repo_dir / "snapshots"
        if not snaps.is_dir():
            continue
        for snap in sorted(p for p in snaps.iterdir() if p.is_dir()):
            is_diffusers = (snap / "model_index.json").is_file()
            members: list[str] = []
            for f in sorted(snap.rglob("*")):
                if not (f.is_file() or f.is_symlink()):
                    continue
                if f.suffix.lower() not in WEIGHT_EXTS:
                    continue
                real = f.resolve()
                if not real.is_file():
                    continue  # dangling snapshot link (blob pruned) -- not a source
                if real.parent.name == "blobs":
                    identity = "hf:" + real.name  # blob filename IS its sha256
                else:  # non-standard layout; fall back to a stat key
                    st = real.stat()
                    identity = f"hf:{real.name}|{st.st_size}|{st.st_mtime_ns}"
                members.append(identity)
                if identity in sources:
                    continue
                rel = f.relative_to(snap)
                sources[identity] = SourceFile(
                    identity=identity,
                    path=real,
                    link_name=preferred_link_name(repo_tail, rel),
                    display=f"{repo_display}/{rel}",
                    rel_dir_parts=tuple(s.lower() for s in rel.parts[:-1]),
                    config_dir=f.parent,
                    origin="hf",
                    sharded=bool(SHARDED_RE.search(f.name)),
                    component=(rel.parts[0] if is_diffusers and len(rel.parts) > 1
                               else None),
                )
            if is_diffusers:
                units.append(RepoUnit(
                    identity=f"diffusers:{repo_display}@{snap.name}",
                    display=f"{repo_display}@{snap.name[:8]}",
                    source=str(snap),
                    members=sorted(set(members)),
                ))
    return list(sources.values()), units


def scan_models_dir(models_dir: Path) -> list[SourceFile]:
    """Walk the optional local models dir. Absent dir -> silently empty."""
    sources: list[SourceFile] = []
    if not models_dir.is_dir():
        return sources
    for f in sorted(models_dir.rglob("*")):
        if not f.is_file() or f.suffix.lower() not in WEIGHT_EXTS:
            continue
        rel = f.relative_to(models_dir)
        st = f.stat()
        sources.append(SourceFile(
            identity=f"local:{rel}|{st.st_size}|{st.st_mtime_ns}",
            path=f.resolve(),
            link_name=preferred_link_name(None, rel),
            display=str(rel),
            rel_dir_parts=tuple(s.lower() for s in rel.parts[:-1]),
            config_dir=f.parent,
            origin="local",
            sharded=bool(SHARDED_RE.search(f.name)),
        ))
    return sources


def collect_sources(cache_dir: Path,
                    models_dir: Path) -> tuple[list[SourceFile], list[RepoUnit]]:
    files, units = scan_hf_cache(cache_dir)
    return files + scan_models_dir(models_dir), units


# --------------------------------------------------------------------- format readers
# Module-level so tests can wrap them with counters (incremental runs must not call them).

def read_safetensors_header(path: Path) -> dict:
    """Read ONLY the 8-byte length + JSON header of a .safetensors file."""
    with open(path, "rb") as f:
        raw = f.read(8)
        if len(raw) < 8:
            raise ValueError("truncated safetensors")
        n = int.from_bytes(raw, "little")
        if not 0 < n <= 100_000_000:
            raise ValueError(f"implausible safetensors header length {n}")
        return json.loads(f.read(n))


def read_gguf_metadata(path: Path, max_kv: int = 256) -> dict:
    """Read GGUF magic + metadata key/values (scalar + string; arrays skipped)."""
    meta: dict = {}
    with open(path, "rb") as f:
        if f.read(4) != b"GGUF":
            raise ValueError("not a GGUF file")
        version, = struct.unpack("<I", f.read(4))
        if version < 2:
            raise ValueError(f"unsupported GGUF version {version}")
        _tensor_count, kv_count = struct.unpack("<QQ", f.read(16))

        scalar = {0: ("<B", 1), 1: ("<b", 1), 2: ("<H", 2), 3: ("<h", 2),
                  4: ("<I", 4), 5: ("<i", 4), 6: ("<f", 4), 7: ("<?", 1),
                  10: ("<Q", 8), 11: ("<q", 8), 12: ("<d", 8)}

        def read_str() -> str:
            n, = struct.unpack("<Q", f.read(8))
            if n > 65536:
                raise ValueError("implausible GGUF string length")
            return f.read(n).decode("utf-8", "replace")

        def read_value(vtype: int, store: bool):
            if vtype in scalar:
                fmt, size = scalar[vtype]
                v, = struct.unpack(fmt, f.read(size))
                return v if store else None
            if vtype == 8:  # string
                s = read_str()
                return s if store else None
            if vtype == 9:  # array: elem-type u32 + count u64 + elems (skipped)
                etype, = struct.unpack("<I", f.read(4))
                count, = struct.unpack("<Q", f.read(8))
                if count > 1_000_000:
                    raise ValueError("implausible GGUF array length")
                for _ in range(count):
                    read_value(etype, store=False)
                return None
            raise ValueError(f"unknown GGUF value type {vtype}")

        for _ in range(min(kv_count, max_kv)):
            key = read_str()
            vtype, = struct.unpack("<I", f.read(4))
            val = read_value(vtype, store=True)
            if val is not None:
                meta[key] = val
            if "general.architecture" in meta:
                break  # all we need; stop early
    return meta


def read_config_json(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


# ----------------------------------------------------------------------- classification

def classify_by_segments(src: SourceFile) -> str | None:
    for seg in src.rel_dir_parts:
        cat = SEGMENT_MAP.get(seg)
        if cat:
            return cat
    return None


def classify_by_filename(name: str) -> str | None:
    norm = re.sub(r"[^a-z0-9]+", "_", Path(name).stem.lower()).strip("_")
    tokens = set(t for t in norm.split("_") if t)
    if "clip_vision" in norm:                       # before any bare-clip rule
        return "clip_vision"
    if tokens & {"lora", "loras"} or "lightning_lora" in norm:
        return "loras"
    if "taesd" in tokens:
        return "vae_approx"
    if "vae" in tokens:
        return "vae"
    if "controlnet" in norm or tokens & {"canny", "depth"}:
        return "controlnet"
    if (tokens & {"t5", "umt5", "byt5", "t5xxl"}
            or "clip_l" in norm or "clip_g" in norm
            or "text_encoder" in norm or "qwen_2_5_vl" in norm):
        return "text_encoders"
    # ---- auxiliary detector / estimator models (Impact-Pack / ReActor / ControlNet-aux).
    # These are .pt/.pth pickles the scanner never content-sniffs, so filename is the
    # only signal -- same idiom as the esrgan/taesd rules above.
    # Ultralytics / YOLO detectors -> Impact-Pack models/ultralytics/{bbox,segm}
    # ("yolo" substring catches yolov5/8/9, yolo11, yolov11, yolox, ... in one shot).
    if "yolo" in norm:
        if "segm" in norm or tokens & {"seg", "segmentation"}:
            return "ultralytics/segm"
        return "ultralytics/bbox"
    # SAM (Segment Anything) -> Impact-Pack models/sams
    if ("sam_vit" in norm or "mobile_sam" in norm or "sam_hq" in norm
            or "sam2_hiera" in norm or "sam2" in tokens
            or ("sam" in tokens and "vit" in tokens)):
        return "sams"
    # BiSeNet / ParseNet face-parsing (facexlib) -> models/facedetection
    if tokens & {"parsenet", "bisenet"} or "parsing_parsenet" in norm \
            or "parsing_bisenet" in norm:
        return "facedetection"
    # OpenPose / DWPose body & hand estimators -> ControlNet-aux annotator weights
    if ("openpose" in norm or "dwpose" in norm
            or "body_pose" in norm or "hand_pose" in norm
            or ("pose" in tokens and tokens & {"body", "hand", "model"})):
        return "controlnet_aux"
    # ESRGAN-family upscalers, incl. arch-unknown 4x/2x/x4 .pth files
    if ("esrgan" in norm or tokens & {"upscale", "upscaler"}
            or UPSCALE_SCALE_RE.search(norm)):
        return "upscale_models"
    if tokens & {"embedding", "embeddings"} or "textual_inversion" in norm:
        return "embeddings"
    if "hypernetwork" in norm:
        return "hypernetworks"
    if "photomaker" in tokens:
        return "photomaker"
    if "gligen" in tokens:
        return "gligen"
    return None


def classify_safetensors_header(header: dict) -> str | None:
    meta = header.get("__metadata__") or {}
    arch = str(meta.get("modelspec.architecture", "")).lower()
    if arch:
        if "lora" in arch:
            return "loras"
        if "controlnet" in arch:
            return "controlnet"
        if arch.endswith("vae") or "/vae" in arch:
            return "vae"

    keys = [k for k in header if k != "__metadata__"]
    prefixes = set()
    for k in keys:
        prefixes.add(k.split(".", 1)[0] + ".")

    def any_start(*pfx):
        return any(k.startswith(pfx) for k in keys)

    # loras first: lora tensor names embed base-model names (double_blocks etc.)
    if any(k.startswith(("lora_unet_", "lora_te")) or ".lora_A" in k
           or ".lora_B" in k or ".lora_down" in k or ".lora_up" in k for k in keys):
        return "loras"
    if any_start("model.diffusion_model."):        # full checkpoint bundle
        return "checkpoints"
    if any_start("control_model."):
        return "controlnet"
    if any_start("vision_model."):
        return "clip_vision"
    if any_start("diffusion_model.", "double_blocks.", "joint_blocks."):
        return "diffusion_models"
    if any_start("encoder.block.", "decoder.block."):   # T5-style, before generic vae
        return "text_encoders"
    if any_start("first_stage_model."):
        return "vae"
    if "decoder." in prefixes and "encoder." in prefixes:
        return "vae"
    if any_start("text_model.", "t5.", "enc."):
        return "text_encoders"
    return None


def classify_config(config: dict) -> str | None:
    archs = config.get("architectures") or []
    for a in archs:
        if a in CONFIG_TEXT_ARCHS:
            return "text_encoders"
        if a in CONFIG_VISION_ARCHS:
            return "clip_vision"
        if a.startswith("Autoencoder"):
            return "vae"
        if a.endswith(CONFIG_LLM_SUFFIXES):
            return "llm"  # HF-format LLM repo: vllm's model, inventory-only
    return None


def classify_gguf(meta: dict) -> str:
    arch = str(meta.get("general.architecture", "")).lower()
    if arch in GGUF_DIFFUSION_ARCHS:
        return "diffusion_models"
    if arch in GGUF_TEXT_ARCHS:
        return "text_encoders"
    if arch in GGUF_LLM_ARCHS:
        return "gguf-llm"  # llama/ds4's models, inventory-only
    return UNCLASSIFIED  # unknown architecture: a true heuristics gap


def classify(src: SourceFile) -> str:
    """Heuristic ladder, cheapest first; no confident signal -> unclassified."""
    ext = Path(src.link_name).suffix.lower()

    # 0. diffusers component role (repo has a root model_index.json)
    if src.component:
        cat = DIFFUSERS_COMPONENT_MAP.get(src.component.lower())
        if cat:
            return cat

    cat = classify_by_segments(src)
    if cat:
        return cat

    cat = classify_by_filename(src.link_name)
    if cat:
        return cat

    # CTranslate2 layout: model.bin + signature vocabulary sibling (cheap stats)
    if (Path(src.display).name.lower() == "model.bin"
            and any((src.config_dir / v).is_file() for v in CT2_SIBLINGS)):
        return "ctranslate2"

    if ext == ".safetensors":
        try:
            cat = classify_safetensors_header(read_safetensors_header(src.path))
            if cat:
                return cat
        except (OSError, ValueError, json.JSONDecodeError) as e:
            log(f"WARN  unreadable safetensors header {src.display}: {e}")

    cfg = src.config_dir / "config.json"
    if cfg.is_file():
        try:
            cat = classify_config(read_config_json(cfg))
            if cat:
                return cat
        except (OSError, ValueError, json.JSONDecodeError) as e:
            log(f"WARN  unreadable config.json next to {src.display}: {e}")

    if ext == ".gguf":
        try:
            cat = classify_gguf(read_gguf_metadata(src.path))
            if cat != UNCLASSIFIED:
                return cat
        except (OSError, ValueError, struct.error) as e:
            log(f"WARN  unreadable GGUF metadata {src.display}: {e}")
        if src.sharded:
            return "gguf-split"  # identified as a split-GGUF part, role unknown
        return UNCLASSIFIED

    return UNCLASSIFIED


# --------------------------------------------------------------------------- registry

@dataclass
class Registry:
    entries: dict = field(default_factory=dict)   # identity -> entry dict
    heuristics: int = HEURISTICS_VERSION

    def owned_links(self) -> set[str]:
        owned: set[str] = set()
        for e in self.entries.values():
            owned.update(e.get("links") or [])
        return owned


def load_registry(path: Path) -> Registry:
    if not path.is_file():
        return Registry()
    try:
        data = yaml.safe_load(path.read_text()) or {}
        entries = data.get("entries") or {}
        if not isinstance(entries, dict):
            raise ValueError("entries is not a mapping")
        return Registry(entries=entries,
                        heuristics=int(data.get("heuristics", 0)))
    except Exception as e:  # corrupt registry -> start fresh; links fail safe to "user's"
        log(f"WARN  unreadable registry {path} ({e}); starting fresh "
            f"(existing links are treated as user-owned)")
        return Registry()


def save_registry(path: Path, reg: Registry) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    doc = {"version": REGISTRY_VERSION, "heuristics": HEURISTICS_VERSION,
           "entries": reg.entries}
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(yaml.safe_dump(doc, sort_keys=True, default_flow_style=False))
    tmp.replace(path)


# ------------------------------------------------------------------------ linking core
# Classification-agnostic: a future manifest-driven explicit layer can reuse these.

def link_matches(dst: Path, target: Path, hardlink: bool) -> bool:
    if dst.is_symlink():
        try:
            return dst.resolve() == target
        except OSError:
            return False
    if hardlink and dst.is_file():
        try:
            return dst.stat().st_ino == target.stat().st_ino
        except OSError:
            return False
    return False


def ensure_link(target: Path, dst: Path, owned: bool, hardlink: bool, dry: bool) -> str:
    """Returns: ok | linked | relinked | conflict."""
    if dst.is_symlink() or dst.exists():
        if link_matches(dst, target, hardlink):
            return "ok"
        if not owned:
            return "conflict"       # unowned file/link already there: never clobber
        if not dst.is_symlink():
            # A non-matching REGULAR file is never replaced, even if the path is
            # owned: we cannot tell a stale hardlink of ours from a real file the
            # user dropped over our link. Safety wins.
            return "conflict"
        if not dry:
            dst.unlink()
            (os.link if hardlink else os.symlink)(target, dst)
        return "relinked"
    if not dry:
        dst.parent.mkdir(parents=True, exist_ok=True)
        (os.link if hardlink else os.symlink)(target, dst)
    return "linked"


def prune_link(tree: Path, rel: str, dry: bool) -> str:
    """Remove an OWNED link whose source vanished. Returns: pruned | kept | gone."""
    dst = tree / rel
    if dst.is_symlink():
        if not dry:
            dst.unlink()
        return "pruned"
    if dst.exists():
        return "kept"   # real file (user replacement or hardlink holding data): keep
    return "gone"


# ------------------------------------------------------------------------------- sync

def linkable(src: SourceFile, category: str) -> bool:
    return category in CATEGORIES and not src.sharded


def plan_links(sources: list[SourceFile], categories: dict[str, str]) -> dict[str, SourceFile]:
    """Map tree-relative link path -> source. Name collisions between two different
    sources get a deterministic provenance-prefixed name."""
    desired: dict[str, SourceFile] = {}
    for src in sources:
        cat = categories[src.identity]
        if not linkable(src, cat):
            continue
        rel = f"{cat}/{src.link_name}"
        if rel in desired and desired[rel].identity != src.identity:
            prefix = _sanitize(str(Path(src.display).parent))
            alt = f"{cat}/{prefix}--{src.link_name}" if prefix else None
            if not alt or alt in desired:
                log(f"SKIP  name collision for {src.display} -> {rel}")
                continue
            rel = alt
        if rel not in desired:
            desired[rel] = src
    return desired


def cmd_sync(args) -> int:
    t0 = time.monotonic()
    tree: Path = args.tree
    dry = args.dry_run

    old = load_registry(args.registry)
    reuse_cache = old.heuristics == HEURISTICS_VERSION
    if old.entries and not reuse_cache:
        log(f"INFO  heuristics changed ({old.heuristics} -> {HEURISTICS_VERSION}); "
            f"reclassifying everything")
    owned = old.owned_links()

    files, units = collect_sources(args.cache_dir, args.models_dir)

    # classify only the DELTA; known identities skip straight to link-verify
    categories: dict[str, str] = {}
    n_new = 0
    for src in files:
        prev = old.entries.get(src.identity) if reuse_cache else None
        if prev and prev.get("category"):
            categories[src.identity] = prev["category"]
        else:
            n_new += 1
            cat = classify(src)
            categories[src.identity] = cat
            log(f"CLASSIFY  {src.display} -> {cat}"
                f"{' [sharded]' if src.sharded else ''}")

    desired = plan_links(files, categories)

    # link phase
    new = Registry()
    counts = {"ok": 0, "linked": 0, "relinked": 0, "conflict": 0}
    entry_links: dict[str, list[str]] = {s.identity: [] for s in files}
    for rel in sorted(desired):
        src = desired[rel]
        state = ensure_link(src.path, tree / rel, owned=rel in owned,
                            hardlink=args.hardlink, dry=dry)
        counts[state] += 1
        if state == "conflict":
            log(f"CONFLICT  {rel} exists and is not ours; leaving it alone "
                f"(wanted -> {src.display})")
        else:
            if state != "ok":
                log(f"{'DRY-' if dry else ''}{state.upper()}  {rel} -> {src.path}")
            entry_links[src.identity].append(rel)

    for src in files:
        entry = {
            "origin": src.origin,
            "category": categories[src.identity],
            "source": str(src.path),
            "display": src.display,
            "links": entry_links[src.identity],
        }
        if src.sharded:
            entry["sharded"] = True
        new.entries[src.identity] = entry
        if categories[src.identity] == UNCLASSIFIED and src.identity not in old.entries:
            log(f"UNCLASSIFIED  {src.display} (not linked; heuristics gap)")

    # repo-level diffusers units: re-detected each run (one stat), never linked
    for u in units:
        if u.identity not in old.entries:
            log(f"REPO  {u.display} -> diffusers unit ({len(u.members)} member files)")
        new.entries[u.identity] = {
            "origin": "hf",
            "category": "diffusers",
            "source": u.source,
            "display": u.display,
            "links": [],
            "members": u.members,
        }

    # prune phase: owned links whose source vanished or whose target moved category
    n_pruned = 0
    desired_rels = set(desired)
    for ident, entry in old.entries.items():
        stale = [rel for rel in (entry.get("links") or []) if rel not in desired_rels]
        for rel in stale:
            if not args.prune:
                continue
            state = prune_link(tree, rel, dry)
            if state == "pruned":
                n_pruned += 1
                log(f"{'DRY-' if dry else ''}PRUNE  {rel} (source gone or reclassified)")
            elif state == "kept":
                log(f"KEEP  {rel} is a real file now; not pruning (dropped from registry)")
        if not args.prune and ident not in new.entries:
            # keep ownership of not-yet-pruned links for a later prune
            new.entries[ident] = entry

    # report broken symlinks that are NOT ours (never touched)
    if tree.is_dir():
        new_owned = new.owned_links()
        for p in sorted(tree.rglob("*")):
            if p.is_symlink() and not p.exists():
                rel = str(p.relative_to(tree))
                if rel not in new_owned and rel not in owned:
                    log(f"NOTE  broken unowned symlink (left alone): {rel}")

    if not dry:
        save_registry(args.registry, new)

    dt = time.monotonic() - t0
    n_uncls = sum(1 for c in categories.values() if c == UNCLASSIFIED)
    n_inv = sum(1 for s in files
                if categories[s.identity] != UNCLASSIFIED
                and not linkable(s, categories[s.identity])) + len(units)
    log(f"model-scanner: {len(files)} files + {len(units)} repo units ({n_new} new), "
        f"{counts['ok']} ok, {counts['linked']} linked, {counts['relinked']} relinked, "
        f"{n_pruned} pruned, {counts['conflict']} conflicts, "
        f"{n_inv} inventory-only, {n_uncls} unclassified"
        f"{' [dry-run]' if dry else ''} ({dt:.2f}s)")
    return 0


# ------------------------------------------------------------------------------ status

def cmd_status(args) -> int:
    tree: Path = args.tree
    reg = load_registry(args.registry)
    files, units = collect_sources(args.cache_dir, args.models_dir)
    displays = {s.identity: s.display for s in files}
    displays.update({u.identity: u.display for u in units})

    new = [i for i in sorted(displays) if i not in reg.entries]
    gone = [i for i in reg.entries if i not in displays]
    owned = reg.owned_links()

    n_ok = n_broken = n_missing = 0
    for ident, entry in reg.entries.items():
        for rel in entry.get("links") or []:
            dst = tree / rel
            if not (dst.is_symlink() or dst.exists()):
                n_missing += 1
                log(f"MISSING  {rel} (owned link absent; sync will recreate)")
            elif dst.is_symlink() and not dst.exists():
                n_broken += 1
                log(f"BROKEN  {rel} (owned; sync --prune will remove)")
            else:
                n_ok += 1
        if entry.get("category") == UNCLASSIFIED:
            log(f"UNCLASSIFIED  {entry.get('display', ident)}"
                f"{'' if ident in displays else ' (source gone)'}")

    n_user_files = n_user_broken = 0
    if tree.is_dir():
        for p in sorted(tree.rglob("*")):
            if p.is_dir():
                continue
            rel = str(p.relative_to(tree))
            if rel in owned:
                continue
            if p.is_symlink() and not p.exists():
                n_user_broken += 1
                log(f"NOTE  broken unowned symlink (left alone): {rel}")
            else:
                n_user_files += 1
                log(f"USER  {rel} (not ours; never touched)")

    for i in new:
        log(f"NEW  {displays[i]} (will classify on next sync)")
    for i in gone:
        log(f"GONE  {reg.entries[i].get('display', i)} (source vanished; "
            f"sync --prune will clean)")

    log(f"model-scanner status: {len(files)} files + {len(units)} repo units "
        f"({len(new)} new, {len(gone)} gone from registry), "
        f"links: {n_ok} ok / {n_broken} broken / {n_missing} missing, "
        f"user items: {n_user_files} files+links, {n_user_broken} broken")
    return 0


# --------------------------------------------------------------------------------- cli

def build_parser() -> argparse.ArgumentParser:
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--cache-dir", type=lambda s: Path(s).expanduser(),
                        default=Path(os.environ.get("HF_HUB_CACHE",
                                                    DEFAULT_CACHE_DIR)).expanduser(),
                        help=f"HF hub cache (default {DEFAULT_CACHE_DIR}, "
                             f"honors $HF_HUB_CACHE)")
    common.add_argument("--models-dir", type=lambda s: Path(s).expanduser(),
                        default=Path(DEFAULT_MODELS_DIR),
                        help=f"optional local models dir (default {DEFAULT_MODELS_DIR}; "
                             f"skipped if absent)")
    common.add_argument("--tree", type=lambda s: Path(s).expanduser(),
                        default=Path(DEFAULT_TREE),
                        help=f"symlink tree to maintain (default {DEFAULT_TREE})")
    common.add_argument("--registry", type=lambda s: Path(s).expanduser(),
                        default=Path(DEFAULT_REGISTRY),
                        help=f"registry YAML (default {DEFAULT_REGISTRY})")

    p = argparse.ArgumentParser(
        prog="model-scanner",
        description="Maintain a classified inventory of the shared HF hub cache "
                    "(+ optional local models dir) and a ComfyUI-friendly symlink "
                    "tree over it. Runs at container start; near-instant when "
                    "nothing changed.")
    sub = p.add_subparsers(dest="cmd")

    s = sub.add_parser("sync", parents=[common],
                       help="scan, classify the delta, and reconcile the link tree "
                            "(default verb)")
    s.add_argument("-n", "--dry-run", action="store_true",
                   help="report what would change without touching anything")
    s.add_argument("--no-prune", dest="prune", action="store_false", default=True,
                   help="do not remove owned links whose source vanished")
    s.add_argument("--hardlink", action="store_true",
                   help="hardlink blobs instead of symlinking")
    s.set_defaults(fn=cmd_sync)

    st = sub.add_parser("status", parents=[common],
                        help="report registry vs tree vs sources (read-only)")
    st.set_defaults(fn=cmd_status)
    return p


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if not argv or argv[0].startswith("-") and argv[0] not in ("-h", "--help"):
        argv.insert(0, "sync")  # sync is the default verb
    args = build_parser().parse_args(argv)
    if not getattr(args, "fn", None):
        build_parser().print_help()
        return 2
    return args.fn(args)


if __name__ == "__main__":
    sys.exit(main())
