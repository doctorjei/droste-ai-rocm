#!/usr/bin/env python3
"""droste-hf-adopt: adopt already-downloaded model files into the HF hub cache.

WHY: droste boxes share a single HuggingFace hub cache; anything in it is
cached content for every container (and for hf/huggingface_hub generally).
Model files that arrived some other way -- browser download, wget, rsync
from another machine -- sit OUTSIDE that cache, so each consumer either
re-downloads multi-GB weights or needs an ad-hoc bind. This tool moves such
files INTO the hub cache, but only when they are PROVABLY hub content:
the file's hash must match what the HF API says the repo contains at the
current (or requested) revision. Everything else is refused with a reason;
its home is the /opt/models bind, not the cache.

HOW: for --repo <org>/<name> we fetch the repo manifest from the HF API
(?blobs=true -- metadata only, never file content). LFS siblings carry
their content sha256 (lfs.oid); small non-LFS siblings carry their git
blob sha1 (blobId). Each local candidate is hashed once (streamed, both
digests in one pass) and matched. A match is adopted exactly the way
`hf download` would have stored it:

Without --repo each file goes through an IDENTIFY phase first: candidate
repos are PROPOSED by several signals, tried in priority order -- a
sibling transformers config.json carrying an absolute org/name
_name_or_path, a curated ecosystem map of well-known aux-model filenames
(ECOSYSTEM_MAP: ComfyUI text/vision encoders, detectors, ControlNets,
annotators), any provenance embedded in a GGUF header, and finally
search terms derived from the filename fed to the HF model-search API.
Every candidate faces the same gate: the repo's manifest must publish
the hash of the local file's actual bytes. Identification is therefore
exactly as strict as --repo -- a repo is only ever accepted on hash
proof, never on name similarity or hint say-so, so a stale or wrong
hint can only fail to confirm and fall through. Files in one directory
may identify to different repos; adoption then runs per identified
repo. Unidentifiable files are refused.

    <cache>/models--<org>--<name>/
        blobs/<digest>                      content-addressed blob
        snapshots/<commit>/<repo-path>  ->  relative symlink into blobs/
        refs/<branch>                       commit sha

Blobs are staged as blobs/.tmp-* and renamed into place, so there is never
partial state; existing blobs are never overwritten; user files are never
deleted except under --move, and only after successful placement.

DEFAULT IS DRY-RUN: nothing is touched until you pass --apply.
"""

import argparse
import errno
import hashlib
import json
import os
import re
import shutil
import sys
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

CHUNK = 1 << 20  # 1 MiB read chunks for streamed hashing
API_BASE = "https://huggingface.co/api/models"
FIXTURE_ENV = "DROSTE_ADOPT_API_FIXTURE"  # test hook; deliberately not in --help
SEARCH_LIMIT = 20  # candidate repos returned per search query
MAX_SEARCHES = 3  # search attempts per file before giving up (stay polite)
HASH_PROGRESS_MIN = 256 << 20  # smaller files hash too fast to show progress
HASH_PROGRESS_STEP = 128 << 20  # progress update cadence while hashing

EXAMPLES = """\
examples:
  # Adopt a GGUF you fetched with wget into the shared cache (dry-run first):
  droste-hf-adopt --repo Qwen/Qwen2.5-0.5B-Instruct-GGUF \\
      ~/Downloads/qwen2.5-0.5b-instruct-q4_k_m.gguf
  droste-hf-adopt --apply --repo Qwen/Qwen2.5-0.5B-Instruct-GGUF \\
      ~/Downloads/qwen2.5-0.5b-instruct-q4_k_m.gguf

  # Don't know (or trust) the repo? Omit --repo: each file is IDENTIFIED
  # (sibling config.json, ecosystem map, GGUF provenance, HF search) and
  # adopted only if a repo's published hashes prove it contains this
  # exact content:
  droste-hf-adopt ~/Downloads/qwen2.5-0.5b-instruct-q4_k_m.gguf

  # Adopt a whole diffusers checkout (nested dirs need --recursive),
  # reclaiming the disk space afterwards:
  droste-hf-adopt --apply --move --recursive \\
      --repo stabilityai/stable-diffusion-xl-base-1.0 ~/sdxl-download/

Only files byte-identical to the repo's content are adopted; everything
else is refused (its home is the /opt/models bind instead). Small repo
files you don't have locally are reported as GAP -- a later
`hf download <repo>` fills those cheaply without re-downloading weights.
"""


def log(args, level, msg):
    """level 0 = always, 1 = normal, 2 = verbose-only."""
    verbosity = 1 + args.verbose - args.quiet
    if level <= verbosity:
        progress_clear()  # never print over a transient status line
        print(msg)


def die(msg, code=2):
    progress_clear()
    print(f"droste-hf-adopt: error: {msg}", file=sys.stderr)
    sys.exit(code)


_progress = 0  # width of the transient stderr status line; 0 = none showing


def progress(args, msg):
    """Transient in-place status line (long hashes, API waits) -- shown
    only on an interactive stderr and never under -q, so pipes, logs and
    cron output are untouched. Overwritten via \\r; progress_clear()
    wipes it before any real output can collide with it."""
    global _progress
    if args.quiet or not sys.stderr.isatty():
        return
    pad = max(_progress - len(msg), 0)
    sys.stderr.write("\r" + msg + " " * pad)
    sys.stderr.flush()
    _progress = len(msg)


def progress_clear():
    global _progress
    if _progress:
        sys.stderr.write("\r" + " " * _progress + "\r")
        sys.stderr.flush()
        _progress = 0


# ---------------------------------------------------------------- cache/auth

def resolve_cache(override):
    """Standard HF cache resolution: flag > HF_HUB_CACHE > HF_HOME/hub >
    ~/.cache/huggingface/hub."""
    if override:
        return Path(override).expanduser()
    if os.environ.get("HF_HUB_CACHE"):
        return Path(os.environ["HF_HUB_CACHE"]).expanduser()
    if os.environ.get("HF_HOME"):
        return Path(os.environ["HF_HOME"]).expanduser() / "hub"
    return Path.home() / ".cache" / "huggingface" / "hub"


def find_token(cache):
    """HF_TOKEN env var, else the token file next to the cache dir
    (~/.cache/huggingface/token in the default layout)."""
    tok = os.environ.get("HF_TOKEN", "").strip()
    if tok:
        return tok
    tok_file = cache.parent / "token"
    try:
        return tok_file.read_text().strip() or None
    except OSError:
        return None


# ------------------------------------------------------------------- HF API

def fetch_repo_info(args, repo, revision, token, fatal=True):
    """GET /api/models/<repo>/revision/<rev>?blobs=true -- metadata only.

    Returns the parsed JSON dict (keys of interest: sha, siblings[]).
    With fatal=False (identify-phase candidates) any failure returns None
    instead of exiting -- the sweep just moves to the next candidate.
    With DROSTE_ADOPT_API_FIXTURE=<dir> set, reads <dir>/<org>__<name>.json
    instead of the network (test hook).
    """
    fixture_dir = os.environ.get(FIXTURE_ENV)
    if fixture_dir:
        org, name = repo.split("/", 1)
        path = Path(fixture_dir) / f"{org}__{name}.json"
        log(args, 2, f"# fixture mode: reading {path} (revision ignored)")
        try:
            return json.loads(path.read_text())
        except OSError as e:
            if fatal:
                die(f"fixture read failed: {e}")
            log(args, 2, f"# fixture manifest missing for {repo}; skipping")
            return None

    quoted_rev = urllib.parse.quote(revision, safe="")
    url = f"{API_BASE}/{repo}/revision/{quoted_rev}?blobs=true"
    req = urllib.request.Request(url, headers={
        "User-Agent": "droste-hf-adopt/0.1",
        **({"Authorization": f"Bearer {token}"} if token else {}),
    })
    log(args, 2, f"# GET {url} (auth: {'token' if token else 'anonymous'})")
    try:
        with urllib.request.urlopen(req, timeout=60) as resp:
            return json.loads(resp.read())
    except urllib.error.HTTPError as e:
        if not fatal:
            log(args, 2, f"# HTTP {e.code} for {repo}@{revision}; "
                         f"skipping candidate")
            return None
        if e.code in (401, 403):
            die(f"HTTP {e.code} for {repo}@{revision}: gated or private "
                "repo -- set HF_TOKEN (or log in with `hf auth login`) "
                "and accept the license on huggingface.co if required")
        if e.code == 404:
            die(f"repo or revision not found: {repo}@{revision}")
        die(f"HF API returned HTTP {e.code} for {repo}@{revision}")
    except urllib.error.URLError as e:
        if not fatal:
            log(args, 2, f"# cannot reach HF API for {repo} ({e.reason}); "
                         f"skipping candidate")
            return None
        die(f"cannot reach HF API ({e.reason}); nothing was changed")
    except TimeoutError:
        if not fatal:
            log(args, 2, f"# HF API timed out for {repo}; skipping candidate")
            return None
        die("HF API request timed out; nothing was changed")


def safe_rfilename(rfilename):
    """Validate a repo-relative path from the API before using it on disk.

    Rejects absolute paths, '..' traversal, empty components, backslashes
    and other Windows-hostile characters. Returns a relative POSIX path or
    None if unsafe.
    """
    if not rfilename or not isinstance(rfilename, str):
        return None
    if rfilename.startswith("/") or "\\" in rfilename or ":" in rfilename:
        return None
    parts = rfilename.split("/")
    if any(p in ("", ".", "..") for p in parts):
        return None
    if any(ch in p for p in parts for ch in '<>"|?*\0'):
        return None
    return rfilename


def is_hex(s, n):
    return isinstance(s, str) and len(s) == n and \
        all(c in "0123456789abcdef" for c in s.lower())


def index_siblings(args, info, repo):
    """Build hash->sibling lookup tables from the API manifest.

    Returns (commit, lfs_by_sha256, small_by_sha1, sizes, all_sizes_known,
    skipped_unsafe).
    """
    commit = info.get("sha")
    if not is_hex(commit, 40):
        die(f"API response for {repo} has no valid commit sha")
    lfs_by_sha256 = {}
    small_by_sha1 = {}
    sizes = set()
    all_sizes_known = True
    skipped_unsafe = 0
    for sib in info.get("siblings", []):
        rfn = safe_rfilename(sib.get("rfilename"))
        if rfn is None:
            skipped_unsafe += 1
            log(args, 2, f"# skipping unsafe repo path from API: "
                         f"{sib.get('rfilename')!r}")
            continue
        lfs = sib.get("lfs") or {}
        # The live API spells the LFS content hash "sha256"; older
        # payloads/docs say "oid". Accept both.
        oid = lfs.get("oid") or lfs.get("sha256")
        size = lfs.get("size", sib.get("size"))
        if size is None:
            all_sizes_known = False
        else:
            sizes.add(size)
        if is_hex(oid, 64):
            lfs_by_sha256[oid.lower()] = (rfn, size)
        elif not lfs:
            # blobId of an LFS sibling is the sha1 of its POINTER file,
            # not the content -- only index blobId for non-LFS siblings
            # (and never for an LFS sibling whose content hash we could
            # not read: adopting a 134-byte pointer as a weight file
            # must be impossible).
            blob_id = sib.get("blobId")
            if is_hex(blob_id, 40):
                small_by_sha1[blob_id.lower()] = (rfn, size)
    return (commit, lfs_by_sha256, small_by_sha1, sizes, all_sizes_known,
            skipped_unsafe)


# ------------------------------------------------------------------ hashing

def hash_file(path, size, args=None):
    """One streamed pass computing BOTH digests we may need:
    content sha256 (matches lfs.oid) and git blob sha1
    ('blob <size>\\0' + content, matches blobId). With args, large files
    show a transient progress line while hashing (interactive only)."""
    h256 = hashlib.sha256()
    h1 = hashlib.sha1()
    h1.update(b"blob %d\x00" % size)
    show = args is not None and size >= HASH_PROGRESS_MIN
    done = next_tick = 0
    with open(path, "rb") as f:
        while chunk := f.read(CHUNK):
            h256.update(chunk)
            h1.update(chunk)
            if show:
                done += len(chunk)
                if done >= next_tick or done == size:
                    gib = 1 << 30
                    progress(args, f"  hashing {Path(path).name}: "
                                   f"{done * 100 // size}% "
                                   f"({done / gib:.1f}/{size / gib:.1f} GiB)")
                    next_tick = done + HASH_PROGRESS_STEP
    if show:
        progress_clear()
    return h256.hexdigest(), h1.hexdigest()


def memo_hash(path, size, memo, args=None):
    """hash_file, cached per run: the identify phase may consult many
    candidate repos and adoption runs after it -- each file is hashed
    exactly ONCE."""
    key = str(path)
    if key not in memo:
        memo[key] = hash_file(path, size, args)
    return memo[key]


# ----------------------------------------------------------- identification

# Quant/dtype/format tokens: useful for filename matching, but repo names
# usually carry only the model name -- broadened searches strip these.
QUANT_RE = re.compile(
    r"(?i)^(i?q\d[a-z0-9_]*|fp\d+|bf16|f16|f32|f64|int[48]|[48]bit|e\dm\d|"
    r"gguf|ggml|safetensors|gptq|awq|exl2|imatrix)$")
GGUF_HINT_KEYS = ("general.source.huggingface.repository",
                  "general.source.url", "general.name", "general.basename")


def valid_repo_id(s):
    """org/name shape check, shared by --repo and identify candidates."""
    s = (s or "").strip("/")
    return s if s.count("/") == 1 and all(s.split("/")) else None


# A sibling config's _name_or_path only counts as a repo hint when it is
# an ABSOLUTE org/name id. transformers writes relative paths like
# "./image_encoder" (or a bare local dir) for local checkouts -- those
# identify nothing and are ignored.
CONFIG_NAME_OR_PATH_RE = re.compile(r"^[A-Za-z0-9._-]+/[A-Za-z0-9._-]+$")
CONFIG_SIDECAR_MAX = 1 << 20  # a transformers config.json is a few KiB


def config_repo_hints(f):
    """Candidate signal 1: a sibling transformers config.json.

    For a model file X.ext, real-world trees spell the sidecar several
    ways: 'X..json' (double-dot quirk some managers write), 'X.ext.json',
    'X.json', and the standard transformers layout's plain 'config.json'
    next to the weights. A sidecar contributes a candidate only if it
    parses as JSON with an absolute org/name `_name_or_path`; relative
    paths ('./image_encoder') are ignored. Returns a deduped list of
    (repo_id, sidecar_name) -- proposals only, the hash gate still rules.
    """
    stem = f.name[:f.name.rfind(".")] if "." in f.name else f.name
    sidecars = (f.with_name(stem + "..json"),   # X..json (double-dot)
                f.with_name(f.name + ".json"),  # X.<ext>.json
                f.with_name(stem + ".json"),    # X.json
                f.with_name("config.json"))     # transformers layout
    out, seen = [], set()
    for sc in sidecars:
        if sc == f or not sc.is_file():
            continue
        try:
            if sc.stat().st_size > CONFIG_SIDECAR_MAX:
                continue
            data = json.loads(sc.read_text())
        except (OSError, ValueError):
            continue
        name = data.get("_name_or_path") if isinstance(data, dict) else None
        if not isinstance(name, str) \
                or not CONFIG_NAME_OR_PATH_RE.match(name) \
                or any(p in (".", "..") for p in name.split("/")):
            continue  # relative ('./image_encoder') or malformed
        if name not in seen:
            seen.add(name)
            out.append((name, sc.name))
    return out


# Candidate signal 2: curated ecosystem map -- filename fingerprint ->
# candidate HF repo(s), covering ubiquitous ComfyUI aux models that a
# name-search cannot find (renamed/loose/component files). Entries are
# ONLY proposals: adoption still requires the file's computed sha256 (or
# git sha1) in the candidate's manifest, so a wrong or stale entry can
# never place a file -- it just fails to confirm and falls through.
# Extend freely; keep the provenance comment per entry.
ECOSYSTEM_MAP = (
    # (key, filename regex, candidate repos tried in order)

    # Flux/SD3 text encoders: ComfyUI docs point clip_l / t5xxl_* here.
    ("flux-text-encoders",
     re.compile(r"(?i)^(clip_l|t5xxl)([._-]|$)"),
     ("comfyanonymous/flux_text_encoders",)),
    # IP-Adapter's CLIP-ViT-bigG/14 image encoder (SDXL flavour), lives
    # under sdxl_models/image_encoder. Checked before ViT-H: bigG names
    # can also carry H-ish tokens.
    ("clip-vision-bigG",
     re.compile(r"(?i)clip[-_.]?vi(t|sion).*(big[-_]?g|[-_]g([-_.]|$))"),
     ("h94/IP-Adapter",)),
    # IP-Adapter's CLIP-ViT-H/14 image encoder (LAION-2B;
    # CLIPVisionModelWithProjection, hidden 1280), lives under
    # models/image_encoder.
    ("clip-vision-H",
     re.compile(r"(?i)clip[-_.]?vi(t|sion).*[-_]h([-_.\d]|$)"),
     ("h94/IP-Adapter",)),
    # adetailer's face/hand/person YOLOv8/9/11 detectors (bbox + seg).
    ("adetailer-yolo",
     re.compile(r"(?i)(face|hand|person)[-_]?yolov?(8|9|11)"),
     ("Bingsu/adetailer",)),
    # Ultralytics base YOLO11 weights (yolo11n/s/m/l/x, -seg/-pose/...).
    ("ultralytics-yolo11",
     re.compile(r"(?i)^yolo11"),
     ("Ultralytics/YOLO11",)),
    # ControlNet v1.1 for SD1.5 (control_v11p_sd15_canny.pth etc.,
    # including the f1/e/s2 variants).
    ("controlnet-v1-1",
     re.compile(r"(?i)^control_v11[a-z0-9]*_sd15"),
     ("lllyasviel/ControlNet-v1-1",)),
    # ControlNet-aux annotators. lllyasviel/Annotators aggregates most of
    # them; specific upstreams added where known.
    ("depth-anything",
     re.compile(r"(?i)depth[-_]?anything"),
     ("lllyasviel/Annotators", "LiheYoung/depth_anything")),
    ("midas-dpt",
     re.compile(r"(?i)(midas|(^|[._-])dpt[._-])"),
     ("lllyasviel/Annotators", "Intel/dpt-hybrid-midas")),
    ("zoe-depth",
     re.compile(r"(?i)zoed(epth)?([._-]|$)"),  # ZoeD_M12_N.pt et al.
     ("lllyasviel/Annotators",)),
    ("openpose",
     re.compile(r"(?i)(openpose|body_pose|hand_pose|facenet)"),
     ("lllyasviel/Annotators",)),
    ("pidinet",
     re.compile(r"(?i)pidinet"),  # table5_pidinet.pth
     ("lllyasviel/Annotators",)),
    ("lineart",
     re.compile(r"(?i)lineart"),
     ("lllyasviel/Annotators",)),
)


def ecosystem_candidates(name):
    """Candidate signal 2 lookup: ECOSYSTEM_MAP entries matching a
    filename. Returns a deduped list of (repo_id, map_key) in map order
    -- proposals only, the hash gate still rules."""
    out, seen = [], set()
    for key, rx, repos in ECOSYSTEM_MAP:
        if not rx.search(name):
            continue
        for repo in repos:
            if repo not in seen:
                seen.add(repo)
                out.append((repo, key))
    return out


def derive_term_sets(name):
    """Filename -> ladder of HF search queries, most specific first:
    the full stem, then quant/dtype tokens stripped, then the trailing
    token dropped as well."""
    stem = name[:name.rfind(".")] if "." in name else name
    tokens = [t for t in re.split(r"[-_. ]+", stem) if t]
    plain, quant_tail = [], False
    for t in tokens:
        if QUANT_RE.match(t):
            quant_tail = True
            continue
        if quant_tail and re.fullmatch(r"(?i)([a-z]{1,3}|\d)", t):
            continue  # the _K / _M / _XXS pieces of q4_K_M-style suffixes
        quant_tail = False
        plain.append(t)
    sets = [stem]
    if plain and plain != tokens:
        sets.append(" ".join(plain))
    if len(plain) > 1:
        sets.append(" ".join(plain[:-1]))
    out = []
    for s in sets:
        if s not in out:
            out.append(s)
    return out[:MAX_SEARCHES]


def gguf_repo_hint(path):
    """Best-effort GGUF provenance: bounded header parse looking for
    general.source.huggingface.repository / general.name style keys.
    Returns (repo_id_or_None, name_or_None). Purely a hint feeding the
    hash-verified pipeline; any parse trouble degrades silently."""
    found = {}
    try:
        with open(path, "rb") as fh:
            magic = fh.read(4)
            if magic != b"GGUF":
                return None, None
            buf = magic + fh.read((1 << 20) - 4)  # bounded prefix
        version = int.from_bytes(buf[4:8], "little")
        if version < 2:  # v1 used 32-bit counts; not worth supporting
            return None, None
        n_kv = int.from_bytes(buf[16:24], "little")
        pos = 24
        scalar = {0: 1, 1: 1, 2: 2, 3: 2, 4: 4, 5: 4, 6: 4, 7: 1,
                  10: 8, 11: 8, 12: 8}  # gguf value type -> byte width

        def take(n):
            nonlocal pos
            if n < 0 or pos + n > len(buf):
                raise ValueError("out of bounds")
            out = buf[pos:pos + n]
            pos += n
            return out

        def take_str():
            return take(int.from_bytes(take(8), "little")) \
                .decode("utf-8", "replace")

        for _ in range(min(n_kv, 512)):
            key = take_str()
            vtype = int.from_bytes(take(4), "little")
            if vtype == 8:  # string
                val = take_str()
                if key in GGUF_HINT_KEYS:
                    found[key] = val.strip()
            elif vtype == 9:  # array: elem type + count + elems
                etype = int.from_bytes(take(4), "little")
                count = int.from_bytes(take(8), "little")
                if etype == 8:
                    for _ in range(count):
                        take_str()
                else:
                    take(scalar[etype] * count)
            else:
                take(scalar[vtype])
    except Exception:
        pass  # hint only -- keep whatever was parsed before the trouble
    src = found.get("general.source.huggingface.repository") \
        or found.get("general.source.url") or ""
    if src.startswith("http"):
        src = "/".join(urllib.parse.urlparse(src).path.strip("/").split("/")[:2])
    return (valid_repo_id(src),
            found.get("general.name") or found.get("general.basename"))


def search_models(args, query, token, searches):
    """GET /api/models?search=... -- repo summaries ({id, downloads, ...}),
    most-downloaded first. Raises URLError/TimeoutError; the caller
    refuses that file rather than crashing the run.

    Fixture (DROSTE_ADOPT_API_FIXTURE=<dir>): reads <dir>/search.json --
    either a list (returned for every query) or a dict of query -> list
    (with optional "*" default); {"error": msg} simulates a network
    failure.
    """
    if query in searches:
        return searches[query]
    fixture_dir = os.environ.get(FIXTURE_ENV)
    if fixture_dir:
        path = Path(fixture_dir) / "search.json"
        log(args, 2, f"# fixture mode: reading {path} for search {query!r}")
        try:
            data = json.loads(path.read_text())
        except OSError as e:
            die(f"fixture read failed: {e}")
        if isinstance(data, dict):
            if "error" in data:
                raise urllib.error.URLError(data["error"])
            data = data.get(query, data.get("*", []))
        searches[query] = data
        return data
    url = (f"{API_BASE}?search={urllib.parse.quote(query)}"
           f"&limit={SEARCH_LIMIT}&sort=downloads&direction=-1")
    req = urllib.request.Request(url, headers={
        "User-Agent": "droste-hf-adopt/0.1",
        **({"Authorization": f"Bearer {token}"} if token else {}),
    })
    log(args, 2, f"# GET {url}")
    with urllib.request.urlopen(req, timeout=30) as resp:
        data = json.loads(resp.read())
    searches[query] = data
    return data


def get_manifest(args, repo, revision, token, manifests):
    """Fetch+index a candidate repo's manifest, cached per run (identify
    sweeps and multi-file invocations must not re-fetch). Returns
    (info, index_tuple) or None; failures are negative-cached."""
    if repo in manifests:
        return manifests[repo]
    entry = None
    info = fetch_repo_info(args, repo, revision, token, fatal=False)
    if info is not None and is_hex(info.get("sha"), 40):
        entry = (info, index_siblings(args, info, repo))
    elif info is not None:
        log(args, 2, f"# {repo}: manifest has no valid commit sha; skipping")
    manifests[repo] = entry
    return entry


def identify_file(args, f, size, token, manifests, searches, hash_memo):
    """IDENTIFY phase for one file (no --repo): find HF repos whose
    published hashes prove they contain this exact content. Candidate
    repos are proposed by several signals, tried in priority order --
    sibling config `_name_or_path`, the curated ECOSYSTEM_MAP, an
    embedded GGUF provenance hint, then the filename-search ladder --
    all deduped and all facing the same hash gate. Returns
    (repo, note, alternatives) on success, (None, refusal_reason, [])
    otherwise."""

    def hash_matches(idx):
        _commit, lfs_by_sha256, small_by_sha1, sizes, all_known, _ = idx
        if all_known and size not in sizes:
            return False  # cheap pre-filter: no sibling can match
        # sizes incomplete -> fall through to the (memoized) hash compare
        sha256, git_sha1 = memo_hash(f, size, hash_memo, args)
        return sha256 in lfs_by_sha256 or git_sha1 in small_by_sha1

    seen = set()  # every candidate proposed so far (dedup across signals)
    tried = []    # candidates whose manifest we actually hash-checked

    def proves(repo):
        """Fetch one candidate's manifest and run the hash gate."""
        if repo in seen:
            return False  # already failed under a higher-priority signal
        seen.add(repo)
        progress(args, f"  checking {repo}...")
        entry = get_manifest(args, repo, args.revision, token, manifests)
        if entry is None:
            return False  # manifest fetch failed -> next candidate
        tried.append(repo)
        return hash_matches(entry[1])

    # Signal 1: sibling transformers config with an absolute
    # _name_or_path -- the strongest local provenance we can have.
    for repo, src in config_repo_hints(f):
        log(args, 2, f"# {f.name}: config _name_or_path ({src}) -> {repo}")
        if proves(repo):
            return repo, (f"via config:_name_or_path ({src}); "
                          f"{len(tried)} candidate(s) tried"), []

    # Signal 2: curated ecosystem map (filename fingerprint).
    for repo, key in ecosystem_candidates(f.name):
        log(args, 2, f"# {f.name}: ecosystem map [{key}] -> {repo}")
        if proves(repo):
            return repo, (f"via ecosystem-map:{key}; "
                          f"{len(tried)} candidate(s) tried"), []

    # Signal 3: GGUF provenance hint: an embedded source repo, once
    # hash-proven, is definitive -- no search needed. A stale/wrong hint
    # just falls through to the search path.
    hint_repo, hint_name = gguf_repo_hint(f)
    if hint_repo:
        log(args, 2, f"# {f.name}: gguf hint -> {hint_repo}")
        if proves(hint_repo):
            return hint_repo, (f"via gguf hint; "
                               f"{len(tried)} candidate(s) tried"), []

    # Signal 4 (fallback): the filename-search ladder.
    term_sets = derive_term_sets(f.name)
    if hint_name and hint_name not in term_sets:
        term_sets.insert(0, hint_name)  # e.g. gguf general.name
    results = []
    for query in term_sets[:MAX_SEARCHES]:
        log(args, 2, f"# {f.name}: search {query!r}")
        progress(args, f'  identifying {f.name}: searching "{query}"...')
        try:
            results = search_models(args, query, token, searches)
        except (urllib.error.URLError, TimeoutError) as e:
            log(args, 0, f"warning: HF search failed "
                         f"({getattr(e, 'reason', e)})")
            return None, "HF search unavailable; pass --repo explicitly", []
        if results:
            break

    downloads, candidates = {}, []
    for r in results:
        rid = valid_repo_id(r.get("id")) if isinstance(r, dict) else None
        if rid and rid not in downloads:
            downloads[rid] = r.get("downloads") or 0
            candidates.append(rid)
    log(args, 2, f"# {f.name}: candidates: "
                 f"{', '.join(candidates) or '(none)'}")

    matches = []
    for repo in candidates:
        if repo in seen:
            continue  # already hash-checked under a higher-priority signal
        seen.add(repo)
        progress(args, f"  checking {repo}...")
        entry = get_manifest(args, repo, args.revision, token, manifests)
        if entry is None:
            continue  # manifest fetch failed -> next candidate
        tried.append(repo)
        if hash_matches(entry[1]):
            matches.append(repo)
    progress_clear()
    if not matches:
        if tried:
            return None, (f"no candidate repo's manifest contained this "
                          f"file's sha256 (tried: {', '.join(tried)}); "
                          f"pass --repo, or the file may have been "
                          f"re-saved (hash drift)"), []
        return None, ("no candidate repos found (config/ecosystem/search "
                      "all came up empty); pass --repo explicitly or "
                      "place it in /opt/models"), []
    matches.sort(key=lambda r: -downloads[r])  # stable: keeps search order
    return (matches[0], f"via search; {len(tried)} candidate(s) tried",
            matches[1:])


# ---------------------------------------------------------------- placement

def place_blob(args, src, blob_path, mode):
    """Get src's content to blob_path atomically. Returns a short verb
    describing what happened ('hardlinked' / 'copied' / ...).

    Stage into blobs/.tmp-<digest>-<pid>, then rename -- a reader never
    sees a partial blob and a crash leaves only a .tmp-* to sweep."""
    blob_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = blob_path.parent / f".tmp-{blob_path.name}-{os.getpid()}"
    verb = None
    try:
        if mode in ("link", "move"):
            try:
                os.link(src, tmp)
                verb = "hardlinked"
            except OSError as e:
                if e.errno not in (errno.EXDEV, errno.EPERM, errno.EMLINK):
                    raise
                log(args, 1, f"  note: hardlink not possible "
                             f"({errno.errorcode.get(e.errno, e.errno)}); "
                             f"copying instead")
        if verb is None:
            shutil.copyfile(src, tmp)
            verb = "copied"
        os.rename(tmp, blob_path)  # same dir -> atomic
    finally:
        tmp.unlink(missing_ok=True)
    return verb


def ensure_snapshot_link(args, repo_dir, commit, rfn, blob_path, apply_mode):
    """Create snapshots/<commit>/<rfn> as a RELATIVE symlink to the blob
    (../../blobs/<digest>, more ..s for nested paths). Never clobbers a
    path that exists and is not our symlink."""
    link = repo_dir / "snapshots" / commit / rfn
    target = os.path.relpath(blob_path, link.parent)
    if link.is_symlink():
        if os.readlink(link) == target:
            return "kept"
        if not apply_mode:
            return "would-fix"
        link.unlink()
        link.symlink_to(target)
        return "fixed"
    if link.exists():
        log(args, 0, f"  warning: {link} exists and is not a symlink; "
                     f"leaving it alone")
        return "blocked"
    if not apply_mode:
        return "would-create"
    link.parent.mkdir(parents=True, exist_ok=True)
    link.symlink_to(target)
    return "created"


def ensure_ref(args, repo_dir, revision, commit, apply_mode):
    """Write/update refs/<revision> -> commit (skipped when the requested
    revision IS a commit sha). An existing ref with a different sha is
    updated to the API's current sha -- matches `hf download` -- but the
    old snapshot dir is left untouched."""
    if is_hex(revision, 40):
        return
    ref = repo_dir / "refs" / revision
    old = None
    try:
        old = ref.read_text().strip()
    except OSError:
        pass
    if old == commit:
        return
    if old:
        log(args, 1, f"  refs/{revision}: {old[:7]} -> {commit[:7]} "
                     f"(old snapshot kept)")
    if apply_mode:
        ref.parent.mkdir(parents=True, exist_ok=True)
        ref.write_text(commit)


# ----------------------------------------------------------- file gathering

def gather_files(args, paths):
    """Expand CLI paths: files as-is; dirs scanned (non-recursive unless
    --recursive). Skips .cache/ metadata dirs that `hf download
    --local-dir` leaves behind."""
    out = []
    for p in paths:
        p = Path(p).expanduser()
        if p.is_dir():
            if args.recursive:
                for root, dirs, files in os.walk(p):
                    dirs[:] = sorted(d for d in dirs if d != ".cache")
                    out.extend(Path(root) / f for f in sorted(files))
            else:
                out.extend(sorted(c for c in p.iterdir() if c.is_file()))
        elif p.is_file():
            out.append(p)
        else:
            die(f"no such file or directory: {p}")
    files = [f for f in out if f.is_file()]
    if not files:
        die("no candidate files found (empty dir? need --recursive?)")
    return files


# --------------------------------------------------------------------- main

def build_parser():
    p = argparse.ArgumentParser(
        prog="droste-hf-adopt",
        description="Adopt already-downloaded model files into the "
                    "HuggingFace hub cache -- no re-download. Only files "
                    "byte-identical to the repo's published content are "
                    "adopted; everything else is refused. Without --repo "
                    "each file is first IDENTIFIED -- candidate repos come "
                    "from a sibling config.json, a curated ecosystem map, "
                    "GGUF provenance and HF search, accepted only on hash "
                    "proof, never name-guessed. DRY-RUN by default: pass "
                    "--apply to actually change the cache.",
        epilog=EXAMPLES,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--repo", metavar="ORG/NAME",
                   help="HF repo id the files claim to belong to "
                        "(omit to identify the repo per file via local "
                        "hints + HF search, always hash-proven)")
    p.add_argument("paths", nargs="+", metavar="PATH",
                   help="files to adopt, or a directory to scan")
    p.add_argument("--apply", action="store_true",
                   help="actually modify the cache (default: dry-run)")
    p.add_argument("--dry-run", action="store_true",
                   help="report only, change nothing (this is the default)")
    mode = p.add_mutually_exclusive_group()
    mode.add_argument("--link", dest="mode", action="store_const",
                      const="link",
                      help="hardlink into the cache (default; falls back "
                           "to copy across filesystems)")
    mode.add_argument("--copy", dest="mode", action="store_const",
                      const="copy", help="copy into the cache")
    mode.add_argument("--move", dest="mode", action="store_const",
                      const="move",
                      help="move into the cache (source removed only "
                           "after successful placement)")
    p.set_defaults(mode="link")
    p.add_argument("--recursive", action="store_true",
                   help="recurse into directories")
    p.add_argument("--cache", metavar="DIR",
                   help="hub cache dir (default: $HF_HUB_CACHE, "
                        "$HF_HOME/hub, or ~/.cache/huggingface/hub)")
    p.add_argument("--revision", default="main", metavar="REV",
                   help="branch or commit sha to match against "
                        "(default: main)")
    p.add_argument("-q", "--quiet", action="count", default=0,
                   help="only refusals, warnings and the summary")
    p.add_argument("-v", "--verbose", action="count", default=0,
                   help="show hashes, API/search details, per-step "
                        "decisions")
    return p


def adopt_group(args, cache, repo, idx, files, hash_memo):
    """Match/place/ref/GAP flow for one (repo, revision) group -- the
    whole adoption pass that used to be main()'s body. Returns
    (adopted, already, refused)."""
    apply_mode = args.apply
    (commit, lfs_by_sha256, small_by_sha1, sizes, all_sizes_known,
     _skipped_unsafe) = idx

    repo_dir = cache / ("models--" + repo.replace("/", "--"))
    at = f"{repo}@{commit[:7]}"
    log(args, 1, f"repo {at} | {len(lfs_by_sha256)} LFS + "
                 f"{len(small_by_sha1)} small file(s) | mode: {args.mode}")

    adopted = already = refused = 0
    satisfied = set()  # rfilenames whose blob is (or will be) in the cache
    ref_needed = False

    for f in files:
        size = f.stat().st_size
        shown = str(f)
        if all_sizes_known and size not in sizes:
            # Cheap pre-filter: no repo file has this size, so no hash
            # can match. Skip hashing multi-GB non-members.
            log(args, 0, f"REFUSE  {shown}: not byte-identical to any "
                         f"file in {at} (no size match)")
            refused += 1
            continue
        sha256, git_sha1 = memo_hash(f, size, hash_memo, args)
        log(args, 2, f"# {f.name}: sha256={sha256[:12]}.. "
                     f"gitsha1={git_sha1[:12]}.. size={size}")
        hit = lfs_by_sha256.get(sha256)
        digest = sha256
        kind = "lfs sha256"
        if hit is None:
            hit = small_by_sha1.get(git_sha1)
            digest = git_sha1
            kind = "git blobId"
        if hit is None:
            log(args, 0, f"REFUSE  {shown}: not byte-identical to any "
                         f"file in {at}")
            refused += 1
            continue

        rfn, _ = hit
        blob = repo_dir / "blobs" / digest
        ref_needed = True
        satisfied.add(rfn)
        if blob.exists():
            if blob.stat().st_size != size:
                log(args, 0, f"warning: cached blob {digest[:12]}.. has "
                             f"unexpected size; not touching it and not "
                             f"removing the source")
                refused += 1
                continue
            ensure_snapshot_link(args, repo_dir, commit, rfn, blob,
                                 apply_mode)
            log(args, 1, f"ALREADY {shown}")
            log(args, 1, f"        == {rfn} (blob already cached)")
            already += 1
            if apply_mode and args.mode == "move" \
                    and f.resolve() != blob.resolve():
                f.unlink()
                log(args, 1, "        source removed (--move)")
        else:
            if apply_mode:
                verb = place_blob(args, f, blob, args.mode)
                ensure_snapshot_link(args, repo_dir, commit, rfn, blob,
                                     apply_mode)
                if args.mode == "move":
                    f.unlink()
                    verb += ", source removed"
            else:
                verb = {"link": "would hardlink", "copy": "would copy",
                        "move": "would move"}[args.mode]
                ensure_snapshot_link(args, repo_dir, commit, rfn, blob,
                                     apply_mode)
            log(args, 0, f"ADOPT   {shown}")
            log(args, 0, f"        -> {rfn} ({kind} {digest[:12]}..) "
                         f"[{verb}]")
            adopted += 1

    if ref_needed:
        ensure_ref(args, repo_dir, args.revision, commit, apply_mode)

    # GAP report: repo content still missing from the cache. Small files
    # are listed (a later `hf download` fills them cheaply); missing LFS
    # weights are just counted -- listing 20 quant variants is noise.
    if adopted or already:
        gaps_small = [rfn for _sha, (rfn, _sz) in small_by_sha1.items()
                      if rfn not in satisfied
                      and not (repo_dir / "blobs" / _sha).exists()]
        gaps_lfs = sum(1 for _sha, (rfn, _sz) in lfs_by_sha256.items()
                       if rfn not in satisfied
                       and not (repo_dir / "blobs" / _sha).exists())
        for rfn in sorted(gaps_small):
            log(args, 1, f"GAP     {rfn}: not local; "
                         f"`hf download {repo}` fills this cheaply")
        if gaps_lfs:
            log(args, 1, f"GAP     {gaps_lfs} large (LFS) repo file(s) "
                         f"not in cache (that may be fine)")
    return adopted, already, refused


def main(argv=None):
    args = build_parser().parse_args(argv)
    if args.dry_run and args.apply:
        die("--dry-run and --apply are mutually exclusive")
    apply_mode = args.apply

    cache = resolve_cache(args.cache)
    token = find_token(cache)
    files = gather_files(args, args.paths)
    hash_memo = {}  # each file is hashed once, across identify + adopt
    header = (f"{'DRY RUN (pass --apply to make changes)' if not apply_mode else 'APPLY'}"
              f" | cache: {cache}")

    refused = 0
    groups = []  # (repo, index_tuple, [files])
    if args.repo:
        repo = valid_repo_id(args.repo)
        if repo is None:
            die(f"--repo must look like org/name, got {args.repo!r}")
        progress(args, f"  fetching manifest {repo}...")
        info = fetch_repo_info(args, repo, args.revision, token)
        progress_clear()
        idx = index_siblings(args, info, repo)
        if idx[5]:
            log(args, 0, f"warning: ignored {idx[5]} unsafe path(s) "
                         f"in the API manifest")
        log(args, 1, header)
        groups.append((repo, idx, files))
    else:
        # IDENTIFY phase: per file (one directory may hold files from
        # several repos), then adopt per identified repo group.
        log(args, 1, header)
        manifests, searches = {}, {}  # per-run caches, never re-fetched
        by_repo = {}
        for f in files:
            repo, note, alts = identify_file(args, f, f.stat().st_size,
                                             token, manifests, searches,
                                             hash_memo)
            if repo is None:
                log(args, 0, f"REFUSE  {f}: {note}")
                refused += 1
                continue
            commit = manifests[repo][1][0]
            log(args, 1, f"IDENTIFIED {f} -> {repo} @ {commit[:7]} ({note})")
            if alts:
                log(args, 1, f"        also matches: {', '.join(alts)} "
                             f"(picked by downloads)")
            by_repo.setdefault(repo, []).append(f)
        for repo, group_files in by_repo.items():
            idx = manifests[repo][1]
            if idx[5]:
                log(args, 0, f"warning: ignored {idx[5]} unsafe path(s) "
                             f"in the API manifest for {repo}")
            groups.append((repo, idx, group_files))

    adopted = already = 0
    for repo, idx, group_files in groups:
        a, c, r = adopt_group(args, cache, repo, idx, group_files, hash_memo)
        adopted += a
        already += c
        refused += r

    tag = " (dry-run; nothing was changed)" if not apply_mode else ""
    log(args, 0, f"summary: {adopted} adopted, {already} already cached, "
                 f"{refused} refused{tag}")
    if adopted == 0 and already == 0 and refused > 0:
        sys.exit(1)


if __name__ == "__main__":
    main()
