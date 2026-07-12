#!/usr/bin/env python3
"""Tests for model_scanner.py using a fully synthetic fixture:

- a fake HF hub cache (models--org--repo/snapshots/<rev>/... symlinking into blobs/<sha>)
  populated with tiny REAL-format files (valid safetensors headers + tiny tensors,
  minimal GGUF with magic + general.architecture, config.json repos, model_index.json
  diffusers repos, a CTranslate2 layout);
- a fake /opt/models local dir.

Exercised: fresh sync, incremental no-op (asserting NO header/metadata reads on the
second run via monkeypatched readers), user-file never-clobber, prune on blob removal,
--no-prune ownership carry-forward, dry-run, status, name collisions, and the v2
behaviors: classify-everything inventory (llm / gguf-llm / ctranslate2 / gguf-split),
repo-level diffusers units with member lists, sharded classified-not-linked, the
generic-filename provenance rule, and UNCLASSIFIED restricted to true unknowns --
including a reproduction of the Raiju field-test scenario.

Run:  python3 targets/comfyui/scripts/tests/test_model_scanner.py -v
"""

import contextlib
import hashlib
import io
import json
import os
import shutil
import struct
import sys
import unittest
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import model_scanner as ms  # noqa: E402

import yaml  # noqa: E402


# ------------------------------------------------------------------ fixture builders

def safetensors_bytes(tensor_names, metadata=None) -> bytes:
    header, off = {}, 0
    for n in tensor_names:
        header[n] = {"dtype": "F32", "shape": [1], "data_offsets": [off, off + 4]}
        off += 4
    if metadata:
        header["__metadata__"] = metadata
    hj = json.dumps(header).encode()
    return len(hj).to_bytes(8, "little") + hj + b"\x00" * off


def gguf_bytes(architecture=None, extra_kv=None) -> bytes:
    """Minimal valid GGUF: magic, v3, 0 tensors, metadata kv (string type only)."""
    def s(x: str) -> bytes:
        b = x.encode()
        return len(b).to_bytes(8, "little") + b
    kvs = []
    if architecture is not None:
        kvs.append(("general.architecture", architecture))
    for k, v in (extra_kv or {}).items():
        kvs.append((k, v))
    out = b"GGUF" + struct.pack("<I", 3) + struct.pack("<QQ", 0, len(kvs))
    for k, v in kvs:
        out += s(k) + struct.pack("<I", 8) + s(v)
    return out


class Fixture:
    """Builds a synthetic HF cache + local models dir + data dir in a tmp root."""

    REV = "0123456789abcdef0123456789abcdef01234567"

    def __init__(self, root: Path):
        self.root = root
        self.cache = root / "hub"
        self.models = root / "opt-models"
        self.tree = root / "data" / "model-tree"
        self.registry = root / "data" / "model-registry.yaml"
        self.cache.mkdir(parents=True)

    def add_hf_file(self, repo: str, relpath: str, content: bytes) -> Path:
        """Standard hub layout: blobs/<sha256> + snapshots/<rev>/<relpath> symlink."""
        repo_dir = self.cache / ("models--" + repo.replace("/", "--"))
        sha = hashlib.sha256(content).hexdigest()
        blob = repo_dir / "blobs" / sha
        blob.parent.mkdir(parents=True, exist_ok=True)
        blob.write_bytes(content)
        (repo_dir / "refs").mkdir(exist_ok=True)
        (repo_dir / "refs" / "main").write_text(self.REV)
        link = repo_dir / "snapshots" / self.REV / relpath
        link.parent.mkdir(parents=True, exist_ok=True)
        link.symlink_to(os.path.relpath(blob, link.parent))
        return blob

    def add_hf_aux(self, repo: str, relpath: str, text: str) -> Path:
        """Non-weight sidecar (config.json, model_index.json, vocabulary.txt, ...)."""
        f = (self.cache / ("models--" + repo.replace("/", "--"))
             / "snapshots" / self.REV / relpath)
        f.parent.mkdir(parents=True, exist_ok=True)
        f.write_text(text)
        return f

    def add_local_file(self, relpath: str, content: bytes) -> Path:
        f = self.models / relpath
        f.parent.mkdir(parents=True, exist_ok=True)
        f.write_bytes(content)
        return f

    def args(self, *extra: str) -> list:
        return ["--cache-dir", str(self.cache), "--models-dir", str(self.models),
                "--tree", str(self.tree), "--registry", str(self.registry),
                *extra]

    def sync(self, *extra: str) -> tuple[int, str]:
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            rc = ms.main(["sync", *self.args(*extra)])
        return rc, buf.getvalue()

    def status(self) -> tuple[int, str]:
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            rc = ms.main(["status", *self.args()])
        return rc, buf.getvalue()

    def load_registry(self) -> dict:
        return yaml.safe_load(self.registry.read_text())


def populate_standard(fx: Fixture) -> None:
    """The standard menagerie used by most tests (one file per ladder step)."""
    # step 1: path segment (Comfy-Org split_files repack)
    fx.add_hf_file("Comfy-Org/flux-repack", "split_files/vae/ae.safetensors",
                   safetensors_bytes(["blah.weight"]))
    # step 2: filename keyword
    fx.add_hf_file("acme/style-pack", "pixel-style-lora.safetensors",
                   safetensors_bytes(["whatever.weight"]))
    # step 4: safetensors header (neutral name/path)
    fx.add_hf_file("acme/flux-dev", "flux1-dev.safetensors",
                   safetensors_bytes(["double_blocks.0.img_attn.qkv.weight",
                                      "single_blocks.1.linear1.weight"]))
    fx.add_hf_file("acme/full-sd", "sd15-full.safetensors",
                   safetensors_bytes(["model.diffusion_model.input_blocks.0.0.weight",
                                      "first_stage_model.decoder.conv_in.weight",
                                      "cond_stage_model.transformer.x.weight"]))
    fx.add_hf_file("acme/vision", "model2.safetensors",
                   safetensors_bytes(["vision_model.encoder.layers.0.mlp.fc1.weight"]))
    # step 5: sibling config.json -> HF LLM repo -> inventory category `llm`
    fx.add_hf_file("meta/tiny-llm", "pytorch_model.bin", b"\x80\x02junkjunk")
    fx.add_hf_aux("meta/tiny-llm", "config.json",
                  json.dumps({"architectures": ["LlamaForCausalLM"]}))
    # step 6: GGUF metadata (neutral names so earlier steps pass through)
    fx.add_hf_file("city96/mystery-diffusion", "mystery-model-q4.gguf",
                   gguf_bytes("flux"))
    fx.add_hf_file("bartowski/assistant", "assistant-8b-q4.gguf",
                   gguf_bytes("llama"))
    # unclassifiable safetensors: no signal anywhere -> a TRUE unknown
    fx.add_hf_file("acme/enigma", "mystery.safetensors",
                   safetensors_bytes(["foo.bar"]))
    # local /opt/models: segment-classified + filename-classified (.pth: no header)
    fx.add_local_file("loras/local-thing.safetensors",
                      safetensors_bytes(["anything.weight"]))
    fx.add_local_file("misc/esrgan-4x.pth", b"\x00" * 64)


EXPECTED_LINKS = {
    "vae/ae.safetensors",
    "loras/pixel-style-lora.safetensors",
    "diffusion_models/flux1-dev.safetensors",
    "checkpoints/sd15-full.safetensors",
    "clip_vision/model2.safetensors",
    "diffusion_models/mystery-model-q4.gguf",
    "loras/local-thing.safetensors",
    "upscale_models/esrgan-4x.pth",
}
# v2: only genuinely unknown files stay unclassified
EXPECTED_UNCLASSIFIED_DISPLAYS = {"acme/enigma/mystery.safetensors"}
# v2: identifiable-but-not-comfyui files get real inventory categories, links:[]
EXPECTED_INVENTORY = {
    "meta/tiny-llm/pytorch_model.bin": "llm",
    "bartowski/assistant/assistant-8b-q4.gguf": "gguf-llm",
}


def populate_raiju(fx: Fixture) -> dict:
    """Reproduce the Raiju field-test cache: a diffusers-format FLUX.1-Fill-dev repo
    (generic component filenames) + a CTranslate2 faster-whisper model.bin."""
    flux = "black-forest-labs/FLUX.1-Fill-dev"
    vae_blob = fx.add_hf_file(flux, "vae/diffusion_pytorch_model.safetensors",
                              safetensors_bytes(["decoder.conv_in.weight",
                                                 "encoder.conv_in.weight"]))
    te_blob = fx.add_hf_file(flux, "text_encoder/model.safetensors",
                             safetensors_bytes(["text_model.encoder.layers.0.q.weight"]))
    fx.add_hf_aux(flux, "model_index.json",
                  json.dumps({"_class_name": "FluxFillPipeline"}))
    fx.add_hf_aux(flux, "vae/config.json",
                  json.dumps({"_class_name": "AutoencoderKL"}))
    fx.add_hf_aux(flux, "text_encoder/config.json",
                  json.dumps({"architectures": ["CLIPTextModel"]}))

    whisper = "Systran/faster-whisper-large-v2"
    wh_blob = fx.add_hf_file(whisper, "model.bin", b"ct2-binary-not-a-torch-file")
    fx.add_hf_aux(whisper, "config.json",
                  json.dumps({"alignment_heads": [[5, 3]], "lang_ids": [50259]}))
    fx.add_hf_aux(whisper, "vocabulary.txt", "<|token|>\n")
    fx.add_hf_aux(whisper, "tokenizer.json", "{}")
    return {"vae": "hf:" + vae_blob.name, "text_encoder": "hf:" + te_blob.name,
            "whisper": "hf:" + wh_blob.name,
            "unit": f"diffusers:{flux}@{Fixture.REV}"}


def tree_links(tree: Path) -> set:
    if not tree.is_dir():
        return set()
    return {str(p.relative_to(tree)) for p in tree.rglob("*") if not p.is_dir()}


def no_read_patches():
    return (mock.patch.object(ms, "read_safetensors_header",
                              side_effect=AssertionError("header read on no-op")),
            mock.patch.object(ms, "read_gguf_metadata",
                              side_effect=AssertionError("gguf read on no-op")),
            mock.patch.object(ms, "read_config_json",
                              side_effect=AssertionError("config read on no-op")))


class ScannerTest(unittest.TestCase):
    def setUp(self):
        import tempfile
        self._tmp = tempfile.TemporaryDirectory(prefix="scanner-test-")
        self.fx = Fixture(Path(self._tmp.name))
        self.addCleanup(self._tmp.cleanup)

    # ------------------------------------------------------------------ fresh sync
    def test_fresh_sync_builds_expected_tree(self):
        populate_standard(self.fx)
        rc, out = self.fx.sync()
        self.assertEqual(rc, 0)
        self.assertEqual(tree_links(self.fx.tree), EXPECTED_LINKS)
        for rel in EXPECTED_LINKS:
            dst = self.fx.tree / rel
            self.assertTrue(dst.is_symlink(), rel)
            resolved = dst.resolve()
            self.assertTrue(resolved.is_file(), rel)
            # HF links must point at the resolved blob path (realpath)
            if not rel.startswith(("loras/local-", "upscale_models/")):
                self.assertEqual(resolved.parent.name, "blobs", rel)

        reg = self.fx.load_registry()
        self.assertEqual(reg["version"], 2)
        cats = {e["display"]: e["category"] for e in reg["entries"].values()}
        for display, want in EXPECTED_INVENTORY.items():
            self.assertEqual(cats[display], want, display)
        uncls = {d for d, c in cats.items() if c == "unclassified"}
        self.assertEqual(uncls, EXPECTED_UNCLASSIFIED_DISPLAYS)
        for e in reg["entries"].values():  # inventory + unclassified: never linked
            if e["category"] in ("unclassified", "llm", "gguf-llm"):
                self.assertEqual(e["links"], [])
        owned = {l for e in reg["entries"].values() for l in e["links"]}
        self.assertEqual(owned, EXPECTED_LINKS)

    def test_unclassified_report_lists_only_true_unknowns(self):
        populate_standard(self.fx)
        rc, out = self.fx.sync()
        uncls_lines = [l for l in out.splitlines() if l.startswith("UNCLASSIFIED")]
        self.assertEqual(len(uncls_lines), 1, out)
        self.assertIn("acme/enigma/mystery.safetensors", uncls_lines[0])
        # identifiable non-comfyui files are counted as inventory, not unclassified
        self.assertIn("1 unclassified", out)

    # ------------------------------------------------------------- raiju scenario
    def test_raiju_scenario_diffusers_plus_ctranslate2(self):
        ids = populate_raiju(self.fx)
        rc, out = self.fx.sync()
        self.assertEqual(rc, 0)
        # generic component filenames -> provenance-derived link names
        self.assertEqual(tree_links(self.fx.tree), {
            "vae/FLUX.1-Fill-dev--vae.safetensors",
            "text_encoders/FLUX.1-Fill-dev--text_encoder.safetensors",
        })
        reg = self.fx.load_registry()
        ent = reg["entries"]
        # component entries: role-classified, provenance-named links
        self.assertEqual(ent[ids["vae"]]["category"], "vae")
        self.assertEqual(ent[ids["vae"]]["links"],
                         ["vae/FLUX.1-Fill-dev--vae.safetensors"])
        self.assertEqual(ent[ids["text_encoder"]]["category"], "text_encoders")
        # repo-level diffusers unit with explicit member relationship
        unit = ent[ids["unit"]]
        self.assertEqual(unit["category"], "diffusers")
        self.assertEqual(unit["links"], [])
        self.assertEqual(sorted(unit["members"]),
                         sorted([ids["vae"], ids["text_encoder"]]))
        # ctranslate2 model: real classification, inventory-only
        self.assertEqual(ent[ids["whisper"]]["category"], "ctranslate2")
        self.assertEqual(ent[ids["whisper"]]["links"], [])
        # nothing is unclassified in this cache
        self.assertNotIn("UNCLASSIFIED", out)
        self.assertIn("0 unclassified", out)
        self.assertIn("REPO  black-forest-labs/FLUX.1-Fill-dev@", out)
        # steady state: no re-inspection
        p1, p2, p3 = no_read_patches()
        with p1 as h, p2 as g, p3 as c:
            rc, out2 = self.fx.sync()
        self.assertEqual((h.call_count, g.call_count, c.call_count), (0, 0, 0))
        self.assertIn("0 new", out2)

    def test_detector_aux_family_routes_to_conventional_dirs(self):
        """Raiju s26 gap: adetailer/YOLO, SAM, face-parsing, pose, and arch-unknown
        upscaler .pt/.pth files used to fall through to `unclassified`; they must now
        land in the Impact-Pack / ReActor / ControlNet-aux conventional subdirs."""
        # local /opt/models drop-ins (bare .pt/.pth, never content-sniffed)
        self.fx.add_local_file("face_yolov8n.pt", b"\x80\x02yolo-pickle")
        self.fx.add_local_file("person_yolov8m-seg.pt", b"\x80\x02yolo-pickle")
        self.fx.add_local_file("sam_vit_b_01ec64.pth", b"\x00" * 32)
        self.fx.add_local_file("parsing_parsenet.pth", b"\x00" * 32)
        self.fx.add_local_file("body_pose_model.pth", b"\x00" * 32)
        self.fx.add_local_file("4x-ClearRealityV1.pth", b"\x00" * 32)
        # a HF-cache adetailer detector (Bingsu) reaches the same rule
        self.fx.add_hf_file("Bingsu/adetailer", "hand_yolov8s.pt", b"\x80\x02yolo")
        rc, out = self.fx.sync()
        self.assertEqual(rc, 0)
        self.assertEqual(tree_links(self.fx.tree), {
            "ultralytics/bbox/face_yolov8n.pt",
            "ultralytics/bbox/hand_yolov8s.pt",
            "ultralytics/segm/person_yolov8m-seg.pt",
            "sams/sam_vit_b_01ec64.pth",
            "facedetection/parsing_parsenet.pth",
            "controlnet_aux/body_pose_model.pth",
            "upscale_models/4x-ClearRealityV1.pth",
        })
        # every one is a real classification now -- nothing left unclassified
        self.assertNotIn("UNCLASSIFIED", out)
        self.assertIn("0 unclassified", out)
        reg = self.fx.load_registry()
        cats = {e["display"]: e["category"] for e in reg["entries"].values()}
        self.assertEqual(cats["face_yolov8n.pt"], "ultralytics/bbox")
        self.assertEqual(cats["person_yolov8m-seg.pt"], "ultralytics/segm")
        self.assertEqual(cats["sam_vit_b_01ec64.pth"], "sams")
        self.assertEqual(cats["parsing_parsenet.pth"], "facedetection")
        self.assertEqual(cats["body_pose_model.pth"], "controlnet_aux")
        self.assertEqual(cats["4x-ClearRealityV1.pth"], "upscale_models")

    def test_sharded_diffusers_component_classified_not_linked(self):
        repo = "acme/big-pipeline"
        self.fx.add_hf_aux(repo, "model_index.json",
                           json.dumps({"_class_name": "SomePipeline"}))
        b1 = self.fx.add_hf_file(
            repo, "transformer/diffusion_pytorch_model-00001-of-00002.safetensors",
            safetensors_bytes(["part1.weight"]))
        b2 = self.fx.add_hf_file(
            repo, "transformer/diffusion_pytorch_model-00002-of-00002.safetensors",
            safetensors_bytes(["part2.weight"]))
        self.fx.add_hf_aux(repo,
                           "transformer/diffusion_pytorch_model.safetensors.index.json",
                           "{}")
        rc, out = self.fx.sync()
        self.assertEqual(tree_links(self.fx.tree), set())  # never linked
        reg = self.fx.load_registry()
        for blob in (b1, b2):
            e = reg["entries"]["hf:" + blob.name]
            self.assertEqual(e["category"], "diffusion_models")  # role still known
            self.assertTrue(e.get("sharded"))
            self.assertEqual(e["links"], [])
        unit = reg["entries"][f"diffusers:{repo}@{Fixture.REV}"]
        self.assertEqual(sorted(unit["members"]),
                         sorted(["hf:" + b1.name, "hf:" + b2.name]))
        self.assertNotIn("UNCLASSIFIED", out)

    def test_sharded_llm_repo_is_llm_not_linked(self):
        repo = "meta/big-llm"
        self.fx.add_hf_aux(repo, "config.json",
                           json.dumps({"architectures": ["Qwen2ForCausalLM"]}))
        b1 = self.fx.add_hf_file(repo, "model-00001-of-00002.safetensors",
                                 safetensors_bytes(["model.layers.0.q.weight"]))
        b2 = self.fx.add_hf_file(repo, "model-00002-of-00002.safetensors",
                                 safetensors_bytes(["model.layers.9.q.weight"]))
        rc, out = self.fx.sync()
        self.assertEqual(tree_links(self.fx.tree), set())
        reg = self.fx.load_registry()
        for blob in (b1, b2):
            e = reg["entries"]["hf:" + blob.name]
            self.assertEqual(e["category"], "llm")
            self.assertTrue(e.get("sharded"))
            self.assertEqual(e["links"], [])
        self.assertNotIn("UNCLASSIFIED", out)

    def test_split_gguf_parts(self):
        b1 = self.fx.add_hf_file("tng/huge-llm", "huge-00001-of-00002.gguf",
                                 gguf_bytes("llama", {"split.tensors.count": "42"}))
        # part 2 carries no general.architecture -> identified as a split part
        b2 = self.fx.add_hf_file("tng/huge-llm", "huge-00002-of-00002.gguf",
                                 gguf_bytes(None, {"split.no": "2"}))
        rc, out = self.fx.sync()
        self.assertEqual(tree_links(self.fx.tree), set())
        reg = self.fx.load_registry()
        e1 = reg["entries"]["hf:" + b1.name]
        self.assertEqual(e1["category"], "gguf-llm")
        self.assertTrue(e1.get("sharded"))
        e2 = reg["entries"]["hf:" + b2.name]
        self.assertEqual(e2["category"], "gguf-split")
        self.assertTrue(e2.get("sharded"))
        self.assertNotIn("UNCLASSIFIED", out)
        # a sharded DIFFUSION gguf is role-classified but still not linked
        b3 = self.fx.add_hf_file("tng/huge-flux", "flux-00001-of-00002.gguf",
                                 gguf_bytes("flux"))
        rc, out = self.fx.sync()
        e3 = self.fx.load_registry()["entries"]["hf:" + b3.name]
        self.assertEqual(e3["category"], "diffusion_models")
        self.assertTrue(e3.get("sharded"))
        self.assertEqual(e3["links"], [])
        self.assertEqual(tree_links(self.fx.tree), set())

    # ------------------------------------------------------- generic-filename rule
    def test_generic_name_rule(self):
        # generic basename, no subdir: repo-tail--stem
        self.fx.add_hf_file("acme/cool-vae", "model.safetensors",
                            safetensors_bytes(["first_stage_model.decoder.w"]))
        # generic basename, subdir provenance (non-diffusers repo)
        self.fx.add_hf_file("acme/bundle", "vae/diffusion_pytorch_model.safetensors",
                            safetensors_bytes(["decoder.a", "encoder.b"]))
        # non-generic basename keeps its plain name
        self.fx.add_hf_file("acme/named", "nice-vae.safetensors",
                            safetensors_bytes(["x.weight"]))
        # local generic file under a category subdir: provenance from dirs
        self.fx.add_local_file("vae/model.fp16.safetensors",
                               safetensors_bytes(["y.weight"]))
        rc, out = self.fx.sync()
        self.assertEqual(tree_links(self.fx.tree), {
            "vae/cool-vae--model.safetensors",
            "vae/bundle--vae.safetensors",
            "vae/nice-vae.safetensors",
            "vae/vae--model.fp16.safetensors",
        })

    def test_preferred_link_name_unit(self):
        self.assertEqual(ms.preferred_link_name(
            "FLUX.1-Fill-dev", Path("vae/diffusion_pytorch_model.safetensors")),
            "FLUX.1-Fill-dev--vae.safetensors")
        self.assertEqual(ms.preferred_link_name(
            "FLUX.1-Fill-dev", Path("text_encoder/model.safetensors")),
            "FLUX.1-Fill-dev--text_encoder.safetensors")
        self.assertEqual(ms.preferred_link_name("repo", Path("pytorch_model.bin")),
                         "repo--pytorch_model.bin")
        self.assertEqual(ms.preferred_link_name("repo", Path("flux1-dev.safetensors")),
                         "flux1-dev.safetensors")
        # local file at models root: no provenance to derive -> plain name kept
        self.assertEqual(ms.preferred_link_name(None, Path("model.bin")), "model.bin")
        self.assertEqual(ms.preferred_link_name(None, Path("whisper/model.bin")),
                         "whisper--model.bin")

    # ---------------------------------------------------------- incremental no-op
    def test_incremental_noop_reads_no_headers(self):
        populate_standard(self.fx)
        self.fx.sync()
        p1, p2, p3 = no_read_patches()
        with p1 as h, p2 as g, p3 as c:
            rc, out = self.fx.sync()
        self.assertEqual(rc, 0)
        self.assertEqual((h.call_count, g.call_count, c.call_count), (0, 0, 0))
        self.assertIn("0 new", out)
        self.assertIn("0 linked", out)
        self.assertEqual(tree_links(self.fx.tree), EXPECTED_LINKS)

    def test_incremental_classifies_only_delta(self):
        populate_standard(self.fx)
        self.fx.sync()
        self.fx.add_hf_file("acme/new-vae", "shiny.safetensors",
                            safetensors_bytes(["first_stage_model.decoder.w"]))
        real_read = ms.read_safetensors_header
        with mock.patch.object(ms, "read_safetensors_header",
                               side_effect=real_read) as h:
            rc, out = self.fx.sync()
        self.assertEqual(rc, 0)
        self.assertEqual(h.call_count, 1)  # only the new file
        self.assertIn("vae/shiny.safetensors", tree_links(self.fx.tree))

    # ------------------------------------------------------------- never-clobber
    def test_never_clobber_user_file(self):
        populate_standard(self.fx)
        user = self.fx.tree / "loras" / "pixel-style-lora.safetensors"
        user.parent.mkdir(parents=True)
        user.write_bytes(b"user data - do not touch")
        rc, out = self.fx.sync()
        self.assertEqual(rc, 0)
        self.assertFalse(user.is_symlink())
        self.assertEqual(user.read_bytes(), b"user data - do not touch")
        self.assertIn("CONFLICT", out)
        # the conflicted path must NOT be claimed in the registry
        reg = self.fx.load_registry()
        owned = {l for e in reg["entries"].values() for l in e["links"]}
        self.assertNotIn("loras/pixel-style-lora.safetensors", owned)
        # and stays untouched on a repeat run
        rc, out = self.fx.sync()
        self.assertEqual(user.read_bytes(), b"user data - do not touch")
        self.assertIn("CONFLICT", out)

    def test_user_dropins_and_unowned_broken_links_survive_prune(self):
        populate_standard(self.fx)
        self.fx.sync()
        dropin = self.fx.tree / "checkpoints" / "my-manual.safetensors"
        dropin.write_bytes(b"manual model")
        dangling = self.fx.tree / "vae" / "old-broken-link.safetensors"
        dangling.symlink_to(self.fx.root / "nowhere.safetensors")
        rc, out = self.fx.sync()
        self.assertEqual(rc, 0)
        self.assertTrue(dropin.is_file())
        self.assertEqual(dropin.read_bytes(), b"manual model")
        self.assertTrue(dangling.is_symlink())  # reported, never removed
        self.assertIn("broken unowned symlink", out)

    # ------------------------------------------------------------------- pruning
    def test_prune_on_blob_removal(self):
        populate_standard(self.fx)
        self.fx.sync()
        shutil.rmtree(self.fx.cache / "models--acme--flux-dev")
        rc, out = self.fx.sync()
        self.assertEqual(rc, 0)
        self.assertIn("PRUNE", out)
        self.assertNotIn("diffusion_models/flux1-dev.safetensors",
                         tree_links(self.fx.tree))
        reg = self.fx.load_registry()
        self.assertNotIn("acme/flux-dev/flux1-dev.safetensors",
                         {e["display"] for e in reg["entries"].values()})

    def test_prune_on_local_file_removed(self):
        populate_standard(self.fx)
        self.fx.sync()
        (self.fx.models / "loras" / "local-thing.safetensors").unlink()
        rc, out = self.fx.sync()
        self.assertNotIn("loras/local-thing.safetensors", tree_links(self.fx.tree))

    def test_no_prune_keeps_ownership_for_later(self):
        populate_standard(self.fx)
        self.fx.sync()
        shutil.rmtree(self.fx.cache / "models--acme--flux-dev")
        rc, out = self.fx.sync("--no-prune")
        stale = self.fx.tree / "diffusion_models" / "flux1-dev.safetensors"
        self.assertTrue(stale.is_symlink())         # kept (broken) ...
        self.assertFalse(stale.exists())
        reg = self.fx.load_registry()               # ... but still OURS
        owned = {l for e in reg["entries"].values() for l in e["links"]}
        self.assertIn("diffusion_models/flux1-dev.safetensors", owned)
        rc, out = self.fx.sync()                    # default prune cleans it up
        self.assertFalse(stale.is_symlink())

    # --------------------------------------------------------------- inventory
    def test_inventory_skipped_and_cached(self):
        populate_standard(self.fx)
        rc, out = self.fx.sync()
        for rel in tree_links(self.fx.tree):
            self.assertNotIn("mystery.safetensors", rel)
            self.assertNotIn("assistant-8b", rel)
            self.assertNotIn("pytorch_model", rel)
            self.assertNotIn("tiny-llm", rel)
        # cached: second run re-inspects nothing (also covered by the no-op test)
        with mock.patch.object(ms, "read_gguf_metadata",
                               side_effect=AssertionError("re-inspected")) as g:
            self.fx.sync()
        self.assertEqual(g.call_count, 0)

    # -------------------------------------------------------------------- dry-run
    def test_dry_run_changes_nothing(self):
        populate_standard(self.fx)
        rc, out = self.fx.sync("--dry-run")
        self.assertEqual(rc, 0)
        self.assertEqual(tree_links(self.fx.tree), set())
        self.assertFalse(self.fx.registry.exists())
        self.assertIn("DRY-LINKED", out)
        self.assertIn("[dry-run]", out)
        # dry-run prune: set up state, remove a source, dry-run again
        self.fx.sync()
        shutil.rmtree(self.fx.cache / "models--acme--flux-dev")
        rc, out = self.fx.sync("-n")
        self.assertIn("DRY-PRUNE", out)
        self.assertTrue((self.fx.tree / "diffusion_models"
                         / "flux1-dev.safetensors").is_symlink())

    # --------------------------------------------------------------------- status
    def test_status_reports(self):
        populate_standard(self.fx)
        self.fx.sync()
        # user drop-in + a vanished source + a new source
        (self.fx.tree / "checkpoints" / "my-manual.safetensors").write_bytes(b"x")
        shutil.rmtree(self.fx.cache / "models--acme--flux-dev")
        self.fx.add_hf_file("acme/new-lora", "fresh-lora.safetensors",
                            safetensors_bytes(["lora_unet_a.lora_down.weight"]))
        rc, out = self.fx.status()
        self.assertEqual(rc, 0)
        self.assertIn("USER  checkpoints/my-manual.safetensors", out)
        self.assertIn("GONE", out)
        self.assertIn("NEW  acme/new-lora/fresh-lora.safetensors", out)
        self.assertIn("BROKEN", out)          # owned link whose blob vanished
        # UNCLASSIFIED lists only the true unknown, not the llm/gguf-llm inventory
        self.assertIn("UNCLASSIFIED  acme/enigma/mystery.safetensors", out)
        self.assertNotIn("UNCLASSIFIED  meta/tiny-llm", out)
        self.assertNotIn("UNCLASSIFIED  bartowski/assistant", out)
        self.assertIn("model-scanner status:", out)
        # status is read-only
        self.assertTrue((self.fx.tree / "diffusion_models"
                         / "flux1-dev.safetensors").is_symlink())

    def test_status_counts_repo_units_without_noise(self):
        populate_raiju(self.fx)
        self.fx.sync()
        rc, out = self.fx.status()
        self.assertEqual(rc, 0)
        self.assertIn("1 repo units", out)
        self.assertIn("0 new, 0 gone", out)   # unit is stable across runs
        self.assertNotIn("UNCLASSIFIED", out)

    # ------------------------------------------------------------ name collisions
    def test_name_collision_gets_disambiguated(self):
        self.fx.add_hf_file("acme/pack-one", "vae/dup.safetensors",
                            safetensors_bytes(["a.weight"]))
        self.fx.add_hf_file("acme/pack-two", "vae/dup.safetensors",
                            safetensors_bytes(["b.weight"]))  # different content
        rc, out = self.fx.sync()
        links = tree_links(self.fx.tree)
        self.assertEqual(len(links), 2, links)
        self.assertIn("vae/dup.safetensors", links)
        self.assertEqual(len([l for l in links if l.endswith("dup.safetensors")]), 2)

    # ------------------------------------------------------- classification units
    def test_classifier_units(self):
        header_cases = [
            (["lora_unet_down.lora_down.weight"], None, "loras"),
            (["model.diffusion_model.x", "first_stage_model.y"], None, "checkpoints"),
            (["diffusion_model.x"], None, "diffusion_models"),
            (["joint_blocks.0.x"], None, "diffusion_models"),
            (["control_model.x"], None, "controlnet"),
            (["vision_model.x"], None, "clip_vision"),
            (["first_stage_model.decoder.x"], None, "vae"),
            (["decoder.conv_in.weight", "encoder.conv_in.weight"], None, "vae"),
            (["encoder.block.0.layer.0.q.weight", "shared.weight"], None,
             "text_encoders"),  # T5-style must NOT hit the generic vae rule
            (["text_model.encoder.x"], None, "text_encoders"),
            (["foo.bar"], None, None),
            (["x.weight"], {"modelspec.architecture": "flux-1-dev/lora"}, "loras"),
        ]
        for names, metadata, want in header_cases:
            header = {n: {"dtype": "F32", "shape": [1], "data_offsets": [0, 4]}
                      for n in names}
            if metadata:
                header["__metadata__"] = metadata
            self.assertEqual(ms.classify_safetensors_header(header), want, names)

        self.assertEqual(ms.classify_by_filename("umt5_xxl_fp8.safetensors"),
                         "text_encoders")
        self.assertEqual(ms.classify_by_filename("qwen_2.5_vl_7b.safetensors"),
                         "text_encoders")
        self.assertEqual(ms.classify_by_filename("clip_vision_h.safetensors"),
                         "clip_vision")
        self.assertEqual(ms.classify_by_filename("taesd_decoder.pth"), "vae_approx")
        self.assertEqual(ms.classify_by_filename("flux1-canny-dev.safetensors"),
                         "controlnet")
        self.assertEqual(ms.classify_by_filename("plain-model.safetensors"), None)

        # ---- auxiliary detector / estimator families (Impact-Pack / ReActor / cnet-aux)
        # Ultralytics / YOLO: bbox vs segm split
        for bbox_name in ("face_yolov8n.pt", "hand_yolov8s.pt", "person_yolov8m.pt",
                          "yolov11-face.pt", "yolov5n-face.pt", "yolo11n.pt"):
            self.assertEqual(ms.classify_by_filename(bbox_name), "ultralytics/bbox",
                             bbox_name)
        for segm_name in ("person_yolov8m-seg.pt", "deepfashion2_yolov8s-seg.pt",
                          "yolov8n-segm.pt"):
            self.assertEqual(ms.classify_by_filename(segm_name), "ultralytics/segm",
                             segm_name)
        # SAM (Segment Anything)
        for sam_name in ("sam_vit_b_01ec64.pth", "sam_vit_h_4b8939.pth",
                         "sam_vit_l_0b3195.pth", "mobile_sam.pt", "sam2_hiera_large.pt"):
            self.assertEqual(ms.classify_by_filename(sam_name), "sams", sam_name)
        # BiSeNet / ParseNet face-parsing
        self.assertEqual(ms.classify_by_filename("parsing_parsenet.pth"), "facedetection")
        self.assertEqual(ms.classify_by_filename("parsing_bisenet.pth"), "facedetection")
        # OpenPose / DWPose estimators
        self.assertEqual(ms.classify_by_filename("body_pose_model.pth"), "controlnet_aux")
        self.assertEqual(ms.classify_by_filename("hand_pose_model.pth"), "controlnet_aux")
        self.assertEqual(ms.classify_by_filename("dwpose.pth"), "controlnet_aux")
        # ESRGAN-family upscalers, incl. arch-unknown 4x/2x/x4 .pth files
        self.assertEqual(ms.classify_by_filename("4x-ClearRealityV1.pth"),
                         "upscale_models")
        self.assertEqual(ms.classify_by_filename("4x_foolhardy_Remacri.pth"),
                         "upscale_models")
        self.assertEqual(ms.classify_by_filename("RealESRGAN_x4plus.pth"),
                         "upscale_models")
        self.assertEqual(ms.classify_by_filename("2x_APISR_RRDB.pth"), "upscale_models")
        # conservative: a plain checkpoint / size tag is NOT swept into upscale
        self.assertIsNone(ms.classify_by_filename("dreamshaper_8.safetensors"))
        self.assertIsNone(ms.classify_by_filename("sdxl_base_1.0.safetensors"))

        self.assertEqual(ms.classify_gguf({"general.architecture": "wan"}),
                         "diffusion_models")
        self.assertEqual(ms.classify_gguf({"general.architecture": "t5"}),
                         "text_encoders")
        # v2: plain-LLM GGUFs are gguf-llm, not unclassified
        self.assertEqual(ms.classify_gguf({"general.architecture": "qwen2"}),
                         "gguf-llm")
        self.assertEqual(ms.classify_gguf({"general.architecture": "brandnew"}),
                         "unclassified")

        self.assertEqual(ms.classify_config({"architectures": ["T5EncoderModel"]}),
                         "text_encoders")
        self.assertEqual(ms.classify_config({"architectures": ["AutoencoderKL"]}),
                         "vae")
        # v2: HF LLM repos are `llm`, not unclassified
        self.assertEqual(ms.classify_config({"architectures": ["Qwen2ForCausalLM"]}),
                         "llm")
        self.assertIsNone(ms.classify_config({"architectures": ["SomethingNew"]}))

    # --------------------------------------------------------- misc robustness
    def test_missing_models_dir_is_silently_skipped(self):
        # fx.models never created; cache has one file
        self.fx.add_hf_file("acme/solo", "split_files/vae/solo.safetensors",
                            safetensors_bytes(["x.weight"]))
        rc, out = self.fx.sync()
        self.assertEqual(rc, 0)
        self.assertEqual(tree_links(self.fx.tree), {"vae/solo.safetensors"})

    def test_registry_lost_means_orphans_not_clobbers(self):
        populate_standard(self.fx)
        self.fx.sync()
        self.fx.registry.unlink()  # simulate registry loss
        shutil.rmtree(self.fx.cache / "models--acme--flux-dev")
        rc, out = self.fx.sync()
        # the now-sourceless link is ORPHANED (treated as user's), never removed
        self.assertTrue((self.fx.tree / "diffusion_models"
                         / "flux1-dev.safetensors").is_symlink())
        self.assertIn("broken unowned symlink", out)
        # still-valid links are re-adopted without clobbering (they match: "ok")
        self.assertIn("vae/ae.safetensors", tree_links(self.fx.tree))

    def test_v1_registry_migrates_via_heuristics_bump(self):
        ids = populate_raiju(self.fx)
        # simulate the v1 state: old heuristics + the old generic-named link on disk
        blob = (self.fx.cache / "models--black-forest-labs--FLUX.1-Fill-dev"
                / "blobs" / ids["text_encoder"][3:])
        old_link = self.fx.tree / "text_encoders" / "model.safetensors"
        old_link.parent.mkdir(parents=True)
        old_link.symlink_to(blob)
        self.fx.registry.parent.mkdir(parents=True, exist_ok=True)
        self.fx.registry.write_text(yaml.safe_dump({
            "version": 1, "heuristics": 1,
            "entries": {ids["text_encoder"]: {
                "origin": "hf", "category": "text_encoders",
                "source": str(blob),
                "display": ("black-forest-labs/FLUX.1-Fill-dev/"
                            "text_encoder/model.safetensors"),
                "links": ["text_encoders/model.safetensors"],
            }}}))
        rc, out = self.fx.sync()
        self.assertEqual(rc, 0)
        self.assertIn("reclassifying everything", out)
        links = tree_links(self.fx.tree)
        # old generic-named link (ours) was pruned; provenance name took over
        self.assertNotIn("text_encoders/model.safetensors", links)
        self.assertIn("text_encoders/FLUX.1-Fill-dev--text_encoder.safetensors", links)

    def test_default_verb_is_sync_and_help_works(self):
        populate_standard(self.fx)
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            rc = ms.main(self.fx.args())  # no verb at all
        self.assertEqual(rc, 0)
        self.assertEqual(tree_links(self.fx.tree), EXPECTED_LINKS)
        with self.assertRaises(SystemExit) as cm:
            with contextlib.redirect_stdout(io.StringIO()):
                ms.main(["--help"])
        self.assertEqual(cm.exception.code, 0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
