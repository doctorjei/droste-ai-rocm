#!/usr/bin/env python3
"""Tests for droste-civitai-adopt using the DROSTE_CIVITAI_API_FIXTURE
hook -- NO live network. A fixture dir stands in for the CivitAI API:

- by-hash.json        batch by-hash endpoint: {sha256: version, ...} or
                      a plain list of version objects; {"error": msg}
                      simulates a network failure.
- version-<id>.json   GET /model-versions/<id> (missing file = 404).

Covers the taxonomy/sniff rework AND the three-file sidecar scheme
(TWEAK 2): pure .civitai.info, objective .meta.droste, ingested
.user.droste; per-format preference extraction; the `unmatched`
discovery bucket (+ on-screen note) across all three source formats;
monotonic merge; idempotent write-if-changed sync (incl. the ALREADY
branch); the user-data overwrite guard + --force -- plus the retained
behaviors (identity gate, taxonomy dirs, sniff routing, normalization,
never-clobber model file, --version-id, preview carry, network-error
refusal, progress helpers, restricted unpickler).

Run:  python3 tests/test_civitai_adopt.py -v
"""

import contextlib
import errno
import hashlib
import importlib.machinery
import importlib.util
import io
import json
import os
import pickle
import sys
import time
import types
import unittest
import zipfile
from collections import OrderedDict
from pathlib import Path
from unittest import mock

SCRIPT = Path(__file__).resolve().parents[1] / "droste-civitai-adopt"
loader = importlib.machinery.SourceFileLoader("droste_civitai_adopt",
                                              str(SCRIPT))
spec = importlib.util.spec_from_loader("droste_civitai_adopt", loader)
mod = importlib.util.module_from_spec(spec)
loader.exec_module(mod)


# ------------------------------------------------------------------ fixture builders

def sha256(content: bytes) -> str:
    return hashlib.sha256(content).hexdigest()


def safetensors_bytes(tensors, metadata=None) -> bytes:
    """Real safetensors header (8-byte LE length + JSON), no tensor body
    needed for sniffing. `tensors` = list of names or {name: (dtype, shape)}."""
    if isinstance(tensors, (list, tuple)):
        tensors = {n: ("F16", [1]) for n in tensors}
    header = {}
    for n, (dt, shape) in tensors.items():
        header[n] = {"dtype": dt, "shape": list(shape), "data_offsets": [0, 0]}
    if metadata:
        header["__metadata__"] = metadata
    hj = json.dumps(header).encode()
    return len(hj).to_bytes(8, "little") + hj


def pickle_statedict(keys, ordered=False, zipped=False) -> bytes:
    d = OrderedDict() if ordered else {}
    for k in keys:
        d[k] = 0
    raw = pickle.dumps(d)
    if not zipped:
        return raw
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as z:
        z.writestr("archive/data.pkl", raw)
        z.writestr("archive/data/0", b"\x00" * 8)
    return buf.getvalue()


def file_entry(name: str, content: bytes, ftype: str = "Model") -> dict:
    # the live API reports SHA256 uppercase: exercise case-insensitivity
    return {"name": name, "type": ftype, "sizeKB": len(content) / 1024,
            "hashes": {"SHA256": sha256(content).upper(),
                       "AutoV2": sha256(content)[:10].upper()}}


def version_obj(vid: int, model_name: str, vname: str, mtype: str,
                files: list, base: str = "SDXL 1.0", air: str = None,
                **extra) -> dict:
    return {"id": vid, "modelId": vid * 10, "name": vname,
            "baseModel": base, "trainedWords": [],
            **({"air": air} if air else {}),
            "model": {"name": model_name, "type": mtype, "nsfw": False},
            "files": files,
            "images": [{"url": "https://example.invalid/x.jpg"}],
            **extra}


class Fixture:
    """Synthetic API fixture dir + cache root + local download dir."""

    def __init__(self, root: Path):
        self.root = root
        self.api = root / "api-fixture"
        self.cache = root / "cache"
        self.downloads = root / "downloads"
        for d in (self.api, self.cache, self.downloads):
            d.mkdir(parents=True)

    def set_by_hash(self, data):
        (self.api / "by-hash.json").write_text(json.dumps(data))

    def add_version(self, version: dict):
        (self.api / f"version-{version['id']}.json").write_text(
            json.dumps(version))

    def add_download(self, relpath: str, content: bytes) -> Path:
        f = self.downloads / relpath
        f.parent.mkdir(parents=True, exist_ok=True)
        f.write_bytes(content)
        return f

    def env(self) -> dict:
        env = {k: v for k, v in os.environ.items()
               if k not in ("CIVITAI_API_TOKEN", "DROSTE_CIVITAI_CACHE")}
        env[mod.FIXTURE_ENV] = str(self.api)
        return env

    def run(self, *argv: str) -> tuple:
        out, err = io.StringIO(), io.StringIO()
        rc = 0
        with mock.patch.dict(os.environ, self.env(), clear=True), \
                contextlib.redirect_stdout(out), \
                contextlib.redirect_stderr(err):
            try:
                mod.main(["--cache", str(self.cache), *argv])
            except SystemExit as e:
                rc = e.code or 0
        return rc, out.getvalue(), err.getvalue()

    def dest(self, rel: str, name: str) -> Path:
        return self.cache / rel / name

    def civ(self, rel: str, stem: str) -> Path:
        return self.cache / rel / (stem + ".civitai.info")

    def meta(self, rel: str, stem: str) -> dict:
        return json.loads((self.cache / rel / (stem + ".meta.droste"))
                          .read_text())

    def user(self, rel: str, stem: str) -> Path:
        return self.cache / rel / (stem + ".user.droste")


class CivitaiAdoptTest(unittest.TestCase):
    def setUp(self):
        import tempfile
        self._tmp = tempfile.TemporaryDirectory(prefix="civitai-adopt-test-")
        self.fx = Fixture(Path(self._tmp.name))
        self.addCleanup(self._tmp.cleanup)

    def simple_checkpoint(self, content: bytes, vid: int = 100) -> dict:
        v = version_obj(vid, "Great Model", "v1.0", "Checkpoint",
                        [file_entry("greatModel_v10.safetensors", content)],
                        air=f"urn:air:sdxl:checkpoint:civitai:{vid * 10}@{vid}")
        self.fx.set_by_hash({sha256(content): v})
        return v

    # --------------------------------------------------- identify: batch hit
    def test_batch_identify_dry_run_then_apply(self):
        content = b"checkpoint bytes " * 4
        f = self.fx.add_download("renamed-download.safetensors", content)
        self.simple_checkpoint(content)

        rc, out, err = self.fx.run(str(f))
        self.assertEqual(rc, 0, err)
        self.assertIn("DRY RUN", out)
        self.assertIn(f"IDENTIFIED {f} -> Great Model / v1.0 "
                      f"(Checkpoint, SDXL 1.0) "
                      f"[AIR urn:air:sdxl:checkpoint:civitai:1000@100] "
                      f"(via hash lookup)", out)
        self.assertIn("-> models/Stable-diffusion/Great-Model_v1.0.safetensors",
                      out)
        self.assertIn("1 adopted, 0 already cached, 0 refused", out)
        self.assertEqual(list(self.fx.cache.rglob("*")), [])  # dry-run

        rc, out, err = self.fx.run("--apply", str(f))
        self.assertEqual(rc, 0, err)
        dest = self.fx.dest("models/Stable-diffusion",
                            "Great-Model_v1.0.safetensors")
        self.assertEqual(dest.read_bytes(), content)
        self.assertTrue(f.exists())  # --link never removes the source

    # ------------------------------------------------ three-file scheme
    def test_civitai_info_is_pure_and_meta_holds_our_block(self):
        content = safetensors_bytes([
            "model.diffusion_model.input_blocks.0.0.weight",
            "conditioner.embedders.1.model.ln_final.weight",
            "first_stage_model.decoder.conv_in.weight"])
        f = self.fx.add_download("x.safetensors", content)
        v = self.simple_checkpoint(content)
        rc, out, err = self.fx.run("--apply", str(f))
        self.assertEqual(rc, 0, err)
        rel, stem = "models/Stable-diffusion", "Great-Model_v1.0"
        # 1) .civitai.info is the PURE API response -- nothing of ours
        info = json.loads(self.fx.civ(rel, stem).read_text())
        self.assertEqual(info, v)
        self.assertNotIn("extensions", info)
        # 2) .meta.droste carries the objective block, minus private fields
        m = self.fx.meta(rel, stem)
        self.assertEqual(m["tool"], mod.TOOL_ID)
        self.assertEqual(m["sha256"], sha256(content))
        self.assertEqual(m["normalized_name"], "Great-Model_v1.0.safetensors")
        self.assertEqual(m["modelId"], v["modelId"])
        self.assertEqual(m["modelVersionId"], v["id"])
        self.assertEqual(m["api_type"], "Checkpoint")
        self.assertEqual(m["resolved_type"], "Checkpoint")
        self.assertEqual(m["routing"], "api")
        self.assertNotIn("adopted_from", m)
        self.assertNotIn("original_name", m)
        self.assertNotIn("sniffed_base_model", m)
        self.assertEqual(m["detected_base_model"], "SDXL")
        self.assertEqual(m["sniff"]["base_model"],
                         {"value": "SDXL", "confidence": "absolute"})
        # 3) no source prefs -> no .user.droste (never created empty)
        self.assertFalse(self.fx.user(rel, stem).exists())

    # --------------------------------------- user-prefs ingestion per format
    def test_ingest_a1111_json_prefs(self):
        content = b"a1111 lora"
        f = self.fx.add_download("styl.safetensors", content)
        self.fx.add_download("styl.json", json.dumps({
            "description": "standard blurb", "sd version": "SDXL",
            "activation text": "styl, masterpiece",
            "preferred weight": 0.8, "notes": "works best at 0.8",
            "my_bespoke_flag": True}).encode())
        v = version_obj(910, "Styl", "v1", "LORA",
                        [file_entry("styl.safetensors", content)])
        self.fx.set_by_hash({sha256(content): v})
        rc, out, err = self.fx.run("--apply", str(f))
        self.assertEqual(rc, 0, err)
        u = json.loads(self.fx.user("models/Lora", "Styl_v1").read_text())
        # trigger_words is now the user-distinct DELTA vs trainedWords ([] here)
        self.assertEqual(u["trigger_words"], ["styl", "masterpiece"])
        self.assertEqual(u["preferred_weight"], 0.8)
        self.assertEqual(u["notes"], "works best at 0.8")
        # standard fields (description, sd version) are NOT copied up ...
        self.assertNotIn("description", u)
        # ... and only the UNMATCHED field is kept -- not the whole raw file
        self.assertEqual(u["unmatched"]["styl.json"], {"my_bespoke_flag": True})
        # discovery note names the unmatched key at normal log level
        self.assertIn("note: unrecognized field(s) in styl.json: "
                      "my_bespoke_flag — kept under 'unmatched'", out)

    def test_ingest_comfyui_metadata_json_prefs(self):
        content = b"comfy lora"
        f = self.fx.add_download("cf.safetensors", content)
        self.fx.add_download("cf.metadata.json", json.dumps({
            "civitai": {"id": 42, "some": "standard dump"},  # standard: skip
            "preview": "cf.png",
            "notes": "my note", "usage_tips": "cfg 4", "tags": ["anime"],
            "favorite": True, "date_added": "2026-01-01"}).encode())
        v = version_obj(911, "Cf", "v2", "LORA",
                        [file_entry("cf.safetensors", content)])
        self.fx.set_by_hash({sha256(content): v})
        rc, out, err = self.fx.run("--apply", str(f))
        self.assertEqual(rc, 0, err)
        u = json.loads(self.fx.user("models/Lora", "Cf_v2").read_text())
        self.assertEqual(u["notes"], "my note")
        self.assertEqual(u["usage_tips"], "cfg 4")
        self.assertEqual(u["tags"], ["anime"])
        self.assertTrue(u["favorite"])
        # standard civitai/preview are NOT preserved; only the unmatched
        # bespoke field (the timestamp) is kept + noted
        self.assertEqual(u["unmatched"]["cf.metadata.json"],
                         {"date_added": "2026-01-01"})
        self.assertNotIn("civitai", u["unmatched"]["cf.metadata.json"])
        self.assertIn("note: unrecognized field(s) in cf.metadata.json: "
                      "date_added — kept under 'unmatched'", out)

    def test_ingest_cm_info_prefs_and_unmatched(self):
        # CRITICAL: the former gap -- an unmapped .cm-info.json user field is
        # now preserved under `unmatched` and surfaced, not silently dropped.
        content = b"stability matrix lora"
        f = self.fx.add_download("sm.safetensors", content)
        self.fx.add_download("sm.cm-info.json", json.dumps({
            "ModelName": "SM", "VersionName": "v1",  # standard mirror: skip
            "Notes": "curated note", "IsFavorite": True,
            "MyCustomField": "keepme"}).encode())     # unmapped -> unmatched
        v = version_obj(912, "SM", "v1", "LORA",
                        [file_entry("sm.safetensors", content)])
        self.fx.set_by_hash({sha256(content): v})
        rc, out, err = self.fx.run("--apply", str(f))
        self.assertEqual(rc, 0, err)
        u = json.loads(self.fx.user("models/Lora", "SM_v1").read_text())
        self.assertEqual(u["notes"], "curated note")
        self.assertTrue(u["favorite"])
        self.assertEqual(u["unmatched"]["sm.cm-info.json"],
                         {"MyCustomField": "keepme"})
        self.assertIn("note: unrecognized field(s) in sm.cm-info.json: "
                      "MyCustomField — kept under 'unmatched'", out)

    # ----------------------------------- 4-outcome: user deltas / title / etc.
    def test_trigger_words_delta_vs_trainedwords(self):
        content = b"trigger delta lora"
        f = self.fx.add_download("t.safetensors", content)
        self.fx.add_download("t.json", json.dumps(
            {"activation text": "foo, baz, qux"}).encode())
        v = version_obj(1010, "TW", "v1", "LORA",
                        [file_entry("t.safetensors", content)])
        v["trainedWords"] = ["Foo", "bar"]  # case-insensitive overlap
        self.fx.set_by_hash({sha256(content): v})
        rc, out, err = self.fx.run("--apply", str(f))
        self.assertEqual(rc, 0, err)
        u = json.loads(self.fx.user("models/Lora", "TW_v1").read_text())
        self.assertEqual(u["trigger_words"], ["baz", "qux"])  # foo dropped

    def test_trigger_words_delta_all_overlap_no_field(self):
        content = b"trigger all overlap"
        f = self.fx.add_download("t2.safetensors", content)
        self.fx.add_download("t2.json",
                             json.dumps({"activation text": "foo, bar"}).encode())
        v = version_obj(1011, "TW2", "v1", "LORA",
                        [file_entry("t2.safetensors", content)])
        v["trainedWords"] = ["foo", "bar"]
        self.fx.set_by_hash({sha256(content): v})
        rc, out, err = self.fx.run("--apply", str(f))
        self.assertEqual(rc, 0, err)
        # every trigger overlaps trainedWords -> empty delta -> no user file
        self.assertFalse(self.fx.user("models/Lora", "TW2_v1").exists())

    def test_tags_delta_vs_civitai_tags(self):
        content = b"tags delta lora"
        f = self.fx.add_download("tg.safetensors", content)
        self.fx.add_download("tg.metadata.json",
                             json.dumps({"tags": ["Anime", "user-only"]}).encode())
        v = version_obj(1012, "TG", "v1", "LORA",
                        [file_entry("tg.safetensors", content)])
        v["model"]["tags"] = ["anime", "style"]  # case-insensitive overlap
        self.fx.set_by_hash({sha256(content): v})
        rc, out, err = self.fx.run("--apply", str(f))
        self.assertEqual(rc, 0, err)
        u = json.loads(self.fx.user("models/Lora", "TG_v1").read_text())
        self.assertEqual(u["tags"], ["user-only"])  # anime dropped

    def test_title_kept_when_different_dropped_when_equal(self):
        # kept
        c1 = b"title diff"
        f1 = self.fx.add_download("td.safetensors", c1)
        self.fx.add_download("td.cm-info.json",
                             json.dumps({"UserTitle": "My Nickname"}).encode())
        v1 = version_obj(1013, "Official Name", "v1", "LORA",
                         [file_entry("td.safetensors", c1)])
        # dropped (equal to model.name, case-insensitive)
        c2 = b"title same"
        f2 = self.fx.add_download("ts.safetensors", c2)
        self.fx.add_download("ts.cm-info.json",
                             json.dumps({"UserTitle": "official NAME"}).encode())
        v2 = version_obj(1014, "Official Name", "v2", "LORA",
                         [file_entry("ts.safetensors", c2)])
        self.fx.set_by_hash({sha256(c1): v1, sha256(c2): v2})
        rc, out, err = self.fx.run("--apply", str(f1), str(f2))
        self.assertEqual(rc, 0, err)
        u1 = json.loads(self.fx.user("models/Lora", "Official-Name_v1")
                        .read_text())
        self.assertEqual(u1["title"], "My Nickname")
        # equal-to-model.name UserTitle -> no title -> no user file at all
        self.assertFalse(self.fx.user("models/Lora", "Official-Name_v2")
                         .exists())

    def test_inference_defaults_structured_passthrough(self):
        content = b"inference defaults"
        f = self.fx.add_download("inf.safetensors", content)
        idefs = {"cfg": 7, "steps": 20, "sampler": "DPM++ 2M"}
        self.fx.add_download("inf.cm-info.json",
                             json.dumps({"InferenceDefaults": idefs}).encode())
        v = version_obj(1015, "Inf", "v1", "LORA",
                        [file_entry("inf.safetensors", content)])
        self.fx.set_by_hash({sha256(content): v})
        rc, out, err = self.fx.run("--apply", str(f))
        self.assertEqual(rc, 0, err)
        u = json.loads(self.fx.user("models/Lora", "Inf_v1").read_text())
        self.assertEqual(u["inference_defaults"], idefs)  # structured verbatim

    # ------------------------------------------ 4-outcome: ENRICH .civitai.info
    def test_enrich_fills_blank_scalars_and_marks(self):
        content = b"enrich blanks"
        f = self.fx.add_download("e.safetensors", content)
        self.fx.add_download("e.metadata.json", json.dumps({
            "modelDescription": "local description",
            "base_model": "Pony"}).encode())
        v = version_obj(1020, "E", "v1", "LORA",
                        [file_entry("e.safetensors", content)], base="")
        v["description"] = ""  # blank -> fill
        self.fx.set_by_hash({sha256(content): v})
        rc, out, err = self.fx.run("--apply", str(f))
        self.assertEqual(rc, 0, err)
        info = json.loads(self.fx.civ("models/Lora", "E_v1").read_text())
        self.assertEqual(info["description"], "local description")
        self.assertEqual(info["baseModel"], "Pony")
        enriched = info["extensions"]["droste"]["enriched"]
        self.assertEqual(sorted(enriched["fields"]), ["baseModel", "description"])
        self.assertEqual(enriched["images"], [])
        # marker carries NO source filenames (privacy)
        self.assertNotIn("e.metadata.json", json.dumps(enriched))

    def test_enrich_does_not_overwrite_populated_and_stays_pure(self):
        content = b"enrich populated"
        f = self.fx.add_download("ep.safetensors", content)
        self.fx.add_download("ep.metadata.json", json.dumps({
            "modelDescription": "local", "base_model": "SD 1.5"}).encode())
        v = version_obj(1021, "EP", "v1", "LORA",
                        [file_entry("ep.safetensors", content)], base="SDXL 1.0")
        v["description"] = "the official description"
        self.fx.set_by_hash({sha256(content): v})
        rc, out, err = self.fx.run("--apply", str(f))
        self.assertEqual(rc, 0, err)
        info = json.loads(self.fx.civ("models/Lora", "EP_v1").read_text())
        # populated CivitAI values control -> unchanged, no marker, byte-pure
        self.assertEqual(info, v)
        self.assertNotIn("extensions", info)

    def test_enrich_description_from_a1111_json(self):
        # A1111 .json `description` fills a blank CivitAI description (no
        # ComfyUI source present) and shows up in the marker.
        content = b"a1111 enrich desc"
        f = self.fx.add_download("ad.safetensors", content)
        self.fx.add_download("ad.json", json.dumps({
            "description": "a1111 model description",
            "sd version": "SDXL"}).encode())
        v = version_obj(1026, "AD", "v1", "LORA",
                        [file_entry("ad.safetensors", content)])
        v["description"] = ""  # blank -> fill from A1111
        self.fx.set_by_hash({sha256(content): v})
        rc, out, err = self.fx.run("--apply", str(f))
        self.assertEqual(rc, 0, err)
        info = json.loads(self.fx.civ("models/Lora", "AD_v1").read_text())
        self.assertEqual(info["description"], "a1111 model description")
        self.assertEqual(info["extensions"]["droste"]["enriched"]["fields"],
                         ["description"])

    def test_enrich_description_comfy_wins_over_a1111(self):
        # BOTH sources present -> ComfyUI modelDescription wins, A1111 is
        # only the fallback (order-independent).
        content = b"both desc sources"
        f = self.fx.add_download("bd.safetensors", content)
        self.fx.add_download("bd.json",
                             json.dumps({"description": "from a1111"}).encode())
        self.fx.add_download("bd.metadata.json",
                             json.dumps({"modelDescription": "from comfy"}).encode())
        v = version_obj(1027, "BD", "v1", "LORA",
                        [file_entry("bd.safetensors", content)])
        v["description"] = ""
        self.fx.set_by_hash({sha256(content): v})
        rc, out, err = self.fx.run("--apply", str(f))
        self.assertEqual(rc, 0, err)
        info = json.loads(self.fx.civ("models/Lora", "BD_v1").read_text())
        self.assertEqual(info["description"], "from comfy")

    def test_enrich_a1111_description_does_not_overwrite_populated(self):
        content = b"a1111 desc populated"
        f = self.fx.add_download("adp.safetensors", content)
        self.fx.add_download("adp.json",
                             json.dumps({"description": "local a1111"}).encode())
        v = version_obj(1028, "ADP", "v1", "LORA",
                        [file_entry("adp.safetensors", content)])
        v["description"] = "the official description"
        self.fx.set_by_hash({sha256(content): v})
        rc, out, err = self.fx.run("--apply", str(f))
        self.assertEqual(rc, 0, err)
        info = json.loads(self.fx.civ("models/Lora", "ADP_v1").read_text())
        # populated CivitAI description untouched -> byte-pure, no marker
        self.assertEqual(info, v)
        self.assertNotIn("extensions", info)

    def test_enrich_basemodel_absolute_sniff_override(self):
        # tensors say FLUX; CivitAI (wrongly) says SD 1.5 -> absolute override
        content = safetensors_bytes(["double_blocks.0.img_attn.qkv.weight",
                                     "single_blocks.0.linear1.weight"])
        f = self.fx.add_download("ov.safetensors", content)
        v = version_obj(1022, "OV", "v1", "Checkpoint",
                        [file_entry("ov.safetensors", content)], base="SD 1.5")
        self.fx.set_by_hash({sha256(content): v})
        rc, out, err = self.fx.run("--apply", str(f))
        self.assertEqual(rc, 0, err)
        info = json.loads(self.fx.civ("models/Stable-diffusion", "OV_v1")
                          .read_text())
        self.assertEqual(info["baseModel"], "FLUX.1")
        self.assertIn("baseModel",
                      info["extensions"]["droste"]["enriched"]["fields"])

    def test_enrich_image_union(self):
        content = b"image union"
        f = self.fx.add_download("im.safetensors", content)
        self.fx.add_download("im.metadata.json", json.dumps({
            "preview_url": "https://cdn.invalid/new.png",
            "preview_nsfw_level": 4}).encode())
        self.fx.add_download("im.cm-info.json", json.dumps({
            "ThumbnailImageUrl": "https://cdn.invalid/thumb.png"}).encode())
        v = version_obj(1023, "IM", "v1", "LORA",
                        [file_entry("im.safetensors", content)])
        v["images"] = [{"url": "https://example.invalid/existing.jpg",
                        "nsfwLevel": 1}]
        self.fx.set_by_hash({sha256(content): v})
        rc, out, err = self.fx.run("--apply", str(f))
        self.assertEqual(rc, 0, err)
        info = json.loads(self.fx.civ("models/Lora", "IM_v1").read_text())
        urls = {im["url"]: im for im in info["images"]}
        self.assertIn("https://example.invalid/existing.jpg", urls)  # kept
        # ComfyUI preview: nsfw rides
        self.assertEqual(urls["https://cdn.invalid/new.png"],
                         {"url": "https://cdn.invalid/new.png", "nsfwLevel": 4})
        # cm-info thumbnail: no source nsfw -> nsfwLevel OMITTED (honest
        # "unknown rating"), not defaulted
        self.assertEqual(urls["https://cdn.invalid/thumb.png"],
                         {"url": "https://cdn.invalid/thumb.png"})
        self.assertEqual(
            sorted(info["extensions"]["droste"]["enriched"]["images"]),
            ["https://cdn.invalid/new.png", "https://cdn.invalid/thumb.png"])

    def test_enrich_image_skips_present_and_local_paths(self):
        content = b"image skip"
        f = self.fx.add_download("is.safetensors", content)
        self.fx.add_download("is.metadata.json", json.dumps({
            "preview_url": "/home/user/private/preview.png"}).encode())  # local
        v = version_obj(1024, "IS", "v1", "LORA",
                        [file_entry("is.safetensors", content)])
        v["images"] = [{"url": "https://example.invalid/keep.jpg"}]
        self.fx.set_by_hash({sha256(content): v})
        rc, out, err = self.fx.run("--apply", str(f))
        self.assertEqual(rc, 0, err)
        info = json.loads(self.fx.civ("models/Lora", "IS_v1").read_text())
        # local-path preview dropped -> nothing added -> byte-pure
        self.assertEqual(info, v)
        self.assertNotIn("extensions", info)

    def test_idempotent_enriched_rerun_writes_nothing(self):
        content = b"idempotent enriched"
        f = self.fx.add_download("ie.safetensors", content)
        self.fx.add_download("ie.metadata.json", json.dumps({
            "modelDescription": "desc",
            "preview_url": "https://cdn.invalid/x.png"}).encode())
        v = version_obj(1025, "IE", "v1", "LORA",
                        [file_entry("ie.safetensors", content)])
        v["description"] = ""
        self.fx.set_by_hash({sha256(content): v})
        rc, out, err = self.fx.run("--apply", str(f))
        self.assertEqual(rc, 0, err)
        civ = self.fx.civ("models/Lora", "IE_v1")
        before = (civ.read_bytes(), civ.stat().st_mtime_ns)
        time.sleep(0.01)
        rc, out, err = self.fx.run("--apply", str(f))  # identical inputs
        self.assertEqual(rc, 0, err)
        self.assertIn("1 already cached", out)
        self.assertEqual(civ.read_bytes(), before[0])
        self.assertEqual(civ.stat().st_mtime_ns, before[1])  # no rewrite

    # ---------------------------------------- 4-outcome: sub_type -> .meta
    def test_sub_type_fine_to_meta_coarse_not_stored(self):
        # fine subtype -> meta.sub_type
        c1 = b"fine subtype"
        f1 = self.fx.add_download("st1.safetensors", c1)
        self.fx.add_download("st1.metadata.json",
                             json.dumps({"sub_type": "text_encoder"}).encode())
        v1 = version_obj(1030, "ST1", "v1", "LORA",
                         [file_entry("st1.safetensors", c1)])
        # coarse subtype -> NOT stored (redundant with model.type)
        c2 = b"coarse subtype"
        f2 = self.fx.add_download("st2.safetensors", c2)
        self.fx.add_download("st2.metadata.json",
                             json.dumps({"sub_type": "lora"}).encode())
        v2 = version_obj(1031, "ST2", "v1", "LORA",
                         [file_entry("st2.safetensors", c2)])
        self.fx.set_by_hash({sha256(c1): v1, sha256(c2): v2})
        rc, out, err = self.fx.run("--apply", str(f1), str(f2))
        self.assertEqual(rc, 0, err)
        self.assertEqual(self.fx.meta("models/Lora", "ST1_v1")["sub_type"],
                         "text_encoder")
        self.assertNotIn("sub_type", self.fx.meta("models/Lora", "ST2_v1"))

    # ------------------------------------------------- 4-outcome: DROP bucket
    def test_drop_fields_absent_everywhere_incl_file_path(self):
        secret = "/home/someone/private/models/thing.safetensors"
        content = b"drop bucket"
        f = self.fx.add_download("d.safetensors", content)
        self.fx.add_download("d.metadata.json", json.dumps({
            "sha256": "abc", "size": 1234, "file_name": "thing",
            "file_path": secret, "from_civitai": True, "db_checked": True,
            "hash_status": "ok", "metadata_source": "civitai",
            "last_checked_at": "2026", "modified": 1.0, "exclude": False,
            "skip_metadata_refresh": False, "civitai_deleted": False}).encode())
        v = version_obj(1040, "D", "v1", "LORA",
                        [file_entry("d.safetensors", content)])
        self.fx.set_by_hash({sha256(content): v})
        rc, out, err = self.fx.run("--apply", str(f))
        self.assertEqual(rc, 0, err)
        # nothing surfaced: no user file, no note, no meta sub_type, no enrich
        self.assertFalse(self.fx.user("models/Lora", "D_v1").exists())
        self.assertNotIn("unrecognized field", out)
        self.assertNotIn("sub_type", self.fx.meta("models/Lora", "D_v1"))
        info = json.loads(self.fx.civ("models/Lora", "D_v1").read_text())
        self.assertEqual(info, v)  # no enrichment
        # the local path leaked into NO written sidecar
        for p in self.fx.cache.rglob("*"):
            if p.is_file():
                self.assertNotIn(secret, p.read_text(errors="ignore"), p.name)

    # -------------------------------------- guard now covers title + inference
    def test_guard_covers_title_and_inference_defaults(self):
        content = b"title inference guard"
        f = self.fx.add_download("gi.safetensors", content)
        self.fx.add_download("gi.cm-info.json", json.dumps({
            "UserTitle": "Original", "InferenceDefaults": {"cfg": 7}}).encode())
        v = version_obj(1050, "Model", "v1", "LORA",
                        [file_entry("gi.safetensors", content)])
        self.fx.set_by_hash({sha256(content): v})
        rc, out, err = self.fx.run("--apply", str(f))
        self.assertEqual(rc, 0, err)
        rel, stem = "models/Lora", "Model_v1"
        # change BOTH guarded fields -> conflict, refuse without --force
        (self.fx.downloads / "gi.cm-info.json").write_text(json.dumps({
            "UserTitle": "Renamed", "InferenceDefaults": {"cfg": 3}}))
        rc, out, err = self.fx.run("--apply", str(f))
        self.assertEqual(rc, 1)
        self.assertIn("sheepishly refusing", err)
        self.assertIn("title", err)
        self.assertIn("inference_defaults", err)
        # --force overwrites both
        rc, out, err = self.fx.run("--apply", "--force", str(f))
        self.assertEqual(rc, 0, err)
        u = json.loads(self.fx.user(rel, stem).read_text())
        self.assertEqual(u["title"], "Renamed")
        self.assertEqual(u["inference_defaults"], {"cfg": 3})

    def test_pure_standard_source_makes_no_user_file_or_note(self):
        content = b"no prefs here"
        f = self.fx.add_download("p.safetensors", content)
        # an A1111 sidecar with ONLY standard fields -> nothing to preserve
        self.fx.add_download("p.json", json.dumps({
            "description": "std", "sd version": "SDXL"}).encode())
        v = version_obj(913, "Plain", "v1", "LORA",
                        [file_entry("p.safetensors", content)])
        self.fx.set_by_hash({sha256(content): v})
        rc, out, err = self.fx.run("--apply", str(f))
        self.assertEqual(rc, 0, err)
        self.assertFalse(self.fx.user("models/Lora", "Plain_v1").exists())
        self.assertNotIn("unrecognized field", out)

    def test_unmatched_note_prints_in_dry_run(self):
        # dry-run is exactly when the user wants to preview what's
        # unrecognized: the note prints, but nothing is written.
        content = b"dry run note test"
        f = self.fx.add_download("dr.safetensors", content)
        self.fx.add_download("dr.json",
                             json.dumps({"notes": "n", "mystery": 1}).encode())
        v = version_obj(915, "Dryly", "v1", "LORA",
                        [file_entry("dr.safetensors", content)])
        self.fx.set_by_hash({sha256(content): v})
        rc, out, err = self.fx.run(str(f))  # NO --apply -> dry-run default
        self.assertEqual(rc, 0, err)
        self.assertIn("DRY RUN", out)
        self.assertIn("note: unrecognized field(s) in dr.json: "
                      "mystery — kept under 'unmatched'", out)
        # dry-run: no .user.droste (nor any sidecar) actually written
        self.assertFalse(self.fx.user("models/Lora", "Dryly_v1").exists())
        self.assertEqual(list(self.fx.cache.rglob("*")), [])

    def test_quiet_consolidates_unrecognized_fields(self):
        # -q data-dump view: per-file inline notes AND the ADOPT lines are
        # suppressed; instead ONE consolidated, filename-sorted list of
        # unrecognized fields prints (level 0), plus the summary.
        a = b"quiet file A"
        b = b"quiet file B"
        fa = self.fx.add_download("aaa.safetensors", a)
        fb = self.fx.add_download("bbb.safetensors", b)
        self.fx.add_download("aaa.json",
                             json.dumps({"notes": "n", "zeta": 1}).encode())
        self.fx.add_download("bbb.json",
                             json.dumps({"alpha": 2, "beta": 3}).encode())
        va = version_obj(914, "QA", "v1", "LORA",
                         [file_entry("aaa.safetensors", a)])
        vb = version_obj(916, "QB", "v1", "LORA",
                         [file_entry("bbb.safetensors", b)])
        self.fx.set_by_hash({sha256(a): va, sha256(b): vb})
        rc, out, err = self.fx.run("--apply", "-q", str(fa), str(fb))
        self.assertEqual(rc, 0, err)
        # consolidated block, entries sorted by source filename
        self.assertIn("unrecognized fields:\n"
                      "  aaa.json: zeta\n"
                      "  bbb.json: alpha, beta\n", out)
        # the inline per-file wording is NOT used under -q ...
        self.assertNotIn("kept under 'unmatched'", out)
        # ... nor the per-file adopt / identify noise; summary still shows
        self.assertNotIn("ADOPT", out)
        self.assertNotIn("IDENTIFIED", out)
        self.assertIn("2 adopted, 0 already cached, 0 refused", out)
        # the block sits just before the summary line
        self.assertLess(out.index("unrecognized fields:"),
                        out.index("summary:"))
        # data was still preserved on disk
        u = json.loads(self.fx.user("models/Lora", "QA_v1").read_text())
        self.assertEqual(u["unmatched"]["aaa.json"], {"zeta": 1})

    def test_quiet_no_unrecognized_fields_prints_no_block(self):
        content = b"clean quiet"
        f = self.fx.add_download("c.safetensors", content)
        v = version_obj(917, "Clean", "v1", "LORA",
                        [file_entry("c.safetensors", content)])
        self.fx.set_by_hash({sha256(content): v})
        rc, out, err = self.fx.run("--apply", "-q", str(f))
        self.assertEqual(rc, 0, err)
        self.assertNotIn("unrecognized fields:", out)  # nothing to list
        self.assertNotIn("ADOPT", out)
        self.assertIn("1 adopted, 0 already cached, 0 refused", out)

    # ----------------------------------------------- idempotent sync
    def test_idempotent_sync_second_run_writes_nothing(self):
        content = b"idempotent bytes"
        f = self.fx.add_download("i.safetensors", content)
        self.fx.add_download("i.json",
                             json.dumps({"notes": "n"}).encode())
        v = version_obj(920, "Idem", "v1", "LORA",
                        [file_entry("i.safetensors", content)])
        self.fx.set_by_hash({sha256(content): v})
        rc, out, err = self.fx.run("--apply", str(f))
        self.assertEqual(rc, 0, err)
        rel, stem = "models/Lora", "Idem_v1"
        sidecars = [self.fx.civ(rel, stem),
                    self.cache_meta(rel, stem), self.fx.user(rel, stem)]
        before = {p: (p.read_bytes(), p.stat().st_mtime_ns) for p in sidecars}
        time.sleep(0.01)
        rc, out, err = self.fx.run("--apply", str(f))  # identical inputs
        self.assertEqual(rc, 0, err)
        self.assertIn("0 adopted, 1 already cached, 0 refused", out)
        for p, (data, mtime) in before.items():
            self.assertEqual(p.read_bytes(), data, p.name)
            self.assertEqual(p.stat().st_mtime_ns, mtime, p.name)

    def cache_meta(self, rel, stem):
        return self.fx.cache / rel / (stem + ".meta.droste")

    def test_already_branch_refreshes_metadata_when_api_changes(self):
        content = b"cached model bytes"
        f = self.fx.add_download("r.safetensors", content)
        v1 = version_obj(930, "Refresh", "v1", "LORA",
                         [file_entry("r.safetensors", content)])
        self.fx.set_by_hash({sha256(content): v1})
        rc, out, err = self.fx.run("--apply", str(f))
        self.assertEqual(rc, 0, err)
        rel, stem = "models/Lora", "Refresh_v1"
        info_before = self.fx.civ(rel, stem).read_text()
        # API grows a field; model file already cached -> ALREADY branch
        v2 = dict(v1, description="freshly documented on civitai")
        self.fx.set_by_hash({sha256(content): v2})
        rc, out, err = self.fx.run("--apply", str(f))
        self.assertEqual(rc, 0, err)
        self.assertIn("ALREADY", out)
        info_after = json.loads(self.fx.civ(rel, stem).read_text())
        self.assertNotEqual(self.fx.civ(rel, stem).read_text(), info_before)
        self.assertEqual(info_after["description"],
                         "freshly documented on civitai")

    def test_monotonic_merge_retains_vanished_and_adds_new(self):
        content = b"monotonic bytes"
        f = self.fx.add_download("m.safetensors", content)
        # first source carries a normalized field AND an unmatched one
        self.fx.add_download("m.json", json.dumps(
            {"notes": "keep me", "weird_key": "x"}).encode())
        v = version_obj(940, "Mono", "v1", "LORA",
                        [file_entry("m.safetensors", content)])
        self.fx.set_by_hash({sha256(content): v})
        rc, out, err = self.fx.run("--apply", str(f))
        self.assertEqual(rc, 0, err)
        rel, stem = "models/Lora", "Mono_v1"
        # source sidecar vanishes, a DIFFERENT source appears with a NEW field
        (self.fx.downloads / "m.json").unlink()
        self.fx.add_download("m.metadata.json",
                             json.dumps({"usage_tips": "cfg 5"}).encode())
        rc, out, err = self.fx.run("--apply", str(f))
        self.assertEqual(rc, 0, err)
        u = json.loads(self.fx.user(rel, stem).read_text())
        self.assertEqual(u["notes"], "keep me")     # retained though source gone
        self.assertEqual(u["usage_tips"], "cfg 5")   # newly added
        # the unmatched entry from the now-vanished source is retained
        self.assertEqual(u["unmatched"]["m.json"], {"weird_key": "x"})

    # ------------------------------------------------- user-data guard
    def test_user_data_guard_refuses_and_prints_diff(self):
        content = b"guard bytes"
        f = self.fx.add_download("g.safetensors", content)
        self.fx.add_download("g.json", json.dumps({
            "notes": "original", "preferred weight": 0.5}).encode())
        v = version_obj(950, "Guard", "v1", "LORA",
                        [file_entry("g.safetensors", content)])
        self.fx.set_by_hash({sha256(content): v})
        rc, out, err = self.fx.run("--apply", str(f))
        self.assertEqual(rc, 0, err)
        rel, stem = "models/Lora", "Guard_v1"
        user_path = self.fx.user(rel, stem)
        before = user_path.read_bytes()
        model_dest = self.fx.dest(rel, "Guard_v1.safetensors")
        # source prefs now CONFLICT with the stored ones
        (self.fx.downloads / "g.json").write_text(json.dumps({
            "notes": "rewritten", "preferred weight": 0.9}))
        rc, out, err = self.fx.run("--apply", "--move", str(f))
        self.assertEqual(rc, 1)  # nothing adopted, one refused -> exit 1
        # exact message on stderr, side-by-side per conflicting field
        self.assertIn("Error: sheepishly refusing to overwrite existing "
                      f"user data in {user_path}.", err)
        self.assertIn("old: original", err)
        self.assertIn("new: rewritten", err)
        self.assertIn("old: 0.5", err)
        self.assertIn("new: 0.9", err)
        self.assertIn("add the --force flag to ignore this error", err)
        # ENTIRE adoption aborted: user file untouched, source not moved
        self.assertEqual(user_path.read_bytes(), before)
        self.assertTrue(f.exists())              # --move did NOT remove source
        self.assertTrue(model_dest.exists())     # model file still present
        self.assertIn("0 adopted, 0 already cached, 1 refused", out)

    def test_force_overrides_user_guard(self):
        content = b"force bytes"
        f = self.fx.add_download("fo.safetensors", content)
        self.fx.add_download("fo.json", json.dumps({"notes": "original"}).encode())
        v = version_obj(951, "Forced", "v1", "LORA",
                        [file_entry("fo.safetensors", content)])
        self.fx.set_by_hash({sha256(content): v})
        self.fx.run("--apply", str(f))
        rel, stem = "models/Lora", "Forced_v1"
        (self.fx.downloads / "fo.json").write_text(json.dumps({"notes": "new"}))
        rc, out, err = self.fx.run("--apply", "--force", str(f))
        self.assertEqual(rc, 0, err)
        u = json.loads(self.fx.user(rel, stem).read_text())
        self.assertEqual(u["notes"], "new")

    def test_force_does_not_imply_apply(self):
        content = b"force-no-apply"
        f = self.fx.add_download("fna.safetensors", content)
        self.fx.add_download("fna.json", json.dumps({"notes": "original"}).encode())
        v = version_obj(952, "FNA", "v1", "LORA",
                        [file_entry("fna.safetensors", content)])
        self.fx.set_by_hash({sha256(content): v})
        self.fx.run("--apply", str(f))
        rel, stem = "models/Lora", "FNA_v1"
        user_path = self.fx.user(rel, stem)
        before = user_path.read_bytes()
        (self.fx.downloads / "fna.json").write_text(json.dumps({"notes": "new"}))
        # --force WITHOUT --apply: guard is overridden (no refuse) but it is
        # still a dry-run, so nothing is written
        rc, out, err = self.fx.run("--force", str(f))
        self.assertEqual(rc, 0, err)
        self.assertNotIn("sheepishly refusing", err)
        self.assertEqual(user_path.read_bytes(), before)  # unchanged (dry-run)

    def test_force_does_not_bypass_identity_gate(self):
        content = b"genuine"
        other = b"not in this version"
        f = self.fx.add_download("id.safetensors", other)
        self.fx.add_version(version_obj(
            953, "Ident", "v1", "LORA",
            [file_entry("ident.safetensors", content)]))
        rc, out, err = self.fx.run("--apply", "--force",
                                   "--version-id", "953", str(f))
        self.assertEqual(rc, 1)
        self.assertIn("not byte-identical to any file in version 953", out)

    def test_force_does_not_bypass_different_content_model_refusal(self):
        content = b"canon"
        f = self.fx.add_download("dc.safetensors", content)
        v = version_obj(954, "DiffC", "v1", "LORA",
                        [file_entry("dc.safetensors", content)])
        self.fx.set_by_hash({sha256(content): v})
        rel = "models/Lora"
        dest = self.fx.dest(rel, "DiffC_v1.safetensors")
        dest.parent.mkdir(parents=True)
        dest.write_bytes(b"USER DATA - different")
        rc, out, err = self.fx.run("--apply", "--force", str(f))
        self.assertEqual(rc, 1)
        self.assertIn("exists with DIFFERENT content; refusing to overwrite",
                      out)
        self.assertEqual(dest.read_bytes(), b"USER DATA - different")

    def test_guard_refuses_only_conflicting_file_in_batch(self):
        ca = b"clean file bytes"
        cc = b"conflicting file bytes"
        fa = self.fx.add_download("clean.safetensors", ca)
        fc = self.fx.add_download("conf.safetensors", cc)
        self.fx.add_download("conf.json", json.dumps({"notes": "orig"}).encode())
        va = version_obj(960, "Clean", "v1", "LORA",
                         [file_entry("clean.safetensors", ca)])
        vc = version_obj(961, "Conf", "v1", "LORA",
                         [file_entry("conf.safetensors", cc)])
        self.fx.set_by_hash({sha256(ca): va, sha256(cc): vc})
        self.fx.run("--apply", str(fc))  # seed Conf's user file
        (self.fx.downloads / "conf.json").write_text(
            json.dumps({"notes": "changed"}))
        rc, out, err = self.fx.run("--apply", str(fa), str(fc))
        # clean adopts, conflicting refuses, run continues
        self.assertIn("ADOPT", out)
        self.assertTrue(self.fx.dest("models/Lora",
                                     "Clean_v1.safetensors").exists())
        self.assertIn("sheepishly refusing", err)
        self.assertIn("1 adopted, 0 already cached, 1 refused", out)

    # ------------------------------------------------- sidecar id fallback
    def test_sidecar_fallback_each_format(self):
        cases = [
            (".civitai.info", lambda vid: {"id": vid, "modelId": vid * 10,
                                           "files": []}),
            (".cm-info.json", lambda vid: {"VersionId": vid}),
            (".metadata.json", lambda vid: {"modelVersionId": vid}),
            (".json", lambda vid: {"modelVersionId": str(vid)}),
        ]
        for i, (suffix, payload) in enumerate(cases):
            with self.subTest(sidecar=suffix):
                vid = 500 + i
                content = f"old unhashed file {i}".encode()
                stem = f"oldie{i}"
                f = self.fx.add_download(f"{stem}.safetensors", content)
                (self.fx.downloads / f"{stem}{suffix}").write_text(
                    json.dumps(payload(vid)))
                self.fx.set_by_hash({})  # by-hash knows nothing (old file)
                self.fx.add_version(version_obj(
                    vid, f"Oldie {i}", "v1", "LORA",
                    [file_entry(f"oldie{i}.safetensors", content)]))
                rc, out, err = self.fx.run(str(f))
                self.assertEqual(rc, 0, err)
                self.assertIn(f"via local sidecar -> version {vid}", out)
                self.assertIn(f"-> models/Lora/Oldie-{i}_v1.safetensors", out)

    def test_sidecar_fallback_rejected_without_hash_proof(self):
        content = b"tampered or mislabelled bytes"
        f = self.fx.add_download("fake.safetensors", content)
        (self.fx.downloads / "fake.metadata.json").write_text(
            json.dumps({"modelVersionId": 777}))
        self.fx.set_by_hash({})
        self.fx.add_version(version_obj(
            777, "Real Model", "v2", "LORA",
            [file_entry("real.safetensors", b"the REAL bytes")]))
        rc, out, err = self.fx.run(str(f))
        self.assertEqual(rc, 1)
        self.assertIn(f"REFUSE  {f}: sidecar points at version 777 but "
                      f"this file's hash is not among its published files",
                      out)

    def test_unknown_hash_no_sidecar_refuses(self):
        f = self.fx.add_download("mystery.safetensors", b"nobody knows")
        self.fx.set_by_hash({})
        rc, out, err = self.fx.run(str(f))
        self.assertEqual(rc, 1)
        self.assertIn(f"REFUSE  {f}: not found by hash on CivitAI and no "
                      f"provable local metadata (old unhashed file? pass "
                      f"--version-id)", out)

    # ------------------------------------------------------------ --version-id
    def test_version_id_match_and_refusal(self):
        good = b"the genuine old checkpoint"
        bad = b"something else entirely!!"
        fg = self.fx.add_download("good.ckpt", good)
        fb = self.fx.add_download("bad.ckpt", bad)
        self.fx.add_version(version_obj(
            128713, "Ancient", "v1", "Checkpoint",
            [file_entry("ancient_v1.ckpt", good)]))
        rc, out, err = self.fx.run("--apply", "--version-id", "128713",
                                   str(fg), str(fb))
        self.assertEqual(rc, 0, err)
        self.assertIn("-> models/Stable-diffusion/Ancient_v1.ckpt", out)
        self.assertIn(f"REFUSE  {fb}: not byte-identical to any file in "
                      f"version 128713", out)
        self.assertIn("1 adopted, 0 already cached, 1 refused", out)
        self.assertTrue(self.fx.dest("models/Stable-diffusion",
                                     "Ancient_v1.ckpt").exists())

    def test_version_id_not_found_dies(self):
        f = self.fx.add_download("x.ckpt", b"x")
        rc, out, err = self.fx.run("--version-id", "999999", str(f))
        self.assertEqual(rc, 2)
        self.assertIn("model version not found: 999999", err)

    # ---------------------------------------------------- type-dir taxonomy
    def test_all_type_dirs_incl_new_and_root_level(self):
        cases = [
            ("Checkpoint", "models/Stable-diffusion"),
            ("LORA", "models/Lora"),
            ("LoCon", "models/LyCORIS"),
            ("DoRA", "models/DoRA"),
            ("TextualInversion", "embeddings"),
            ("Hypernetwork", "models/hypernetworks"),
            ("VAE", "models/VAE"),
            ("Controlnet", "models/ControlNet"),
            ("MotionModule", "models/motion_modules"),
            ("AestheticGradient", "models/aesthetic_embeddings"),
            ("Detection", "models/detection"),
            ("Poses", "poses"),
            ("Wildcards", "wildcards"),
            ("Workflows", "workflows"),
        ]
        for i, (mtype, rel) in enumerate(cases):
            with self.subTest(mtype=mtype):
                content = f"content for {mtype} {i}".encode()
                f = self.fx.add_download(f"dl{i}.pt", content)
                v = version_obj(600 + i, f"{mtype}Model", "v1", mtype,
                                [file_entry(f"orig{i}.pt", content)])
                self.fx.set_by_hash({sha256(content): v})
                rc, out, err = self.fx.run("--apply", str(f))
                self.assertEqual(rc, 0, err)
                dest = self.fx.dest(rel, f"{mtype}Model_v1.pt")
                self.assertEqual(dest.read_bytes(), content, mtype)
        self.assertFalse((self.fx.cache / "models" / "embeddings").exists())

    def test_unknown_api_type_goes_to_other_never_refused(self):
        content = b"some future modality"
        f = self.fx.add_download("weird.pt", content)
        v = version_obj(660, "Frobnicator", "v2", "QuantumEmbedding",
                        [file_entry("frob.pt", content)])
        self.fx.set_by_hash({sha256(content): v})
        rc, out, err = self.fx.run("--apply", str(f))
        self.assertEqual(rc, 0, err)  # NOT refused
        dest = self.fx.dest("other/QuantumEmbedding", "Frobnicator_v2.pt")
        self.assertEqual(dest.read_bytes(), content)
        m = self.fx.meta("other/QuantumEmbedding", "Frobnicator_v2")
        self.assertEqual(m["resolved_type"], "QuantumEmbedding")

    def test_vae_file_entry_overrides_model_type(self):
        ckpt = b"main checkpoint bytes"
        vae = b"the bundled VAE bytes"
        fc = self.fx.add_download("model.safetensors", ckpt)
        fv = self.fx.add_download("model.vae.safetensors", vae)
        v = version_obj(400, "Bundle", "v3", "Checkpoint", [
            file_entry("bundle_v3.safetensors", ckpt, ftype="Model"),
            file_entry("bundle_v3.vae.safetensors", vae, ftype="VAE"),
        ])
        self.fx.set_by_hash({sha256(ckpt): v, sha256(vae): v})
        rc, out, err = self.fx.run("--apply", str(fc), str(fv))
        self.assertEqual(rc, 0, err)
        self.assertTrue(self.fx.dest("models/Stable-diffusion",
                                     "Bundle_v3.safetensors").exists())
        # split_ext keeps the multi-dot .vae.safetensors extension whole,
        # so the bundled VAE's marker survives into the normalized name
        self.assertTrue(self.fx.dest("models/VAE",
                                     "Bundle_v3.vae.safetensors").exists())

    # ---------------------------------------------- content sniff: routing
    def test_controlnet_vs_t2i_adapter_sniff(self):
        cn = safetensors_bytes(["control_model.input_blocks.0.0.weight",
                                "controlnet_cond_embedding.conv_in.weight"])
        t2i = safetensors_bytes(["adapter.body.0.block1.weight",
                                 "adapter.body.1.block2.weight"])
        fcn = self.fx.add_download("cn.safetensors", cn)
        ft2i = self.fx.add_download("t2i.safetensors", t2i)
        vcn = version_obj(810, "MyControl", "v1", "Controlnet",
                          [file_entry("mycn.safetensors", cn)])
        vt2i = version_obj(811, "MyAdapter", "v1", "Controlnet",
                           [file_entry("myt2i.safetensors", t2i)])
        self.fx.set_by_hash({sha256(cn): vcn, sha256(t2i): vt2i})
        rc, out, err = self.fx.run("--apply", str(fcn), str(ft2i))
        self.assertEqual(rc, 0, err)
        self.assertTrue(self.fx.dest("models/ControlNet",
                                     "MyControl_v1.safetensors").exists())
        self.assertTrue(self.fx.dest("models/T2IAdapter",
                                     "MyAdapter_v1.safetensors").exists())
        self.assertIn("sniff-override", out)
        m = self.fx.meta("models/T2IAdapter", "MyAdapter_v1")
        self.assertEqual(m["api_type"], "Controlnet")
        self.assertEqual(m["resolved_type"], "T2IAdapter")
        self.assertTrue(m["routing"].startswith("sniff-override"))
        self.assertEqual(m["sniff"]["kind"],
                         {"value": "t2i_adapter", "confidence": "absolute"})

    def test_upscaler_arch_split_and_catchall(self):
        cases = [
            (["layers.0.residual_group.blocks.0.attn."
              "relative_position_bias_table", "conv_first.weight"],
             "models/SwinIR"),
            (["m_head.0.weight", "m_body.0.weight", "m_tail.0.weight"],
             "models/ScuNET"),
            (["model.0.weight", "model.1.sub.0.RDB1.conv1.0.weight"],
             "models/ESRGAN"),
            (["totally.unknown.arch.weight", "mystery.block.bias"],
             "models/upscale_models"),
        ]
        for i, (keys, rel) in enumerate(cases):
            with self.subTest(dir=rel):
                content = safetensors_bytes(keys)
                f = self.fx.add_download(f"up{i}.safetensors", content)
                v = version_obj(820 + i, f"Scaler{i}", "v1", "Upscaler",
                                [file_entry(f"s{i}.safetensors", content)])
                self.fx.set_by_hash({sha256(content): v})
                rc, out, err = self.fx.run("--apply", str(f))
                self.assertEqual(rc, 0, err)
                self.assertTrue(self.fx.dest(rel, f"Scaler{i}_v1.safetensors")
                                .exists(), rel)

    def test_sniff_uncertain_keeps_api_type(self):
        content = safetensors_bytes(["model.diffusion_model.x.weight",
                                     "lora_unet_down.lora_down.weight"])
        f = self.fx.add_download("thing.safetensors", content)
        v = version_obj(830, "Styler", "v1", "LORA",
                        [file_entry("styler.safetensors", content)])
        self.fx.set_by_hash({sha256(content): v})
        rc, out, err = self.fx.run("--apply", str(f))
        self.assertEqual(rc, 0, err)
        self.assertTrue(self.fx.dest("models/Lora",
                                     "Styler_v1.safetensors").exists())
        m = self.fx.meta("models/Lora", "Styler_v1")
        self.assertEqual(m["routing"], "api")
        self.assertEqual(m["resolved_type"], "LORA")
        self.assertEqual(m["sniff"]["base_model"]["confidence"], "uncertain")

    def test_base_model_sniff_flux(self):
        content = safetensors_bytes(["double_blocks.0.img_attn.qkv.weight",
                                     "single_blocks.0.linear1.weight"])
        f = self.fx.add_download("flux.safetensors", content)
        v = version_obj(840, "FluxThing", "v1", "Checkpoint",
                        [file_entry("flux.safetensors", content)],
                        base="Flux.1 D")
        self.fx.set_by_hash({sha256(content): v})
        rc, out, err = self.fx.run("--apply", str(f))
        self.assertEqual(rc, 0, err)
        m = self.fx.meta("models/Stable-diffusion", "FluxThing_v1")
        self.assertEqual(m["detected_base_model"], "FLUX.1")
        self.assertEqual(m["sniff"]["base_model"]["confidence"], "absolute")

    # --------------------------------------------- restricted unpickler
    def test_restricted_unpickler_recovers_keys(self):
        keys = ["layers.0.weight", "layers.0.bias", "layers.1.weight"]
        self.assertEqual(sorted(mod._restricted_unpickle_keys(
            io.BytesIO(pickle_statedict(keys)))), sorted(keys))
        self.assertEqual(sorted(mod._restricted_unpickle_keys(
            io.BytesIO(pickle_statedict(keys, ordered=True)))), sorted(keys))
        raw = self.fx.add_download("raw.pt", pickle_statedict(keys))
        self.assertEqual(sorted(mod.sniff_pickle_keys(raw)), sorted(keys))
        zp = self.fx.add_download("z.pt", pickle_statedict(keys, zipped=True))
        self.assertEqual(sorted(mod.sniff_pickle_keys(zp)), sorted(keys))

    def test_malicious_pickle_executes_nothing(self):
        marker = self.fx.root / "PWNED"

        class _Evil:
            def __reduce__(self):
                import os as _os
                return (_os.system, (f"touch {marker}",))

        evil = self.fx.add_download("evil.pt", pickle.dumps(_Evil()))
        keys = mod.sniff_pickle_keys(evil)
        self.assertFalse(marker.exists(), "restricted unpickler executed code")
        self.assertIsInstance(keys, (list, type(None)))
        content = evil.read_bytes()
        v = version_obj(850, "Trap", "v1", "LORA",
                        [file_entry("evil.pt", content)])
        self.fx.set_by_hash({sha256(content): v})
        rc, out, err = self.fx.run("--apply", str(evil))
        self.assertEqual(rc, 0, err)
        self.assertFalse(marker.exists())

    # ----------------------------------------------------- normalization
    def test_normalize_sanitizes_names(self):
        content = b"unicode and unsafe chars"
        f = self.fx.add_download("dl.safetensors", content)
        v = version_obj(860, "Modèl: Bad/Name*  Two", "v9 <final>",
                        "Checkpoint",
                        [file_entry("whatever.safetensors", content)])
        self.fx.set_by_hash({sha256(content): v})
        rc, out, err = self.fx.run("--apply", str(f))
        self.assertEqual(rc, 0, err)
        dest = self.fx.dest("models/Stable-diffusion",
                            "Modèl-BadName-Two_v9-final.safetensors")
        self.assertTrue(dest.exists(),
                        sorted(str(p.relative_to(self.fx.cache))
                               for p in self.fx.cache.rglob("*.safetensors")))

    def test_multifile_same_dir_disambiguation(self):
        full = b"the full unpruned checkpoint bytes"
        pruned = b"the pruned checkpoint bytes"
        ff = self.fx.add_download("full.safetensors", full)
        fp = self.fx.add_download("pruned.safetensors", pruned)
        v = version_obj(870, "Dream", "v8", "Checkpoint", [
            file_entry("full.safetensors", full, ftype="Model"),
            file_entry("pruned.safetensors", pruned, ftype="Model"),
        ])
        self.fx.set_by_hash({sha256(full): v, sha256(pruned): v})
        rc, out, err = self.fx.run("--apply", str(ff), str(fp))
        self.assertEqual(rc, 0, err)
        self.assertTrue(self.fx.dest("models/Stable-diffusion",
                                     "Dream_v8-full.safetensors").exists())
        self.assertTrue(self.fx.dest("models/Stable-diffusion",
                                     "Dream_v8-pruned.safetensors").exists())

    def test_unsafe_api_name_neutralized_not_placed_outside_cache(self):
        content = b"escape attempt"
        f = self.fx.add_download("dl.safetensors", content)
        v = version_obj(880, "Evil", "v1", "LORA", [
            {"name": "../../escape.safetensors", "type": "Model",
             "hashes": {"SHA256": sha256(content).upper()}}])
        self.fx.set_by_hash({sha256(content): v})
        rc, out, err = self.fx.run("--apply", str(f))
        self.assertEqual(rc, 0, err)
        self.assertTrue(self.fx.dest("models/Lora",
                                     "Evil_v1.safetensors").exists())
        self.assertFalse((self.fx.root / "escape.safetensors").exists())

    # ------------------------------------------------------- never-clobber
    def test_never_clobber_same_and_different_content(self):
        content = b"canonical bytes"
        f = self.fx.add_download("dl.safetensors", content)
        self.simple_checkpoint(content)
        dest = self.fx.dest("models/Stable-diffusion",
                            "Great-Model_v1.0.safetensors")
        dest.parent.mkdir(parents=True)
        dest.write_bytes(content)
        rc, out, err = self.fx.run("--apply", str(f))
        self.assertEqual(rc, 0, err)
        self.assertIn("ALREADY", out)
        self.assertIn("0 adopted, 1 already cached, 0 refused", out)
        dest.write_bytes(b"USER DATA - different")
        rc, out, err = self.fx.run("--apply", str(f))
        self.assertEqual(rc, 1)
        self.assertIn("exists with DIFFERENT content; refusing to overwrite",
                      out)
        self.assertEqual(dest.read_bytes(), b"USER DATA - different")

    # ------------------------------------------------------- preview carry
    def test_preview_carried_and_never_clobbered(self):
        content = b"lora bytes"
        f = self.fx.add_download("mylora.safetensors", content)
        self.fx.add_download("mylora.preview.png", b"PNGDATA")
        v = version_obj(700, "My Lora", "v1", "LORA",
                        [file_entry("myLora_v1.safetensors", content)])
        self.fx.set_by_hash({sha256(content): v})
        rc, out, err = self.fx.run("--apply", str(f))
        self.assertEqual(rc, 0, err)
        pdest = self.fx.dest("models/Lora", "My-Lora_v1.preview.png")
        self.assertEqual(pdest.read_bytes(), b"PNGDATA")
        pdest.write_bytes(b"CURATED")
        f2 = self.fx.add_download("again/mylora.safetensors", content)
        self.fx.add_download("again/mylora.png", b"OTHERPNG")
        rc, out, err = self.fx.run("--apply", str(f2))
        self.assertEqual(rc, 0, err)
        self.assertEqual(pdest.read_bytes(), b"CURATED")
        self.assertIn("preview exists; leaving it", out)

    # --------------------------------------------- directories / grouping
    def test_mixed_version_directory_and_companions_skipped(self):
        ca = b"checkpoint A bytes"
        cb = b"lora B bytes"
        fa = self.fx.add_download("a.safetensors", ca)
        fb = self.fx.add_download("b.safetensors", cb)
        self.fx.add_download("a.civitai.info", b"{}")  # companions: skipped
        self.fx.add_download("b.preview.png", b"PNG")
        va = version_obj(801, "Model A", "v1", "Checkpoint",
                         [file_entry("modelA.safetensors", ca)])
        vb = version_obj(802, "Model B", "v2", "LORA",
                         [file_entry("modelB.safetensors", cb)])
        self.fx.set_by_hash({sha256(ca): va, sha256(cb): vb})
        rc, out, err = self.fx.run("--apply", str(self.fx.downloads))
        self.assertEqual(rc, 0, err)
        self.assertIn(f"IDENTIFIED {fa} -> Model A / v1", out)
        self.assertIn(f"IDENTIFIED {fb} -> Model B / v2", out)
        self.assertIn("2 adopted, 0 already cached, 0 refused", out)
        self.assertTrue(self.fx.dest("models/Stable-diffusion",
                                     "Model-A_v1.safetensors").exists())
        self.assertTrue(self.fx.dest("models/Lora",
                                     "Model-B_v2.safetensors").exists())
        self.assertTrue(self.fx.dest("models/Lora",
                                     "Model-B_v2.preview.png").exists())

    def test_move_reclaims_source_and_preview(self):
        content = b"movable bytes"
        f = self.fx.add_download("m.safetensors", content)
        prev = self.fx.add_download("m.preview.png", b"PNG")
        v = version_obj(900, "Mover", "v1", "LORA",
                        [file_entry("mover_v1.safetensors", content)])
        self.fx.set_by_hash({sha256(content): v})
        rc, out, err = self.fx.run("--apply", "--move", str(f))
        self.assertEqual(rc, 0, err)
        self.assertFalse(f.exists())
        self.assertFalse(prev.exists())
        self.assertEqual(self.fx.dest("models/Lora", "Mover_v1.safetensors")
                         .read_bytes(), content)
        self.assertEqual(self.fx.dest("models/Lora", "Mover_v1.preview.png")
                         .read_bytes(), b"PNG")

    # ------------------------------------------------------ network errors
    def test_network_error_refuses_per_file_run_survives(self):
        f1 = self.fx.add_download("one.safetensors", b"one")
        f2 = self.fx.add_download("two.safetensors", b"two")
        self.fx.set_by_hash({"error": "connection reset"})
        rc, out, err = self.fx.run(str(f1), str(f2))
        self.assertEqual(rc, 1)
        self.assertIn("warning: CivitAI hash lookup failed", out)
        self.assertEqual(out.count("REFUSE"), 2, out)
        self.assertIn("CivitAI lookup unavailable; try again or pass "
                      "--version-id", out)
        self.assertIn("0 adopted, 0 already cached, 2 refused", out)

    # ----------------------------------------------------- progress (TTY only)
    def test_progress_tty_quiet_and_hashing(self):
        class FakeTTY(io.StringIO):
            def isatty(self):
                return True

        args = types.SimpleNamespace(quiet=0, verbose=0)
        plain = io.StringIO()  # non-TTY stderr: strict no-op
        with contextlib.redirect_stderr(plain):
            mod.progress(args, "  looking up 3 hash(es) on CivitAI...")
            mod.progress_clear()
        self.assertEqual(plain.getvalue(), "")
        # progress no longer checks quiet: on a TTY it shows even under -q
        tty = FakeTTY()
        with contextlib.redirect_stderr(tty):
            mod.progress(types.SimpleNamespace(quiet=1, verbose=0),
                         "  hashing under quiet...")
            mod.progress_clear()
        self.assertIn("\r  hashing under quiet...", tty.getvalue())
        tty = FakeTTY()
        with contextlib.redirect_stderr(tty):
            mod.progress(args, "  fetching version 12345...")
            mod.progress_clear()
        raw = tty.getvalue()
        self.assertIn("\r  fetching version 12345...", raw)
        self.assertTrue(raw.endswith("\r"))
        f = self.fx.add_download("big.bin", b"z" * 4096)
        tty = FakeTTY()
        with mock.patch.object(mod, "HASH_PROGRESS_MIN", 1024), \
                mock.patch.object(mod, "HASH_PROGRESS_STEP", 1024), \
                contextlib.redirect_stderr(tty):
            sha, facts = mod.scan_file(f, 4096, args)
        raw = tty.getvalue()
        self.assertEqual(sha, sha256(b"z" * 4096))
        self.assertIn("hashing big.bin: ", raw)
        self.assertIn("100%", raw)
        self.assertTrue(raw.endswith("\r"))
        tty = FakeTTY()
        with contextlib.redirect_stderr(tty):
            mod.hash_file(f, 4096)
        self.assertEqual(tty.getvalue(), "")

    # ------------------------- long names: split_ext / byte budget / refuse
    WAN_MODEL = ("ON-THE-FLY 实时生成！Wan-AI 万相 Wan2.1 Video Model "
                 "(multi-specs) - CausVid&Comfy&Kijai - workflow included")
    WAN_VERSION = "首尾帧-FLF2V-14B-720P"
    # Jei's expected recommendation -- CUMULATIVE cascade: NFKC (step 1)
    # folds the fullwidth '！' to ASCII '!', which then survives step 4's
    # CJK drop; deletions leave double dashes intact (NO re-collapse). Note
    # the kept '!' right after 'ON-THE-FLY-'.
    WAN_RECOMMENDED = ("ON-THE-FLY-!Wan-AI--Wan2.1-Video-Model-(multi-specs)-"
                       "CausVid&Comfy&Kijai-workflow-included_-FLF2V-14B-720P")

    def test_split_ext_variants(self):
        self.assertEqual(mod.split_ext("model.safetensors"),
                         ("model", ".safetensors"))
        self.assertEqual(mod.split_ext("archive.tar.gz"),
                         ("archive", ".tar.gz"))
        self.assertEqual(mod.split_ext("Foo_v1.civitai.info"),
                         ("Foo_v1", ".civitai.info"))
        self.assertEqual(mod.split_ext("bundle_v3.vae.safetensors"),
                         ("bundle_v3", ".vae.safetensors"))
        self.assertEqual(mod.split_ext("noext"), ("noext", ""))
        self.assertEqual(mod.split_ext(".hidden"), (".hidden", ""))
        # a trailing dotted run past the cap is NOT an extension
        long = "file." + "x" * 33
        self.assertEqual(mod.split_ext(long), (long, ""))
        # the cap bounds the WHOLE dotted run, not each segment
        self.assertEqual(mod.split_ext("name.tar.twelvechars0", cap=16),
                         ("name.tar", ".twelvechars0"))
        # path separators never join an extension
        self.assertEqual(mod.split_ext("../../escape.safetensors"),
                         ("../../escape", ".safetensors"))

    def test_stem_budget_is_in_bytes_not_chars(self):
        with mock.patch.object(mod, "_name_max", return_value=255):
            budget = mod.stem_budget_bytes(Path("/nonexistent"),
                                           ".safetensors")
        # reserves the widest family member: the staged .civitai.info
        self.assertEqual(budget,
                         255 - len(f".tmp-.civitai.info-{os.getpid()}"))
        # bytes, never characters: 3-byte CJK burns the budget 3x faster
        self.assertLessEqual(len(("好" * (budget // 3)).encode()), budget)
        self.assertGreater(len(("好" * (budget // 3 + 1)).encode()), budget)
        self.assertLessEqual(len(("x" * budget).encode()), budget)

    def test_name_max_pathconf_and_fallback(self):
        with mock.patch("os.pathconf", return_value=180):
            self.assertEqual(mod._name_max(self.fx.cache), 180)
        with mock.patch("os.pathconf", side_effect=OSError):
            self.assertEqual(mod._name_max(self.fx.cache), 255)
        # a not-yet-created dest dir resolves via its nearest ancestor
        self.assertGreater(mod._name_max(self.fx.cache / "no" / "such"), 0)

    def test_recommend_cascade_each_step(self):
        # 1: NFKC-normalize, then drop historic-script chars (runes vanish,
        #    fullwidth folds to ASCII)
        self.assertEqual(mod.recommend_stem("ᚠᚹᛖ-Model", 6), "-Model")
        self.assertEqual(mod.recommend_stem("Ｍｏｄｅｌ", 5), "Model")
        # 2: planes 2-16 dropped (U+20000 is CJK ext-B) while BMP CJK stays
        self.assertEqual(mod.recommend_stem("A\U00020000B好", 5), "AB好")
        # 3: plane 1 (emoji) dropped, BMP CJK still kept
        self.assertEqual(mod.recommend_stem("A\U0001F600B好", 5), "AB好")
        # 4: 3-byte UTF-8 (all CJK) dropped
        self.assertEqual(mod.recommend_stem("A好B好C", 3), "ABC")
        # 5: non-ASCII dropped
        self.assertEqual(mod.recommend_stem("AéBé", 2), "AB")
        # 6: right-truncate keeping the last 8 chars -- guaranteed fit
        self.assertEqual(mod.recommend_stem("x" * 50 + "LAST8888", 20),
                         "x" * 12 + "LAST8888")
        self.assertEqual(mod.recommend_stem("x" * 50 + "LAST8888", 5),
                         "T8888")

    def test_recommend_cascade_is_cumulative(self):
        # Regression guard against reverting to independent steps: the later
        # plane/byte filters must narrow step 1's NFKC output, NOT the raw
        # stem. A fullwidth '！' (U+FF01) NFKC-folds to ASCII '!' (0x21) in
        # step 1; a BMP CJK '好' (0x597D) is over budget. Step 4 (drop >=
        # 0x800) removes the CJK while the FOLDED '!' survives -- only
        # possible if step 4 sees the folded candidate.
        self.assertEqual(mod.recommend_stem("Ａ！好Ｂ", 4), "A!B")
        # were the steps independent (re-deriving from the raw stem), step 4
        # would drop the raw fullwidth '！' too and yield "AB".
        self.assertNotEqual(mod.recommend_stem("Ａ！好Ｂ", 4), "AB")

    def test_recommend_wan_example_exact(self):
        stem = (mod.sanitize_component(self.WAN_MODEL) + "_"
                + mod.sanitize_component(self.WAN_VERSION))
        self.assertEqual(len(stem.encode()), 135)  # CJK = 3 bytes/char
        # steps 1-3 miss; step 4 lands via the CUMULATIVE path (folded '!'
        # from step-1 NFKC is retained; CJK dropped)
        rec = mod.recommend_stem(stem, 110)
        self.assertEqual(rec, self.WAN_RECOMMENDED)
        self.assertIn("ON-THE-FLY-!Wan-AI", rec)   # folded '!' kept
        # deletions leave surrounding punctuation intact -- NO re-collapse
        self.assertIn("Wan-AI--Wan2.1", rec)
        self.assertIn("included_-FLF2V", rec)

    def test_too_long_name_refused_batch_continues(self):
        # THE original bug: dest.exists() on a > NAME_MAX name raised
        # ENAMETOOLONG and killed the whole batch. Now: the fit check
        # refuses it up front and the NEXT file still adopts.
        long_bytes = b"the too-long model bytes"
        ok_bytes = b"the fine model bytes"
        fl = self.fx.add_download("long.safetensors", long_bytes)
        fk = self.fx.add_download("ok.safetensors", ok_bytes)
        vl = version_obj(2000, "好" * 90, "v1", "LORA",
                         [file_entry("orig-long.safetensors", long_bytes)])
        vk = version_obj(2001, "Fine", "v1", "LORA",
                         [file_entry("ok.safetensors", ok_bytes)])
        self.fx.set_by_hash({sha256(long_bytes): vl, sha256(ok_bytes): vk})
        rc, out, err = self.fx.run("--apply", str(fl), str(fk))
        self.assertEqual(rc, 0, err)  # something adopted -> exit 0
        self.assertIn(f"REFUSE  {fl}", out)
        self.assertIn("over the filesystem's name limit by", out)
        self.assertIn("recommend: --rename", out)
        self.assertIn("1 adopted, 0 already cached, 1 refused", out)
        self.assertTrue(self.fx.dest("models/Lora",
                                     "Fine_v1.safetensors").exists())
        # nothing of the too-long family was written (or even touched)
        self.assertFalse(any("好" in p.name
                             for p in self.fx.cache.rglob("*")))

    def test_wan_refuse_prints_byte_overage_and_recommendation(self):
        content = b"wan flf2v model bytes"
        f = self.fx.add_download("wan.safetensors", content)
        v = version_obj(2100, self.WAN_MODEL, self.WAN_VERSION, "Checkpoint",
                        [file_entry("wan_flf2v.safetensors", content)])
        self.fx.set_by_hash({sha256(content): v})
        # the spec's (abridged) Wan name is 147 bytes -- squeeze NAME_MAX so
        # it overflows and the cascade still lands on step 4
        with mock.patch.object(mod, "_name_max", return_value=140):
            rc, out, err = self.fx.run("--apply", str(f))
        self.assertEqual(rc, 1)  # nothing adopted, one refused
        self.assertIn(f"REFUSE  {f}: name for {self.WAN_MODEL} / "
                      f"{self.WAN_VERSION} (Checkpoint, SDXL 1.0) is over "
                      f"the filesystem's name limit by", out)
        self.assertIn(f"recommend: --rename '{self.WAN_RECOMMENDED}'", out)
        self.assertEqual(list(self.fx.cache.rglob("*")), [])  # untouched

    def test_per_file_oserror_refuses_and_batch_continues(self):
        # containment is broader than the fit check: ANY per-file OSError
        # during placement becomes a REFUSE line, never a dead batch
        a, b = b"oserror one", b"oserror two"
        fa = self.fx.add_download("oa.safetensors", a)
        fb = self.fx.add_download("ob.safetensors", b)
        va = version_obj(2500, "OA", "v1", "LORA",
                         [file_entry("oa.safetensors", a)])
        vb = version_obj(2501, "OB", "v1", "LORA",
                         [file_entry("ob.safetensors", b)])
        self.fx.set_by_hash({sha256(a): va, sha256(b): vb})
        real, tripped = mod.place_file, []

        def boom(args, src, dest, mode):
            if not tripped:
                tripped.append(1)
                raise OSError(errno.ENAMETOOLONG, "File name too long")
            return real(args, src, dest, mode)

        with mock.patch.object(mod, "place_file", side_effect=boom):
            rc, out, err = self.fx.run("--apply", str(fa), str(fb))
        self.assertEqual(rc, 0, err)
        self.assertIn(f"REFUSE  {fa}:", out)
        self.assertIn("File name too long", out)
        self.assertIn("1 adopted, 0 already cached, 1 refused", out)
        self.assertTrue(self.fx.dest("models/Lora",
                                     "OB_v1.safetensors").exists())

    # ------------------------------------------------------------- --rename
    def _long_name_fixture(self, content=b"long-name model bytes", vid=2300):
        """A hash-identified file whose normalized name (90 CJK chars =
        270 bytes) cannot fit NAME_MAX -- the --rename use case."""
        f = self.fx.add_download("long.safetensors", content)
        v = version_obj(vid, "好" * 90, "v1", "LORA",
                        [file_entry("orig.safetensors", content)])
        self.fx.set_by_hash({sha256(content): v})
        return f

    def test_rename_bare_stem_appends_source_ext(self):
        f = self._long_name_fixture()
        rc, out, err = self.fx.run("--apply", "--rename", "Wan21-FLF2V",
                                   str(f))
        self.assertEqual(rc, 0, err)
        self.assertTrue(self.fx.dest("models/Lora",
                                     "Wan21-FLF2V.safetensors").exists())
        # the override is captured as the guarded `filename` user field
        u = json.loads(self.fx.user("models/Lora", "Wan21-FLF2V").read_text())
        self.assertEqual(u["filename"], "Wan21-FLF2V.safetensors")

    def test_rename_full_filename_not_doubled(self):
        # NAME's own trailing extension (16-char detection window) matches
        # the identified file's -> NAME is a whole filename already
        f = self._long_name_fixture()
        rc, out, err = self.fx.run("--apply", "--rename",
                                   "Wan21-FLF2V.safetensors", str(f))
        self.assertEqual(rc, 0, err)
        self.assertTrue(self.fx.dest("models/Lora",
                                     "Wan21-FLF2V.safetensors").exists())
        self.assertFalse(self.fx.dest(
            "models/Lora", "Wan21-FLF2V.safetensors.safetensors").exists())

    def test_rename_different_ext_gets_source_ext_appended(self):
        # .ckpt is an extension, but not THIS file's -- the tool owns the
        # extension of the sha-identified file, so .safetensors is appended
        f = self._long_name_fixture()
        rc, out, err = self.fx.run("--apply", "--rename", "Wan21.ckpt",
                                   str(f))
        self.assertEqual(rc, 0, err)
        self.assertTrue(self.fx.dest("models/Lora",
                                     "Wan21.ckpt.safetensors").exists())

    def test_rename_still_too_long_self_validates(self):
        f = self._long_name_fixture()
        rc, out, err = self.fx.run("--apply", "--rename", "x" * 300, str(f))
        self.assertEqual(rc, 1)
        self.assertIn("over the filesystem's name limit by", out)
        self.assertEqual([p for p in self.fx.cache.rglob("*") if p.is_file()],
                         [])  # never writes an unusable name

    def test_rename_requires_exactly_one_file(self):
        a, b = b"rename file one", b"rename file two"
        fa = self.fx.add_download("one.safetensors", a)
        fb = self.fx.add_download("two.safetensors", b)
        va = version_obj(2400, "One", "v1", "LORA",
                         [file_entry("one.safetensors", a)])
        vb = version_obj(2401, "Two", "v1", "LORA",
                         [file_entry("two.safetensors", b)])
        self.fx.set_by_hash({sha256(a): va, sha256(b): vb})
        rc, out, err = self.fx.run("--apply", "--rename", "X",
                                   str(fa), str(fb))
        self.assertEqual(rc, 2)
        self.assertIn("--rename applies to a single file; 2 matched", err)
        self.assertEqual(list(self.fx.cache.rglob("*")), [])

    # -------------------------------------------- filename capture/read-back
    def test_readback_makes_plain_rerun_already(self):
        f = self._long_name_fixture()
        rc, out, err = self.fx.run("--apply", "--rename", "Shorty", str(f))
        self.assertEqual(rc, 0, err)
        # plain re-run, NO --rename: the routed dir's .meta.droste sha scan
        # reads the recorded filename back -> ALREADY, not a re-refuse
        rc, out, err = self.fx.run("--apply", str(f))
        self.assertEqual(rc, 0, err)
        self.assertIn("ALREADY", out)
        self.assertIn("0 adopted, 1 already cached, 0 refused", out)
        self.assertNotIn("REFUSE", out)

    def test_rename_conflict_guarded_then_forced(self):
        f = self._long_name_fixture()
        rc, out, err = self.fx.run("--apply", "--rename", "FirstName", str(f))
        self.assertEqual(rc, 0, err)
        # a DIFFERENT --rename conflicts with the recorded filename ->
        # guarded like any user field: refuse without --force
        rc, out, err = self.fx.run("--apply", "--rename", "SecondName",
                                   str(f))
        self.assertEqual(rc, 1)
        self.assertIn("sheepishly refusing", err)
        self.assertIn("filename", err)
        self.assertIn("old: FirstName.safetensors", err)
        self.assertIn("new: SecondName.safetensors", err)
        self.assertFalse(self.fx.dest("models/Lora",
                                      "SecondName.safetensors").exists())
        rc, out, err = self.fx.run("--apply", "--force", "--rename",
                                   "SecondName", str(f))
        self.assertEqual(rc, 0, err)
        self.assertTrue(self.fx.dest("models/Lora",
                                     "SecondName.safetensors").exists())
        u = json.loads(self.fx.user("models/Lora", "SecondName").read_text())
        self.assertEqual(u["filename"], "SecondName.safetensors")

    def test_help_exits_zero(self):
        with self.assertRaises(SystemExit) as cm:
            with contextlib.redirect_stdout(io.StringIO()):
                mod.main(["--help"])
        self.assertEqual(cm.exception.code, 0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
