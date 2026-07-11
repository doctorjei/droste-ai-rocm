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
        self.assertEqual(u["trigger_words"], "styl, masterpiece")
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
        self.assertTrue(self.fx.dest("models/VAE",
                                     "Bundle_v3.safetensors").exists())

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

    def test_help_exits_zero(self):
        with self.assertRaises(SystemExit) as cm:
            with contextlib.redirect_stdout(io.StringIO()):
                mod.main(["--help"])
        self.assertEqual(cm.exception.code, 0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
