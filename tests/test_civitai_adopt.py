#!/usr/bin/env python3
"""Tests for droste-civitai-adopt using the DROSTE_CIVITAI_API_FIXTURE
hook -- NO live network. A fixture dir stands in for the CivitAI API:

- by-hash.json        batch by-hash endpoint: {sha256: version, ...} or
                      a plain list of version objects; {"error": msg}
                      simulates a network failure.
- version-<id>.json   GET /model-versions/<id> (missing file = 404).

Covered: batch identify hit, 404 -> sidecar fallback (every sidecar
format), fallback rejected when our sha isn't in the version, --version-id
(match + refusal), unmapped-type refusal, VAE file-entry override, A1111
placement paths incl. root-level embeddings, dry-run vs --apply layout,
never-clobber (same content = already, different = refuse), preview
carry, sidecar contents, mixed-version directories, --move reclaim,
network error -> per-file refusal, and the transient progress helpers.

Run:  python3 tests/test_civitai_adopt.py -v
"""

import contextlib
import hashlib
import importlib.machinery
import importlib.util
import io
import json
import os
import sys
import types
import unittest
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


def file_entry(name: str, content: bytes, ftype: str = "Model") -> dict:
    # the live API reports SHA256 uppercase: exercise case-insensitivity
    return {"name": name, "type": ftype, "sizeKB": len(content) / 1024,
            "hashes": {"SHA256": sha256(content).upper(),
                       "AutoV2": sha256(content)[:10].upper()}}


def version_obj(vid: int, model_name: str, vname: str, mtype: str,
                files: list, base: str = "SDXL 1.0", air: str = None) -> dict:
    return {"id": vid, "modelId": vid * 10, "name": vname,
            "baseModel": base, "trainedWords": [],
            **({"air": air} if air else {}),
            "model": {"name": model_name, "type": mtype, "nsfw": False},
            "files": files,
            "images": [{"url": "https://example.invalid/x.jpg"}]}


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
        self.assertIn("ADOPT", out)
        self.assertIn("-> models/Stable-diffusion/greatModel_v10.safetensors",
                      out)
        self.assertIn("1 adopted, 0 already cached, 0 refused", out)
        # dry-run: nothing placed
        self.assertEqual(list(self.fx.cache.rglob("*")), [])

        rc, out, err = self.fx.run("--apply", str(f))
        self.assertEqual(rc, 0, err)
        dest = (self.fx.cache / "models" / "Stable-diffusion"
                / "greatModel_v10.safetensors")
        self.assertEqual(dest.read_bytes(), content)
        self.assertTrue(f.exists())  # --link never removes the source

    def test_sidecar_written_and_valid(self):
        content = b"weights!"
        f = self.fx.add_download("x.safetensors", content)
        v = self.simple_checkpoint(content)
        rc, out, err = self.fx.run("--apply", str(f))
        self.assertEqual(rc, 0, err)
        side = (self.fx.cache / "models" / "Stable-diffusion"
                / "greatModel_v10.civitai.info")
        info = json.loads(side.read_text())
        # raw API response preserved (Civitai Helper convention) ...
        self.assertEqual(info["id"], v["id"])
        self.assertEqual(info["model"]["name"], "Great Model")
        self.assertEqual(info["files"][0]["name"],
                         "greatModel_v10.safetensors")
        # ... with our additions namespaced under extensions.droste
        self.assertEqual(info["extensions"]["droste"]["sha256"],
                         sha256(content))
        self.assertEqual(info["extensions"]["droste"]["adopted_from"],
                         str(f))

    # ------------------------------------------------- sidecar id fallback
    def test_sidecar_fallback_each_format(self):
        cases = [
            # (suffix, payload) -- every supported sidecar format
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
                self.assertIn(f"-> models/Lora/oldie{i}.safetensors", out)

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
        self.assertIn("ADOPT", out)
        self.assertIn("-> models/Stable-diffusion/ancient_v1.ckpt", out)
        self.assertIn(f"REFUSE  {fb}: not byte-identical to any file in "
                      f"version 128713", out)
        self.assertIn("1 adopted, 0 already cached, 1 refused", out)
        self.assertTrue((self.fx.cache / "models" / "Stable-diffusion"
                         / "ancient_v1.ckpt").exists())

    def test_version_id_not_found_dies(self):
        f = self.fx.add_download("x.ckpt", b"x")
        rc, out, err = self.fx.run("--version-id", "999999", str(f))
        self.assertEqual(rc, 2)
        self.assertIn("model version not found: 999999", err)

    # ------------------------------------------------------- type mapping
    def test_unmapped_type_refused(self):
        content = b"a pose pack zip"
        f = self.fx.add_download("poses.zip", content)
        v = version_obj(300, "Pose Pack", "v1", "Poses",
                        [file_entry("poses.zip", content, ftype="Archive")])
        self.fx.set_by_hash({sha256(content): v})
        rc, out, err = self.fx.run(str(f))
        self.assertEqual(rc, 1)
        self.assertIn(f"REFUSE  {f}: unmapped CivitAI type 'Poses'; "
                      f"place it manually", out)

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
        self.assertTrue((self.fx.cache / "models" / "Stable-diffusion"
                         / "bundle_v3.safetensors").exists())
        self.assertTrue((self.fx.cache / "models" / "VAE"
                         / "bundle_v3.vae.safetensors").exists())

    def test_placement_paths_incl_root_embeddings(self):
        cases = [  # (model.type, expected relative dir)
            ("LORA", "models/Lora"),
            ("LoCon", "models/Lora"),
            ("DoRA", "models/Lora"),
            ("TextualInversion", "embeddings"),  # root level, not models/
            ("Hypernetwork", "models/hypernetworks"),
            ("VAE", "models/VAE"),
            ("Controlnet", "models/ControlNet"),
            ("Upscaler", "models/ESRGAN"),
            ("MotionModule", "models/animatediff"),
        ]
        for i, (mtype, rel) in enumerate(cases):
            with self.subTest(mtype=mtype):
                content = f"content for {mtype}".encode()
                f = self.fx.add_download(f"dl{i}.bin", content)
                v = version_obj(600 + i, mtype + " thing", "v1", mtype,
                                [file_entry(f"thing{i}.pt", content)])
                self.fx.set_by_hash({sha256(content): v})
                rc, out, err = self.fx.run("--apply", str(f))
                self.assertEqual(rc, 0, err)
                dest = self.fx.cache / rel / f"thing{i}.pt"
                self.assertEqual(dest.read_bytes(), content)
        self.assertFalse((self.fx.cache / "models" / "embeddings").exists())

    # ------------------------------------------------------- never-clobber
    def test_never_clobber_same_and_different_content(self):
        content = b"canonical bytes"
        f = self.fx.add_download("dl.safetensors", content)
        self.simple_checkpoint(content)
        dest = (self.fx.cache / "models" / "Stable-diffusion"
                / "greatModel_v10.safetensors")
        dest.parent.mkdir(parents=True)
        # same content already at destination -> ALREADY, not re-placed
        dest.write_bytes(content)
        rc, out, err = self.fx.run("--apply", str(f))
        self.assertEqual(rc, 0, err)
        self.assertIn("ALREADY", out)
        self.assertIn("0 adopted, 1 already cached, 0 refused", out)
        # different content at destination -> loud refusal, file untouched
        dest.write_bytes(b"USER DATA - different")
        rc, out, err = self.fx.run("--apply", str(f))
        self.assertEqual(rc, 1)
        self.assertIn("exists with DIFFERENT content; refusing to overwrite",
                      out)
        self.assertEqual(dest.read_bytes(), b"USER DATA - different")

    def test_existing_sidecar_left_alone(self):
        content = b"bytes"
        f = self.fx.add_download("dl.safetensors", content)
        self.simple_checkpoint(content)
        side = (self.fx.cache / "models" / "Stable-diffusion"
                / "greatModel_v10.civitai.info")
        side.parent.mkdir(parents=True)
        side.write_text('{"mine": true}')
        rc, out, err = self.fx.run("--apply", str(f))
        self.assertEqual(rc, 0, err)
        self.assertEqual(json.loads(side.read_text()), {"mine": True})
        self.assertIn("sidecar exists; leaving it", out)

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
        pdest = self.fx.cache / "models" / "Lora" / "myLora_v1.preview.png"
        self.assertEqual(pdest.read_bytes(), b"PNGDATA")
        # a second source with the SAME dest preview: never clobbered
        pdest.write_bytes(b"CURATED")
        f2 = self.fx.add_download("again/mylora.safetensors", content)
        self.fx.add_download("again/mylora.png", b"OTHERPNG")
        rc, out, err = self.fx.run("--apply", str(f2))
        self.assertEqual(rc, 0, err)  # dest model matches -> ALREADY
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
        self.assertIn("version Model A / v1", out)  # one group line each
        self.assertIn("version Model B / v2", out)
        self.assertIn("2 adopted, 0 already cached, 0 refused", out)
        self.assertNotIn("a.civitai.info:", out)  # never a candidate
        self.assertTrue((self.fx.cache / "models" / "Stable-diffusion"
                         / "modelA.safetensors").exists())
        self.assertTrue((self.fx.cache / "models" / "Lora"
                         / "modelB.safetensors").exists())
        # b's preview was carried alongside modelB
        self.assertTrue((self.fx.cache / "models" / "Lora"
                         / "modelB.preview.png").exists())

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
        self.assertEqual((self.fx.cache / "models" / "Lora"
                          / "mover_v1.safetensors").read_bytes(), content)
        self.assertEqual((self.fx.cache / "models" / "Lora"
                          / "mover_v1.preview.png").read_bytes(), b"PNG")

    # ------------------------------------------------------ network errors
    def test_network_error_refuses_per_file_run_survives(self):
        f1 = self.fx.add_download("one.safetensors", b"one")
        f2 = self.fx.add_download("two.safetensors", b"two")
        self.fx.set_by_hash({"error": "connection reset"})
        rc, out, err = self.fx.run(str(f1), str(f2))
        self.assertEqual(rc, 1)  # refused, never crashed
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
        plain = io.StringIO()  # non-TTY: strict no-op
        with contextlib.redirect_stderr(plain):
            mod.progress(args, "  looking up 3 hash(es) on CivitAI...")
            mod.progress_clear()
        self.assertEqual(plain.getvalue(), "")
        tty = FakeTTY()  # -q suppresses even on a TTY
        with contextlib.redirect_stderr(tty):
            mod.progress(types.SimpleNamespace(quiet=1, verbose=0), "  x")
            mod.progress_clear()
        self.assertEqual(tty.getvalue(), "")
        tty = FakeTTY()  # TTY: \r-updated in place, wiped clean
        with contextlib.redirect_stderr(tty):
            mod.progress(args, "  fetching version 12345...")
            mod.progress_clear()
        raw = tty.getvalue()
        self.assertIn("\r  fetching version 12345...", raw)
        self.assertTrue(raw.endswith("\r"))
        # hashing progress with thresholds patched small
        f = self.fx.add_download("big.bin", b"z" * 4096)
        tty = FakeTTY()
        with mock.patch.object(mod, "HASH_PROGRESS_MIN", 1024), \
                mock.patch.object(mod, "HASH_PROGRESS_STEP", 1024), \
                contextlib.redirect_stderr(tty):
            mod.hash_file(f, 4096, args)
        raw = tty.getvalue()
        self.assertIn("hashing big.bin: ", raw)
        self.assertIn("100%", raw)
        self.assertTrue(raw.endswith("\r"))
        tty = FakeTTY()  # without args: silent even on a TTY
        with contextlib.redirect_stderr(tty):
            mod.hash_file(f, 4096)
        self.assertEqual(tty.getvalue(), "")

    # ------------------------------------------------------------------- misc
    def test_unsafe_api_filename_refused(self):
        content = b"evil"
        f = self.fx.add_download("dl.safetensors", content)
        v = version_obj(950, "Evil", "v1", "LORA", [
            {"name": "../../escape.safetensors", "type": "Model",
             "hashes": {"SHA256": sha256(content).upper()}}])
        self.fx.set_by_hash({sha256(content): v})
        rc, out, err = self.fx.run("--apply", str(f))
        self.assertEqual(rc, 1)
        self.assertIn("unsafe filename from the API", out)
        self.assertEqual(list(self.fx.cache.rglob("*")), [])

    def test_help_exits_zero(self):
        with self.assertRaises(SystemExit) as cm:
            with contextlib.redirect_stdout(io.StringIO()):
                mod.main(["--help"])
        self.assertEqual(cm.exception.code, 0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
