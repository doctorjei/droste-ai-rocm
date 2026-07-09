#!/usr/bin/env python3
"""Tests for droste-hf-adopt using the DROSTE_ADOPT_API_FIXTURE hook --
NO live network. A fixture dir stands in for the HF API:

- <org>__<name>.json           repo manifest (?blobs=true shape)
- search.json                  /api/models?search= results: a list
                               (every query), a {query: [...]} map with
                               optional "*" default, or {"error": msg}
                               to simulate a network failure.

Covered: --repo mode unchanged (adopt/refuse/cache layout, and that it
never touches the search endpoint), identify single-candidate, multiple
matching repos (downloads tiebreak), identify miss -> refusal, mixed-repo
directory, search API error -> per-file refusal, candidate manifest
error -> skip to next candidate, GGUF provenance hint short-circuiting
the search, term derivation, and one-hash-per-file memoization.

Run:  python3 tests/test_hf_adopt.py -v
"""

import contextlib
import hashlib
import importlib.machinery
import importlib.util
import io
import json
import os
import sys
import unittest
from pathlib import Path
from unittest import mock

SCRIPT = Path(__file__).resolve().parents[1] / "droste-hf-adopt"
loader = importlib.machinery.SourceFileLoader("droste_hf_adopt", str(SCRIPT))
spec = importlib.util.spec_from_loader("droste_hf_adopt", loader)
mod = importlib.util.module_from_spec(spec)
loader.exec_module(mod)

REV_A = "a" * 40
REV_B = "b" * 40


# ------------------------------------------------------------------ fixture builders

def git_sha1(content: bytes) -> str:
    return hashlib.sha1(b"blob %d\x00" % len(content) + content).hexdigest()


def lfs_sibling(rfilename: str, content: bytes) -> dict:
    return {"rfilename": rfilename, "size": len(content),
            "lfs": {"oid": hashlib.sha256(content).hexdigest(),
                    "size": len(content)}}


def small_sibling(rfilename: str, content: bytes) -> dict:
    return {"rfilename": rfilename, "size": len(content),
            "blobId": git_sha1(content)}


def gguf_bytes(kv: dict, pad: bytes = b"") -> bytes:
    """Minimal valid GGUF v3 header: magic, 0 tensors, string kvs only."""
    def s(x: str) -> bytes:
        b = x.encode()
        return len(b).to_bytes(8, "little") + b
    out = b"GGUF" + (3).to_bytes(4, "little") \
        + (0).to_bytes(8, "little") + len(kv).to_bytes(8, "little")
    for k, v in kv.items():
        out += s(k) + (8).to_bytes(4, "little") + s(v)
    return out + pad


class Fixture:
    """Synthetic API fixture dir + hub cache + local download dir."""

    def __init__(self, root: Path):
        self.root = root
        self.api = root / "api-fixture"
        self.cache = root / "hub"
        self.downloads = root / "downloads"
        for d in (self.api, self.cache, self.downloads):
            d.mkdir(parents=True)

    def add_manifest(self, repo: str, siblings: list, sha: str = REV_A):
        org, name = repo.split("/")
        (self.api / f"{org}__{name}.json").write_text(
            json.dumps({"sha": sha, "siblings": siblings}))

    def set_search(self, data):
        (self.api / "search.json").write_text(json.dumps(data))

    def add_download(self, relpath: str, content: bytes) -> Path:
        f = self.downloads / relpath
        f.parent.mkdir(parents=True, exist_ok=True)
        f.write_bytes(content)
        return f

    def env(self) -> dict:
        env = {k: v for k, v in os.environ.items()
               if k not in ("HF_TOKEN", "HF_HUB_CACHE", "HF_HOME")}
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

    def blob(self, repo: str, content: bytes) -> Path:
        return (self.cache / ("models--" + repo.replace("/", "--"))
                / "blobs" / hashlib.sha256(content).hexdigest())

    def ref(self, repo: str) -> Path:
        return (self.cache / ("models--" + repo.replace("/", "--"))
                / "refs" / "main")


class AdoptTest(unittest.TestCase):
    def setUp(self):
        import tempfile
        self._tmp = tempfile.TemporaryDirectory(prefix="hf-adopt-test-")
        self.fx = Fixture(Path(self._tmp.name))
        self.addCleanup(self._tmp.cleanup)

    # -------------------------------------------------- --repo mode (regression)
    def test_repo_mode_adopts_dry_run_then_apply(self):
        content = b"weights-weights-weights"
        f = self.fx.add_download("model-q4_k_m.gguf", content)
        self.fx.add_manifest("acme/tiny", [
            lfs_sibling("model-q4_k_m.gguf", content),
            small_sibling("config.json", b"{}"),
        ])
        rc, out, err = self.fx.run("--repo", "acme/tiny", str(f))
        self.assertEqual(rc, 0, err)
        self.assertIn("DRY RUN", out)
        self.assertIn("ADOPT", out)
        self.assertIn("1 adopted, 0 already cached, 0 refused", out)
        self.assertFalse(self.fx.blob("acme/tiny", content).exists())

        rc, out, err = self.fx.run("--apply", "--repo", "acme/tiny", str(f))
        self.assertEqual(rc, 0, err)
        blob = self.fx.blob("acme/tiny", content)
        self.assertEqual(blob.read_bytes(), content)
        link = (self.fx.cache / "models--acme--tiny" / "snapshots" / REV_A
                / "model-q4_k_m.gguf")
        self.assertTrue(link.is_symlink())
        self.assertEqual(link.resolve(), blob.resolve())
        self.assertEqual(self.fx.ref("acme/tiny").read_text(), REV_A)
        self.assertTrue(f.exists())  # --link never removes the source

    def test_repo_mode_refuses_nonmember(self):
        f = self.fx.add_download("random.bin", b"not repo content")
        self.fx.add_manifest("acme/tiny", [lfs_sibling("real.bin", b"real")])
        rc, out, err = self.fx.run("--repo", "acme/tiny", str(f))
        self.assertEqual(rc, 1)
        self.assertIn("REFUSE", out)
        self.assertIn("no size match", out)

    def test_repo_mode_never_searches(self):
        # no search.json in the fixture dir: touching the search endpoint
        # would die with 'fixture read failed' (rc 2)
        content = b"repo-mode-content"
        f = self.fx.add_download("thing.bin", content)
        self.fx.add_manifest("acme/tiny", [lfs_sibling("thing.bin", content)])
        rc, out, err = self.fx.run("--repo", "acme/tiny", str(f))
        self.assertEqual(rc, 0, err)
        self.assertNotIn("IDENTIFIED", out)

    def test_repo_must_look_like_org_name(self):
        f = self.fx.add_download("x.bin", b"x")
        rc, out, err = self.fx.run("--repo", "not-a-repo", str(f))
        self.assertEqual(rc, 2)
        self.assertIn("org/name", err)

    # -------------------------------------------------------- identify: basics
    def test_identify_single_candidate(self):
        content = b"unique gguf payload " * 4
        f = self.fx.add_download("tinymodel-q4_k_m.gguf", content)
        self.fx.set_search([{"id": "acme/tinymodel-GGUF", "downloads": 10}])
        self.fx.add_manifest("acme/tinymodel-GGUF",
                             [lfs_sibling("tinymodel-q4_k_m.gguf", content)])
        rc, out, err = self.fx.run(str(f))
        self.assertEqual(rc, 0, err)
        self.assertIn(f"IDENTIFIED {f} -> acme/tinymodel-GGUF @ {REV_A[:7]} "
                      f"(via search; 1 candidate(s) tried)", out)
        self.assertIn("ADOPT", out)
        self.assertIn("1 adopted", out)

        rc, out, err = self.fx.run("--apply", str(f))
        self.assertEqual(rc, 0, err)
        self.assertTrue(self.fx.blob("acme/tinymodel-GGUF", content).exists())
        self.assertEqual(self.fx.ref("acme/tinymodel-GGUF").read_text(), REV_A)

    def test_identify_multiple_matches_downloads_tiebreak(self):
        content = b"shared across a mirror and the official repo"
        f = self.fx.add_download("model.safetensors", content)
        # lower-downloads candidate listed FIRST: the tiebreak must reorder
        self.fx.set_search([{"id": "mirror/copy", "downloads": 5},
                            {"id": "official/model", "downloads": 500}])
        sib = [lfs_sibling("model.safetensors", content)]
        self.fx.add_manifest("mirror/copy", sib, sha=REV_B)
        self.fx.add_manifest("official/model", sib, sha=REV_A)
        rc, out, err = self.fx.run(str(f))
        self.assertEqual(rc, 0, err)
        self.assertIn("-> official/model @", out)
        self.assertIn("also matches: mirror/copy", out)
        self.assertIn("via search; 2 candidate(s) tried", out)

    def test_identify_miss_refuses(self):
        f = self.fx.add_download("mystery.bin", b"nobody publishes this")
        self.fx.set_search([{"id": "acme/other", "downloads": 3}])
        self.fx.add_manifest("acme/other",
                             [lfs_sibling("other.bin", b"different stuff")])
        rc, out, err = self.fx.run(str(f))
        self.assertEqual(rc, 1)
        self.assertIn(f"REFUSE  {f}: could not identify a HF repo containing "
                      f"this exact content (tried 1 repo(s)); pass --repo "
                      f"explicitly or place it in /opt/models", out)
        self.assertIn("0 adopted, 0 already cached, 1 refused", out)

    def test_identify_no_search_hits_refuses(self):
        f = self.fx.add_download("obscure-thing.bin", b"zzz")
        self.fx.set_search([])  # every broadened query comes back empty
        rc, out, err = self.fx.run(str(f))
        self.assertEqual(rc, 1)
        self.assertIn("tried 0 repo(s)", out)

    # ---------------------------------------------------- identify: directories
    def test_mixed_repo_directory(self):
        content_a = b"file that belongs to repo A"
        content_b = b"file that belongs to repo B, different bytes"
        fa = self.fx.add_download("a-model.gguf", content_a)
        fb = self.fx.add_download("b-model.gguf", content_b)
        self.fx.set_search({"*": [{"id": "org/repo-a", "downloads": 9},
                                  {"id": "org/repo-b", "downloads": 8}]})
        self.fx.add_manifest("org/repo-a",
                             [lfs_sibling("a-model.gguf", content_a)],
                             sha=REV_A)
        self.fx.add_manifest("org/repo-b",
                             [lfs_sibling("b-model.gguf", content_b)],
                             sha=REV_B)
        rc, out, err = self.fx.run("--apply", str(self.fx.downloads))
        self.assertEqual(rc, 0, err)
        self.assertIn(f"IDENTIFIED {fa} -> org/repo-a @ {REV_A[:7]}", out)
        self.assertIn(f"IDENTIFIED {fb} -> org/repo-b @ {REV_B[:7]}", out)
        self.assertIn("2 adopted", out)
        self.assertTrue(self.fx.blob("org/repo-a", content_a).exists())
        self.assertTrue(self.fx.blob("org/repo-b", content_b).exists())
        self.assertEqual(self.fx.ref("org/repo-a").read_text(), REV_A)
        self.assertEqual(self.fx.ref("org/repo-b").read_text(), REV_B)

    # -------------------------------------------------------- identify: errors
    def test_search_error_refuses_per_file(self):
        f1 = self.fx.add_download("one.bin", b"one")
        f2 = self.fx.add_download("two.bin", b"two")
        self.fx.set_search({"error": "connection reset"})
        rc, out, err = self.fx.run(str(f1), str(f2))
        self.assertEqual(rc, 1)  # refused, never crashed
        self.assertEqual(out.count("warning: HF search failed"), 2)
        self.assertEqual(
            out.count("REFUSE"), 2, out)
        self.assertIn("HF search unavailable; pass --repo explicitly", out)
        self.assertIn("0 adopted, 0 already cached, 2 refused", out)

    def test_candidate_manifest_error_skips_to_next(self):
        content = b"content in the second candidate only"
        f = self.fx.add_download("some-model.bin", content)
        # ghost/missing has no manifest fixture -> fetch fails -> skipped
        self.fx.set_search([{"id": "ghost/missing", "downloads": 999},
                            {"id": "real/repo", "downloads": 5}])
        self.fx.add_manifest("real/repo",
                             [lfs_sibling("some-model.bin", content)])
        rc, out, err = self.fx.run(str(f))
        self.assertEqual(rc, 0, err)
        self.assertIn("-> real/repo @", out)
        self.assertIn("via search; 1 candidate(s) tried", out)

    # ------------------------------------------------------------ gguf hint
    def test_gguf_hint_short_circuits_search(self):
        # no search.json at all: reaching the search endpoint would rc-2 die
        content = gguf_bytes(
            {"general.architecture": "llama",
             "general.source.huggingface.repository": "acme/gguf-home"},
            pad=b"tensor-data")
        f = self.fx.add_download("renamed-beyond-recognition.gguf", content)
        self.fx.add_manifest("acme/gguf-home",
                             [lfs_sibling("original-name.gguf", content)])
        rc, out, err = self.fx.run(str(f))
        self.assertEqual(rc, 0, err)
        self.assertIn(f"-> acme/gguf-home @ {REV_A[:7]} "
                      f"(via gguf hint; 1 candidate tried)", out)

    def test_gguf_hint_wrong_falls_through_to_search(self):
        content = gguf_bytes(
            {"general.source.huggingface.repository": "stale/moved"},
            pad=b"other-tensor-data")
        f = self.fx.add_download("model.gguf", content)
        self.fx.set_search([{"id": "fresh/home", "downloads": 7}])
        # stale hint repo exists but does NOT contain the content
        self.fx.add_manifest("stale/moved", [lfs_sibling("x.gguf", b"nope")])
        self.fx.add_manifest("fresh/home", [lfs_sibling("model.gguf", content)])
        rc, out, err = self.fx.run(str(f))
        self.assertEqual(rc, 0, err)
        self.assertIn("-> fresh/home @", out)
        self.assertIn("via search", out)

    def test_gguf_repo_hint_unit(self):
        d = self.fx.downloads
        url_form = d / "url.gguf"
        url_form.write_bytes(gguf_bytes(
            {"general.source.url": "https://huggingface.co/org/name/tree/main",
             "general.name": "Nice Model"}))
        self.assertEqual(mod.gguf_repo_hint(url_form), ("org/name", "Nice Model"))
        plain = d / "plain.gguf"
        plain.write_bytes(gguf_bytes({"general.architecture": "llama"}))
        self.assertEqual(mod.gguf_repo_hint(plain), (None, None))
        notgguf = d / "not.gguf"
        notgguf.write_bytes(b"just bytes")
        self.assertEqual(mod.gguf_repo_hint(notgguf), (None, None))
        truncated = d / "trunc.gguf"
        truncated.write_bytes(gguf_bytes(
            {"general.source.huggingface.repository": "a/b"})[:30])
        self.assertEqual(mod.gguf_repo_hint(truncated), (None, None))

    # -------------------------------------------------------- unit: term ladder
    def test_derive_term_sets(self):
        self.assertEqual(
            mod.derive_term_sets("qwen2.5-0.5b-instruct-q4_k_m.gguf"),
            ["qwen2.5-0.5b-instruct-q4_k_m",
             "qwen2 5 0 5b instruct",
             "qwen2 5 0 5b"])
        self.assertEqual(mod.derive_term_sets("llama-3-8b-fp16.safetensors"),
                         ["llama-3-8b-fp16", "llama 3 8b", "llama 3"])
        # no quant tokens: full stem, then trailing token dropped
        self.assertEqual(mod.derive_term_sets("stable-diffusion-xl.bin"),
                         ["stable-diffusion-xl", "stable diffusion"])
        self.assertEqual(mod.derive_term_sets("model.bin"), ["model"])

    # -------------------------------------------------------- hash memoization
    def test_file_hashed_once_across_identify_and_adopt(self):
        content = b"hash me exactly once please"
        f = self.fx.add_download("once.bin", content)
        # two candidates with a matching size force two hash_matches calls,
        # then adopt_group needs the digests again
        self.fx.set_search([{"id": "a/one", "downloads": 2},
                            {"id": "b/two", "downloads": 1}])
        self.fx.add_manifest("a/one", [lfs_sibling("once.bin", content)])
        self.fx.add_manifest("b/two", [lfs_sibling("other.bin",
                                                   b"x" * len(content))])
        real = mod.hash_file
        with mock.patch.object(mod, "hash_file", side_effect=real) as h:
            rc, out, err = self.fx.run(str(f))
        self.assertEqual(rc, 0, err)
        self.assertIn("-> a/one @", out)
        self.assertEqual(h.call_count, 1)

    # ------------------------------------------------------------------- misc
    def test_help_exits_zero(self):
        with self.assertRaises(SystemExit) as cm:
            with contextlib.redirect_stdout(io.StringIO()):
                mod.main(["--help"])
        self.assertEqual(cm.exception.code, 0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
