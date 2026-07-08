#!/usr/bin/env python3
"""gen_vllm_config.py — emit the default vllm_config.yaml at IMAGE BUILD time.

Runs DURING the vLLM image build (see targets/Container.vllm). It parses the
VENDORED, PINNED upstream MODEL_TABLE (targets/vllm/upstream/models.py, copied into
the image) and writes /opt/resources/templates/vllm_config.yaml — the template that
templates.yaml seeds to /opt/data/vllm_config.yaml on first run (if_missing).

The generated YAML mirrors `vllm serve` CLI args (each key = a long flag without the
leading `--`). Everything is COMMENTED except the active serving defaults (host/port):
the user uncomments a MODEL_TABLE stanza (or writes their own) to choose a model.
vLLM refuses to start without a model, so leaving it commented is a self-explanatory
"you must pick a model" gate rather than a silent wrong default.

Hermetic: parses the vendored file (no network at build). Drift is visible in git via
the vendored models.py. The MODEL_TABLE is executed (not regex-parsed) so structured
fields (valid_tp, env, extra_flags, ...) come straight from the upstream dict.
"""

import argparse
import os
import runpy
import sys

DEFAULT_MODELS = "/opt/resources/scripts/vllm_models_pinned.py"
DEFAULT_OUT = "/opt/resources/templates/vllm_config.yaml"

# Serving defaults that are ACTIVE (uncommented) in the emitted config.
ACTIVE_HOST = "0.0.0.0"
ACTIVE_PORT = 8000


def load_model_table(models_path):
    """Execute the vendored models.py and return (MODEL_TABLE, globals-dict)."""
    ns = runpy.run_path(models_path)
    table = ns.get("MODEL_TABLE")
    if not isinstance(table, dict):
        raise SystemExit(f"{models_path}: MODEL_TABLE dict not found")
    return table, ns


def parse_extra_flags(flags):
    """Turn upstream extra_flags (argv-style list) into [(key, value_or_None)].

    A token starting with `--` is a flag; if the NEXT token is not a flag it is that
    flag's value, else the flag is boolean (value None -> `key: true`).
    """
    out = []
    i = 0
    n = len(flags)
    while i < n:
        tok = flags[i]
        if tok.startswith("--"):
            key = tok[2:]
            if i + 1 < n and not flags[i + 1].startswith("--"):
                out.append((key, flags[i + 1]))
                i += 2
            else:
                out.append((key, None))
                i += 1
        else:
            # stray positional; skip defensively
            i += 1
    return out


def emit_header(out):
    out.append("# vllm_config.yaml — vLLM serve configuration (passed as `vllm serve --config <this>`).")
    out.append("#")
    out.append("# This YAML mirrors the `vllm serve` command-line arguments: each key is a long")
    out.append("# CLI flag without the leading `--` (e.g. `max-model-len:` == `--max-model-len`).")
    out.append("# These values are the LOWEST precedence — anything you pass on the command line or")
    out.append("# via $VLLM_EXTRA_ARGS overrides what is set here (vLLM's config-merge order).")
    out.append("#")
    out.append("# The container serves THIS file by default ($VLLM_CONFIG=/opt/data/vllm_config.yaml).")
    out.append("# Edit it in place; it lives on the persisted /opt/data volume.")
    out.append("#")
    out.append("# Stale compiled-graph note: this image sets VLLM_DISABLE_COMPILE_CACHE=1, so vLLM")
    out.append("# does not persist torch.compile graphs. If vLLM crashes right after a version bump,")
    out.append("# clear any leftover JIT cache with:  rm -rf ~/.cache/vllm")
    out.append("#")
    out.append("# " + "-" * 75)
    out.append("# REQUIRED: choose a model. vLLM will NOT start until `model:` is set. Uncomment one")
    out.append("# MODEL_TABLE stanza below (or add your own line); a model is a HuggingFace repo id")
    out.append("# or an absolute local path:")
    out.append("#")
    out.append("# model: meta-llama/Meta-Llama-3.1-8B-Instruct   # REQUIRED — vllm won't start until set")
    out.append("# " + "-" * 75)
    out.append("")
    out.append("# ── Active serving defaults (uncommented = in effect) ────────────────────────")
    out.append(f'host: "{ACTIVE_HOST}"')
    out.append(f"port: {ACTIVE_PORT}")
    out.append("")
    out.append("# Suggested global defaults (uncomment to apply; from the upstream toolbox):")
    out.append("# gpu-memory-utilization: 0.90")
    out.append("# max-num-batched-tokens: 8192")
    out.append("")


def emit_model(out, repo, spec):
    out.append("# " + "=" * 75)
    out.append(f"# {repo}")

    valid_tp = spec.get("valid_tp")
    if valid_tp:
        out.append(f"#   valid tensor-parallel sizes on Strix Halo: {valid_tp}")

    env = spec.get("env") or {}
    if env:
        out.append("#   env (shell vars — set BEFORE launch, they cannot live in this yaml):")
        for k, v in env.items():
            out.append(f"#     export {k}={v}")
    else:
        out.append("#   env (shell vars — set BEFORE launch): <none>")

    ctx = spec.get("ctx")
    if ctx:
        out.append(f"#   ctx (upstream field): {ctx}  (raise max-model-len below to use full context)")

    out.append("# " + "-" * 75)
    out.append(f"# model: {repo}")

    tp_default = valid_tp[0] if valid_tp else 1
    out.append(f"# tensor-parallel-size: {tp_default}")

    if spec.get("max_num_seqs") is not None:
        out.append(f"# max-num-seqs: {spec['max_num_seqs']}")
    if spec.get("max_tokens") is not None:
        out.append(f"# max-model-len: {spec['max_tokens']}")

    if "trust_remote" in spec:
        out.append(f"# trust-remote-code: {str(bool(spec['trust_remote'])).lower()}")
    if spec.get("enforce_eager"):
        out.append("# enforce-eager: true")

    for key, val in parse_extra_flags(spec.get("extra_flags") or []):
        if val is None:
            out.append(f"# {key}: true")
        else:
            out.append(f"# {key}: {val}")

    out.append("")


def generate(models_path):
    table, _ = load_model_table(models_path)
    out = []
    emit_header(out)
    out.append("# ══ MODEL_TABLE — verified Strix Halo models (uncomment ONE stanza) ══════════")
    out.append(f"#   harvested at build from the pinned upstream scripts/models.py ({len(table)} models)")
    out.append("")
    for repo, spec in table.items():
        emit_model(out, repo, spec)
    return "\n".join(out) + "\n"


def main(argv):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--models", default=DEFAULT_MODELS,
                    help=f"vendored models.py to parse (default: {DEFAULT_MODELS})")
    ap.add_argument("--out", default=DEFAULT_OUT,
                    help=f"output yaml path (default: {DEFAULT_OUT})")
    args = ap.parse_args(argv)

    if not os.path.isfile(args.models):
        raise SystemExit(f"models file not found: {args.models}")

    text = generate(args.models)
    out_dir = os.path.dirname(args.out)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as fh:
        fh.write(text)
    print(f"gen_vllm_config: wrote {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
