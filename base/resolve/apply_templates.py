#!/usr/bin/env python3
"""apply_templates.py — seed baked default files into runtime locations.

Shared template applier baked into the runtime base image. Reads a manifest
``templates.yaml`` in the templates directory and applies two seeding rules.

RESTRICTED YAML SUBSET (intentional — NO pyyaml dependency in the lean base):
This parser understands ONLY this exact shape, and keeps the ``.yaml`` extension
so it can be swapped for a real YAML loader later without touching callers:

    # comment lines and blank lines are ignored
    if_empty:
      src_dir: /absolute/dest/dir
    if_missing:
      src_file.yaml: /absolute/dest/file.yaml

  * Exactly two top-level keys: ``if_empty`` and ``if_missing`` (either may be
    absent). Each is a FLAT one-level map of ``src: dest``.
  * ``src`` is relative to the templates directory (this script's argv[1]).
  * ``dest`` is an absolute path.
  * ``#`` full-line comments and blank lines are ignored. No nesting, no lists,
    no quoting, no inline structures.

SEMANTICS:
  if_empty   — copy src (dir or file) into dest IFF dest is a COMPLETELY EMPTY
               directory. "Empty" = no entries at all (dotfiles count as content).
               A missing dest counts as empty (created). Any entry → skip.
  if_missing — copy src to dest IFF dest does NOT exist. Never overwrites.

Copies preserve tree structure (shutil.copytree/copy2). Idempotent; prints one
line per action, silent when there is nothing to do.

Exit status: 0 on success or no-op (a missing manifest is fine). Nonzero only on
real errors (e.g. a manifest src that does not exist).
"""

import os
import shutil
import sys

TOOL = "apply_templates"
SECTIONS = ("if_empty", "if_missing")


def warn(msg):
    print(f"{TOOL}: {msg}", file=sys.stderr)


def parse_manifest(path):
    """Parse the restricted-subset manifest into {section: {src: dest}}."""
    sections = {name: {} for name in SECTIONS}
    current = None
    with open(path, encoding="utf-8") as fh:
        for lineno, raw in enumerate(fh, 1):
            line = raw.rstrip("\n")
            stripped = line.strip()
            if not stripped or stripped.startswith("#"):
                continue
            if not line[0].isspace():
                # top-level key, e.g. "if_empty:"
                key = stripped.rstrip(":").strip()
                if key not in sections:
                    raise ValueError(
                        f"{path}:{lineno}: unknown top-level key '{key}' "
                        f"(expected one of {', '.join(SECTIONS)})"
                    )
                current = key
                continue
            # indented "src: dest" entry
            if current is None:
                raise ValueError(f"{path}:{lineno}: entry before any section header")
            if ":" not in stripped:
                raise ValueError(f"{path}:{lineno}: expected 'src: dest', got '{stripped}'")
            src, _, dest = stripped.partition(":")
            sections[current][src.strip()] = dest.strip()
    return sections


def is_empty_dir(dest):
    """True if dest is missing, or an existing directory with no entries."""
    if not os.path.exists(dest):
        return True
    if os.path.isdir(dest):
        return len(os.listdir(dest)) == 0
    return False  # an existing file is not an empty dir


def copy_tree_or_file(src, dest):
    """Copy src (dir or file) to dest, preserving structure. src must exist."""
    if os.path.isdir(src):
        shutil.copytree(src, dest, dirs_exist_ok=True)
    elif os.path.isfile(src):
        parent = os.path.dirname(dest)
        if parent:
            os.makedirs(parent, exist_ok=True)
        shutil.copy2(src, dest)
    else:
        raise FileNotFoundError(src)


def apply(templates_dir):
    manifest = os.path.join(templates_dir, "templates.yaml")
    if not os.path.isfile(manifest):
        # No manifest is a legitimate no-op.
        return 0

    sections = parse_manifest(manifest)

    # if_empty: seed only into a completely empty (or missing) destination.
    for src, dest in sections["if_empty"].items():
        src_path = os.path.join(templates_dir, src)
        if not os.path.exists(src_path):
            warn(f"if_empty src does not exist: {src_path}")
            return 1
        if not is_empty_dir(dest):
            continue
        copy_tree_or_file(src_path, dest)
        print(f"{TOOL}: seeded {dest} from {src} (if_empty)")

    # if_missing: copy only when the destination does not exist. Never overwrite.
    for src, dest in sections["if_missing"].items():
        src_path = os.path.join(templates_dir, src)
        if not os.path.exists(src_path):
            warn(f"if_missing src does not exist: {src_path}")
            return 1
        if os.path.exists(dest):
            continue
        copy_tree_or_file(src_path, dest)
        print(f"{TOOL}: created {dest} from {src} (if_missing)")

    return 0


def main(argv):
    templates_dir = argv[1] if len(argv) > 1 else "/opt/resources/templates"
    try:
        return apply(templates_dir)
    except (ValueError, OSError) as exc:
        warn(str(exc))
        return 1


if __name__ == "__main__":
    sys.exit(main(sys.argv))
