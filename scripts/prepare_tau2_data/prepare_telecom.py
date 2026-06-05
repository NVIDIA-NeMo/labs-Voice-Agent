#!/usr/bin/env python3
# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Prepare tau2-bench telecom domain data for the eval framework.

Reads upstream tau2-bench's ``data/tau2/domains/telecom/`` and writes a
normalized copy under ``evaluation/data/tau2_telecom/``. Three transformations:

1. ``db.toml`` and ``user_db.toml`` are parsed via ``tomllib`` (stdlib in
   Python 3.11+) and re-serialized as JSON so the bot-side loader in
   ``rtvi_actions.create_update_system_prompt_action`` can use the same
   ``json.load`` path as airline / retail. Avoids adding a TOML branch on
   the bot side.
2. ``tasks.json`` is filtered to only the 114 entries whose ``id`` is in
   ``split_tasks.json["base"]`` (14 MB → ~660 KB). The 2171 non-base task
   definitions are dead weight in our repo; the base-split eval surface
   never references them.
3. ``tasks_voice.json`` is filtered to only the 114 base ids in its
   ``configs`` map (62 MB → ~3 MB). Hard-fails if any base id is missing
   from upstream's tasks_voice — guards against silently dropping
   eval-surface ids.

All other files (split_tasks.json, audio_difficulty.json, *.md,
workflows/) are copied verbatim. ``split_tasks.json`` keeps all split
keys for debug context (~336 KB). ``audio_difficulty.json`` (886 KB) is
traceability-only (not consumed by the eval pipeline today); its
``tasks`` sub-dict keys are positional indices, not task IDs, so a
base-ID filter doesn't apply — keep verbatim until the upstream key
mapping is decoded (if ever needed).

Idempotent: re-running overwrites outputs. Safe to re-run after upstream
re-syncs at the pinned commit.

Usage:

    python scripts/prepare_tau2_data/prepare_telecom.py
    python scripts/prepare_tau2_data/prepare_telecom.py --source /custom/path/tau2-bench
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
import sys
import tomllib
from pathlib import Path
from typing import Optional


# Upstream commit we pin against. M2 / M3 used this for airline + retail; M5+
# uses the same pin for telecom. Bump only when re-syncing to a newer tau2
# voice-user-sim release (and update README.md accordingly).
PINNED_COMMIT = "17e07b1da2bbc0cadfddeea36412686e0604127b"

# Default upstream checkout location. Override via --source.
DEFAULT_SOURCE = Path("/home/heh/github/tau2-bench")

# Files to convert (TOML → JSON). Keys are source filenames; values are
# destination filenames (extension swapped to .json).
TOML_TO_JSON = {
    "db.toml": "db.json",
    "user_db.toml": "user_db.json",
}

# Files to copy verbatim. Includes both data (.json) and policy text (.md).
# ``tasks.json`` / ``tasks_voice.json`` are NOT in this list — they're filtered
# (see ``_filter_tasks``, ``_filter_tasks_voice``). ``audio_difficulty.json``
# IS verbatim — its inner ``tasks`` dict is keyed by positional index, not
# task id, so a base-ID filter doesn't apply.
VERBATIM_FILES = [
    "split_tasks.json",
    "audio_difficulty.json",
    "main_policy.md",
    "tech_support_workflow.md",
    "tech_support_manual.md",
]

# Directories to copy verbatim (recursive).
VERBATIM_DIRS = [
    "workflows",
]


def _verify_commit(source: Path) -> Optional[str]:
    """Return the upstream HEAD commit; warn (don't fail) if it diverges from
    the pinned value. Returns ``None`` if not a git repo or git unavailable."""
    try:
        result = subprocess.run(
            ["git", "-C", str(source), "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
        )
        return result.stdout.strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return None


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()[:12]


def _convert_toml_to_json(src: Path, dst: Path) -> None:
    """Parse a TOML file and re-serialize as indented JSON.

    Indent=2 + sort_keys=False to keep the output diff-readable. We avoid
    sort_keys because tomllib already returns a dict in document order;
    re-sorting would mask upstream re-orderings during re-sync diffs.
    """
    with src.open("rb") as f:
        data = tomllib.load(f)
    dst.parent.mkdir(parents=True, exist_ok=True)
    with dst.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, sort_keys=False)
        f.write("\n")


def _copy_file(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)


def _copy_dir(src: Path, dst: Path) -> None:
    if dst.exists():
        shutil.rmtree(dst)
    shutil.copytree(src, dst)


def _filter_tasks(src: Path, dst: Path, base_ids: set[str]) -> tuple[int, int]:
    """Filter ``tasks.json`` (a list of task dicts) to entries whose ``id`` is in ``base_ids``.

    Upstream ships 2285 task definitions in a single list; we only need the
    114 base-split ids. Filtering trims ~14 MB → ~660 KB.

    Hard-fails if any base id is missing from upstream's tasks list — that
    would mean upstream is internally inconsistent (split_tasks references
    an id that has no definition).

    Returns ``(n_kept, n_dropped)``.
    """
    with src.open("rb") as f:
        full = json.load(f)
    if not isinstance(full, list):
        raise RuntimeError(f"tasks.json is not a list at top level (got {type(full).__name__})")
    filtered = [t for t in full if t.get("id") in base_ids]
    kept_ids = {t["id"] for t in filtered}
    missing = base_ids - kept_ids
    if missing:
        raise RuntimeError(
            f"tasks.json filter failed — {len(missing)} base-split ids have no "
            f"task definition upstream. First 3 missing: {sorted(missing)[:3]}"
        )
    dst.parent.mkdir(parents=True, exist_ok=True)
    with dst.open("w", encoding="utf-8") as f:
        json.dump(filtered, f, indent=2, sort_keys=False)
        f.write("\n")
    return len(filtered), len(full) - len(filtered)


def _filter_tasks_voice(src: Path, dst: Path, base_ids: set[str]) -> tuple[int, int]:
    """Filter ``tasks_voice.json`` to only entries whose task id is in ``base_ids``.

    Upstream's tasks_voice.json carries acoustic/voice config for ALL 2285
    telecom tasks (~62 MB). We only ship the base-split eval surface, so the
    other 2171 entries are dead weight in our repo. The base split (114 ids)
    is documented in ``split_tasks.json["base"]``; bring-up debug subsets
    like ``split_tasks.json["small"]`` are intentionally NOT included (they
    are upstream-debug-only and disjoint from base by design).

    Hard-fails if any base id is missing from upstream's tasks_voice entries
    — that would mean our eval surface is broken (we'd try to instantiate a
    scenario for which we have no voice config).

    Returns ``(n_kept, n_dropped)``.
    """
    with src.open("rb") as f:
        full = json.load(f)
    full_configs = full.get("configs") or {}
    filtered_configs = {tid: cfg for tid, cfg in full_configs.items() if tid in base_ids}

    missing = base_ids - set(filtered_configs)
    if missing:
        raise RuntimeError(
            f"tasks_voice.json filter failed — {len(missing)} base-split ids have "
            f"no voice config upstream. First 3 missing: {sorted(missing)[:3]}"
        )

    # Preserve top-level keys; replace ``configs`` with the filtered subset.
    filtered = {**full, "configs": filtered_configs}
    dst.parent.mkdir(parents=True, exist_ok=True)
    with dst.open("w", encoding="utf-8") as f:
        json.dump(filtered, f, indent=2, sort_keys=False)
        f.write("\n")
    return len(filtered_configs), len(full_configs) - len(filtered_configs)


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--source",
        type=Path,
        default=DEFAULT_SOURCE,
        help=f"Path to local tau2-bench checkout (default: {DEFAULT_SOURCE})",
    )
    parser.add_argument(
        "--dest",
        type=Path,
        default=None,
        help="Destination root (default: evaluation/data/tau2_telecom/ relative to repo root)",
    )
    args = parser.parse_args(argv)

    # Resolve destination relative to repo root (parent-of-parent of this
    # script: scripts/prepare_tau2_data/prepare_telecom.py → repo root).
    repo_root = Path(__file__).resolve().parents[2]
    dest = args.dest or (repo_root / "evaluation" / "data" / "tau2_telecom")

    source = args.source.resolve()
    source_telecom = source / "data" / "tau2" / "domains" / "telecom"
    if not source_telecom.exists():
        print(f"ERROR: upstream telecom data not found at {source_telecom}", file=sys.stderr)
        print(f"       (looking for tau2-bench checkout under --source {args.source})", file=sys.stderr)
        return 1

    # Soft commit check — warn but don't fail. Lets re-syncs work without
    # editing this script for one-off testing.
    commit = _verify_commit(source)
    if commit is None:
        print(f"WARN: could not verify git commit at {source} (not a git repo?)")
    elif commit != PINNED_COMMIT:
        print(f"WARN: upstream HEAD is {commit}; expected pinned {PINNED_COMMIT}")
        print(f"      (continuing anyway; update PINNED_COMMIT in this script if intentional)")
    else:
        print(f"OK:   upstream tau2-bench at pinned commit {commit[:12]}")

    print(f"Source: {source_telecom}")
    print(f"Dest:   {dest}")
    print()

    # 1. TOML → JSON conversions
    for src_name, dst_name in TOML_TO_JSON.items():
        src = source_telecom / src_name
        dst = dest / dst_name
        if not src.exists():
            print(f"ERROR: required source missing: {src}", file=sys.stderr)
            return 1
        _convert_toml_to_json(src, dst)
        print(f"  convert  {src_name:<32} → {dst_name:<32} ({_sha256(dst)})")

    # 2. Verbatim files
    for name in VERBATIM_FILES:
        src = source_telecom / name
        dst = dest / name
        if not src.exists():
            print(f"WARN: optional file missing: {src} (skipping)")
            continue
        _copy_file(src, dst)
        print(f"  copy     {name:<32} → {name:<32} ({_sha256(dst)})")

    # 2a. Filtered per-task files — keep only base-split entries.
    # split_tasks.json is the source of truth for "base"; it was copied
    # verbatim to dest in step 2 so we can read either side. We read from
    # source for canonical correctness (avoids dest-then-source drift on
    # partial failures).
    split_src = source_telecom / "split_tasks.json"
    with split_src.open() as f:
        base_ids = set(json.load(f).get("base") or [])
    if not base_ids:
        print(f"ERROR: split_tasks.json has no 'base' key or it is empty", file=sys.stderr)
        return 1

    filter_steps = [
        ("tasks.json", _filter_tasks),
        ("tasks_voice.json", _filter_tasks_voice),
    ]
    for name, fn in filter_steps:
        src = source_telecom / name
        dst = dest / name
        if not src.exists():
            print(f"ERROR: required {name} missing at {src}", file=sys.stderr)
            return 1
        n_kept, n_dropped = fn(src, dst, base_ids)
        src_size = src.stat().st_size
        dst_size = dst.stat().st_size
        print(
            f"  filter   {name:<32} → {name:<32} ({_sha256(dst)})  "
            f"kept={n_kept} dropped={n_dropped}  "
            f"size: {src_size:,}B → {dst_size:,}B "
            f"({100*dst_size/src_size:.1f}%)"
        )

    # 3. Verbatim directories
    for name in VERBATIM_DIRS:
        src = source_telecom / name
        dst = dest / name
        if not src.exists():
            print(f"WARN: optional directory missing: {src} (skipping)")
            continue
        _copy_dir(src, dst)
        n_files = sum(1 for _ in dst.rglob("*") if _.is_file())
        print(f"  copydir  {name + '/':<32} → {name + '/':<32} ({n_files} files)")

    print()
    print(f"Done. {len(TOML_TO_JSON)} conversions + {len(VERBATIM_FILES)} files + {len(VERBATIM_DIRS)} dirs.")
    print(f"Output dir: {dest}")
    print()
    print("Next: append a tau2_telecom section to evaluation/data/README.md "
          "(source URL, license, commit pin) if this is a first import.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
