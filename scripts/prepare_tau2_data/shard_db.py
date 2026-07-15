# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
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

"""Shard a tau2-style ``db.json`` into per-table files under ``db/``.

Some upstream DBs exceed the 5 MB per-file cap in the repository
(notably ``tau2_airline/db.json`` at 6.8 MB). Splitting top-level keys
into individual ``db/<table>.json`` files keeps the repos byte-identical
at the in-memory level while staying well under the limit.

Idempotent: re-running on an already-sharded directory rebuilds the
shards from the source file if one is present, otherwise no-ops.

Usage::

    python scripts/prepare_tau2_data/shard_db.py nemo_voice_agent/evaluation/data/tau2_airline/db.json
    # → writes nemo_voice_agent/evaluation/data/tau2_airline/db/{flights,users,reservations}.json
    # → deletes the original db.json (unless --keep-source is passed)

The matching loader is ``nemo_voice_agent.evaluation.load_db_artifact``
which probes ``<path>.json`` then ``<path>/`` so both layouts work.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def shard(src: Path, keep_source: bool = False) -> Path:
    if src.suffix != ".json":
        raise ValueError(f"Expected a .json file, got {src}")
    if not src.is_file():
        raise FileNotFoundError(src)
    data = json.loads(src.read_text())
    if not isinstance(data, dict):
        raise ValueError(f"{src} top-level must be a dict; got {type(data).__name__}")

    out_dir = src.with_suffix("")
    out_dir.mkdir(exist_ok=True)
    for key, value in data.items():
        if not isinstance(key, str) or "/" in key or key.startswith("."):
            raise ValueError(f"Unsafe top-level key for sharding: {key!r}")
        shard_path = out_dir / f"{key}.json"
        shard_path.write_text(json.dumps(value, indent=2) + "\n")
        size_mb = shard_path.stat().st_size / 1024 / 1024
        print(f"  wrote {shard_path}  ({size_mb:.2f} MB)")

    if not keep_source:
        src.unlink()
        print(f"  removed {src}")
    return out_dir


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("path", type=Path, help="Path to db.json to shard")
    ap.add_argument(
        "--keep-source",
        action="store_true",
        help="Keep the original db.json (default: delete after sharding)",
    )
    args = ap.parse_args(argv)
    out = shard(args.path, keep_source=args.keep_source)
    print(f"Sharded into {out}/")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
