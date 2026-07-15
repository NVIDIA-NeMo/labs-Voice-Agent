# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
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

import json
import os
from pathlib import Path
from typing import Any, Dict, Union


def get_eval_data_root() -> Path:
    """Resolve the root directory for evaluation fixture data.

    Checks `$EVAL_DATA_ROOT` first; falls back to packaged data under
    `nemo_voice_agent/evaluation/data`, then the legacy `<repo>/evaluation/data`
    layout.
    Lazy (function, not module constant) so env-var changes after import take
    effect — useful for tests and for bridge/server processes setting it
    differently.

    Convention: `db_path` and similar fixture-path values stored in
    `shared_state_init` are always **relative** to this root, so bridge and
    server can resolve to different absolute roots.
    """
    if env := os.environ.get("EVAL_DATA_ROOT"):
        return Path(env)
    package_data = Path(__file__).resolve().parent / "data"
    if package_data.exists():
        return package_data
    # parents[2]: __init__.py -> evaluation -> nemo_voice_agent -> repo root
    return Path(__file__).resolve().parents[2] / "evaluation" / "data"


def load_db_artifact(path: Union[str, Path]) -> Dict[str, Any]:
    """Load a tau2-style DB artifact, transparently handling sharded form.

    Some upstream ``db.json`` files exceed file size limits (e.g.,
    5 MB per-file cap), so they're sharded into ``<name>/<table>.json``
    one-file-per-top-level-key. This helper accepts either form:

    - ``<path>.json`` file → ``json.loads(read_text())``
    - ``<path>/`` directory → ``{stem: json.loads(read_text())}`` for
      every ``*.json`` file under it (top-level only; no recursion).

    The two layouts are byte-equivalent at the in-memory ``dict`` level
    so callers (gold replay, bot-side state, predicates) don't care
    which on-disk shape a given domain uses.

    The caller passes a path WITHOUT the ``.json`` suffix (the artifact
    name). For backward compatibility with callers that still include
    ``.json``, the suffix is stripped before probing.

    Raises ``FileNotFoundError`` if neither form exists.
    """
    p = Path(path)
    if p.suffix == ".json":
        p = p.with_suffix("")
    file_form = p.with_suffix(".json")
    if file_form.is_file():
        return json.loads(file_form.read_text())
    if p.is_dir():
        out: Dict[str, Any] = {}
        for shard in sorted(p.glob("*.json")):
            out[shard.stem] = json.loads(shard.read_text())
        return out
    raise FileNotFoundError(f"No DB artifact at {file_form} or {p}/ (tried single-file and sharded forms).")
