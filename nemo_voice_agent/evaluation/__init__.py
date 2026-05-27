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

import os
from pathlib import Path


def get_eval_data_root() -> Path:
    """Resolve the root directory for evaluation fixture data.

    Checks `$EVAL_DATA_ROOT` first; falls back to `<repo>/evaluation/data`.
    Lazy (function, not module constant) so env-var changes after import take
    effect — useful for tests and for bridge/server processes setting it
    differently.

    Convention: `db_path` and similar fixture-path values stored in
    `shared_state_init` are always **relative** to this root, so bridge and
    server can resolve to different absolute roots.
    """
    if env := os.environ.get("EVAL_DATA_ROOT"):
        return Path(env)
    # parents[2]: __init__.py → evaluation → nemo_voice_agent → repo root
    return Path(__file__).resolve().parents[2] / "evaluation" / "data"
