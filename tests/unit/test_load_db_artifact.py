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

"""Tests for ``load_db_artifact`` — the shared loader that transparently
handles single-file ``db.json`` and sharded ``db/<table>.json`` layouts.

The sharded layout exists so ``tau2_airline/db.json`` (6.8 MB) can be
mirrored to GitLab (5 MB per-file cap) without LFS. Both shapes must
produce byte-identical in-memory dicts so the matching DB hash remains
stable across the migration.
"""

import json

import pytest

from nemo_voice_agent.evaluation import get_eval_data_root, load_db_artifact
from nemo_voice_agent.evaluation.db_hash import get_dict_hash


def test_load_db_artifact_single_file(tmp_path):
    db = {"users": {"u1": {}}, "orders": {"#A": {}}}
    f = tmp_path / "db.json"
    f.write_text(json.dumps(db))
    assert load_db_artifact(tmp_path / "db") == db
    # ``.json``-suffixed input is also accepted (suffix stripped before probing).
    assert load_db_artifact(tmp_path / "db.json") == db


def test_load_db_artifact_sharded_dir(tmp_path):
    d = tmp_path / "db"
    d.mkdir()
    (d / "users.json").write_text(json.dumps({"u1": {}}))
    (d / "orders.json").write_text(json.dumps({"#A": {}}))
    out = load_db_artifact(tmp_path / "db")
    assert out == {"users": {"u1": {}}, "orders": {"#A": {}}}


def test_load_db_artifact_single_file_and_sharded_match(tmp_path):
    """Both layouts produce the same in-memory dict → same DB hash."""
    db = {"users": {"u1": {"name": "x"}}, "orders": {"#A": {"qty": 1}}}
    # Layout 1: single file
    f1 = tmp_path / "single" / "db.json"
    f1.parent.mkdir()
    f1.write_text(json.dumps(db))
    # Layout 2: sharded
    d2 = tmp_path / "sharded" / "db"
    d2.mkdir(parents=True)
    for k, v in db.items():
        (d2 / f"{k}.json").write_text(json.dumps(v))
    assert load_db_artifact(f1.parent / "db") == load_db_artifact(d2.parent / "db")
    assert get_dict_hash(load_db_artifact(f1.parent / "db")) == get_dict_hash(load_db_artifact(d2.parent / "db"))


def test_load_db_artifact_missing_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_db_artifact(tmp_path / "nope")


def test_tau2_airline_db_loads_via_sharded_layout():
    """Regression: the live tau2_airline DB is sharded on disk; the loader
    must reassemble it into the same shape the scenario base expects.

    The 3 expected tables are ``flights``, ``users``, ``reservations``.
    """
    db = load_db_artifact(get_eval_data_root() / "tau2_airline" / "db")
    assert set(db.keys()) == {"flights", "users", "reservations"}
    assert isinstance(db["flights"], dict) and len(db["flights"]) > 0
    assert isinstance(db["users"], dict) and len(db["users"]) > 0
    assert isinstance(db["reservations"], dict) and len(db["reservations"]) > 0
