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

"""Smoke tests for the bridge-pull infrastructure landed in commit A:

  1. ``db_hash.get_dict_hash`` — determinism, ``HASH_EXCLUDED_KEYS`` exclusion,
     ``normalize_for_comparison`` (1.0/1, "none"/None, recursion).
  2. ``db_hash.compute_db_diff`` — structured diff shape on table/record/field
     differences and order-independent list handling.
  3. ``rtvi_actions.create_get_scenario_summary_action`` — handler returns the
     state ref's ``actions`` and ``db`` keys; uninitialized state returns empty.
  4. ``Scenario.expected_scenario_db`` field — default None; set via __init__
     or class attribute; doesn't break existing scenarios.
"""

import asyncio
import sys

import pytest

from nemo_voice_agent.evaluation.db_hash import (
    HASH_EXCLUDED_KEYS,
    ORDER_INDEPENDENT_LIST_FIELDS,
    compute_db_diff,
    get_dict_hash,
    normalize_for_comparison,
)
from nemo_voice_agent.evaluation.scenarios.classes import (
    Actions,
    Persona,
    Resources,
    Scenario,
    SuccessSignal,
    Task,
)
from nemo_voice_agent.pipecat.processors.frameworks.rtvi_actions import (
    SharedStateRef,
    create_get_scenario_summary_action,
)

# ---------------------------------------------------------------------------
# 1. db_hash hashing
# ---------------------------------------------------------------------------


def test_get_dict_hash_deterministic():
    """Same content, different key order → same hash."""
    a = {"x": 1, "y": [1, 2, 3], "z": {"k": "v"}}
    b = {"z": {"k": "v"}, "y": [1, 2, 3], "x": 1}
    assert get_dict_hash(a) == get_dict_hash(b)


def test_get_dict_hash_excludes_session():
    """``HASH_EXCLUDED_KEYS`` keys at the top level don't affect the hash."""
    assert "session" in HASH_EXCLUDED_KEYS
    a = {"x": 1}
    b = {"x": 1, "session": {"id": "abc-123"}}
    assert get_dict_hash(a) == get_dict_hash(b)


def test_get_dict_hash_normalizes_floats_and_strings():
    """1.0 ↔ 1 and 'none' ↔ None don't cause mismatches."""
    a = {"qty": 1, "meal": None, "fares": [1.0, 2.0]}
    b = {"qty": 1.0, "meal": "none", "fares": [1, 2]}
    assert get_dict_hash(a) == get_dict_hash(b)


def test_normalize_for_comparison_recurses():
    src = {"a": [{"b": 1.0, "c": "NULL"}, 2.0], "d": "Null"}
    out = normalize_for_comparison(src)
    assert out == {"a": [{"b": 1, "c": None}, 2], "d": None}


def test_normalize_handles_order_independent_list():
    """Lists under ``ORDER_INDEPENDENT_LIST_FIELDS`` are sorted before serialization."""
    assert "bookings" in ORDER_INDEPENDENT_LIST_FIELDS
    a = {"bookings": [{"id": "X"}, {"id": "Y"}]}
    b = {"bookings": [{"id": "Y"}, {"id": "X"}]}
    assert get_dict_hash(a) == get_dict_hash(b)


def test_get_dict_hash_detects_value_change():
    a = {"x": 1}
    b = {"x": 2}
    assert get_dict_hash(a) != get_dict_hash(b)


# ---------------------------------------------------------------------------
# 2. db_hash diff
# ---------------------------------------------------------------------------


def test_compute_db_diff_identical_returns_empty():
    a = {"reservations": {"R1": {"status": "confirmed"}}}
    b = {"reservations": {"R1": {"status": "confirmed"}}}
    diff = compute_db_diff(a, b)
    assert diff == {"tables_added": [], "tables_removed": [], "tables_modified": {}}


def test_compute_db_diff_table_added_and_removed():
    a = {"reservations": {}, "journeys": {}}
    b = {"reservations": {}, "refunds": {}}
    diff = compute_db_diff(a, b)
    assert diff["tables_added"] == ["refunds"]
    assert diff["tables_removed"] == ["journeys"]
    assert diff["tables_modified"] == {}


def test_compute_db_diff_record_modified_field_value():
    a = {"reservations": {"R1": {"status": "confirmed", "fare_paid": 100}}}
    b = {"reservations": {"R1": {"status": "cancelled", "fare_paid": 100}}}
    diff = compute_db_diff(a, b)
    assert "reservations" in diff["tables_modified"]
    table_diff = diff["tables_modified"]["reservations"]
    assert table_diff["records_added"] == []
    assert table_diff["records_removed"] == []
    assert "R1" in table_diff["records_modified"]
    record_diff = table_diff["records_modified"]["R1"]
    assert "status" in record_diff["fields_modified"]
    assert record_diff["fields_modified"]["status"] == {
        "type": "value_mismatch",
        "expected": "confirmed",
        "actual": "cancelled",
    }


def test_compute_db_diff_record_added():
    a = {"reservations": {"R1": {"status": "confirmed"}}}
    b = {"reservations": {"R1": {"status": "confirmed"}, "R2": {"status": "confirmed"}}}
    diff = compute_db_diff(a, b)
    assert diff["tables_modified"]["reservations"]["records_added"] == ["R2"]


# ---------------------------------------------------------------------------
# 3. SharedStateRef + get_scenario_summary action
# ---------------------------------------------------------------------------


def test_shared_state_ref_default_empty():
    ref = SharedStateRef()
    assert ref.state == {}


def test_get_scenario_summary_action_returns_actions_and_hash():
    """Handler returns the action list verbatim + the SHA-256 of the DB.

    The DB itself does NOT travel through the response — only its hash —
    so the response stays small regardless of DB size (tau2's 7MB DB
    would exceed the 1MB WebSocket frame limit if sent inline).
    """
    from nemo_voice_agent.evaluation.db_hash import get_dict_hash

    db_contents = {"reservations": {"R1": {}}}
    ref = SharedStateRef()
    ref.state = {
        "actions": [{"action_type": "rebook_flight", "x": 1}],
        "db": db_contents,
        "_call_counts": {"rebook_flight": 1},  # internal marker, not in response
    }
    action = create_get_scenario_summary_action(None, ref)
    result = asyncio.run(action.handler(None, "context", {}))
    assert result["actions"] == [{"action_type": "rebook_flight", "x": 1}]
    assert result["db_hash"] == get_dict_hash(db_contents)
    # Each bot returns only its own db; user_db labeling happens at the
    # bridge boundary when the bridge dual-pulls (added with the telecom
    # port). No `user_db_hash` in the per-bot response.
    assert "user_db_hash" not in result
    # Internal markers stay on the bot — not in the response payload.
    assert "db" not in result
    assert "_call_counts" not in result


def test_get_scenario_summary_action_uninitialized_state():
    """Empty state returns ``{actions: [], db_hash: None}`` — minimal shape."""
    ref = SharedStateRef()
    action = create_get_scenario_summary_action(None, ref)
    result = asyncio.run(action.handler(None, "context", {}))
    assert result == {"actions": [], "db_hash": None}


def test_get_scenario_summary_action_metadata():
    """Action declares the correct service / action / argument schema.

    The single optional ``include_db: bool`` argument (default ``False``)
    lets the runner ask the bot to inline this bot's ``db`` dict (not
    just the hash) when a scenario carries ``db_state_assertions``. Each
    bot returns only its own DB; the bridge labels them by source.
    """
    ref = SharedStateRef()
    action = create_get_scenario_summary_action(None, ref)
    assert action.service == "context"
    assert action.action == "get_scenario_summary"
    assert len(action.arguments) == 1
    arg = action.arguments[0]
    assert arg.name == "include_db"
    assert arg.type == "bool"
    # Default ``include_db=False`` preserves retail's hash-out behavior;
    # only opt-in domains (telecom) ask for the inline DB.


# ---------------------------------------------------------------------------
# 4. Scenario.expected_scenario_db field
# ---------------------------------------------------------------------------


def _make_dummy_scenario(**kwargs) -> Scenario:
    """Minimal Scenario subclass that satisfies all abstract properties."""

    class _Dummy(Scenario):
        name = "test__dummy"
        success_signals = (SuccessSignal.ACTION_MATCH,)  # satisfies non-empty check

        @property
        def user_persona(self) -> Persona:
            return Persona(role="u", name="U", background="b", personality="p")

        @property
        def agent_persona(self) -> Persona:
            return Persona(role="a", name="A", background="b", personality="p")

        @property
        def user_task(self) -> Task:
            return Task(goal="g")

        @property
        def agent_task(self) -> Task:
            return Task(goal="g")

        @property
        def user_actions(self) -> Actions:
            return Actions(instructions=["i"])

        @property
        def agent_actions(self) -> Actions:
            return Actions(instructions=["i"])

        @property
        def user_resources(self) -> Resources:
            return Resources()

        @property
        def agent_resources(self) -> Resources:
            return Resources()

    return _Dummy(**kwargs)


def test_scenario_expected_db_default_is_none():
    s = _make_dummy_scenario()
    assert s.expected_scenario_db is None


def test_scenario_expected_db_via_init_kwarg():
    expected = {"reservations": {"R1": {"status": "confirmed"}}}
    s = _make_dummy_scenario(expected_scenario_db=expected)
    assert s.expected_scenario_db == expected


def test_scenario_expected_db_class_attribute_takes_precedence():
    """Subclasses can set ``expected_scenario_db`` as a class attribute (or
    cached_property). The ``__init__`` parameter is the fallback."""

    class _WithClassAttr(Scenario):
        name = "with_attr"
        success_signals = (SuccessSignal.ACTION_MATCH,)
        expected_scenario_db = {"reservations": {"X": {}}}

        @property
        def user_persona(self) -> Persona:
            return Persona(role="u", name="U", background="b", personality="p")

        @property
        def agent_persona(self) -> Persona:
            return Persona(role="a", name="A", background="b", personality="p")

        @property
        def user_task(self) -> Task:
            return Task(goal="g")

        @property
        def agent_task(self) -> Task:
            return Task(goal="g")

        @property
        def user_actions(self) -> Actions:
            return Actions(instructions=["i"])

        @property
        def agent_actions(self) -> Actions:
            return Actions(instructions=["i"])

        @property
        def user_resources(self) -> Resources:
            return Resources()

        @property
        def agent_resources(self) -> Resources:
            return Resources()

    # Class attribute survives — __init__'s kwarg doesn't overwrite (existing pattern via hasattr).
    s = _WithClassAttr(expected_scenario_db={"different": {}})
    assert s.expected_scenario_db == {"reservations": {"X": {}}}


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
