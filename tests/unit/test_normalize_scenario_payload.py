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

"""Tests for ``normalize_scenario_payload`` — the shape-equivalence rule
shared by ``check_if_task_success`` and the LLM judge prep path in the runner.

Locks in the four rule branches:
  1. List of exactly one dict → unwrap to that dict.
  2. Single dict → return as-is.
  3. List of multiple dicts → return as-is (legitimate push-path output).
  4. Anything else → return as-is.
"""

import pytest

from nemo_voice_agent.evaluation.utils import normalize_scenario_payload


# ---------------------------------------------------------------------------
# Rule 1: list-of-one-dict unwraps
# ---------------------------------------------------------------------------


def test_list_of_one_dict_unwraps_to_dict():
    payload = [{"actions": [{"name": "rebook"}]}]
    result = normalize_scenario_payload(payload)
    assert result == {"actions": [{"name": "rebook"}]}
    assert isinstance(result, dict)


def test_list_of_one_empty_dict_unwraps():
    payload = [{}]
    result = normalize_scenario_payload(payload)
    assert result == {}
    assert isinstance(result, dict)


def test_unwrap_makes_tau2_pred_match_tau2_ref():
    """The exact tau2 case that motivated this helper.

    Tau2's reference is ``{"actions": [...]}`` (a dict). The bridge writes
    the prediction as ``[{"actions": [...]}]`` (list-of-1-dict) for legacy
    push-path compat. After normalization both sides are dicts with identical
    structure, so the LLM judge sees the same shape.
    """
    actions = [
        {"action_type": "cancel_reservation", "arguments": {"reservation_id": "XEHM4B"}, "side": "agent"},
        {"action_type": "cancel_reservation", "arguments": {"reservation_id": "59XX6W"}, "side": "agent"},
    ]
    tau2_ref = {"actions": actions}
    tau2_pred = [{"actions": actions}]

    ref_norm = normalize_scenario_payload(tau2_ref)
    pred_norm = normalize_scenario_payload(tau2_pred)

    assert ref_norm == pred_norm
    assert isinstance(ref_norm, dict)
    assert isinstance(pred_norm, dict)


# ---------------------------------------------------------------------------
# Rule 2: single dict pass-through
# ---------------------------------------------------------------------------


def test_single_dict_returns_as_is():
    payload = {"actions": [{"name": "rebook"}]}
    result = normalize_scenario_payload(payload)
    # Identity is fine — the helper isn't required to copy
    assert result is payload


def test_empty_dict_returns_as_is():
    payload = {}
    result = normalize_scenario_payload(payload)
    assert result == {}


# ---------------------------------------------------------------------------
# Rule 3: list-of-multiple-dicts preserves shape (push-path with >1 emission)
# ---------------------------------------------------------------------------


def test_list_of_multiple_dicts_returns_as_is():
    """Push-path predictions with multiple ``<final_response>`` emissions
    must stay as a list — each entry is a distinct event, not redundant
    wrapping."""
    payload = [{"action": "first"}, {"action": "second"}, {"action": "third"}]
    result = normalize_scenario_payload(payload)
    assert result is payload
    assert isinstance(result, list)
    assert len(result) == 3


def test_list_of_two_dicts_does_not_unwrap():
    """Boundary case: 2 dicts is not the unwrap pattern."""
    payload = [{"a": 1}, {"b": 2}]
    result = normalize_scenario_payload(payload)
    assert result is payload


# ---------------------------------------------------------------------------
# Rule 4: other shapes pass through (defensive)
# ---------------------------------------------------------------------------


def test_empty_list_returns_as_is():
    payload = []
    result = normalize_scenario_payload(payload)
    assert result == []


def test_list_of_non_dict_returns_as_is():
    """List of 1 string, list of 1 int, etc. — not a list-of-1-dict, so no unwrap."""
    assert normalize_scenario_payload(["hello"]) == ["hello"]
    assert normalize_scenario_payload([42]) == [42]
    assert normalize_scenario_payload([None]) == [None]


def test_none_returns_as_is():
    assert normalize_scenario_payload(None) is None


def test_scalar_returns_as_is():
    assert normalize_scenario_payload("string") == "string"
    assert normalize_scenario_payload(42) == 42
    assert normalize_scenario_payload(True) is True


# ---------------------------------------------------------------------------
# Integration: check_if_task_success uses the helper transparently
# ---------------------------------------------------------------------------


def test_check_if_task_success_still_matches_after_normalization(tmp_path):
    """The refactor preserves check_if_task_success behavior for the tau2 case:
    dict ref + list-of-1-dict pred should match (was Situation 2 before;
    now normalization collapses to Situation 1)."""
    import json as _json

    from nemo_voice_agent.evaluation.utils import check_if_task_success

    actions = [{"action_type": "cancel_reservation", "arguments": {"reservation_id": "XEHM4B"}}]

    ref_path = tmp_path / "ref.json"
    ref_path.write_text(_json.dumps({"actions": actions}))

    pred_path = tmp_path / "pred.json"
    pred_path.write_text(_json.dumps([{"actions": actions}]))

    assert check_if_task_success(reference=str(ref_path), prediction=str(pred_path)) is True, (
        "tau2-style dict ref + list-of-1-dict pred should still match"
    )
