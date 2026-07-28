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

"""Regression tests for empty-result masking at the pipecat tool boundary.

Pipecat's ``handle_function_call_result`` does ``if frame.result:`` and
substitutes the literal string ``"COMPLETED"`` for any falsy result (see
``pipecat/services/openai/llm.py``). A read tool that returns a bare empty
list (e.g. ``get_customer_by_name`` when name+DOB matches nobody) therefore
gets silently rewritten to ``"COMPLETED"`` — which the LLM reads as *success*.

Observed live in tau2_telecom (2026-07): a nothink agent fabricated a DOB
(``1980-01-01`` vs the real ``1985-06-15``) for John Smith; the lookup returned
``[]``; the LLM saw ``"COMPLETED"`` and believed the customer was verified.

Fix: ``_normalize_empty_result`` wraps falsy results in a non-falsy envelope.
It is applied **centrally**, exactly once, in
``StandardSchemaTool.__call__`` — the single pipecat-facing delivery point —
so no tool can forget it and no result is delivered twice. ``_execute`` is now
pure (kwargs in, value out, never delivers), and the sync ``invoke`` path
(gold replay + shadow-DB sync) keeps returning the raw shape.
"""

import asyncio
from types import SimpleNamespace

from nemo_voice_agent.evaluation.tools import normalize_tool_result
from nemo_voice_agent.utils.tool_calling.base import _normalize_empty_result


# ---------------------------------------------------------------------------
# _normalize_empty_result helper
# ---------------------------------------------------------------------------


def test_empty_list_is_wrapped_non_falsy():
    out = _normalize_empty_result([])
    assert out, "wrapped result must be truthy so pipecat serializes it"
    assert out["status"] == "success"
    assert out["results"] == []
    assert out["count"] == 0
    assert "No matching records" in out["message"]


def test_empty_dict_is_wrapped_non_falsy():
    out = _normalize_empty_result({})
    assert out
    assert out["status"] == "success"
    assert out["results"] == {}


def test_none_empty_string_and_zero_are_wrapped_preserving_value():
    for falsy in (None, "", 0, False):
        out = _normalize_empty_result(falsy)
        assert out, f"{falsy!r} must be wrapped into a truthy envelope"
        assert out["result"] == falsy  # original value preserved
        assert out["status"] == "success"


def test_non_empty_results_pass_through_unchanged():
    # Non-empty list, dict, and string are returned verbatim (identity).
    lst = [{"customer_id": "C1001"}]
    assert _normalize_empty_result(lst) is lst
    d = {"status": "success", "user_id": "u1"}
    assert _normalize_empty_result(d) is d
    s = "Roaming enabled."
    assert _normalize_empty_result(s) is s


def test_deprecated_alias_delegates_to_canonical_helper():
    """``normalize_tool_result`` survives as a back-compat alias for external
    callers; the guard itself now lives in ``tool_calling.base``."""
    assert normalize_tool_result([]) == _normalize_empty_result([])
    lst = [{"customer_id": "C1001"}]
    assert normalize_tool_result(lst) is lst


# ---------------------------------------------------------------------------
# End-to-end through the real tau2 telecom read tool
# ---------------------------------------------------------------------------


def _make_params(arguments):
    """Fake ``FunctionCallParams``: records every delivered result.

    ``captured["result"]`` is the latest delivery; ``captured["results"]`` is
    the full sequence, so tests can assert delivery happens EXACTLY once —
    the invariant ``StandardSchemaTool.__call__`` now owns.
    """
    captured = {"results": []}

    async def result_callback(result):
        captured["results"].append(result)
        captured["result"] = result

    return SimpleNamespace(arguments=arguments, result_callback=result_callback), captured


def _telecom_db_with_john_smith():
    # Minimal shape: GetCustomerByNameTool._do_work matches on full_name + date_of_birth.
    return {
        "customers": [
            {
                "customer_id": "C1001",
                "full_name": "John Smith",
                "date_of_birth": "1985-06-15",
            }
        ]
    }


def test_no_match_call_emits_non_falsy_not_completed():
    """The bug scenario: wrong (fabricated) DOB → no match → must NOT surface as a
    bare ``[]`` (which pipecat would mask as "COMPLETED").

    Drives the public ``__call__`` entry point, so this covers the real
    delivery path including the central normalization.
    """
    from nemo_voice_agent.evaluation.tools.tau2_telecom_tools import GetCustomerByNameTool

    tool = GetCustomerByNameTool(shared_state={"db": _telecom_db_with_john_smith()})
    params, captured = _make_params({"full_name": "john smith", "dob": "1980-01-01"})
    asyncio.run(tool(params))

    assert len(captured["results"]) == 1, "result must be delivered exactly once (no double-delivery)"
    result = captured["result"]
    assert result, "empty match must be surfaced as a truthy payload, never a bare []"
    assert result != "COMPLETED"
    assert isinstance(result, dict)
    assert result["results"] == []
    assert result["count"] == 0
    assert "No matching records" in result["message"]


def test_invoke_still_returns_raw_empty_list_for_gold_replay():
    """The sync ``invoke`` path (gold replay / shadow-DB sync) must be UNCHANGED —
    it still returns the raw empty list, not the normalized envelope."""
    from nemo_voice_agent.evaluation.tools.tau2_telecom_tools import GetCustomerByNameTool

    tool = GetCustomerByNameTool(shared_state={"db": _telecom_db_with_john_smith()})
    raw = tool.invoke(full_name="john smith", dob="1980-01-01")
    assert raw == [], "invoke() must keep raw shape so gold replay / sync are unaffected"


def test_match_call_passes_customer_view_through():
    """A real match returns a non-empty list → passes through unchanged (still a list)."""
    from nemo_voice_agent.evaluation.tools.tau2_telecom_tools import GetCustomerByNameTool

    tool = GetCustomerByNameTool(shared_state={"db": _telecom_db_with_john_smith()})
    params, captured = _make_params({"full_name": "John Smith", "dob": "1985-06-15"})
    asyncio.run(tool(params))

    assert len(captured["results"]) == 1, "result must be delivered exactly once (no double-delivery)"
    result = captured["result"]
    assert isinstance(result, list) and len(result) == 1
    assert result[0]["customer_id"] == "C1001"
