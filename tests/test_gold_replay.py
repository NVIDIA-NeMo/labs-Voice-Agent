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

"""Unit tests for Tau2BaseScenario._gold_replay (M1.G).

Uses a hand-crafted fake subclass that bypasses the data-file loader, so the test
runs without any tau2_* data files on disk (those land in M2/M3/M5).
"""

from typing import Any, Dict

import pytest

from nemo_voice_agent.evaluation.scenarios.data.tau2_common import Tau2BaseScenario


# ---------------------------------------------------------------------------
# Fake tool — minimal interface ``_gold_replay`` requires: ``invoke(**kwargs)``
# that mutates the bound state and (for writes) appends to state["actions"].
# ---------------------------------------------------------------------------


class _FakeTool:
    """In-memory stand-in for a domain tool. Tracks state mutations."""

    def __init__(self, state: dict, name: str, write: bool = True):
        self.state = state
        self.name = name
        self.write = write

    def invoke(self, **kwargs):
        if self.write:
            self.state["db"][f"{self.name}_calls"] = self.state["db"].get(f"{self.name}_calls", 0) + 1
            self.state["actions"].append({"type": self.name, "args": kwargs, "result": "ok"})
        return {"status": "ok"}


class _ErroringTool:
    def __init__(self, state: dict, name: str):
        self.state = state
        self.name = name

    def invoke(self, **kwargs):
        raise RuntimeError(f"{self.name} blew up")


# ---------------------------------------------------------------------------
# Fake scenario — overrides everything that would touch disk.
# ---------------------------------------------------------------------------


class _FakeTau2Scenario(Tau2BaseScenario):
    domain = "_fake"
    tau2_id = "_fake_1"

    _ACTIONS = [
        {
            "action_id": "1_0",
            "name": "rebook_flight",
            "arguments": {"user_id": "u1", "new_flight": "SK999"},
            "info": None,
            "compare_args": None,
        },
        {
            "action_id": "1_1",
            "name": "cancel_reservation",
            "arguments": {"confirmation": "ABC123"},
            "info": None,
            "compare_args": [],
            "requestor": "assistant",
        },
    ]

    @property
    def _index_entry(self) -> Dict[str, Any]:
        return {
            "task": {
                "id": self.tau2_id,
                "evaluation_criteria": {"actions": self._ACTIONS},
            },
            "persona_name": "test_persona",
        }

    @property
    def db(self) -> Dict[str, Any]:
        return {"users": {"u1": {"name": "Test"}}, "reservations": {}}

    @property
    def policy(self) -> str:
        return "fake policy"

    def _build_tool_map(self, state: dict) -> Dict[str, Any]:
        return {
            "rebook_flight": _FakeTool(state, "rebook_flight"),
            "cancel_reservation": _FakeTool(state, "cancel_reservation"),
        }


class _TelecomLikeScenario(_FakeTau2Scenario):
    """Exercises has_user_state + requestor='user' tagging."""

    has_user_state = True

    _ACTIONS = [
        {
            "action_id": "u_0",
            "name": "set_phone_mode",
            "arguments": {"airplane_mode": False},
            "compare_args": None,
            "requestor": "user",
        },
        {
            "action_id": "a_0",
            "name": "lookup_account",
            "arguments": {"phone": "555-1234"},
            "compare_args": None,
            "requestor": "assistant",
        },
    ]

    def setup_shared_state(self, state: dict, side: str) -> None:
        super().setup_shared_state(state, side)
        if side == "user":
            state["user_db"] = {"phone_state": {"airplane_mode": True}}

    def _build_tool_map(self, state: dict) -> Dict[str, Any]:
        return {
            "set_phone_mode": _FakeTool(state, "set_phone_mode"),
            "lookup_account": _FakeTool(state, "lookup_account"),
        }


class _MissingToolScenario(_FakeTau2Scenario):
    """Exercises the 'no tool named X' warning branch."""

    _ACTIONS = [
        {
            "action_id": "x_0",
            "name": "tool_that_doesnt_exist",
            "arguments": {},
            "compare_args": None,
        },
    ]


class _ErroringScenario(_FakeTau2Scenario):
    """Exercises the 'tool.invoke raised' warning branch."""

    _ACTIONS = [
        {
            "action_id": "e_0",
            "name": "boom",
            "arguments": {},
            "compare_args": None,
        },
    ]

    def _build_tool_map(self, state: dict) -> Dict[str, Any]:
        return {"boom": _ErroringTool(state, "boom")}


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_gold_replay_returns_db_user_db_and_actions_consistent_with_replay():
    scenario = _FakeTau2Scenario()
    final_db, final_user_db, actions = scenario._gold_replay
    assert final_db["rebook_flight_calls"] == 1
    assert final_db["cancel_reservation_calls"] == 1
    assert final_user_db is None  # single-side domain
    assert [a["type"] for a in actions] == ["rebook_flight", "cancel_reservation"]


def test_expected_dbs_and_reference_answer_are_views_of_same_replay():
    # Same execution, three cached_property views — must be byte-identical.
    scenario = _FakeTau2Scenario()
    assert scenario.expected_scenario_db is scenario._gold_replay[0]
    assert scenario.expected_user_db is scenario._gold_replay[1]
    assert scenario.reference_answer is scenario._gold_replay[2]


def test_gold_replay_stamps_side_from_requestor():
    scenario = _FakeTau2Scenario()
    _, _, actions = scenario._gold_replay
    # First action has no requestor → "agent"; second has requestor="assistant" → "agent".
    assert actions[0]["side"] == "agent"
    assert actions[1]["side"] == "agent"


def test_gold_replay_telecom_user_side_tagged():
    scenario = _TelecomLikeScenario()
    final_db, final_user_db, actions = scenario._gold_replay
    # User-side seeding must have happened — the dual-DB telecom path.
    assert final_user_db is not None
    assert final_user_db["phone_state"]["airplane_mode"] is True
    # set_phone_mode write also lands in user_db's actions… actually no, our fake
    # tool writes to state["db"], not state["user_db"]. The side-tag is purely
    # about which side initiated the action, independent of where the write lands.
    assert actions[0]["side"] == "user"
    assert actions[1]["side"] == "agent"


def test_gold_replay_missing_tool_does_not_crash():
    # Should log a warning and produce an empty action list, no exception.
    scenario = _MissingToolScenario()
    final_db, _, actions = scenario._gold_replay
    assert actions == []


def test_gold_replay_erroring_tool_does_not_crash():
    # Should log a warning, record nothing for that action, no exception.
    scenario = _ErroringScenario()
    final_db, _, actions = scenario._gold_replay
    assert actions == []


def test_gold_replay_empty_actions_returns_initial_db():
    class _EmptyScenario(_FakeTau2Scenario):
        _ACTIONS = []

    scenario = _EmptyScenario()
    final_db, _, actions = scenario._gold_replay
    assert actions == []
    # No mutations expected — initial db users dict should still be there.
    assert "users" in final_db
    assert final_db["users"]["u1"]["name"] == "Test"


# ---------------------------------------------------------------------------
# Scenario-contract stubs (agent side) + structured user side
# ---------------------------------------------------------------------------


class _StructuredUserScenario(_FakeTau2Scenario):
    """Adds a realistic user_scenario block so the user-side properties have content."""

    @property
    def _index_entry(self):
        return {
            "task": {
                "id": self.tau2_id,
                "evaluation_criteria": {"actions": self._ACTIONS},
                "user_scenario": {
                    "persona": None,
                    "instructions": {
                        "task_instructions": "Be polite. Insist on a refund.",
                        "domain": "airline",
                        "reason_for_call": "Cancel reservation ABC123 and get a refund.",
                        "known_info": "You are Emma Kim. Your user id is emma_kim_9957.",
                        "unknown_info": None,
                    },
                },
            },
            "persona_name": "emma_kim",
        }


def test_agent_persona_is_a_stub_carrying_persona_name():
    scenario = _FakeTau2Scenario()
    persona = scenario.agent_persona
    # Stub fields are present; persona_name carried into Persona.name for metric slicing.
    assert persona.name == "test_persona"
    assert persona.role.startswith("tau2 ")
    # The agent prompt does NOT come from Persona.to_prompt_section() — it comes
    # from policy.md via get_agent_prompt(). Sanity-check that boundary.
    assert "see get_agent_prompt" in persona.background


def test_agent_stubs_dont_raise_notimplementederror():
    scenario = _FakeTau2Scenario()
    # All four agent-side abstract properties on Scenario must resolve to objects
    # of the right type so anything iterating Scenario subclasses doesn't crash.
    assert scenario.agent_task.goal  # non-empty string ("(see policy.md)")
    assert scenario.agent_actions.instructions == []
    assert scenario.agent_actions.guidelines == []
    assert scenario.agent_resources.tools == {}


def test_get_agent_prompt_returns_policy_verbatim_not_via_persona():
    scenario = _FakeTau2Scenario()
    prompt = scenario.get_agent_prompt()
    # The prompt is policy.md content as-is — no "You are a {role} named {name}." prefix
    # that Persona.to_prompt_section would have added.
    assert prompt == "fake policy"
    assert "You are a tau2" not in prompt
    assert "Keep your responses concise" not in prompt  # the GENERAL_PROMPT suffix


def test_user_persona_pulled_from_user_scenario_instructions():
    scenario = _StructuredUserScenario()
    persona = scenario.user_persona
    assert persona.name == "emma_kim"
    assert "Emma Kim" in persona.background
    assert "Insist on a refund" in persona.personality


def test_user_task_uses_reason_for_call():
    scenario = _StructuredUserScenario()
    assert "Cancel reservation ABC123" in scenario.user_task.goal


def test_user_actions_include_voice_alphanumeric_rule():
    from nemo_voice_agent.evaluation.voice_rules import VOICE_ALPHANUMERIC_RULE

    scenario = _StructuredUserScenario()
    assert VOICE_ALPHANUMERIC_RULE in scenario.user_actions.guidelines


def test_user_side_handles_missing_user_scenario_gracefully():
    # _FakeTau2Scenario has no user_scenario field in its fake task — defensive
    # .get(...) chains in the properties must not crash.
    scenario = _FakeTau2Scenario()
    assert scenario.user_persona.background == ""
    assert scenario.user_persona.personality == ""
    assert scenario.user_task.goal == ""


def test_get_user_prompt_uses_structured_properties_end_to_end():
    # The inherited Scenario.get_user_prompt() composes persona+task+actions+resources.
    # If our structured properties are wired correctly, the user prompt contains both
    # the persona's known_info and the task's reason_for_call.
    scenario = _StructuredUserScenario()
    user_prompt = scenario.get_user_prompt()
    assert "Emma Kim" in user_prompt
    assert "Cancel reservation ABC123" in user_prompt
    # The VOICE_ALPHANUMERIC_RULE is in user_actions.guidelines.
    assert "spell each character one at a time" in user_prompt
