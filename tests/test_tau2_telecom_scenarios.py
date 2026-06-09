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

"""Integration tests for the tau2_telecom scenario port.

Same shape as ``test_tau2_retail_scenarios.py`` but exercises the
dual-state path: ``has_user_state=True``, ``user_db`` cached property,
``initialization_actions`` translation, gold-replay running both init
actions and user-side reference actions, predicate evaluation against
the resulting user_db.
"""

import nemo_voice_agent.evaluation.scenarios.data  # noqa: F401 — triggers registration

from nemo_voice_agent.evaluation.db_state_predicates import evaluate_db_state_assertion
from nemo_voice_agent.evaluation.scenarios import ALL_EVAL_SCENARIOS
from nemo_voice_agent.evaluation.scenarios.data.tau2_common import _normalize_env_record
from nemo_voice_agent.evaluation.scenarios.data.tau2_telecom.base import Tau2TelecomBaseScenario


_FIRST_SCENARIO = "tau2_telecom__mobile_data_issue__airplane_mode_on__data_mode_off"


# ---------------------------------------------------------------------------
# 1. _normalize_env_record helper — unit-level
# ---------------------------------------------------------------------------


def test_normalize_env_record_user_unchanged():
    rec = {"env_type": "user", "func_name": "x", "arguments": {}}
    out = _normalize_env_record(rec)
    assert out == {"side": "user", "func_name": "x", "arguments": {}}
    # Doesn't mutate input
    assert "env_type" in rec


def test_normalize_env_record_assistant_to_agent():
    rec = {"env_type": "assistant", "func_name": "y", "arguments": {"x": 1}}
    out = _normalize_env_record(rec)
    assert out == {"side": "agent", "func_name": "y", "arguments": {"x": 1}}


def test_normalize_env_record_passes_through_other_fields():
    rec = {
        "env_type": "user",
        "func_name": "z",
        "arguments": {},
        "assert_value": True,
        "message": "irrelevant",
    }
    out = _normalize_env_record(rec)
    assert out["assert_value"] is True
    assert out["message"] == "irrelevant"


def test_normalize_env_record_no_env_type_field():
    rec = {"func_name": "noop", "arguments": {}}
    out = _normalize_env_record(rec)
    assert out == rec


# ---------------------------------------------------------------------------
# 2. Scenario registration + instantiation
# ---------------------------------------------------------------------------


def test_first_scenario_registered():
    assert _FIRST_SCENARIO in ALL_EVAL_SCENARIOS


def test_all_114_base_split_scenarios_registered():
    """Auto-scaffolded by ``nemo_experiments/generate_tau2_telecom_scaffolds.py``
    from ``split_tasks.json[base]``. Locks in the count so an accidental
    overwrite or generator regression surfaces in unit tests, not at
    eval-run time."""
    telecom = [n for n in ALL_EVAL_SCENARIOS if n.startswith("tau2_telecom__")]
    assert len(telecom) == 114


def test_every_scaffolded_scenario_instantiates_and_gold_replays():
    """Every scaffold must produce a valid Pydantic-validated tau2 task,
    a loadable policy, and a successful in-process gold replay. Catches
    bad ``env_type`` translations, missing init functions, predicate
    name collisions, and class-name collisions before live runs.
    """
    telecom = [n for n in ALL_EVAL_SCENARIOS if n.startswith("tau2_telecom__")]
    failures: list = []
    for name in telecom:
        try:
            inst = ALL_EVAL_SCENARIOS[name]()
            # Touch every cached_property derived from tau2_task / data so
            # any latent error surfaces (reference_answer runs the gold
            # replay which exercises tool dispatch + init replay).
            _ = (
                inst.policy,
                inst.user_db,
                inst.db_state_assertions,
                inst.initialization_actions,
                inst.nl_assertions,
                inst.user_resources,
                inst.agent_resources,
                inst.reference_answer,
            )
        except Exception as exc:  # noqa: BLE001 — we want to collect all
            failures.append((name, f"{type(exc).__name__}: {exc!s:.200}"))
    assert not failures, f"Scaffolded scenarios failed: {failures[:5]}"


def test_first_scenario_instantiates():
    scenario = ALL_EVAL_SCENARIOS[_FIRST_SCENARIO]()
    assert isinstance(scenario, Tau2TelecomBaseScenario)
    assert scenario.domain == "tau2_telecom"
    assert scenario.has_user_state is True
    assert scenario.tau2_id == "[mobile_data_issue]airplane_mode_on|data_mode_off[PERSONA:None]"


# ---------------------------------------------------------------------------
# 3. Policy concat — markdown, not XML
# ---------------------------------------------------------------------------


def test_policy_concats_main_and_tech_support_with_hrule_separator():
    scenario = ALL_EVAL_SCENARIOS[_FIRST_SCENARIO]()
    policy = scenario.policy
    # Both files' top-level headers present
    assert "# Telecom Agent Policy" in policy
    assert "# Introduction" in policy  # tech_support_manual.md's H1
    # Horizontal-rule separator
    assert "\n\n---\n\n" in policy
    # No XML tags
    assert "<main_policy>" not in policy
    assert "<tech_support_policy>" not in policy


def test_policy_variant_default_is_manual():
    scenario = ALL_EVAL_SCENARIOS[_FIRST_SCENARIO]()
    assert scenario.policy_variant == "manual"
    # The manual variant's distinctive section header should be present
    # ("Understanding and Troubleshooting Your Phone's Cellular Service"
    # exists in tech_support_manual.md but not in tech_support_workflow.md).
    assert "Understanding and Troubleshooting" in scenario.policy


# ---------------------------------------------------------------------------
# 4. db_state_assertions / initialization_actions translation
# ---------------------------------------------------------------------------


def test_db_state_assertions_translated_from_upstream():
    scenario = ALL_EVAL_SCENARIOS[_FIRST_SCENARIO]()
    asserts = scenario.db_state_assertions
    assert asserts is not None
    assert len(asserts) == 2
    # All entries use the renamed key ``side`` (not the upstream ``env_type``)
    for a in asserts:
        assert "side" in a
        assert "env_type" not in a
        assert a["side"] in {"user", "agent"}
    # Specific expected predicates for this scenario
    names = {a["func_name"] for a in asserts}
    assert names == {"assert_mobile_data_status", "assert_internet_speed"}


def test_initialization_actions_translated_from_upstream():
    scenario = ALL_EVAL_SCENARIOS[_FIRST_SCENARIO]()
    actions = scenario.initialization_actions
    assert actions is not None
    assert len(actions) == 3
    for a in actions:
        assert "side" in a
        assert "env_type" not in a
    names = [a["func_name"] for a in actions]
    assert names == ["set_user_info", "turn_airplane_mode_on", "turn_data_off"]


# ---------------------------------------------------------------------------
# 5. user_db property + setup_shared_state (live-runtime path)
# ---------------------------------------------------------------------------


def test_user_db_loads():
    scenario = ALL_EVAL_SCENARIOS[_FIRST_SCENARIO]()
    udb = scenario.user_db
    # Materialized defaults from M5a's Pydantic round-trip
    assert "device" in udb
    assert "surroundings" in udb
    assert "signal_strength" in udb["surroundings"]


def test_setup_shared_state_agent_side_sets_db_path():
    scenario = ALL_EVAL_SCENARIOS[_FIRST_SCENARIO]()
    state: dict = {}
    scenario.setup_shared_state(state, side="agent")
    assert state == {"db_path": "tau2_telecom/db.json"}


def test_setup_shared_state_user_side_sets_db_path():
    """Live-runtime path: ``setup_shared_state`` writes only ``db_path``,
    not an inline ``db`` dict. The bot's path-resolver loads from disk.
    Gold-replay bypasses this and inline-loads ``user_db`` directly in
    the parent's ``_gold_replay``.
    """
    scenario = ALL_EVAL_SCENARIOS[_FIRST_SCENARIO]()
    state: dict = {}
    scenario.setup_shared_state(state, side="user")
    assert state == {"db_path": "tau2_telecom/user_db.json"}


# ---------------------------------------------------------------------------
# 6. _build_tool_map registers the 4 user-side tools
# ---------------------------------------------------------------------------


def test_build_tool_map_returns_all_telecom_tools():
    """30 user-side + 13 agent-side = 43 tools dispatched via the gold-replay map."""
    scenario = ALL_EVAL_SCENARIOS[_FIRST_SCENARIO]()
    tool_map = scenario._build_tool_map({"db": {}, "user_db": {}, "actions": []})
    # User-side LLM tools (30)
    assert "toggle_data" in tool_map
    assert "toggle_airplane_mode" in tool_map
    assert "check_status_bar" in tool_map
    assert "run_speed_test" in tool_map
    assert "set_apn_settings" in tool_map
    assert "reboot_device" in tool_map
    # Agent-side LLM tools (13)
    assert "get_customer_by_phone" in tool_map
    assert "suspend_line" in tool_map
    assert "send_payment_request" in tool_map
    assert "transfer_to_human_agents" in tool_map
    assert len(tool_map) == 43


def test_user_resources_carries_all_30_user_tools():
    """Full upstream surface (14 reads + 16 writes) of user-side LLM tools.
    Keys are snake_case ``cls.name`` — single source of truth for
    registry, LLM schema, and gold-replay dispatch."""
    scenario = ALL_EVAL_SCENARIOS[_FIRST_SCENARIO]()
    tools = scenario.user_resources.tools
    expected = {
        # Reads (14)
        "check_status_bar", "check_network_status", "check_network_mode_preference",
        "check_sim_status", "check_data_restriction_status", "check_apn_settings",
        "check_wifi_status", "check_wifi_calling_status", "check_vpn_status",
        "check_installed_apps", "check_app_status", "check_app_permissions",
        "can_send_mms", "check_payment_request", "run_speed_test",
        # Writes (16)
        "toggle_airplane_mode", "toggle_data", "toggle_roaming",
        "toggle_data_saver_mode", "toggle_wifi", "toggle_wifi_calling",
        "set_network_mode_preference", "reseat_sim_card", "set_apn_settings",
        "reset_apn_settings", "connect_vpn", "disconnect_vpn",
        "grant_app_permission", "reboot_device", "make_payment",
    }
    # Note: "run_speed_test" is read-only — listed in reads above; the count
    # of 15+15 (with run_speed_test in reads) totals 30. Confirm the count:
    assert set(tools.keys()) == expected


def test_tool_name_attribute_matches_llm_schema_name():
    """``cls.name`` propagates through StandardSchemaTool.__init__ to
    ``self.name`` and to the rendered ``FunctionSchema.name`` — that's
    the function name the LLM sees in its tool-call schema."""
    from nemo_voice_agent.evaluation.tools.tau2_telecom_user_tools import ToggleDataTool

    instance = ToggleDataTool()
    assert instance.name == "toggle_data"
    assert instance.schema.name == "toggle_data"


def test_agent_resources_carries_full_telecom_surface():
    """Full 13 agent-side tools + ``EndConversationTool`` voice-harness signal.

    Agent-side tools use snake_case ``cls.name`` keys matching upstream
    method names + policy.md references (so the LLM doesn't hallucinate
    a different naming when reading the policy text).
    ``EndConversationTool`` keeps its PascalCase registry key — it's a
    shared harness tool in the ``"default"`` registry namespace, not
    domain-scoped.
    """
    scenario = ALL_EVAL_SCENARIOS[_FIRST_SCENARIO]()
    tools = scenario.agent_resources.tools
    expected_agent_tools = {
        # Reads (6)
        "get_customer_by_phone", "get_customer_by_id", "get_customer_by_name",
        "get_details_by_id", "get_bills_for_customer", "get_data_usage",
        # Writes (6)
        "suspend_line", "resume_line", "enable_roaming", "disable_roaming",
        "send_payment_request", "refuel_data",
        # Generic (1)
        "transfer_to_human_agents",
        # Harness
        "EndConversationTool",
    }
    assert set(tools.keys()) == expected_agent_tools


def test_user_resources_preserves_parent_info_sections():
    """``user_resources`` inherits ``known_info`` rendering from the parent."""
    scenario = ALL_EVAL_SCENARIOS[_FIRST_SCENARIO]()
    info = scenario.user_resources.info_sections or {}
    assert "Things you know" in info
    # The known_info from this scenario contains "John Smith"
    assert "John Smith" in info["Things you know"]


# ---------------------------------------------------------------------------
# 7. _gold_replay end-to-end — init + reference actions land at expected state
# ---------------------------------------------------------------------------


def test_gold_replay_runs_without_error():
    scenario = ALL_EVAL_SCENARIOS[_FIRST_SCENARIO]()
    gold_db, gold_user_db, recorded = scenario._gold_replay
    assert isinstance(gold_db, dict)
    assert isinstance(gold_user_db, dict)
    assert isinstance(recorded, list)


def test_gold_replay_produces_two_user_side_reference_actions():
    """The first scenario has 2 reference actions, both with
    ``requestor="user"`` upstream. The gold replay should record both
    with side="user" stamped on."""
    scenario = ALL_EVAL_SCENARIOS[_FIRST_SCENARIO]()
    _, _, recorded = scenario._gold_replay
    names = [a.get("name") for a in recorded]
    assert "toggle_airplane_mode" in names
    assert "toggle_data" in names
    sides = {a.get("side") for a in recorded}
    assert sides == {"user"}


def test_gold_replay_final_user_db_is_fixed_state():
    """After applying init actions (broken state) + reference actions
    (user toggles back), the user_db should be in the expected
    'fixed' state — airplane mode off, data enabled on."""
    scenario = ALL_EVAL_SCENARIOS[_FIRST_SCENARIO]()
    _, gold_user_db, _ = scenario._gold_replay
    assert gold_user_db["device"]["airplane_mode"] is False
    assert gold_user_db["device"]["data_enabled"] is True


def test_gold_replay_satisfies_db_state_assertions():
    """The predicates declared on this scenario evaluate ``True`` against
    the gold-replay's final user_db. This is the end-to-end gate for
    the scenario port: init actions transition to broken state, reference
    actions fix it, predicates verify the fix."""
    scenario = ALL_EVAL_SCENARIOS[_FIRST_SCENARIO]()
    gold_db, gold_user_db, _ = scenario._gold_replay
    for assertion in scenario.db_state_assertions:
        verdict = evaluate_db_state_assertion(
            domain="tau2_telecom",
            assertion=assertion,
            db=gold_db,
            user_db=gold_user_db,
        )
        assert verdict["passed"], (
            f"Predicate {assertion['func_name']!r} failed against gold-replay "
            f"user_db: actual={verdict['actual']!r}, error={verdict['error']!r}"
        )


# ---------------------------------------------------------------------------
# 8. reference_answer shape for runner action-list comparison
# ---------------------------------------------------------------------------


def test_reference_answer_has_actions_key():
    """Same shape contract as airline/retail — ``{"actions": [...]}`` so
    the runner's deterministic comparator sees a uniform shape."""
    scenario = ALL_EVAL_SCENARIOS[_FIRST_SCENARIO]()
    ref = scenario.reference_answer
    assert "actions" in ref
    assert len(ref["actions"]) == 2
