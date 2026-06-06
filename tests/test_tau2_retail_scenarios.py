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

"""Integration tests for the tau2_retail scenario port.

These tests touch the real ``evaluation/data/tau2_retail/`` fixtures and
exercise the full chain: data-file load → ``Tau2RetailBaseScenario`` →
``_gold_replay`` against ported tools → ``expected_scenario_db`` hash.

They're integration-flavored — they cover that the 114 base-split scenarios
all instantiate, replay cleanly, and hash deterministically. Unit-level
behavior of individual tools is covered in the existing module-level smoke
checks in ``test_gold_replay.py`` (which uses a hand-crafted fake scenario
that doesn't touch disk).
"""

import nemo_voice_agent.evaluation.scenarios.data.tau2_retail  # noqa: F401  (side-import for @register_eval_scenario)
from nemo_voice_agent.evaluation.db_hash import get_dict_hash
from nemo_voice_agent.evaluation.scenarios import ALL_EVAL_SCENARIOS
from nemo_voice_agent.evaluation.scenarios.data.tau2_retail.base import (
    Tau2RetailBaseScenario,
)


def _retail_scenario_keys():
    """Sorted list of ``tau2_retail__N`` keys, numerically ordered."""
    keys = [k for k in ALL_EVAL_SCENARIOS if k.startswith("tau2_retail__")]
    return sorted(keys, key=lambda k: int(k.split("__")[1]))


def test_114_retail_scenarios_registered():
    """All 114 base-split tau2_retail tasks are discoverable via the registry."""
    keys = _retail_scenario_keys()
    assert len(keys) == 114, f"expected 114 retail scenarios, got {len(keys)}"
    # First and last are continuous integer ids 0..113.
    assert keys[0] == "tau2_retail__0"
    assert keys[-1] == "tau2_retail__113"


def test_every_retail_scenario_instantiates_and_hashes():
    """Every scenario must produce a stable SHA-256 from its expected DB.

    Catches any subclass that breaks the inherited ``_gold_replay`` pipeline.
    """
    for key in _retail_scenario_keys():
        cls = ALL_EVAL_SCENARIOS[key]
        inst = cls()
        h = get_dict_hash(inst.expected_scenario_db)
        assert isinstance(h, str) and len(h) == 64, f"{key}: bad hash {h!r}"


def test_nl_assertion_count_matches_upstream():
    """40 of 114 retail tasks carry nl_assertions; the rest return None.

    Verifies our ``nl_assertions`` cached_property:
    - reads ``evaluation_criteria.nl_assertions`` from the upstream task,
    - normalizes ``null``/``[]`` to ``None`` so the runner's truthy check is correct,
    - returns a populated list for tasks that actually have assertions.
    """
    with_nl = 0
    without_nl = 0
    for key in _retail_scenario_keys():
        inst = ALL_EVAL_SCENARIOS[key]()
        if inst.nl_assertions:
            with_nl += 1
            # Each entry is a non-empty string.
            for a in inst.nl_assertions:
                assert isinstance(a, str) and a.strip(), f"{key}: bad assertion {a!r}"
        else:
            without_nl += 1
    # 40 tasks with assertions (39 mixed + 1 nl-only), 74 without (73 action-only + 1 chitchat).
    assert with_nl == 40, f"expected 40 nl_assertion tasks, got {with_nl}"
    assert without_nl == 74


def test_task_57_chitchat_has_neither_signal():
    """Task 57 is the "polite but brief" no-task scenario (DB unchanged, no nl_assertions)."""
    inst = ALL_EVAL_SCENARIOS["tau2_retail__57"]()
    assert inst.nl_assertions is None
    # expected_scenario_db equals initial db (no actions to replay)
    assert get_dict_hash(inst.expected_scenario_db) == get_dict_hash(inst.db)
    # reference_answer wraps an empty action list
    assert inst.reference_answer == {"actions": []}


def test_task_16_has_one_nl_assertion():
    """Task 16 carries one nl_assertion: 'Agent should tell the user the total refund amount is $8,276.23.'"""
    inst = ALL_EVAL_SCENARIOS["tau2_retail__16"]()
    assert inst.nl_assertions is not None
    assert len(inst.nl_assertions) == 1
    assert "8,276.23" in inst.nl_assertions[0]


def test_agent_resources_lists_all_16_retail_tools_plus_end_call():
    """The 16 ported retail tools + EndConversationTool are exposed via agent_resources.tools."""
    inst = ALL_EVAL_SCENARIOS["tau2_retail__0"]()
    tool_names = set(inst.agent_resources.tools)
    expected = {
        "FindUserIdByNameZipTool",
        "FindUserIdByEmailTool",
        "GetOrderDetailsTool",
        "GetProductDetailsTool",
        "GetItemDetailsTool",
        "GetUserDetailsTool",
        "ListAllProductTypesTool",
        "CancelPendingOrderTool",
        "ExchangeDeliveredOrderItemsTool",
        "ModifyPendingOrderAddressTool",
        "ModifyPendingOrderItemsTool",
        "ModifyPendingOrderPaymentTool",
        "ModifyUserAddressTool",
        "ReturnDeliveredOrderItemsTool",
        "CalculateTool",
        "TransferToHumanAgentsTool",
        "EndConversationTool",
    }
    assert tool_names == expected, f"missing: {expected - tool_names}, extra: {tool_names - expected}"


def test_agent_prompt_is_policy_md_plus_voice_addendum():
    """The agent prompt is policy.md verbatim + ``## Additional Notes to Follow`` section."""
    inst = ALL_EVAL_SCENARIOS["tau2_retail__0"]()
    prompt = inst.get_agent_prompt()
    # policy.md verbatim prefix (first heading)
    assert prompt.startswith("# Retail agent policy")
    # Voice addendum appended after policy.md (header name lives in tau2_common.get_agent_prompt)
    assert "## Additional Notes to Follow" in prompt
    assert "EndConversationTool" in prompt  # the END_CONVERSATION_GUIDELINE addendum


def test_db_path_seeded_for_agent_side():
    """setup_shared_state writes db_path (NOT inline db) for the agent side — avoids 1MB WS frame cap."""
    inst = ALL_EVAL_SCENARIOS["tau2_retail__0"]()
    state: dict = {}
    inst.setup_shared_state(state, side="agent")
    assert state == {"db_path": "tau2_retail/db.json"}, state


def test_domain_attribute_is_tau2_retail():
    """Tau2RetailBaseScenario.domain is the registry-namespace key + data subdir name."""
    assert Tau2RetailBaseScenario.domain == "tau2_retail"
    inst = ALL_EVAL_SCENARIOS["tau2_retail__0"]()
    assert inst.domain == "tau2_retail"
