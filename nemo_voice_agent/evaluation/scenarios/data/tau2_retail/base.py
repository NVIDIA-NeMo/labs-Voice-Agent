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
#
# Scenario fixtures (evaluation/data/tau2_retail/{db,tasks,tasks_voice,split_tasks,policy.md}.json)
# are adapted from https://github.com/sierra-research/tau2-bench/tree/voice-user-sim-v1.0
# (MIT-licensed). See evaluation/data/README.md for the upstream pin (commit 17e07b1).

"""Tau2-retail scenario base.

``Tau2RetailBaseScenario`` mirrors the airline base, plus exposes
``nl_assertions`` from ``evaluation_criteria.nl_assertions`` for the LLM judge.
Each scenario subclass only needs to set ``tau2_id`` (e.g. ``"0"``).
"""

# pylint: disable=line-too-long
# flake8: noqa: E501

from functools import cached_property
from typing import Any, Dict, List, Optional

from nemo_voice_agent.evaluation.scenarios.classes import Resources, SuccessSignal
from nemo_voice_agent.evaluation.scenarios.data.tau2_common import Tau2BaseScenario
from nemo_voice_agent.evaluation.tools.tau2_retail_tools import TAU2_RETAIL_TOOL_NAME_TO_CLASS


class Tau2RetailBaseScenario(Tau2BaseScenario):
    """Base class for scenarios ported from tau2-bench/retail (voice-user-sim-v1.0).

    Subclasses must set ``tau2_id`` (the key from ``tasks.json``). For example::

        @register_eval_scenario
        class Tau2Retail0(Tau2RetailBaseScenario):
            name = "tau2_retail__0"
            tau2_id = "0"

    Adds one signal beyond airline: ``nl_assertions``. 40 of 114 retail tasks
    carry natural-language assertions in ``evaluation_criteria.nl_assertions``
    (e.g. *"Agent should tell the user that there are 10 t-shirt options
    available."*). The runner threads these into ``LLMJudge.judge_scenario``,
    which returns per-assertion verdicts and a per-domain ``nl_assertion_success_rate``.
    """

    domain: str = "tau2_retail"

    # tau2_retail is single-side (agent only). Only telecom uses user_db.
    has_user_state: bool = False

    @cached_property
    def success_signals(self) -> tuple:
        """DB-state hash + (when present) per-assertion NL judge verdicts.

        40 of 114 retail tasks carry ``nl_assertions``; the rest don't. The
        rule is mechanical — derive from ``self.nl_assertions`` so it
        cannot drift from the per-task opt-in. ``JUDGE_PASSED`` (overall
        judge score) is intentionally NOT gating: ``DB_STATE_MATCH`` is
        deterministic, and ``NL_ASSERTION`` is already per-claim judge
        verdicts when present — the overall judge becomes informational.
        """
        if self.nl_assertions:
            return (SuccessSignal.DB_STATE_MATCH, SuccessSignal.NL_ASSERTION, SuccessSignal.CLEAN_EXIT)
        return (SuccessSignal.DB_STATE_MATCH, SuccessSignal.CLEAN_EXIT)

    def _build_tool_map(self, state: dict) -> Dict[str, Any]:
        """Instantiate one of each ported retail tool, bound to the given ``state``.

        Used by ``Tau2BaseScenario._gold_replay`` to dispatch reference actions.
        """
        return {name: cls(shared_state=state) for name, cls in TAU2_RETAIL_TOOL_NAME_TO_CLASS.items()}

    @cached_property
    def nl_assertions(self) -> Optional[List[str]]:
        """Per-task natural-language assertions for the LLM judge.

        Pulled directly from tau2's ``evaluation_criteria.nl_assertions`` —
        a list of strings like *"Agent should tell the user that there are
        10 t-shirt options available."*. ``None`` (not ``[]``) when the task
        has no assertions, so the runner's ``if scenario_nl_assertions:`` guard
        in ``runner.py`` correctly skips verdict aggregation for action-only
        tasks. Empty list from upstream is also normalized to ``None`` for the
        same reason.
        """
        criteria = self.tau2_task.get("evaluation_criteria") or {}
        assertions = criteria.get("nl_assertions")
        if not assertions:
            return None
        return list(assertions)

    @cached_property
    def agent_resources(self) -> Resources:
        """Full tau2_retail tool surface (all 16 ported tools) + per-scenario info.

        OVERRIDES ``Tau2BaseScenario.agent_resources`` (which returns an empty
        stub). Tool keys are the **Tool class names** (registry lookup); each
        maps to an empty kwargs dict because tau2 tools take only ``shared_state``
        which the bot server injects automatically.

        ``EndConversationTool`` resolves via the registry's "default" namespace
        fallback (it lives in basic_tools.py, not tau2_retail_tools.py). The
        paired prompt-side guideline lives in
        ``nemo_voice_agent.evaluation.scenarios.END_CONVERSATION_GUIDELINE``.
        """
        return Resources(
            tools={
                # Reads (7)
                "FindUserIdByNameZipTool": {},
                "FindUserIdByEmailTool": {},
                "GetOrderDetailsTool": {},
                "GetProductDetailsTool": {},
                "GetItemDetailsTool": {},
                "GetUserDetailsTool": {},
                "ListAllProductTypesTool": {},
                # Writes (7)
                "CancelPendingOrderTool": {},
                "ExchangeDeliveredOrderItemsTool": {},
                "ModifyPendingOrderAddressTool": {},
                "ModifyPendingOrderItemsTool": {},
                "ModifyPendingOrderPaymentTool": {},
                "ModifyUserAddressTool": {},
                "ReturnDeliveredOrderItemsTool": {},
                # Generics (2)
                "CalculateTool": {},
                "TransferToHumanAgentsTool": {},
                # Harness tool — default-namespace fallback (lives in basic_tools.py).
                "EndConversationTool": {},
            },
            information=[],
        )
