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
# Scenario fixtures (evaluation/data/tau2_airline/{db,tasks,tasks_voice,split_tasks,policy.md}.json)
# are adapted from https://github.com/sierra-research/tau2-bench/tree/voice-user-sim-v1.0
# (MIT-licensed). See evaluation/data/README.md for the upstream pin (commit 17e07b1).

"""Tau2-airline scenario base + seed scenarios.

``Tau2AirlineBaseScenario`` glues together:

- ``Tau2BaseScenario`` (cross-domain machinery: voice-task index, ``_gold_replay``,
  Persona/Task stubs, structured user side, ``policy.md`` loader)
- ``TAU2_AIRLINE_TOOL_NAME_TO_CLASS`` (the 14 ported tools)

Each scenario subclass only needs to set ``tau2_id`` (e.g. ``"0"``). Everything
else (current_date, tool map, db, policy, expected_scenario_db,
reference_answer, user_persona / user_task / user_actions) derives from the
upstream data files via cached properties.

Hand-authored seed scenarios live at the bottom of this file. Auto-scaffolded
scenarios land in ``group_Nx.py`` siblings via the gitignored scaffold
generator in ``nemo_experiments/``.
"""

# pylint: disable=line-too-long
# flake8: noqa: E501

from functools import cached_property
from typing import Any, Dict

from nemo_voice_agent.evaluation.scenarios.classes import Resources, SuccessSignal
from nemo_voice_agent.evaluation.scenarios.data.tau2_common import Tau2BaseScenario
from nemo_voice_agent.evaluation.tools.tau2_airline_tools import TAU2_AIRLINE_TOOL_NAME_TO_CLASS


class Tau2AirlineBaseScenario(Tau2BaseScenario):
    """Base class for scenarios ported from tau2-bench/airline (voice-user-sim-v1.0).

    Subclasses must set ``tau2_id`` (the key from ``tasks.json``). For example::

        @register_eval_scenario
        class Tau2Airline0(Tau2AirlineBaseScenario):
            name = "tau2_airline__0"
            tau2_id = "0"

    Everything else (current_date, tool registry, expected DB, reference action
    list, user persona/task) derives from the upstream files via
    ``Tau2BaseScenario`` machinery.
    """

    domain: str = "tau2_airline"
    # Path-independent DB-hash equality is the principled outcome signal for
    # tau2_airline. No predicates / NL assertions; judge becomes informational
    # so the verdict doesn't depend on judge availability.
    success_signals = (SuccessSignal.DB_STATE_MATCH,)

    # tau2_airline is single-side (agent only); user side is text-driven by the
    # user simulator but the tau2 toolset doesn't include user-side tools.
    has_user_state: bool = False

    def _build_tool_map(self, state: dict) -> Dict[str, Any]:
        """Instantiate one of each ported tool, bound to the given ``state``.

        Used by ``Tau2BaseScenario._gold_replay`` to dispatch reference actions.
        For live runs the agent bot constructs its own tool instances via the
        usual ``register_schema_tools_to_llm`` path with ``shared_state=state``.
        """
        return {name: cls(shared_state=state) for name, cls in TAU2_AIRLINE_TOOL_NAME_TO_CLASS.items()}

    @cached_property
    def agent_resources(self) -> Resources:
        """Full tau2_airline tool surface (all 14 ported tools) + per-scenario info.

        Tool keys are the **Tool class names** (matching the ``@register_schema_tool_for_eval``
        registry keys); each maps to an empty kwargs dict because tau2 tools take only
        ``shared_state`` which the bot server injects automatically. The scenario DB
        flows in via ``setup_shared_state`` (seeded with a deep copy of ``db.json``).

        Note: this OVERRIDES ``Tau2BaseScenario.agent_resources`` (which returns an
        empty stub). The stub exists for Scenario-contract introspection; the live
        run needs the populated dict to actually register tools.
        """
        return Resources(
            tools={
                # Reads (6)
                "GetUserDetailsTool": {},
                "GetReservationDetailsTool": {},
                "ListAllAirportsTool": {},
                "SearchDirectFlightTool": {},
                "SearchOnestopFlightTool": {},
                "GetFlightStatusTool": {},
                # Writes (6)
                "BookReservationTool": {},
                "CancelReservationTool": {},
                "SendCertificateTool": {},
                "UpdateReservationBaggagesTool": {},
                "UpdateReservationFlightsTool": {},
                "UpdateReservationPassengersTool": {},
                # Generics (2)
                "CalculateTool": {},
                "TransferToHumanAgentsTool": {},
                # Harness tool — resolved via the registry's "default" namespace
                # fallback (lives in basic_tools.py, not tau2_airline_tools.py).
                # Tau2's policy.md is text-mode authored and never mentions an
                # end-of-call signal; the voice harness needs one so the bridge
                # can pull the summary. See ``END_CONVERSATION_GUIDELINE`` in
                # ``nemo_voice_agent.evaluation.scenarios`` for the prompt-side
                # guidance that pairs with this.
                "EndConversationTool": {},
            },
            information=[],
        )
