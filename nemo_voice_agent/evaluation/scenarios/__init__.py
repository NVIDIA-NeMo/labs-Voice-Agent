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

from typing import Dict, List, Optional

from nemo_voice_agent.evaluation.scenarios.classes import Scenario

ALL_EVAL_SCENARIOS: Dict[str, Scenario] = {}


# Agent-behavior guideline paired with ``EndConversationTool`` (a harness tool
# in ``nemo_voice_agent.evaluation.tools``). Voice conversations need an
# explicit termination signal so the bridge can detect end-of-scenario and
# pull the per-scenario summary; without this guideline the agent tends to
# trail off without firing the tool, which the bridge then reports as
# ``[TIMEOUT]``. Shared across all eval domains (tau2_*, eva_airline, etc.) —
# any scenario whose ``agent_resources.tools`` includes ``EndConversationTool``
# should append this to its agent prompt.
#
# Lives in ``scenarios/__init__.py`` (not ``utils/``) because it's tied to an
# evaluation-framework tool that doesn't exist in production. The pure
# voice-realization rules (concise speech, spell-out-IDs) live in
# ``nemo_voice_agent.utils.voice_prompts`` — those apply to any voice agent.
END_CONVERSATION_GUIDELINE = (
    "When the user has indicated they have no further requests and you have "
    "exchanged goodbyes, call `EndConversationTool` to end the call. Do not "
    "call it before the user has explicitly confirmed they are done — premature "
    "termination is a failure mode. Typical flow: (a) confirm all actions are "
    "complete, (b) ask 'is there anything else you need help with?', (c) wait "
    "for the user's response, (d) if they say goodbye or say nothing is needed, "
    "say goodbye back and then call `EndConversationTool`. This tool should also "
    "be called if you have transferred the call to a human agent successfully."
)


# Agent-behavior guideline that prevents the "claim-without-doing" failure
# mode: the agent discusses an action, the user pivots before confirming, the
# agent never fires the corresponding tool, but later includes the action in a
# summary as if it had been completed. Surfaced live on 2026-06-03 in
# tau2_retail__16 (the agent reported a $4,777.75 refund for an order it had
# never cancelled and even fed that amount into ``CalculateTool``).
#
# **Opt-in, eval-only, tool-coupled** — explicitly references tools, so it
# only makes sense for agents whose ``agent_resources.tools`` is non-empty.
# Putting this in ``GENERAL_PROMPT`` (in ``utils.voice_prompts``) would
# paralyze tool-less scenarios (qa, fastbite) where no tool call can ever
# satisfy "a step counts as completed only when you successfully called a
# tool that performed it". The user simulator also never includes this rule —
# it isn't executing actions, just describing intent.
EXECUTION_HONESTY_GUIDELINE = (
    "Distinguish 'discussed' from 'done': a step counts as completed only when "
    "you successfully called a tool that performed it. In summaries, totals, "
    "and status updates, include only items that were actually executed via "
    "tools — never items that were merely planned, attempted, or unconfirmed "
    "by the user."
)


def register_eval_scenario(cls):
    """Class decorator that registers a scenario class into ALL_EVAL_SCENARIOS."""
    if not issubclass(cls, Scenario):
        raise ValueError(f"Class {cls.__name__} is not a subclass of Scenario")
    key = getattr(cls, "name", cls.__name__)
    ALL_EVAL_SCENARIOS[key] = cls
    return cls


def get_eval_scenario(name: str, **kwargs) -> Optional[Scenario]:
    """
    Get an evaluation scenario by name.
    """
    if name not in ALL_EVAL_SCENARIOS:
        return None
    return ALL_EVAL_SCENARIOS[name](**kwargs)


def list_eval_scenarios() -> List[str]:
    """
    List all evaluation scenarios.
    """
    return list(ALL_EVAL_SCENARIOS.keys())


# Import data subpackage to trigger @register_eval_scenario decorators.
# Must be at the end to avoid circular imports (data modules import register_eval_scenario).
import nemo_voice_agent.evaluation.scenarios.data  # noqa: E402, F401
