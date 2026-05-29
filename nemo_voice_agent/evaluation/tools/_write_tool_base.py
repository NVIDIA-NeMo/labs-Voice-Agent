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

"""Domain-agnostic base class for write-tools that record actions.

Promoted out of ``eva_airline_tools.WriteAirlineTool`` so that tau2 domains
(airline / retail / telecom) can reuse the same recording machinery without
duplicating ``_record_action`` / ``_next_call_index`` logic. Each domain ships
its own ACTION_TYPES vocabulary (e.g. ``AIRLINE_ACTION_TYPES``,
``TAU2_AIRLINE_ACTION_TYPES``) and a thin subclass that binds it via the
``ACTION_TYPES`` ClassVar.

Side-tagging is **not** done here — tool code cannot know which bot (user vs.
agent) it's running inside. The bridge stamps ``side`` at scenario-end merge
time based on which WebSocket produced the record; the gold-replay path stamps
``side`` based on the action's ``requestor`` field. See plan §7 Q3.
"""

from typing import ClassVar, List

from loguru import logger

from nemo_voice_agent.utils.tool_calling import StandardSchemaTool


class WriteScenarioTool(StandardSchemaTool):
    """Base class for evaluation tools that mutate shared_state and emit an action record.

    On a successful tool call, the subclass calls ``self._record_action(record)`` with
    a dict whose ``action_type`` matches one of the domain's ``ACTION_TYPES`` entries.
    Records are accumulated in ``shared_state["actions"]``; the bridge pulls them at
    end-of-scenario via the ``get_scenario_summary`` RTVI action.

    Read-only tools subclass ``StandardSchemaTool`` directly — only writes record.

    Subclasses must override ``ACTION_TYPES`` (e.g. ``ACTION_TYPES = AIRLINE_ACTION_TYPES``).
    The base's empty default means an unconfigured subclass logs a warning on every
    record but doesn't raise — preserving forward compatibility while making the bug
    visible in logs.
    """

    ACTION_TYPES: ClassVar[List[str]] = []

    def _record_action(self, action: dict) -> None:
        """Append a structured action record to shared_state['actions']."""
        if action.get("action_type") not in self.ACTION_TYPES:
            logger.warning(
                f"{type(self).__name__}._record_action: action_type "
                f"{action.get('action_type')!r} not in ACTION_TYPES"
            )
        self.state.setdefault("actions", []).append(action)

    def _next_call_index(self, tool_name: str) -> int:
        """Increment and return the call counter for ``tool_name``.

        Used by tools that mint unique IDs (refund_id, transfer_id, etc.) and need
        a stable per-scenario counter.
        """
        counts = self.state.setdefault("_call_counts", {})
        counts[tool_name] = counts.get(tool_name, 0) + 1
        return counts[tool_name]
