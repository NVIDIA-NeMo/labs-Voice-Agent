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

import asyncio
from typing import ClassVar, List

import pipecat.processors.frameworks.rtvi.models as RTVI
from loguru import logger
from pipecat.processors.frameworks.rtvi import RTVIProcessor

from nemo_voice_agent.utils.tool_calling import StandardSchemaTool


_EXIT_START = "<exit>"
_EXIT_END = "</exit>"


class WriteScenarioTool(StandardSchemaTool):
    """Base class for evaluation tools that mutate shared_state and emit an action record.

    On a successful tool call, the subclass calls ``self._record_action(record)`` with
    a dict whose ``action_type`` matches one of the domain's ``ACTION_TYPES`` entries.
    Records are accumulated in ``shared_state["actions"]``; the bridge pulls them at
    end-of-scenario via the ``get_scenario_summary`` RTVI action.

    Additionally — when the bot's ``shared_state`` contains the
    ``__rtvi__`` sentinel set by ``rtvi_actions.create_update_system_prompt_action``
    — ``_record_action`` ALSO pushes an ``action-applied``
    ``RTVI.ServerMessage`` to the bridge. The bridge listens for it and
    triggers the cross-side sync pipeline (replay action onto shadow
    DBs → run ``scenario.sync_state`` → dispatch deltas to the other
    bot). Domains that don't opt into sync (single-side eva / airline /
    retail) ignore the message harmlessly.

    Read-only tools subclass ``StandardSchemaTool`` directly — only writes record.

    Subclasses must override ``ACTION_TYPES`` (e.g. ``ACTION_TYPES = AIRLINE_ACTION_TYPES``).
    The base's empty default means an unconfigured subclass logs a warning on every
    record but doesn't raise — preserving forward compatibility while making the bug
    visible in logs.
    """

    ACTION_TYPES: ClassVar[List[str]] = []

    def _record_action(self, action: dict) -> None:
        """Append a structured action record to shared_state['actions'].

        Also emits an ``action-applied`` RTVI.ServerMessage when the bot
        has a registered RTVI processor (stashed under
        ``shared_state["__rtvi__"]``). Bridge consumes this to drive
        cross-side sync. The push is fire-and-forget — failures
        are logged but don't block the tool's normal return path.
        """
        if action.get("action_type") not in self.ACTION_TYPES:
            logger.warning(
                f"{type(self).__name__}._record_action: action_type {action.get('action_type')!r} not in ACTION_TYPES"
            )
        self.state.setdefault("actions", []).append(action)
        self._emit_action_applied(action)

    def _emit_action_applied(self, action: dict) -> None:
        """Push ``action-applied`` RTVI.ServerMessage if RTVI is available.

        The bot side has the RTVI processor under
        ``shared_state["__rtvi__"]`` (stashed in
        ``create_update_system_prompt_action``). When this sentinel is
        absent — for example in unit tests that don't construct a bot
        server, or in gold-replay paths that run tools in-process — we
        silently skip the push.
        """
        rtvi = self.state.get("__rtvi__")
        if rtvi is None:
            return

        if not isinstance(rtvi, RTVIProcessor):
            logger.warning(f"RTVI processor not found in shared_state, got {type(rtvi)}: {rtvi}")
            return

        try:
            domain = self.state.get("__tool_domain__", "default")
            payload = {
                "type": "action-applied",
                "action": action,
                "tool_domain": domain,
            }
            message = RTVI.ServerMessage(data=payload)
            # ``push_transport_message`` is the pipecat API for sending
            # an ``RTVI.ServerMessage`` to the WebSocket client (the
            # bridge in our setup). ``RTVI.ServerMessage`` is NOT a
            # ``Frame`` subclass — ``push_frame`` fails with
            # ``AttributeError: 'RTVI.ServerMessage' object has no
            # attribute 'transport_destination'`` because pipecat's
            # frame plumbing expects Frame fields. Same call shape used
            # by ``SendRTVIMessageTool.send_rtvi_message`` for the
            # ``<exit>`` / ``<final_response>`` signals.
            coro = rtvi.push_transport_message(message, exclude_none=True)
            try:
                loop = asyncio.get_running_loop()
                loop.create_task(coro)
            except RuntimeError:
                # No running loop — best-effort schedule on a fresh one.
                # In practice this branch only fires in unit tests; the
                # bot server always has an event loop.
                try:
                    asyncio.get_event_loop().create_task(coro)
                except RuntimeError:
                    logger.debug("action-applied push skipped: no event loop available")
        except Exception as exc:
            logger.warning(f"action-applied push failed for {action.get('name')!r}: {type(exc).__name__}: {exc}")

    async def _send_exit_message(self) -> None:
        """Emit ``<exit>`` to the bridge, ending the scenario immediately.

        Called by terminal tools (transfer-to-agent/human) from their
        ``_after_result`` override, which ``StandardSchemaTool.__call__``
        invokes only once the tool result has been delivered — so pipecat
        commits the tool-call record before the bridge tears down the
        session. Do **not** call this from ``_execute``: that runs before
        delivery and would race the teardown against the result. Silently
        no-ops when RTVI is absent (unit tests, gold-replay).
        """
        rtvi = self.state.get("__rtvi__")
        if rtvi is None:
            return
        text = f"{_EXIT_START}Transfer complete.{_EXIT_END}"
        msg = RTVI.ServerMessage(data=RTVI.TextMessageData(text=text))
        try:
            await rtvi.push_transport_message(msg, exclude_none=True)
        except Exception as exc:
            logger.warning(f"exit-message push failed: {type(exc).__name__}: {exc}")

    def _next_call_index(self, tool_name: str) -> int:
        """Increment and return the call counter for ``tool_name``.

        Used by tools that mint unique IDs (refund_id, transfer_id, etc.) and need
        a stable per-scenario counter.
        """
        counts = self.state.setdefault("_call_counts", {})
        counts[tool_name] = counts.get(tool_name, 0) + 1
        return counts[tool_name]
