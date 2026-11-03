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

# Scenario definitions contain long prose strings (personas, instructions, account data);
# wrapping every one hurts readability without improving correctness.
# pylint: disable=line-too-long
# flake8: noqa: E501
from typing import Any, Dict, List, Optional

import pipecat.processors.frameworks.rtvi.models as RTVI
from loguru import logger
from pipecat.processors.frameworks.rtvi import RTVIProcessor
from pipecat.services.llm_service import FunctionCallParams

from nemo_voice_agent.evaluation.tools import register_schema_tool_for_eval
from nemo_voice_agent.utils.tool_calling import StandardSchemaTool


FINAL_RESPONSE_START_TAG = "<final_response>"
FINAL_RESPONSE_END_TAG = "</final_response>"
EXIT_MESSAGE_START_TAG = "<exit>"
EXIT_MESSAGE_END_TAG = "</exit>"


@register_schema_tool_for_eval
class SendRTVIMessageTool(StandardSchemaTool):
    """
    Send a scenario finished message to the evaluator.
    """

    DESCRIPTION: str = """
        Send a message to the orchestrator.
        """

    def __init__(self, *, description: Optional[str] = None, rtvi: Optional[RTVIProcessor] = None):
        if description is None:
            description = self.DESCRIPTION
        if rtvi is None:
            rtvi = RTVIProcessor()
        super().__init__(description=description)
        self._rtvi = rtvi

    @property
    def properties(self) -> Dict[str, Any]:
        """
        Return the properties for the tool.
        """
        return {
            "message": {
                "type": "string",
                "description": "The message to be sent in the required format.",
            },
        }

    @property
    def required_properties(self) -> List[str]:
        """
        Return the required properties for the tool.
        """
        return ["message"]

    async def send_rtvi_message(self, message: str) -> None:
        """
        Send a message.

        Args:
            message: The message to be sent.
        """
        message = RTVI.ServerMessage(data=RTVI.TextMessageData(text=message))
        await self._rtvi.push_transport_message(message, exclude_none=True)

    async def _execute(self, message: Optional[str] = None, **kwargs: Any) -> Any:
        """
        Send a message.

        Args:
            message: The message to be sent.
            kwargs: Any extra LLM-supplied arguments, ignored.

        Returns:
            The tool result delivered by ``StandardSchemaTool.__call__``.
        """
        await self.send_rtvi_message(message)
        return {"success": True, "message": "message sent to the RTVIclient."}


@register_schema_tool_for_eval
class SendScenarioSummaryTool(SendRTVIMessageTool):
    """
    Send a "Scnario Summary" message after the user has no more requests
    and the agent has answered all the user's questions The input message should contain all required information
    in the required format.
    """

    def __init__(self, *, rtvi: Optional[RTVIProcessor] = None, description: Optional[str] = None):
        if description is None:
            description = """
            Send a "Task Summary" message to summarize how the agent has helped the user to finish the task.
            """
        super().__init__(description=description, rtvi=rtvi)

    async def send_scenario_summary(self, message: str) -> None:
        """
        Send a "Scnario Summary" message.

        Args:
            message: The message to be sent.
        """
        message = f"{FINAL_RESPONSE_START_TAG}{message}{FINAL_RESPONSE_END_TAG}"
        logger.debug(f"Sending scenario summary message: {message}")
        await self.send_rtvi_message(message)

    async def _execute(self, message: Optional[str] = None, **kwargs: Any) -> Any:
        """
        Send a "Scnario Summary" message to the client, which
        should contain all required information for the evaluation.

        Args:
            message: The summary message to be sent.
            kwargs: Any extra LLM-supplied arguments, ignored.

        Returns:
            The tool result delivered by ``StandardSchemaTool.__call__``.
        """
        await self.send_scenario_summary(message)
        return {"success": True, "message": "Scenario summary message sent."}


@register_schema_tool_for_eval
class SendExitMessageTool(SendRTVIMessageTool):
    """
    Send an "Exit" message to indicate that the scenario is finished.
    """

    def __init__(self, rtvi: RTVIProcessor, description: Optional[str] = None):
        if description is None:
            description = (
                'Send an "Exit" message to the orchestrator to indicate that the task is finished, '
                "and it's safe to stop the pipeline. This tool should only be used when the user "
                "has no more requests and the agent has answered all the user's questions."
            )
        super().__init__(description=description, rtvi=rtvi)

    @property
    def properties(self) -> Dict[str, Any]:
        return {}

    @property
    def required_properties(self) -> List[str]:
        return []

    async def send_exit_message(self, message: str = "The task is finished.") -> None:
        """
        Send an "Exit" message.

        Args:
            message: The message to be sent.
        """
        message = f"{EXIT_MESSAGE_START_TAG}{message}{EXIT_MESSAGE_END_TAG}"
        logger.debug(f"Sending exit message: {message}")
        await self.send_rtvi_message(message)

    async def _execute(self, **kwargs: Any) -> Any:
        """
        Acknowledge the exit request. The "Exit" message itself is emitted from
        :meth:`_after_result`, i.e. *after* the result has been delivered.

        Args:
            kwargs: Any extra LLM-supplied arguments, ignored (this tool takes none).

        Returns:
            The tool result delivered by ``StandardSchemaTool.__call__``.
        """
        return {"success": True, "message": "Exit message sent."}

    async def _after_result(self, params: FunctionCallParams) -> None:
        """
        Send the "Exit" message once the function-call result is delivered.

        Order matters: ack the function call FIRST so pipecat dispatches
        ``FunctionCallResultFrame`` and the assistant aggregator commits
        both the assistant message (with tool_calls) and the tool reply
        before the bridge sees ``<exit>``. If ``send_exit_message`` ran
        first, the bridge would race the commit cycle, snapshot the
        context mid-flight, and the captured llm_context.json would be
        missing the ``EndConversationTool`` entry. That ordering is why this
        lives in the post-delivery hook rather than in :meth:`_execute`.
        """
        message = "The task is finished."
        await self.send_exit_message(message)
