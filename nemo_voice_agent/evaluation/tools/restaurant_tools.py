# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from typing import Any, Dict, List, Optional

from loguru import logger
from pipecat.services.llm_service import FunctionCallParams

from nemo_voice_agent.evaluation.tools import register_schema_tool_for_eval
from nemo_voice_agent.utils.tool_calling import StandardSchemaTool


@register_schema_tool_for_eval
class GetMenuTool(StandardSchemaTool):
    """Returns the restaurant menu. Menu content is configured per scenario."""

    def __init__(self, *, menu: str = "", description: Optional[str] = None):
        super().__init__(description=description or "Get the restaurant menu to see available items and prices.")
        self.menu = menu

    @property
    def properties(self) -> Dict[str, Any]:
        return {}

    @property
    def required_properties(self) -> List[str]:
        return []

    async def _execute(self, params: FunctionCallParams) -> None:
        logger.debug(f"GetMenuTool returning menu ({len(self.menu)} chars)")
        await params.result_callback({"menu": self.menu})
