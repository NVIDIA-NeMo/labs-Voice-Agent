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

import inspect
from typing import Dict, List, Optional

from pipecat.processors.frameworks.rtvi import RTVIProcessor

from nemo_voice_agent.utils.tool_calling.base import StandardSchemaTool

ALL_SCHEMA_TOOLS_FOR_EVAL: Dict[str, StandardSchemaTool] = {}


def register_schema_tool_for_eval(cls):
    """Class decorator that registers a tool class into ``ALL_SCHEMA_TOOLS_FOR_EVAL``.

    Registry key is ``cls.name`` if set as a class attribute, otherwise ``cls.__name__``.

    **Naming convention** (enforced by the collision check below):

    Tool classes must be prefixed with their data source + domain so the global
    registry doesn't have ambiguous keys. Pattern: ``<SourceDomain><BaseToolName>``.

    Examples:
        - ``EvaAirlineGetReservationTool``      (source: eva, domain: airline)
        - ``Tau2AirlineGetUserDetailsTool``     (source: tau2, domain: airline)
        - ``Tau2RetailGetOrderDetailsTool``     (source: tau2, domain: retail)

    Without the prefix, identically-named tools from different domains
    (e.g., ``GetReservationTool`` in eva vs ``GetReservationDetailsTool`` in tau2)
    would silently overwrite each other in this dict at import time, leaving
    only the last-imported one usable.

    Raises:
        ValueError: if a tool with the same name is already registered.
    """
    if not issubclass(cls, StandardSchemaTool):
        raise ValueError(f"Class {cls.__name__} is not a subclass of StandardSchemaTool")
    key = getattr(cls, "name", cls.__name__)
    if key in ALL_SCHEMA_TOOLS_FOR_EVAL:
        existing = ALL_SCHEMA_TOOLS_FOR_EVAL[key]
        raise ValueError(
            f"Tool name collision in ALL_SCHEMA_TOOLS_FOR_EVAL: '{key}' is already "
            f"registered by {existing.__module__}.{existing.__name__}; cannot also "
            f"register {cls.__module__}.{cls.__name__}. Per the naming convention, "
            f"prefix tool classes with their data source + domain "
            f"(e.g., 'Tau2AirlineGetUserDetailsTool', 'EvaAirlineGetReservationTool')."
        )
    ALL_SCHEMA_TOOLS_FOR_EVAL[key] = cls
    return cls


def get_schema_tool_for_eval(
    name: str,
    rtvi: Optional[RTVIProcessor] = None,
    shared_state: Optional[dict] = None,
    **kwargs,
) -> StandardSchemaTool:
    """
    Get a schema tool for evaluation by name, and initialize the tool with the given arguments.

    Args:
        name: The name of the tool.
        rtvi: The RTVI processor to use for sending messages to the evaluator.
        shared_state: A shared mutable dict for tools within the same scenario to exchange state.
            Created once per scenario and passed to all tools that accept it.
        kwargs: The additional keyword arguments to pass to the tool constructor.

    Returns:
        The schema tool for evaluation.
    """
    if name not in ALL_SCHEMA_TOOLS_FOR_EVAL:
        return None
    tool_class = ALL_SCHEMA_TOOLS_FOR_EVAL[name]
    sig = inspect.signature(tool_class)
    inject_kwargs = {}
    if "rtvi" in sig.parameters:
        inject_kwargs["rtvi"] = rtvi
    if "shared_state" in sig.parameters:
        inject_kwargs["shared_state"] = shared_state
    return tool_class(**inject_kwargs, **kwargs)


def list_schema_tools_for_eval() -> List[StandardSchemaTool]:
    """
    List all schema tools for evaluation.
    """
    return list(ALL_SCHEMA_TOOLS_FOR_EVAL.keys())


import nemo_voice_agent.evaluation.tools.basic_tools
import nemo_voice_agent.evaluation.tools.customer_service_tools  # noqa: E402, F401
import nemo_voice_agent.evaluation.tools.eva_airline_tools  # noqa: E402, F401
import nemo_voice_agent.evaluation.tools.restaurant_tools  # noqa: E402, F401
import nemo_voice_agent.evaluation.tools.tau2_airline_tools  # noqa: E402, F401

# Import subpackages to trigger @register_schema_tool_for_eval decorators.
# Must be at the end to avoid circular imports (data modules import register_schema_tool_for_eval).
import nemo_voice_agent.evaluation.tools.rtvi_control
import nemo_voice_agent.evaluation.tools.waitlist_tools  # noqa: E402, F401
