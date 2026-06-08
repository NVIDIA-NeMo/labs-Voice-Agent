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
from typing import Dict, List, Optional, Type

from pipecat.processors.frameworks.rtvi import RTVIProcessor

from nemo_voice_agent.utils.tool_calling.base import StandardSchemaTool

# Per-domain tool registry: outer key is the domain (matching ``Scenario.domain``),
# inner key is the tool class name (``cls.__name__`` unless ``name`` class
# attribute is set). Tools with the same name in different domains coexist as
# distinct entries — that's the entire point. Within a single domain, duplicate
# names raise ``ValueError`` at registration time.
ALL_SCHEMA_TOOLS_FOR_EVAL: Dict[str, Dict[str, Type[StandardSchemaTool]]] = {}


def _do_register_tool(cls, domain: str):
    """Internal: validate + register a tool class into ``ALL_SCHEMA_TOOLS_FOR_EVAL[domain]``."""
    if not issubclass(cls, StandardSchemaTool):
        raise ValueError(f"Class {cls.__name__} is not a subclass of StandardSchemaTool")
    key = getattr(cls, "name", cls.__name__)
    bucket = ALL_SCHEMA_TOOLS_FOR_EVAL.setdefault(domain, {})
    if key in bucket:
        existing = bucket[key]
        raise ValueError(
            f"Tool name collision in domain '{domain}': '{key}' is already "
            f"registered by {existing.__module__}.{existing.__name__}; cannot "
            f"also register {cls.__module__}.{cls.__name__}. Rename one of the "
            f"classes, or move one to a different domain."
        )
    bucket[key] = cls
    return cls


def register_schema_tool_for_eval(arg=None, *, domain: str = "default"):
    """Class-decorator factory that registers a tool class into a per-domain bucket.

    Usage::

        @register_schema_tool_for_eval(domain="tau2_airline")
        class GetUserDetailsTool(_Tau2ReadTool):
            ...

        # Or with the positional shortcut:
        @register_schema_tool_for_eval("tau2_airline")
        class CancelReservationTool(_Tau2WriteTool):
            ...

        # Legacy: bare ``@register_schema_tool_for_eval`` (no parens) still
        # registers under ``"default"`` — used by tool modules that haven't
        # been migrated yet.
        @register_schema_tool_for_eval
        class EndConversationTool(SendExitMessageTool):
            ...

    **Why per-domain registries:** with a single global ``{name → class}`` dict,
    tools with the same short name in different domains (e.g., ``CancelReservationTool``
    in both eva and tau2) silently overwrote each other at import time. The
    alternative — long unique class names like ``Tau2AirlineCancelReservationTool``
    — crowded the LLM's tool surface. Namespacing by domain lets tools keep
    natural short names while remaining unambiguous.

    Args:
        arg: Either the class being decorated (bare ``@register_schema_tool_for_eval``)
            or a positional domain string (``@register_schema_tool_for_eval("tau2_airline")``).
        domain: Registry namespace. Defaults to ``"default"`` for tools not
            tied to a specific evaluation domain (harness tools in
            ``basic_tools.py`` / ``rtvi_control.py`` plus customer_service /
            restaurant / qa / waitlist tools).

    Raises:
        ValueError: if the class isn't a ``StandardSchemaTool`` subclass, or if
            a tool with the same name is already registered in this domain.
    """
    # Bare ``@register_schema_tool_for_eval`` (no parens) — arg is the class.
    if isinstance(arg, type):
        return _do_register_tool(arg, domain)

    # ``@register_schema_tool_for_eval("tau2_airline")`` (positional string).
    if isinstance(arg, str):
        domain = arg

    # ``@register_schema_tool_for_eval()`` / ``@register_schema_tool_for_eval(domain="X")``.
    def _decorator(cls):
        return _do_register_tool(cls, domain)

    return _decorator


def get_schema_tool_for_eval(
    name: str,
    domain: str = "default",
    rtvi: Optional[RTVIProcessor] = None,
    shared_state: Optional[dict] = None,
    **kwargs,
) -> StandardSchemaTool:
    """Instantiate a registered tool by ``(domain, name)``, falling back to ``"default"``.

    Lookup order:
      1. ``ALL_SCHEMA_TOOLS_FOR_EVAL[domain][name]``  — exact match.
      2. ``ALL_SCHEMA_TOOLS_FOR_EVAL["default"][name]`` — fallback for shared
         harness tools (``EndConversationTool``, ``SendScenarioSummaryTool``, …)
         that aren't per-domain. A warning is logged so silent fallthroughs are
         visible in the bot logs.

    Raises ``KeyError`` if neither the specified domain nor ``"default"`` has
    a tool by that name. Catches typos (``domain="tau2_arline"``) and missing-
    tool issues at the tool-instantiation step rather than silently down the line.

    Args:
        name: Tool class name (e.g., ``"GetUserDetailsTool"``).
        domain: Registry namespace to look in first (e.g., ``"tau2_airline"``).
            Defaults to ``"default"`` (which makes the fallback path a no-op).
        rtvi: RTVI processor passed to the tool if its constructor accepts it.
        shared_state: Per-scenario mutable state dict passed to the tool if
            its constructor accepts it.
        **kwargs: Additional keyword arguments forwarded to the tool constructor.

    Raises:
        KeyError: if ``name`` isn't registered in ``domain`` or ``"default"``.
    """
    # Lazy-import to avoid coupling the registry to loguru at module load.
    from loguru import logger

    bucket = ALL_SCHEMA_TOOLS_FOR_EVAL.get(domain, {})
    tool_class = bucket.get(name)
    if tool_class is None and domain != "default":
        default_bucket = ALL_SCHEMA_TOOLS_FOR_EVAL.get("default", {})
        tool_class = default_bucket.get(name)
        if tool_class is not None:
            logger.warning(
                f"Tool '{name}' not found in domain '{domain}', falling back to "
                f"'default' (resolved to {tool_class.__module__}.{tool_class.__name__}). "
                f"Move the tool to '{domain}' or treat as a shared harness tool."
            )
    if tool_class is None:
        available_in_domain = sorted(bucket.keys()) if bucket else []
        available_in_default = sorted(ALL_SCHEMA_TOOLS_FOR_EVAL.get("default", {}).keys())
        raise KeyError(
            f"Tool '{name}' not found in domain '{domain}' or 'default'. "
            f"Available in '{domain}': {available_in_domain}. "
            f"Available in 'default': {available_in_default}."
        )
    sig = inspect.signature(tool_class)
    inject_kwargs = {}
    if "rtvi" in sig.parameters:
        inject_kwargs["rtvi"] = rtvi
    if "shared_state" in sig.parameters:
        inject_kwargs["shared_state"] = shared_state
    return tool_class(**inject_kwargs, **kwargs)


def list_schema_tools_for_eval(domain: Optional[str] = None) -> List[str]:
    """List registered tool names.

    Args:
        domain: If provided, list tools in that domain only. If ``None``,
            returns ``"domain.name"`` strings across all domains for debugging.
    """
    if domain is not None:
        return list(ALL_SCHEMA_TOOLS_FOR_EVAL.get(domain, {}).keys())
    return [f"{d}.{n}" for d, bucket in ALL_SCHEMA_TOOLS_FOR_EVAL.items() for n in bucket]


import nemo_voice_agent.evaluation.tools.basic_tools
import nemo_voice_agent.evaluation.tools.customer_service_tools  # noqa: E402, F401
import nemo_voice_agent.evaluation.tools.eva_airline_tools  # noqa: E402, F401
import nemo_voice_agent.evaluation.tools.restaurant_tools  # noqa: E402, F401
import nemo_voice_agent.evaluation.tools.tau2_airline_tools  # noqa: E402, F401
import nemo_voice_agent.evaluation.tools.tau2_retail_tools  # noqa: E402, F401
import nemo_voice_agent.evaluation.tools.tau2_telecom_tools  # noqa: E402, F401
import nemo_voice_agent.evaluation.tools.tau2_telecom_user_tools  # noqa: E402, F401

# Import subpackages to trigger @register_schema_tool_for_eval decorators.
# Must be at the end to avoid circular imports (data modules import register_schema_tool_for_eval).
import nemo_voice_agent.evaluation.tools.rtvi_control
import nemo_voice_agent.evaluation.tools.waitlist_tools  # noqa: E402, F401
