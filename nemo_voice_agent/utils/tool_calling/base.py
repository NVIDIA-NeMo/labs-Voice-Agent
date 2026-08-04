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

from typing import Any, Dict, List, Optional

from loguru import logger
from pipecat.adapters.schemas.function_schema import FunctionSchema
from pipecat.adapters.schemas.tools_schema import ToolsSchema
from pipecat.processors.aggregators.llm_context import LLMContext
from pipecat.services.llm_service import FunctionCallParams
from pipecat.services.openai.llm import OpenAILLMService


def _normalize_empty_result(result: Any) -> Any:
    """Guard a tool result against pipecat's empty-result masking.

    Pipecat's function-call handlers substitute the literal string
    ``"COMPLETED"`` whenever a result is falsy — the check is ``if
    frame.result:`` (e.g. ``pipecat/services/openai/llm.py``
    ``handle_function_call_result``). An empty ``list``/``dict`` is a
    *meaningful* result for a read tool ("no records / no match found"), so
    returning a bare ``[]`` or ``{}`` gets silently rewritten to
    ``"COMPLETED"`` — which the LLM reads as *success*. That is the failure
    behind an agent believing a name+DOB customer lookup that matched nobody
    had "completed" successfully (observed in tau2_telecom).

    Wrap any falsy result in an explicit, non-falsy envelope so pipecat
    serializes it verbatim and the model gets an honest "empty" signal.
    Truthy results pass through unchanged.

    Applied by ``StandardSchemaTool.__call__`` at the single pipecat-facing
    boundary, so no tool can forget it. It must NOT be applied inside
    ``_execute``/``_do_work``/``invoke``: the sync ``invoke`` path feeds gold
    replay and shadow-DB cross-side sync, which expect the raw return shape
    (e.g. a bare list of matches); rewrapping there would corrupt DB-hash
    comparison and ``sync_state``.
    """
    if result:  # non-empty / truthy — leave untouched
        return result
    if isinstance(result, list):
        return {"status": "success", "results": [], "count": 0, "message": "No matching records found."}
    if isinstance(result, dict):  # empty dict
        return {"status": "success", "results": {}, "message": "No matching records found."}
    # None / "" / 0 / False — preserve the value, just make the envelope non-falsy.
    return {"status": "success", "result": result, "message": "Completed with no data returned."}


class StandardSchemaTool:
    """
    Base class for all standard tools with FunctionSchema.
    """

    def __init__(self, *, description: Optional[str] = None, name: Optional[str] = None):
        if name is None:
            # Honor a class-level ``name`` attribute when set (e.g.
            # ``class ToggleDataTool(...): name = "toggle_data"``). The
            # eval tool registry at ``nemo_voice_agent.evaluation.tools``
            # also reads ``cls.name`` for its registry key, so setting
            # this single class attribute unifies the registry key, the
            # LLM-visible function name, and the ``_build_tool_map``
            # gold-replay key under one snake_case identifier — matching
            # how the upstream tau2 policy documents reference tools
            # (e.g. ``toggle_airplane_mode()``).
            #
            # Filter to ``str`` so type-annotation-only declarations like
            # ``name: ClassVar[Optional[str]] = None`` don't get picked up
            # as the literal value ``None``. Falls back to the Python
            # class name for tools that don't set this attribute (every
            # airline / retail / harness tool today).
            cls_name = getattr(type(self), "name", None)
            name = cls_name if isinstance(cls_name, str) else self.__class__.__name__
        self.name = name
        self.description = description if description is not None else ""
        if not self.name:
            raise ValueError(f"Name is required for tool {self.__class__}")
        if not self.description:
            raise ValueError(f"Description is required for tool {self.__class__}")

    @property
    def schema(self) -> FunctionSchema:
        """
        Return the FunctionSchema for the tool. Refer to
        https://docs.pipecat.ai/guides/learn/function-calling#using-the-standard-schema-recommended
        for more details.

        An example of the FunctionSchema:
        ```
        schema = FunctionSchema(
            name="get_current_weather",
            description="Get the current weather in a location",
            properties={
                "location": {
                    "type": "string",
                    "description": "The city and state, e.g. San Francisco, CA",
                },
                "format": {
                    "type": "string",
                    "enum": ["celsius", "fahrenheit"],
                    "description": "The temperature unit to use.",
                },
            },
            required=["location", "format"]
        )
        ```
        """
        return FunctionSchema(
            name=self.name,
            description=self.description,
            properties=self.properties,
            required=self.required_properties,
        )

    async def __call__(self, params: FunctionCallParams) -> None:
        """Pipecat entry point. Owns everything framework-facing.

        This is the *only* place a tool result is delivered. Subclasses
        implement :meth:`_execute`, which takes the call arguments as plain
        keyword arguments and returns a plain result — it never touches
        ``params`` and never delivers.

        Keeping delivery here makes double-delivery structurally impossible.
        Previously ``_execute`` received ``params`` and each implementation
        called ``params.result_callback`` itself, while this method *also*
        delivered ``_execute``'s return value — so every call sent its real
        result and then a second, spurious ``None`` for the same
        ``tool_call_id``. Pipecat <1.0 ignored the duplicate; 1.x tracks
        in-flight tool calls and rejects it with "tool_call_id ... is not
        running", which strands the aggregator's deferred context push and
        stops the LLM being re-invoked with the tool output.
        """
        try:
            result = await self._execute(**(params.arguments or {}))
        except Exception as e:
            # `logger.opt(exception=True)`, not `exc_info=True`: loguru treats
            # any extra kwarg as a `str.format()` argument, so a message
            # containing braces (e.g. "the item is: {}.") would raise
            # IndexError from inside the error handler and mask the real error.
            logger.opt(exception=True).error(f"Error in tool `{self.name}`: {e}")
            await params.result_callback({"error": str(e)})
            return
        # normalize_tool_result equivalent: an empty list/dict is a meaningful
        # answer ("no matches"), but pipecat's `if frame.result:` check would
        # rewrite it to the literal "COMPLETED", which the LLM reads as
        # success. Applied here so no tool can forget it.
        await params.result_callback(_normalize_empty_result(result))
        await self._after_result(params)

    async def _after_result(self, params: FunctionCallParams) -> None:
        """Hook for side effects that must run *after* the result is delivered.

        Default is a no-op. Used by the end-conversation tools, which emit an
        exit message once the LLM has been given the tool result.
        """
        return None

    @property
    def properties(self) -> Dict[str, Any]:
        """
        Return the properties for the tool.

        An example of the properties:
        ```
        properties = {
            "location": {
                "type": "string",
                "description": "The city and state, e.g. San Francisco, CA",
            },
            "format": {
                    "type": "string",
                    "enum": ["celsius", "fahrenheit"],
                    "description": "The temperature unit to use.",
            },
        }
        ```
        """
        raise NotImplementedError("Subclasses must implement this method to return the properties for the tool.")

    @property
    def required_properties(self) -> List[str]:
        """
        Return the required properties for the tool.

        An example of the required properties:
        ```
        required_properties = ["location", "format"]
        ```
        """
        raise NotImplementedError(
            "Subclasses must implement this method to return the required properties for the tool."
        )

    async def _execute(self, **kwargs: Any) -> Any:
        """Run the tool and RETURN its result. Pure logic, no framework types.

        Receives the LLM-supplied call arguments as keyword arguments and
        returns the result to send back. Do **not** import or touch
        ``FunctionCallParams``, and do **not** deliver the result yourself —
        :meth:`__call__` owns delivery and calls it exactly once.

        Raising is fine: ``__call__`` converts an exception into a structured
        ``{"error": ...}`` result so the LLM gets a usable signal and the
        aggregator is never left waiting.

        An example of a get_current_weather tool:
        ```
        async def _execute(self, location: str, format: str = "celsius") -> dict:
            return {"location": location, "format": format}
        ```
        """
        raise NotImplementedError("Subclasses must implement this method to implement the tool logic.")


def _current_context_tool_names(context: Any) -> List[str]:
    """Extract the tool names the LLM actually sees in its current schema.

    ``context._tools`` (set by ``LLMContext.set_tools``) is the LLM's
    canonical view; ``llm._functions`` is the Python-side registry which
    accumulates entries across bootstrap + per-scenario RTVI re-registrations.
    The two diverge whenever ``register_schema_tools_to_llm`` is called with
    ``keep_existing_tools=False`` (the per-scenario path in
    ``rtvi_actions.create_update_system_prompt_action``): ``context._tools``
    gets fully replaced, but Python-side ``llm._functions`` still has the
    bootstrap entries hanging around. For an "unknown tool" message that
    actually helps the LLM self-correct, we need the LLM's view.

    Handles both shapes ``context._tools`` can take:
    - ``ToolsSchema(standard_tools=[FunctionSchema, ...])`` — set by our
      ``register_schema_tools_to_llm``.
    - ``List[ChatCompletionToolParam]`` — raw OpenAI dicts when the caller
      bypasses ``ToolsSchema``.

    Returns an empty list (NOT a fallback to the registry) when the context
    has no tools — the LLM was told it has no tools, so that's what we report.
    """
    tools = getattr(context, "_tools", None)
    if tools is None:
        return []
    standard_tools = getattr(tools, "standard_tools", None)
    if standard_tools:
        return [t.name for t in standard_tools if getattr(t, "name", None)]
    if isinstance(tools, list):
        names: List[str] = []
        for entry in tools:
            if isinstance(entry, dict):
                # OpenAI shape: {"type": "function", "function": {"name": "...", ...}}
                fn = entry.get("function") if "function" in entry else entry
                if isinstance(fn, dict) and "name" in fn:
                    names.append(fn["name"])
        return names
    return []


async def _unknown_tool_handler(params: FunctionCallParams) -> None:
    """Catch-all handler for tool names the LLM hallucinated.

    Pipecat's ``LLMService.run_function_calls`` broadcasts a
    ``FunctionCallsStartedFrame`` listing every call the LLM made — including
    ones whose names aren't registered. The downstream aggregator stuffs each
    ``tool_call_id`` into ``_function_calls_in_progress`` on receipt of that
    frame. The service then loops and ``continue``\\s past the unregistered call
    without producing a ``FunctionCallResultFrame``, so the aggregator's
    in-progress set stays non-empty forever — ``run_llm = not bool(...)``
    evaluates ``False`` for every subsequent message and the pipeline wedges.

    The escape hatch pipecat designed for this is to register a function with
    ``function_name=None`` — the service routes any unmatched name to it
    (``llm_service.py:449``). We give the LLM a structured error listing the
    **actual context-visible tools** so it can self-correct on the next turn.

    Why we read from ``context._tools`` and not ``llm._functions``: see the
    ``_current_context_tool_names`` docstring. Surfaced live on 2026-06-03
    during tau2_retail bring-up — the first iteration of this handler read
    ``llm._functions``, which still contained the bootstrap
    ``GetCityWeatherTool``. The agent then announced "My available tools are
    limited to functions like checking city weather", confusing the user
    instead of redirecting to the actual retail toolset.
    """
    available = sorted(_current_context_tool_names(params.context))
    logger.warning(
        f"Unknown tool '{params.function_name}' called by LLM; returning structured error. "
        f"Context-visible tools: {available}"
    )
    await params.result_callback(
        {
            "status": "error",
            "error_type": "unknown_tool",
            "message": (f"There is no tool named '{params.function_name}'. Use one of the available tools instead."),
            "available_tools": available,
        }
    )


def register_schema_tools_to_llm(
    llm: OpenAILLMService,
    context: LLMContext,
    tools: List[StandardSchemaTool],
    cancel_on_interruption: bool = True,
    keep_existing_tools: bool = True,
    register_unknown_tool_handler: bool = True,
) -> None:
    """
    Register standard schema tools to the LLM.
    Args:
        llm: The LLM service to use.
        context: The LLM context to use.
        tools: The list of tools to register.
        cancel_on_interruption: Whether to cancel an in-flight tool call when the
            user interrupts. Keep the ``True`` default for ordinary synchronous
            tools. In pipecat >=1.0 this flag does double duty: ``False`` also
            marks the tool **asynchronous**, meaning the LLM does not wait for
            the result — it continues the conversation immediately, the ``tool``
            message it receives is a ``{"status": "running"}`` placeholder, and
            the real payload is injected later as a ``developer`` message. Only
            pass ``False`` for a genuinely long-running, fire-and-forget tool
            whose result the model must not block on. (In pipecat 0.x the flag
            meant only "survive interruption"; there is no longer a way to get
            that behaviour without also opting into the async protocol.)
        keep_existing_tools: Whether to keep the existing tools in the context.
        register_unknown_tool_handler: When True (default), registers a catch-all
            handler under ``function_name=None`` so any hallucinated tool call
            gets a structured error result instead of wedging the aggregator.
            See ``_unknown_tool_handler`` docstring for the deadlock chain this
            avoids. Disable only if you've already registered a custom catch-all
            via ``llm.register_function(function_name=None, ...)``.
    """
    all_schemas = []
    for tool in tools:
        if not isinstance(tool, StandardSchemaTool):
            logger.warning(f"Tool {tool.__class__.__name__} is not a `StandardSchemaTool`, skipping...")
            continue
        all_schemas.append(tool.schema)
        logger.info(f"Registering standard schematool `{tool.name}` with schema properties: {tool.schema.properties}")
        llm.register_function(
            function_name=tool.name,
            handler=tool,
            cancel_on_interruption=cancel_on_interruption,
        )
    if register_unknown_tool_handler and None not in llm._functions:
        llm.register_function(
            function_name=None,
            handler=_unknown_tool_handler,
            cancel_on_interruption=cancel_on_interruption,
        )
        logger.info("Registered catch-all handler for unknown tool names (pipecat deadlock guard).")
    if keep_existing_tools:
        existing_tools = context.tools
        if not isinstance(existing_tools, list):
            existing_tools = []
        all_schemas.extend(existing_tools)
    tools_schema = ToolsSchema(standard_tools=all_schemas)
    context.set_tools(tools_schema)
