# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for the catch-all unknown-tool handler in register_schema_tools_to_llm.

This handler prevents a pipecat deadlock: when the LLM hallucinates a tool name,
``LLMService.run_function_calls`` broadcasts a ``FunctionCallsStartedFrame``
containing the (unregistered) tool_call_id BEFORE checking the registry. The
downstream aggregator marks the call as in-progress; the service then
``continue``s past it without emitting a ``FunctionCallResultFrame``; the
aggregator waits forever.

Surfaced live on 2026-06-03 during tau2_retail bring-up — the agent invented
``GetUserOrders`` (not in our 16-tool surface) and the pipeline stalled at
``FunctionCallsStartedFrame``.

Source-of-truth note: the handler reads tool names from ``context._tools``
(the LLM's actual schema set via ``set_tools``) — NOT from ``llm._functions``
(the Python registry). The two diverge after per-scenario re-registration
because ``keep_existing_tools=False`` replaces the context tools but doesn't
clear the function registry. First iteration of this handler read the
registry and surfaced bootstrap-time ``GetCityWeatherTool`` to the LLM during
a retail scenario, which the agent then echoed verbatim to the user.
"""

import asyncio
from types import SimpleNamespace

from nemo_voice_agent.utils.tool_calling.base import (
    _current_context_tool_names,
    _unknown_tool_handler,
)


class _FakeFunctionSchema:
    """Mimics ``pipecat.adapters.schemas.function_schema.FunctionSchema``."""

    def __init__(self, name: str):
        self.name = name


class _FakeToolsSchema:
    """Mimics ``pipecat.adapters.schemas.tools_schema.ToolsSchema``."""

    def __init__(self, names):
        self.standard_tools = [_FakeFunctionSchema(n) for n in names]


def _make_params(function_name: str, context_tools):
    """Build a duck-typed FunctionCallParams.

    ``context_tools`` is what we stuff into ``context._tools`` to simulate
    different registration shapes (ToolsSchema, list-of-dicts, None).
    """
    captured = {}

    async def result_callback(result):
        captured["result"] = result

    fake_context = SimpleNamespace(_tools=context_tools)
    return (
        SimpleNamespace(
            function_name=function_name,
            tool_call_id="tc-1",
            arguments={},
            llm=SimpleNamespace(_functions={}),  # deliberately empty — handler must NOT consult this
            context=fake_context,
            result_callback=result_callback,
        ),
        captured,
    )


# ---------------------------------------------------------------------------
# _current_context_tool_names helper
# ---------------------------------------------------------------------------


def test_extracts_names_from_tools_schema():
    """``context._tools`` is the ToolsSchema set by ``register_schema_tools_to_llm``."""
    schema = _FakeToolsSchema(["GetOrderDetailsTool", "CancelPendingOrderTool"])
    fake_context = SimpleNamespace(_tools=schema)
    assert _current_context_tool_names(fake_context) == ["GetOrderDetailsTool", "CancelPendingOrderTool"]


def test_extracts_names_from_openai_dict_list():
    """OpenAI-shaped ``[{"type": "function", "function": {"name": ..., ...}}, ...]``."""
    raw = [
        {"type": "function", "function": {"name": "ToolA", "parameters": {}}},
        {"type": "function", "function": {"name": "ToolB"}},
    ]
    fake_context = SimpleNamespace(_tools=raw)
    assert _current_context_tool_names(fake_context) == ["ToolA", "ToolB"]


def test_extracts_names_from_flat_dict_list():
    """Some adapters emit a flat ``[{"name": ..., ...}]`` shape — handle that too."""
    raw = [{"name": "FlatTool"}]
    fake_context = SimpleNamespace(_tools=raw)
    assert _current_context_tool_names(fake_context) == ["FlatTool"]


def test_empty_tools_returns_empty_list():
    """No tools in context → empty list (NOT a fallback to the registry)."""
    fake_context = SimpleNamespace(_tools=None)
    assert _current_context_tool_names(fake_context) == []
    fake_context._tools = _FakeToolsSchema([])
    assert _current_context_tool_names(fake_context) == []


def test_missing_attribute_returns_empty_list():
    """Context object without a ``_tools`` attribute at all — defensive return [] rather than raising."""
    assert _current_context_tool_names(SimpleNamespace()) == []


# ---------------------------------------------------------------------------
# _unknown_tool_handler
# ---------------------------------------------------------------------------


def test_unknown_tool_emits_structured_error_from_context_tools():
    """The handler must read from ``context._tools`` (LLM's actual view), NOT ``llm._functions``."""
    params, captured = _make_params(
        "GetUserOrders",
        context_tools=_FakeToolsSchema(["GetUserDetailsTool", "GetOrderDetailsTool", "ListAllProductTypesTool"]),
    )
    asyncio.run(_unknown_tool_handler(params))
    assert "result" in captured, "handler must emit a result via result_callback to unblock the aggregator"
    result = captured["result"]
    assert result["status"] == "error"
    assert result["error_type"] == "unknown_tool"
    assert "GetUserOrders" in result["message"]
    assert set(result["available_tools"]) == {
        "GetUserDetailsTool",
        "GetOrderDetailsTool",
        "ListAllProductTypesTool",
    }


def test_handler_ignores_llm_functions_registry():
    """Regression for the 2026-06-03 bug: even if ``llm._functions`` has stale entries
    (like ``GetCityWeatherTool`` from bootstrap), the handler must report only what
    the LLM was actually told it has access to via ``context._tools``.
    """
    params, captured = _make_params(
        "Hallucinated",
        context_tools=_FakeToolsSchema(["RealRetailTool"]),
    )
    # Stale bootstrap entry hangs around in llm._functions
    params.llm._functions["tool_get_city_weather"] = object()
    params.llm._functions[None] = object()  # the catch-all itself
    asyncio.run(_unknown_tool_handler(params))
    # Output should reflect only the context-visible tool, NOT the stale registry entries.
    assert captured["result"]["available_tools"] == ["RealRetailTool"]


def test_handler_sorts_available_tools_for_stable_output():
    """LLM nudges work better with deterministic ordering — verify sorted output."""
    params, captured = _make_params(
        "FakeName",
        context_tools=_FakeToolsSchema(["ZZZ_Tool", "AAA_Tool", "MMM_Tool"]),
    )
    asyncio.run(_unknown_tool_handler(params))
    assert captured["result"]["available_tools"] == ["AAA_Tool", "MMM_Tool", "ZZZ_Tool"]


def test_handler_with_no_context_tools_reports_empty_list():
    """If the LLM was told it has no tools, the handler reflects that honestly."""
    params, captured = _make_params(
        "Anything",
        context_tools=None,
    )
    asyncio.run(_unknown_tool_handler(params))
    assert captured["result"]["available_tools"] == []
    assert captured["result"]["error_type"] == "unknown_tool"
