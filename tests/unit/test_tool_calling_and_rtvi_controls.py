# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Unit tests for generic tool-calling and RTVI control-tool behavior."""

import asyncio
from types import SimpleNamespace

import pytest

from nemo_voice_agent.evaluation.tools._write_tool_base import WriteScenarioTool
from nemo_voice_agent.evaluation.tools.rtvi_control import (
    EXIT_MESSAGE_END_TAG,
    EXIT_MESSAGE_START_TAG,
    FINAL_RESPONSE_END_TAG,
    FINAL_RESPONSE_START_TAG,
    SendExitMessageTool,
    SendRTVIMessageTool,
    SendScenarioSummaryTool,
)
from nemo_voice_agent.utils import tool_calling as registry
from nemo_voice_agent.utils.tool_calling.base import (
    StandardSchemaTool,
    _current_context_tool_names,
    _unknown_tool_handler,
    register_schema_tools_to_llm,
)
from nemo_voice_agent.utils.tool_calling.mixins import ToolCallingMixin, register_direct_tools_to_llm


class _CapturingRTVI:
    """Fake RTVI processor that records transport messages."""

    def __init__(self):
        """Initialize an empty message capture."""
        self.messages = []

    async def push_transport_message(self, message, exclude_none=True):
        """Record the message payload and exclude_none flag."""
        self.messages.append((message, exclude_none))


class _Params:
    """Duck-typed FunctionCallParams replacement for tool unit tests."""

    def __init__(self, arguments=None):
        """Create params with arguments and result capture."""
        self.arguments = arguments or {}
        self.results = []

    async def result_callback(self, result):
        """Capture one tool result."""
        self.results.append(result)


class _GoodTool(StandardSchemaTool):
    """Minimal successful StandardSchemaTool implementation."""

    def __init__(self, *, description="good", name=None):
        """Initialize the tool with a configurable name for schema assertions."""
        super().__init__(description=description, name=name)

    @property
    def properties(self):
        """Return one synthetic schema property."""
        return {"value": {"type": "string"}}

    @property
    def required_properties(self):
        """Return the required synthetic schema property."""
        return ["value"]

    async def _execute(self, **kwargs):
        """Echo the provided arguments."""
        return {"arguments": kwargs}


class _FailingTool(_GoodTool):
    """Tool implementation whose execution raises."""

    async def _execute(self, **kwargs):
        """Raise an error to exercise StandardSchemaTool.__call__ error handling."""
        raise RuntimeError("boom")


class _EmptyResultTool(_GoodTool):
    """Tool implementation returning a falsy result that must be normalized."""

    def __init__(self, empty_value, **kwargs):
        """Store the falsy value that _execute should return."""
        super().__init__(**kwargs)
        self._empty_value = empty_value

    async def _execute(self, **kwargs):
        """Return the configured falsy value without delivering it."""
        return self._empty_value


def test_standard_schema_tool_uses_class_name_and_builds_schema():
    """Tools default to their class name and expose a FunctionSchema-compatible schema."""
    tool = _GoodTool(description="A useful tool")

    schema = tool.schema

    assert tool.name == "_GoodTool"
    assert schema.name == "_GoodTool"
    assert schema.description == "A useful tool"
    assert schema.properties == {"value": {"type": "string"}}
    assert schema.required == ["value"]


def test_standard_schema_tool_call_emits_success_result():
    """Calling a StandardSchemaTool awaits _execute and forwards its result exactly once."""
    tool = _GoodTool()
    params = _Params(arguments={"value": "abc"})

    asyncio.run(tool(params))

    assert params.results == [{"arguments": {"value": "abc"}}]


def test_standard_schema_tool_call_wraps_execution_errors():
    """Execution errors are returned as structured error payloads instead of escaping."""
    tool = _FailingTool()
    params = _Params(arguments={"value": "abc"})

    asyncio.run(tool(params))

    assert params.results == [{"error": "boom"}]


@pytest.mark.parametrize(
    "empty_value, expected",
    [
        ([], {"status": "success", "results": [], "count": 0, "message": "No matching records found."}),
        ({}, {"status": "success", "results": {}, "message": "No matching records found."}),
        (None, {"status": "success", "result": None, "message": "Completed with no data returned."}),
        ("", {"status": "success", "result": "", "message": "Completed with no data returned."}),
    ],
)
def test_standard_schema_tool_call_normalizes_empty_results(empty_value, expected):
    """Falsy _execute results are wrapped in a non-falsy envelope so pipecat cannot mask them."""
    tool = _EmptyResultTool(empty_value)
    params = _Params(arguments={"value": "abc"})

    asyncio.run(tool(params))

    assert params.results == [expected]


def test_schema_tools_register_as_synchronous_by_default():
    """Registered tools must not trip pipecat's asynchronous-tool protocol.

    In pipecat >=1.0 ``cancel_on_interruption=False`` does double duty: besides
    surviving interruptions it marks the tool *asynchronous*, so the LLM is told
    not to wait for the result. The tool message it receives becomes a
    ``{"status": "running"}`` placeholder and the real payload arrives later as a
    ``developer`` message. Eval tools are synchronous DB lookups the agent must
    have in hand before it speaks, so the default must keep them synchronous.

    Asserted against a real ``LLMService`` rather than a double, because the
    behaviour lives in pipecat's ``_has_async_tools()`` — a fake would happily
    record the flag and prove nothing.
    """
    from pipecat.services.llm_service import LLMService

    class _FakeContext:
        tools = []

        def set_tools(self, tools_schema):
            self._tools = tools_schema

    llm = LLMService()
    register_schema_tools_to_llm(llm, _FakeContext(), [_GoodTool(name="good_tool")], keep_existing_tools=False)

    assert llm._has_async_tools() is False, (
        "schema tools registered as asynchronous — the LLM will answer from a "
        "'status: running' placeholder instead of the real tool result"
    )


def test_register_schema_tools_to_llm_sets_context_and_unknown_handler_once():
    """Schema registration installs valid tools, skips invalid objects, and registers the catch-all once."""
    tool = _GoodTool(name="good_tool")
    registered = []

    class _FakeLLM:
        """Fake LLM service that records registered function names."""

        def __init__(self):
            """Initialize an empty function registry."""
            self._functions = {}

        def register_function(self, function_name, handler, cancel_on_interruption=True):
            """Record a function registration in the same shape used by the production service."""
            self._functions[function_name] = handler
            registered.append((function_name, handler, cancel_on_interruption))

    class _FakeContext:
        """Fake OpenAI context that captures the latest tools schema."""

        tools = []

        def set_tools(self, tools_schema):
            """Capture the schema passed by register_schema_tools_to_llm."""
            self._tools = tools_schema

    llm = _FakeLLM()
    context = _FakeContext()

    register_schema_tools_to_llm(
        llm,
        context,
        [object(), tool],
        cancel_on_interruption=False,
        keep_existing_tools=False,
    )
    register_schema_tools_to_llm(llm, context, [tool], register_unknown_tool_handler=True)

    assert "good_tool" in llm._functions
    assert None in llm._functions
    assert [entry[0] for entry in registered].count(None) == 1
    assert context._tools.standard_tools[0].name == "good_tool"


def test_standard_schema_tool_registry_registers_lists_and_instantiates(monkeypatch):
    """The package-level registry decorator validates classes and returns configured instances."""
    saved_registry = dict(registry.ALL_STANDARD_SCHEMA_TOOLS)
    registry.ALL_STANDARD_SCHEMA_TOOLS.clear()

    try:
        registered = registry.register_standard_schema_tool(_GoodTool)

        assert registered is _GoodTool
        assert registry.list_standard_schema_tools() == ["_GoodTool"]
        assert isinstance(registry.get_standard_schema_tool("_GoodTool", description="configured"), _GoodTool)
        assert registry.get_standard_schema_tool("missing") is None
        with pytest.raises(ValueError, match="not a subclass"):
            registry.register_standard_schema_tool(object)
    finally:
        registry.ALL_STANDARD_SCHEMA_TOOLS.clear()
        registry.ALL_STANDARD_SCHEMA_TOOLS.update(saved_registry)


def test_current_context_tool_names_reads_tools_schema_and_openai_dicts():
    """Unknown-tool helpers report only the tools visible in the current LLM context."""
    tool = _GoodTool(name="visible")

    assert _current_context_tool_names(SimpleNamespace(_tools=SimpleNamespace(standard_tools=[tool.schema]))) == [
        "visible"
    ]
    assert _current_context_tool_names(
        SimpleNamespace(_tools=[{"type": "function", "function": {"name": "dict_tool"}}])
    ) == ["dict_tool"]
    assert _current_context_tool_names(SimpleNamespace()) == []


def test_unknown_tool_handler_returns_structured_context_visible_error():
    """The catch-all unknown-tool handler returns a correction payload instead of raising."""
    params = _Params()
    params.function_name = "hallucinated_tool"
    params.context = SimpleNamespace(_tools=[{"function": {"name": "real_tool"}}])

    asyncio.run(_unknown_tool_handler(params))

    assert params.results == [
        {
            "status": "error",
            "error_type": "unknown_tool",
            "message": "There is no tool named 'hallucinated_tool'. Use one of the available tools instead.",
            "available_tools": ["real_tool"],
        }
    ]


class _DirectToolMixin(ToolCallingMixin):
    """Minimal ToolCallingMixin subclass that registers one direct function."""

    def setup_tool_calling(self):
        """Register one fake direct function."""
        self.register_direct_function("direct_tool", _direct_tool)


async def _direct_tool(params, value: str = "default"):
    """Fake direct function registered into the LLM.

    Args:
        value: Synthetic value used to exercise direct-function schema extraction.
    """


async def _extra_direct_tool(params, value: str = "extra"):
    """Second fake direct function supplied without a mixin.

    Args:
        value: Synthetic value used to exercise direct-function schema extraction.
    """


def test_tool_calling_mixin_and_direct_tool_registration():
    """Direct-function registration combines mixin tools, explicit tools, and existing context tools."""
    mixin = _DirectToolMixin()
    mixin.setup_tool_calling()
    registered = []

    class _FakeLLM:
        """Fake LLM service that records direct-function registrations."""

        def register_direct_function(self, tool, cancel_on_interruption=True):
            """Record the direct function and cancellation setting."""
            registered.append((tool, cancel_on_interruption))

    class _FakeContext:
        """Fake LLM context with one existing direct tool."""

        tools = [_extra_direct_tool]

        def set_tools(self, tools_schema):
            """Capture the combined direct-tool schema."""
            self._tools = tools_schema

    context = _FakeContext()
    register_direct_tools_to_llm(
        llm=_FakeLLM(),
        context=context,
        tool_mixins=[object(), mixin],
        tools=[_extra_direct_tool],
        cancel_on_interruption=False,
    )

    assert mixin.available_tools == {"direct_tool": _direct_tool}
    assert [tool.name for tool in context._tools.standard_tools] == [
        "_direct_tool",
        "_extra_direct_tool",
        "_extra_direct_tool",
    ]
    assert registered == [
        (_direct_tool, False),
        (_extra_direct_tool, False),
        (_extra_direct_tool, False),
    ]


def test_register_direct_tools_to_llm_noops_without_tools():
    """Direct-tool registration returns without mutating the context when no tools are provided."""
    context = SimpleNamespace(tools=[])
    llm = SimpleNamespace(register_direct_function=lambda *args, **kwargs: None)

    assert register_direct_tools_to_llm(llm=llm, context=context) is None
    assert not hasattr(context, "_tools")


def test_send_rtvi_message_tool_sends_plain_message_and_ack():
    """SendRTVIMessageTool pushes the requested text and then acknowledges the call."""
    rtvi = _CapturingRTVI()
    tool = SendRTVIMessageTool(rtvi=rtvi)
    params = _Params(arguments={"message": "hello"})

    asyncio.run(tool(params))

    assert len(rtvi.messages) == 1
    assert rtvi.messages[0][0].data.text == "hello"
    assert rtvi.messages[0][1] is True
    assert params.results == [{"success": True, "message": "message sent to the RTVIclient."}]


def test_send_scenario_summary_tool_wraps_final_response_tags():
    """Scenario summaries are wrapped in final-response tags before transport."""
    rtvi = _CapturingRTVI()
    tool = SendScenarioSummaryTool(rtvi=rtvi)
    params = _Params(arguments={"message": '{"ok": true}'})

    asyncio.run(tool(params))

    assert len(rtvi.messages) == 1
    text = rtvi.messages[0][0].data.text
    assert text == f'{FINAL_RESPONSE_START_TAG}{{"ok": true}}{FINAL_RESPONSE_END_TAG}'
    assert params.results == [{"success": True, "message": "Scenario summary message sent."}]


def test_send_exit_message_tool_acknowledges_before_sending_exit():
    """Exit tool returns the function result before emitting the exit transport message."""
    rtvi = _CapturingRTVI()
    tool = SendExitMessageTool(rtvi=rtvi)
    params = _Params()
    messages_at_result_time = []
    deliver = params.result_callback

    async def _ordering_callback(result):
        """Snapshot how many transport messages existed when the result was delivered."""
        messages_at_result_time.append(len(rtvi.messages))
        await deliver(result)

    params.result_callback = _ordering_callback

    asyncio.run(tool(params))

    assert params.results == [{"success": True, "message": "Exit message sent."}]
    # The exit message is emitted from _after_result, i.e. strictly after delivery.
    assert messages_at_result_time == [0]
    assert len(rtvi.messages) == 1
    assert rtvi.messages[0][0].data.text == f"{EXIT_MESSAGE_START_TAG}The task is finished.{EXIT_MESSAGE_END_TAG}"


class _RecordingWriteTool(WriteScenarioTool):
    """Minimal WriteScenarioTool subclass for action-recording tests."""

    ACTION_TYPES = ["known_action"]

    def __init__(self, state):
        """Bind shared state for the write-tool base methods."""
        self.state = state
        super().__init__(description="records actions")

    @property
    def properties(self):
        """Return no tool-call properties."""
        return {}

    @property
    def required_properties(self):
        """Return no required properties."""
        return []

    async def _execute(self, **kwargs):
        """No-op execute path; tests call base helpers directly."""
        return {}


def test_write_scenario_tool_records_actions_and_call_counts():
    """WriteScenarioTool appends actions and maintains per-tool call counters."""
    state = {}
    tool = _RecordingWriteTool(state)

    tool._record_action({"action_type": "known_action", "name": "first"})

    assert state["actions"] == [{"action_type": "known_action", "name": "first"}]
    assert tool._next_call_index("refund") == 1
    assert tool._next_call_index("refund") == 2
    assert state["_call_counts"] == {"refund": 2}


def test_write_scenario_tool_skips_bad_rtvi_sentinel():
    """A non-RTVI sentinel in shared state does not prevent recording the action."""
    state = {"__rtvi__": object()}
    tool = _RecordingWriteTool(state)

    tool._record_action({"action_type": "unknown_action", "name": "still_recorded"})

    assert state["actions"] == [{"action_type": "unknown_action", "name": "still_recorded"}]


def test_write_scenario_tool_exit_message_noops_without_rtvi():
    """Sending an exit message is a no-op when unit tests do not provide RTVI."""
    tool = _RecordingWriteTool({})

    assert asyncio.run(tool._send_exit_message()) is None
