# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Unit coverage for NVIDIA LLM streaming and tool-call assembly."""

import asyncio
from types import SimpleNamespace

import httpx
from pipecat.frames.frames import LLMFullResponseEndFrame, LLMFullResponseStartFrame, LLMTextFrame
from pipecat.services.openai.llm import OpenAILLMService

from nemo_voice_agent.pipecat.services.nvidia_llm import NvidiaLLMService


class _Context:
    def __init__(self, messages):
        self.messages = messages

    def get_messages(self):
        return self.messages

    def set_messages(self, messages):
        self.messages = messages


def _chunk(content=None, tool_call=None, usage=None, choices=True):
    delta = SimpleNamespace(content=content, tool_calls=[tool_call] if tool_call else None)
    return SimpleNamespace(usage=usage, choices=[SimpleNamespace(delta=delta)] if choices else [])


async def _stream(values):
    for value in values:
        yield value


def _service(values, **attrs):
    service = NvidiaLLMService.__new__(NvidiaLLMService)
    service._name = "test-nvidia-llm"
    service._mistral_model_support = attrs.get("mistral", False)
    service._filter_think_tokens = attrs.get("filter_think", False)
    service._text_aggregator = None
    service._first_sentence_detected = False
    service._first_sentence_start_time = None
    service._prompt_tokens = service._completion_tokens = service._total_tokens = 0
    service._has_reported_prompt_tokens = False
    service._is_processing = False
    service._reset_think_filter_state()
    service.frames, service.calls = [], []

    async def noop(*args, **kwargs):
        return None

    async def push(frame, direction=None):
        service.frames.append(frame)

    async def completions(context):
        return _stream(values)

    async def has_function(name):
        return True

    async def call_function(**kwargs):
        service.calls.append(kwargs)

    service.start_ttfb_metrics = service.stop_ttfb_metrics = noop
    service.start_processing_metrics = service.stop_processing_metrics = noop
    service.stop_all_metrics = noop
    service.push_frame = push
    service._stream_chat_completions_universal_context = completions
    service.has_function = has_function
    service.call_function = call_function
    return service


def test_process_context_streams_content_usage_and_mistral(monkeypatch):
    usage = SimpleNamespace(prompt_tokens=5, completion_tokens=1, total_tokens=6)
    values = [
        _chunk(usage=usage, choices=False),
        _chunk(content="Hello."),
        SimpleNamespace(usage=None, choices=[SimpleNamespace(delta=None)]),
    ]
    service = _service(values, mistral=True)
    reported = []

    async def report(self, tokens):
        reported.append(tokens)

    monkeypatch.setattr(OpenAILLMService, "start_llm_usage_metrics", report)
    context = _Context(
        [{"role": "system", "content": "p"}, {"role": "user", "content": "a"}, {"role": "user", "content": "b"}]
    )
    asyncio.run(service._process_context(context))
    assert context.messages[-1]["content"] == "a b"
    assert [frame.text for frame in service.frames if isinstance(frame, LLMTextFrame)] == ["Hello."]
    assert reported[0].prompt_tokens == 5
    assert service._total_tokens == 6


def test_process_context_assembles_multiple_tool_calls():
    def tool(index, name=None, arguments=None, tool_id=None):
        return SimpleNamespace(index=index, id=tool_id, function=SimpleNamespace(name=name, arguments=arguments))

    values = [
        _chunk(tool_call=tool(0, "lookup", '{"x":', "id-1")),
        _chunk(tool_call=tool(0, "lookup", "1}", "id-1")),
        _chunk(tool_call=tool(1, "update", '{"y":2}', "id-2")),
    ]
    service = _service(values)
    asyncio.run(service._process_context(_Context([{"role": "user", "content": "go"}])))
    assert [call["function_name"] for call in service.calls] == ["lookup", "update"]
    assert service.calls[0]["arguments"] == {"x": 1}
    assert service.calls[1]["tool_call_id"] == "id-2"


def test_process_context_think_fallback_and_system_only():
    service = _service([_chunk(content="thinking only")], filter_think=True)
    asyncio.run(service._process_context(_Context([{"role": "user", "content": "go"}])))
    assert any(isinstance(frame, LLMTextFrame) and frame.text == "thinking only" for frame in service.frames)
    skipped = _service([], mistral=True)
    asyncio.run(skipped._process_context(_Context([{"role": "system", "content": "only"}])))
    assert skipped._is_processing is False


def test_process_context_and_frames_timeout_and_success():
    service = _service([])
    events = []

    async def event(name):
        events.append(name)

    async def timeout(context):
        raise httpx.ReadTimeout("slow")

    service._call_event_handler = event
    service._process_context = timeout
    asyncio.run(service._process_context_and_frames(_Context([])))
    assert events == ["on_completion_timeout"]
    assert isinstance(service.frames[0], LLMFullResponseStartFrame)
    assert isinstance(service.frames[-1], LLMFullResponseEndFrame)
