# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Unit coverage for the custom websocket server transport."""

import asyncio
from types import SimpleNamespace

import pytest
from pipecat.frames.frames import EndFrame, InputAudioRawFrame

from nemo_voice_agent.pipecat.transports.network import websocket_server as ws_module
from nemo_voice_agent.pipecat.transports.network.websocket_server import (
    WebsocketServerInputTransport,
    WebsocketServerParams,
    WebsocketServerTransport,
)


class _Socket:
    def __init__(self, messages=(), *, error=None, closed=False):
        self.messages = iter(messages)
        self.error = error
        self.closed = closed
        self.remote_address = ("client", 1234)
        self.close_calls = 0

    def __aiter__(self):
        return self

    async def __anext__(self):
        if self.error:
            error, self.error = self.error, None
            raise error
        try:
            return next(self.messages)
        except StopIteration:
            raise StopAsyncIteration from None

    async def close(self):
        self.close_calls += 1
        self.closed = True


def _callbacks(events):
    async def record(name, *args):
        events.append((name, *args))

    return SimpleNamespace(
        on_client_connected=lambda socket: record("connected", socket),
        on_client_disconnected=lambda socket: record("disconnected", socket),
        on_session_timeout=lambda socket: record("timeout", socket),
        on_websocket_ready=lambda: record("ready"),
    )


def _input(serializer=None):
    transport = WebsocketServerInputTransport.__new__(WebsocketServerInputTransport)
    transport._name = "test-websocket-input"
    transport._params = SimpleNamespace(serializer=serializer, session_timeout=None)
    transport._callbacks = _callbacks([])
    transport._websocket = None
    transport._monitor_task = None
    transport.audio, transport.frames = [], []

    async def push_audio(frame):
        transport.audio.append(frame)

    async def push(frame):
        transport.frames.append(frame)

    transport.push_audio_frame = push_audio
    transport.push_frame = push
    return transport


def test_client_handler_deserializes_audio_frames_and_replaces_client():
    audio = InputAudioRawFrame(audio=b"pcm", sample_rate=16000, num_channels=1)
    end = EndFrame()

    class Serializer:
        async def deserialize(self, message):
            return {"audio": audio, "empty": None, "end": end}[message]

    service = _input(Serializer())
    events = []
    service._callbacks = _callbacks(events)
    old_socket, socket = _Socket(), _Socket(["audio", "empty", "end"])
    service._websocket = old_socket
    asyncio.run(service._client_handler(socket))
    assert service.audio == [audio] and service.frames == [end]
    assert old_socket.close_calls == 1 and socket.close_calls == 1
    assert [event[0] for event in events] == ["connected", "disconnected"]
    assert service._websocket is None


def test_client_handler_ignores_unserialized_messages_and_receive_errors():
    service = _input()
    events = []
    service._callbacks = _callbacks(events)
    socket = _Socket(["ignored"])
    asyncio.run(service._client_handler(socket))
    assert events[-1][0] == "disconnected"

    socket = _Socket(error=RuntimeError("receive failed"))
    asyncio.run(service._client_handler(socket))
    assert socket.closed


def test_monitor_and_server_handlers(monkeypatch):
    service = _input()
    events = []
    service._callbacks = _callbacks(events)

    async def no_sleep(delay):
        assert delay == 4

    monkeypatch.setattr(ws_module.asyncio, "sleep", no_sleep)
    socket = _Socket(closed=False)
    asyncio.run(service._monitor_websocket(socket, 4))
    assert events == [("timeout", socket)]
    asyncio.run(service._monitor_websocket(_Socket(closed=True), 4))

    async def cancelled(delay):
        raise asyncio.CancelledError

    monkeypatch.setattr(ws_module.asyncio, "sleep", cancelled)
    with pytest.raises(asyncio.CancelledError):
        asyncio.run(service._monitor_websocket(socket, 4))

    class Server:
        async def __aenter__(self):
            events.append(("serve-enter",))

        async def __aexit__(self, *args):
            events.append(("serve-exit",))

    service._host, service._port = "127.0.0.1", 9999
    service._stop_server_event = asyncio.Event()
    service._stop_server_event.set()
    monkeypatch.setattr(ws_module.websockets, "serve", lambda handler, host, port: Server())
    asyncio.run(service._server_task_handler())
    assert ("ready",) in events and ("serve-exit",) in events


def test_transport_lazily_builds_processors_and_forwards_events(monkeypatch):
    params = WebsocketServerParams()
    transport = WebsocketServerTransport(params, host="host", port=42, input_name="in", output_name="out")
    made = []

    def make_input(*args, **kwargs):
        made.append(("input", args, kwargs))
        return SimpleNamespace()

    class Output:
        def __init__(self, *args, **kwargs):
            made.append(("output", args, kwargs))
            self.connections = []

        async def set_client_connection(self, socket):
            self.connections.append(socket)

    monkeypatch.setattr(ws_module, "WebsocketServerInputTransport", make_input)
    monkeypatch.setattr(ws_module, "WebsocketServerOutputTransport", Output)
    assert transport.input() is transport.input()
    output = transport.output()
    assert output is transport.output()

    events = []

    async def call(name, *args):
        events.append((name, *args))

    transport._call_event_handler = call
    socket = _Socket()
    asyncio.run(transport._on_client_connected(socket))
    asyncio.run(transport._on_client_disconnected(socket))
    asyncio.run(transport._on_session_timeout(socket))
    asyncio.run(transport._on_websocket_ready())
    assert output.connections == [socket, None]
    assert [event[0] for event in events] == [
        "on_client_connected",
        "on_client_disconnected",
        "on_session_timeout",
        "on_websocket_ready",
    ]

    transport._output = None
    asyncio.run(transport._on_client_connected(socket))
    asyncio.run(transport._on_client_disconnected(socket))
    assert [item[0] for item in made] == ["input", "output"]


def test_input_lifecycle_initializes_and_stops_tasks(monkeypatch):
    calls = []

    async def base_call(self, frame=None):
        calls.append("base")

    monkeypatch.setattr(ws_module.BaseInputTransport, "start", base_call)
    monkeypatch.setattr(ws_module.BaseInputTransport, "stop", base_call)
    monkeypatch.setattr(ws_module.BaseInputTransport, "cancel", base_call)
    monkeypatch.setattr(ws_module.BaseInputTransport, "cleanup", base_call)

    serializer = SimpleNamespace(setup=lambda frame: _record_async(calls, "serializer"))
    service = _input(serializer)
    service._initialized = False
    service._server_task = None
    service._stop_server_event = asyncio.Event()
    service._transport = SimpleNamespace(cleanup=lambda: _record_async(calls, "transport-cleanup"))
    service.create_task = lambda coroutine: (coroutine.close(), SimpleNamespace())[1]
    service.set_transport_ready = lambda frame: _record_async(calls, "ready")
    service.cancel_task = lambda task: _record_async(calls, "cancel-task")
    service.wait_for_task = lambda task: _record_async(calls, "wait-task")

    asyncio.run(service.start(object()))
    asyncio.run(service.start(object()))
    assert calls.count("serializer") == 1 and calls.count("ready") == 1

    service._monitor_task = SimpleNamespace()
    asyncio.run(service.stop(object()))
    assert service._monitor_task is None and service._server_task is None
    assert service._stop_server_event.is_set()

    service._monitor_task = SimpleNamespace()
    service._server_task = SimpleNamespace()
    asyncio.run(service.cancel(object()))
    asyncio.run(service.cleanup())
    assert calls.count("cancel-task") == 3 and "transport-cleanup" in calls


async def _record_async(calls, value):
    calls.append(value)
